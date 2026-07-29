"""Deterministic, artifact-backed evidence rendering for one strut."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json
from .strut_metrics import read_metrics_csv, stable_frame
from .volume import AXIS_MAPPING, load_volume

EVIDENCE_SCHEMA_VERSION = "part2-strut-evidence/1.0.0"


def _save_image(
    image: np.ndarray,
    path: Path,
    *,
    title: str,
    overwrite: bool,
) -> dict[str, str]:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Evidence artifact exists; enable overwrite explicitly: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(image, cmap="gray", origin="lower")
    axes.set_title(title)
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(
        path,
        dpi=140,
        bbox_inches="tight",
        metadata={"Software": "llnl_nde"},
    )
    plt.close(figure)
    return {"path": str(path), "sha256": sha256_file(path)}


def _profile_for_strut(path: str | Path, strut_id: int) -> dict[str, Any]:
    payload = read_json_object(path)
    for profile in payload.get("profiles", []):
        if int(profile.get("strut_id", -1)) == strut_id:
            return profile
    raise KeyError(f"Strut ID {strut_id} is absent from profile artifact {path}")


def render_strut_evidence(
    ct_path: str | Path,
    localized_graph_path: str | Path,
    profiles_path: str | Path,
    output_directory: str | Path,
    *,
    strut_id: int,
    threshold: float,
    crop_margin_voxels: int = 8,
    metrics_path: str | Path | None = None,
    classifications_path: str | Path | None = None,
    thresholds_path: str | Path | None = None,
    specimen_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write local-frame CT views, a profile plot, and a hashed manifest.

    The local z axis is A-to-B.  Sampling raw CT for display is not a Stage 2
    metric recomputation and cannot alter the supplied classification.
    """

    if crop_margin_voxels < 1:
        raise ValueError("crop_margin_voxels must be positive")
    volume = load_volume(ct_path)
    graph = load_lattice_json(localized_graph_path)
    matches = np.flatnonzero(graph.edge_ids == int(strut_id))
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate strut ID {strut_id}")
    row = int(matches[0])
    endpoints = graph.node_positions_xyz[graph.edge_node_rows[row]]
    metric: dict[str, Any] | None = None
    if metrics_path is not None:
        matches_metrics = [
            item
            for item in read_metrics_csv(metrics_path)
            if int(item["strut_id"]) == int(strut_id)
        ]
        if len(matches_metrics) != 1:
            raise KeyError(f"Expected one Stage 2 metric row for strut {strut_id}")
        metric = matches_metrics[0]
    classification: dict[str, Any] | None = None
    if classifications_path is not None:
        classification_payload = read_json_object(classifications_path)
        matches_classifications = [
            item
            for item in classification_payload.get("classifications", [])
            if int(item.get("strut_id", -1)) == int(strut_id)
        ]
        if len(matches_classifications) != 1:
            raise KeyError(f"Expected one Stage 3 classification for strut {strut_id}")
        classification = matches_classifications[0]
        if not bool(classification.get("evidence_required")):
            raise ValueError(
                f"Strut {strut_id} is not a non-present/bent evidence target"
            )

    basis_x, basis_y, basis_z, length = stable_frame(endpoints[0], endpoints[1])
    center = 0.5 * (endpoints[0] + endpoints[1])
    half_width = float(
        metric["cuboid_half_width_voxels"]
        if metric is not None
        else crop_margin_voxels
    )
    local_x = np.arange(-np.ceil(half_width), np.ceil(half_width) + 1)
    local_y = np.arange(-np.ceil(half_width), np.ceil(half_width) + 1)
    axial_margin = 0.10 * length
    local_z = np.linspace(
        -0.5 * length - axial_margin,
        0.5 * length + axial_margin,
        int(np.ceil(1.20 * length)) + 1,
    )
    zz, yy, xx = np.meshgrid(local_z, local_y, local_x, indexing="ij")
    points = (
        center[None, None, None, :]
        + xx[..., None] * basis_x
        + yy[..., None] * basis_y
        + zz[..., None] * basis_z
    )
    coordinates_zyx = np.vstack(
        [points[..., 2].ravel(), points[..., 1].ravel(), points[..., 0].ravel()]
    )
    aligned = ndimage.map_coordinates(
        volume.array,
        coordinates_zyx,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).reshape(zz.shape)
    if not aligned.size:
        raise ValueError(f"Strut {strut_id} produced an empty aligned CT cuboid")
    destination = Path(output_directory).expanduser().resolve() / f"strut_{strut_id}"
    artifacts: dict[str, dict[str, str]] = {}
    projections = {
        "aligned_xy": np.max(aligned, axis=0),
        "aligned_xz": np.max(aligned, axis=1),
        "aligned_yz": np.max(aligned, axis=2),
    }
    for name, image in projections.items():
        artifacts[name] = _save_image(
            image,
            destination / f"{name}.png",
            title=f"Strut {strut_id}: {name} local-frame maximum projection",
            overwrite=overwrite,
        )
    # Preserve the original evidence lookup names while making the local-frame
    # semantics explicit for new Stage 3 consumers.
    artifacts["axial"] = dict(artifacts["aligned_xy"])
    artifacts["coronal"] = dict(artifacts["aligned_xz"])
    artifacts["sagittal"] = dict(artifacts["aligned_yz"])

    profile = _profile_for_strut(profiles_path, int(strut_id))
    profile_path = destination / "occupancy_profile.png"
    if profile_path.exists() and not overwrite:
        raise FileExistsError(
            f"Evidence artifact exists; enable overwrite explicitly: {profile_path}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(profile["axial_t"], profile["occupancy_profile"], marker="o")
    thresholds_payload: dict[str, Any] | None = None
    if thresholds_path is not None:
        thresholds_payload = read_json_object(thresholds_path)
        policy = thresholds_payload.get("policy", {})
        broken = policy.get("broken", {}) if isinstance(policy, dict) else {}
        if broken and metric is not None:
            central_values = np.asarray(profile["occupancy_profile"], dtype=np.float64)
            axial_values = np.asarray(profile["axial_t"], dtype=np.float64)
            core = central_values[(axial_values >= 0.20) & (axial_values <= 0.80)]
            if core.size:
                reference = float(np.quantile(core, float(broken["central_reference_quantile"])))
                axes.axhline(
                    float(broken["deficit_ratio"]) * reference,
                    color="tab:red",
                    linestyle="--",
                    label="broken deficient-slice cutoff",
                )
                axes.legend(loc="best")
    axes.axhline(0.0, color="black", linewidth=0.5)
    axes.set_ylim(-0.02, 1.02)
    axes.set_xlabel("normalized axial position")
    axes.set_ylabel("foreground fraction")
    axes.set_title(f"Strut {strut_id}: corridor occupancy at threshold {threshold:g}")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        profile_path,
        dpi=140,
        bbox_inches="tight",
        metadata={"Software": "llnl_nde"},
    )
    plt.close(figure)
    artifacts["occupancy_profile"] = {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
    }
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "gate": "pass",
        **({"specimen_id": specimen_id} if specimen_id is not None else {}),
        "strut_id": int(strut_id),
        "junction_ids": graph.edge_node_ids[row].tolist(),
        "endpoints_xyz": endpoints.tolist(),
        "local_frame": {
            "basis_x_xyz": basis_x.tolist(),
            "basis_y_xyz": basis_y.tolist(),
            "basis_z_xyz": basis_z.tolist(),
            "center_xyz": center.tolist(),
            "local_z_is_a_to_b": True,
            "axial_padding_fraction_total": 0.20,
            "half_width_voxels": half_width,
        },
        "threshold": float(threshold),
        "classification_policy": (
            thresholds_payload.get("policy")
            if thresholds_payload is not None
            else None
        ),
        "classification": classification,
        "stage_2_metrics": metric,
        "axis_mapping": AXIS_MAPPING,
        "artifacts": {
            name: {
                **metadata,
                "role": f"strut_evidence_{name}",
                "retention": "committed",
            }
            for name, metadata in artifacts.items()
        },
        "hashes": {
            "ct_sha256": sha256_file(volume.path),
            "localized_graph_sha256": graph.source_sha256,
            "profiles_sha256": sha256_file(profiles_path),
            **(
                {"metrics_sha256": sha256_file(metrics_path)}
                if metrics_path is not None
                else {}
            ),
            **(
                {"classifications_sha256": sha256_file(classifications_path)}
                if classifications_path is not None
                else {}
            ),
            **(
                {"thresholds_sha256": sha256_file(thresholds_path)}
                if thresholds_path is not None
                else {}
            ),
        },
        "provenance": {
            "sealed_labels_read": False,
            "metrics_recomputed": False,
            "classification_recomputed": False,
            "ct_resampled_for_visualization_only": True,
            "render_config_sha256": sha256_json(
                {
                    "threshold": float(threshold),
                    "crop_margin_voxels": int(crop_margin_voxels),
                    "half_width_voxels": half_width,
                    "axial_padding_fraction_total": 0.20,
                }
            ),
        },
        "warnings": [],
    }
    manifest_artifact = write_json_atomic(
        destination / "manifest.json",
        manifest,
        overwrite=overwrite,
    )
    # The production Stage 3 contract registers one repeatable
    # ``evidence_packets`` role per strut.  The image records remain inside the
    # immutable packet manifest for Stage 4, but are not separately registered
    # as top-level Stage 3 receipt artifacts.
    return {
        **manifest,
        "artifacts": {
            "manifest": {
                **manifest_artifact,
                "role": "evidence_packets",
                "retention": "committed",
            }
        },
        "hashes": {
            **manifest["hashes"],
            "manifest_sha256": manifest_artifact["sha256"],
        },
    }
