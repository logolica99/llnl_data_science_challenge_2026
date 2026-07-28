"""Deterministic, artifact-backed evidence rendering for one strut."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json
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
        metadata={"Software": "part2_core"},
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
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write three orthogonal CT views, a profile plot, and a hashed manifest."""

    if crop_margin_voxels < 1:
        raise ValueError("crop_margin_voxels must be positive")
    volume = load_volume(ct_path)
    graph = load_lattice_json(localized_graph_path)
    matches = np.flatnonzero(graph.edge_ids == int(strut_id))
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate strut ID {strut_id}")
    row = int(matches[0])
    endpoints = graph.node_positions_xyz[graph.edge_node_rows[row]]
    shape_xyz = np.asarray(volume.shape[::-1], dtype=int)
    low = np.maximum(
        np.floor(endpoints.min(axis=0) - crop_margin_voxels).astype(int), 0
    )
    high = np.minimum(
        np.ceil(endpoints.max(axis=0) + crop_margin_voxels + 1).astype(int),
        shape_xyz,
    )
    crop = np.asarray(
        volume.array[
            low[2] : high[2],
            low[1] : high[1],
            low[0] : high[0],
        ],
        dtype=np.float32,
    )
    if not crop.size:
        raise ValueError(f"Strut {strut_id} produced an empty CT crop")
    destination = Path(output_directory).expanduser().resolve() / f"strut_{strut_id}"
    artifacts: dict[str, dict[str, str]] = {}
    projections = {
        "axial": np.max(crop, axis=0),
        "coronal": np.max(crop, axis=1),
        "sagittal": np.max(crop, axis=2),
    }
    for name, image in projections.items():
        artifacts[name] = _save_image(
            image,
            destination / f"{name}.png",
            title=f"Strut {strut_id}: {name} maximum projection",
            overwrite=overwrite,
        )

    profile = _profile_for_strut(profiles_path, int(strut_id))
    profile_path = destination / "occupancy_profile.png"
    if profile_path.exists() and not overwrite:
        raise FileExistsError(
            f"Evidence artifact exists; enable overwrite explicitly: {profile_path}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(7, 4))
    axes.plot(profile["axial_t"], profile["occupancy_profile"], marker="o")
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
        metadata={"Software": "part2_core"},
    )
    plt.close(figure)
    artifacts["occupancy_profile"] = {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
    }
    manifest = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "gate": "pass",
        "strut_id": int(strut_id),
        "junction_ids": graph.edge_node_ids[row].tolist(),
        "endpoints_xyz": endpoints.tolist(),
        "crop_bounds_xyz": {
            "minimum": low.tolist(),
            "maximum_exclusive": high.tolist(),
        },
        "threshold": float(threshold),
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
        },
        "provenance": {
            "sealed_labels_read": False,
            "render_config_sha256": sha256_json(
                {
                    "threshold": float(threshold),
                    "crop_margin_voxels": int(crop_margin_voxels),
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
    manifest["artifacts"]["manifest"] = {
        **manifest_artifact,
        "role": "strut_evidence_manifest",
        "retention": "committed",
    }
    manifest["hashes"]["manifest_sha256"] = manifest_artifact["sha256"]
    return manifest
