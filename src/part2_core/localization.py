"""Independent full-resolution CT localization of registered lattice nodes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .artifacts import sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json
from .volume import AXIS_MAPPING, load_volume

LOCALIZATION_SCHEMA_VERSION = "part2-node-localization/1.0.0"
DEFAULT_LOCALIZATION_CONFIG: dict[str, Any] = {
    "patch_radius_voxels": 10,
    "search_radius_voxels": 8.0,
    "minimum_peak_radius_voxels": 1.0,
    "maximum_peak_radius_voxels": 12.0,
    "maximum_shift_voxels": 8.0,
    "peak_centroid_radius_voxels": 3.5,
    "ambiguity_exclusion_radius_voxels": 3.0,
    "maximum_second_to_first_peak_ratio": 0.95,
    "minimum_accepted_fraction": 0.95,
    "maximum_ambiguous_fraction": 0.05,
}


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_LOCALIZATION_CONFIG)
    if config:
        result.update(config)
    return result


def _localize_one(
    volume: np.ndarray,
    prediction_xyz: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    patch_radius = int(config["patch_radius_voxels"])
    search_radius = float(config["search_radius_voxels"])
    shape_xyz = np.asarray(volume.shape[::-1], dtype=np.int64)
    center = np.rint(prediction_xyz).astype(np.int64)
    low = np.maximum(center - patch_radius, 0)
    high = np.minimum(center + patch_radius + 1, shape_xyz)
    boundary_truncated = bool(
        np.any(low != center - patch_radius)
        or np.any(high != center + patch_radius + 1)
    )
    patch = np.asarray(
        volume[
            low[2] : high[2],
            low[1] : high[1],
            low[0] : high[0],
        ]
        >= threshold,
        dtype=bool,
    )
    if not patch.size or not patch.any():
        return prediction_xyz.copy(), {
            "accepted": False,
            "reason": "no_foreground",
            "peak_radius_voxels": 0.0,
            "second_peak_ratio": 1.0,
            "shift_voxels": 0.0,
            "boundary_truncated": boundary_truncated,
        }
    edt = ndimage.distance_transform_edt(patch)
    z_index, y_index, x_index = np.indices(patch.shape)
    global_x = x_index + low[0]
    global_y = y_index + low[1]
    global_z = z_index + low[2]
    distance_squared = (
        (global_x - prediction_xyz[0]) ** 2
        + (global_y - prediction_xyz[1]) ** 2
        + (global_z - prediction_xyz[2]) ** 2
    )
    search = distance_squared <= search_radius**2
    restricted = np.where(search, edt, -1.0)
    peak_index = np.unravel_index(int(np.argmax(restricted)), restricted.shape)
    peak_radius = float(restricted[peak_index])
    if peak_radius <= 0:
        return prediction_xyz.copy(), {
            "accepted": False,
            "reason": "no_peak_in_search_window",
            "peak_radius_voxels": peak_radius,
            "second_peak_ratio": 1.0,
            "shift_voxels": 0.0,
            "boundary_truncated": boundary_truncated,
        }
    peak_zyx = np.asarray(peak_index, dtype=np.float64)
    peak_distance_squared = (
        (x_index - peak_zyx[2]) ** 2
        + (y_index - peak_zyx[1]) ** 2
        + (z_index - peak_zyx[0]) ** 2
    )
    outside_primary = (
        peak_distance_squared > float(config["ambiguity_exclusion_radius_voxels"]) ** 2
    )
    second_peak = float(np.max(np.where(search & outside_primary, edt, 0.0)))
    ambiguity_ratio = second_peak / peak_radius
    centroid_region = (
        peak_distance_squared <= float(config["peak_centroid_radius_voxels"]) ** 2
    )
    weights = (
        np.clip(edt - (peak_radius - 1.0), 0.0, None) ** 2 * search * centroid_region
    )
    if float(weights.sum()) <= 0:
        location = np.asarray(
            [
                peak_index[2] + low[0],
                peak_index[1] + low[1],
                peak_index[0] + low[2],
            ],
            dtype=np.float64,
        )
    else:
        location = (
            np.asarray(
                [
                    np.sum(weights * global_x),
                    np.sum(weights * global_y),
                    np.sum(weights * global_z),
                ],
                dtype=np.float64,
            )
            / weights.sum()
        )
    shift = float(np.linalg.norm(location - prediction_xyz))
    reasons = []
    if not (
        float(config["minimum_peak_radius_voxels"])
        <= peak_radius
        <= float(config["maximum_peak_radius_voxels"])
    ):
        reasons.append("peak_radius_out_of_range")
    if shift > float(config["maximum_shift_voxels"]):
        reasons.append("shift_exceeds_limit")
    if ambiguity_ratio > float(config["maximum_second_to_first_peak_ratio"]):
        reasons.append("ambiguous_peak")
    accepted = not reasons
    return (location if accepted else prediction_xyz.copy()), {
        "accepted": accepted,
        "reason": "accepted" if accepted else ",".join(reasons),
        "peak_radius_voxels": peak_radius,
        "second_peak_ratio": ambiguity_ratio,
        "shift_voxels": shift,
        "boundary_truncated": boundary_truncated,
    }


def localize_lattice_nodes(
    ct_path: str | Path,
    registered_graph_path: str | Path,
    output_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    config: dict[str, Any] | None = None,
    registration_report_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recenter nodes independently and never collapse them to a global fit."""

    if registration_mode not in ("challenge_aligned_json", "autonomous_v2"):
        raise ValueError(f"Unsupported registration mode: {registration_mode}")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    merged = _config(config)
    volume = load_volume(ct_path)
    graph = load_lattice_json(registered_graph_path)
    localized = np.empty_like(graph.node_positions_xyz)
    records: list[dict[str, Any]] = []
    for row, (node_id, prediction) in enumerate(
        zip(graph.node_ids, graph.node_positions_xyz)
    ):
        location, details = _localize_one(
            volume.array, prediction, float(threshold), merged
        )
        localized[row] = location
        records.append(
            {
                "node_id": int(node_id),
                "coarse_xyz": prediction.tolist(),
                "localized_xyz": location.tolist(),
                **details,
            }
        )

    accepted = np.asarray([record["accepted"] for record in records], dtype=bool)
    ambiguous = np.asarray(
        ["ambiguous_peak" in record["reason"] for record in records],
        dtype=bool,
    )
    boundary = np.asarray(
        [record["boundary_truncated"] for record in records], dtype=bool
    )
    accepted_fraction = float(np.mean(accepted))
    ambiguous_fraction = float(np.mean(ambiguous))
    gates = {
        "accepted_fraction_sufficient": bool(
            accepted_fraction >= float(merged["minimum_accepted_fraction"])
        ),
        "ambiguity_fraction_within_limit": bool(
            ambiguous_fraction <= float(merged["maximum_ambiguous_fraction"])
        ),
        "all_localized_positions_finite": bool(np.isfinite(localized).all()),
    }
    if not all(gates.values()):
        gate = "halt"
    elif not np.all(accepted) or np.any(boundary):
        gate = "manual_review"
    else:
        gate = "pass"

    graph_artifact = write_json_atomic(
        output_graph_path,
        graph.document_with_positions(localized),
        overwrite=overwrite,
    )
    registration_report_hash = (
        sha256_file(registration_report_path)
        if registration_report_path is not None
        else None
    )
    hashes = {
        "ct_sha256": sha256_file(volume.path),
        "input_registered_graph_sha256": graph.source_sha256,
        "localized_graph_sha256": graph_artifact["sha256"],
    }
    if registration_report_hash:
        hashes["registration_report_sha256"] = registration_report_hash
    report = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate != "halt",
        "registration_mode": registration_mode,
        "threshold": float(threshold),
        "axis_mapping": AXIS_MAPPING,
        "counts": {
            **graph.counts,
            "accepted_nodes": int(np.count_nonzero(accepted)),
            "fallback_nodes": int(np.count_nonzero(~accepted)),
            "ambiguous_nodes": int(np.count_nonzero(ambiguous)),
            "boundary_truncated_nodes": int(np.count_nonzero(boundary)),
        },
        "localization": {
            "accepted_fraction": accepted_fraction,
            "ambiguous_fraction": ambiguous_fraction,
            "accepted_shift_voxels": {
                "median": (
                    float(
                        np.median(
                            [
                                record["shift_voxels"]
                                for record in records
                                if record["accepted"]
                            ]
                        )
                    )
                    if accepted.any()
                    else None
                ),
                "p95": (
                    float(
                        np.quantile(
                            [
                                record["shift_voxels"]
                                for record in records
                                if record["accepted"]
                            ],
                            0.95,
                        )
                    )
                    if accepted.any()
                    else None
                ),
            },
            "independent_positions_retained": True,
            "global_refit_performed": False,
        },
        "gates": gates,
        "records": records,
        "artifacts": {
            "localized_graph": {
                **graph_artifact,
                "role": "independently_localized_lattice_graph",
                "retention": "regenerable",
            }
        },
        "hashes": hashes,
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "sealed_labels_read": False,
        },
        "warnings": (
            []
            if gate == "pass"
            else [
                f"{int(np.count_nonzero(~accepted))} nodes retained coarse coordinates"
            ]
        ),
    }
    report_artifact = write_json_atomic(
        output_report_path,
        report,
        overwrite=overwrite,
    )
    report["artifacts"]["localization_report"] = {
        **report_artifact,
        "role": "node_localization_report",
        "retention": "committed",
    }
    report["hashes"]["localization_report_sha256"] = report_artifact["sha256"]
    return report
