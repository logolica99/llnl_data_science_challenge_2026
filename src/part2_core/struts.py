"""Per-strut padded-ROI measurements and deterministic classification."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .artifacts import (
    read_json_object,
    require_new_path,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from .lattice import load_lattice_json
from .sampling import longest_false_run, perpendicular_basis, sample_corridor
from .volume import AXIS_MAPPING, load_volume

STRUT_METRICS_SCHEMA_VERSION = "part2-strut-metrics/1.0.0"
CLASSIFICATION_SCHEMA_VERSION = "part2-strut-classification/1.0.0"

DEFAULT_METRICS_CONFIG: dict[str, Any] = {
    "axial_samples": 21,
    "corridor_radius_voxels": 6.0,
    "angular_samples": 12,
    "axial_padding_fraction": 0.2,
    "minimum_axial_foreground_fraction": 0.10,
    "endpoint_axial_sample_count": 3,
    "junction_mask_radius_voxels": 3.0,
    "minimum_valid_roi_fraction": 0.99,
    "radius_foreground_probability": 0.5,
    "centerline_smoothing_passes": 2,
}

METRIC_FIELDS = [
    "strut_id",
    "junction0_id",
    "junction1_id",
    "length_voxels",
    "corridor_foreground_fraction",
    "maximum_axial_gap_samples",
    "maximum_axial_gap_fraction",
    "endpoint0_support_fraction",
    "endpoint1_support_fraction",
    "interior_component_count",
    "largest_component_fraction",
    "edt_radius_median_voxels",
    "centerline_curvature_rms_voxels",
    "roi_in_bounds_fraction",
    "roi_valid",
]


def _metrics_config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_METRICS_CONFIG)
    if config:
        result.update(config)
    return result


def _finite_text(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}" if math.isfinite(float(value)) else ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value)


def _write_metrics_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> dict[str, str]:
    destination = require_new_path(path, overwrite)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite_text(row[key]) for key in METRIC_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": sha256_file(destination)}


def _local_connectivity_and_radius(
    volume_zyx: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    threshold: float,
    radius_voxels: float,
    junction_mask_radius_voxels: float,
    axial_samples: int,
) -> tuple[int, float, float]:
    """Measure connectivity only inside this strut's bounded local corridor."""

    margin = int(math.ceil(radius_voxels)) + 1
    shape_xyz = np.asarray(volume_zyx.shape[::-1], dtype=np.int64)
    low = np.maximum(np.floor(np.minimum(start_xyz, end_xyz) - margin).astype(int), 0)
    high = np.minimum(
        np.ceil(np.maximum(start_xyz, end_xyz) + margin + 1).astype(int),
        shape_xyz,
    )
    patch = np.asarray(
        volume_zyx[
            low[2] : high[2],
            low[1] : high[1],
            low[0] : high[0],
        ]
        >= threshold,
        dtype=bool,
    )
    if not patch.size:
        return 0, 0.0, 0.0
    zz, yy, xx = np.indices(patch.shape)
    points = np.stack(
        (xx + low[0], yy + low[1], zz + low[2]),
        axis=-1,
    ).astype(np.float64)
    direction = end_xyz - start_xyz
    length_squared = float(np.dot(direction, direction))
    t = np.clip(
        np.sum((points - start_xyz) * direction, axis=-1) / length_squared,
        0.0,
        1.0,
    )
    closest = start_xyz + t[..., None] * direction
    corridor = np.sum((points - closest) ** 2, axis=-1) <= radius_voxels**2
    from_start = np.linalg.norm(points - start_xyz, axis=-1)
    from_end = np.linalg.norm(points - end_xyz, axis=-1)
    interior = (
        patch
        & corridor
        & (from_start > junction_mask_radius_voxels)
        & (from_end > junction_mask_radius_voxels)
    )
    labels, component_count = ndimage.label(
        interior, structure=ndimage.generate_binary_structure(3, 3)
    )
    sizes = np.bincount(labels.ravel())[1:]
    foreground_count = int(np.count_nonzero(interior))
    largest_fraction = (
        float(sizes.max() / foreground_count)
        if foreground_count and sizes.size
        else 0.0
    )
    edt = ndimage.distance_transform_edt(patch)
    center_t = np.linspace(0.1, 0.9, axial_samples)
    centers = start_xyz[None, :] + center_t[:, None] * direction[None, :]
    indices = np.rint(centers).astype(int) - low
    valid = np.all((indices >= 0) & (indices < np.asarray(patch.shape[::-1])), axis=1)
    indices = indices[valid]
    radii = (
        edt[indices[:, 2], indices[:, 1], indices[:, 0]]
        if len(indices)
        else np.asarray([], dtype=float)
    )
    positive = radii[radii > 0]
    radius = float(np.median(positive)) if positive.size else 0.0
    return int(component_count), largest_fraction, radius


def _centerline_curvature(
    sample: dict[str, Any],
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    smoothing_passes: int,
) -> float:
    foreground = sample["foreground"]
    coordinates = sample["coordinates_xyz"]
    direction = end_xyz - start_xyz
    first, second = perpendicular_basis(direction)
    centroids: list[np.ndarray] = []
    valid_rows: list[bool] = []
    for coordinate_row, foreground_row in zip(coordinates, foreground):
        if not np.any(foreground_row):
            centroids.append(np.asarray([np.nan, np.nan]))
            valid_rows.append(False)
            continue
        offsets = coordinate_row[foreground_row] - coordinate_row[0]
        centroids.append(
            np.asarray(
                [
                    np.mean(offsets @ first),
                    np.mean(offsets @ second),
                ]
            )
        )
        valid_rows.append(True)
    points = np.asarray(centroids, dtype=np.float64)
    valid = np.asarray(valid_rows, dtype=bool)
    if np.count_nonzero(valid) < 5:
        return 0.0
    x = np.arange(len(points))
    for axis in range(2):
        points[~valid, axis] = np.interp(x[~valid], x[valid], points[valid, axis])
    for _ in range(max(0, int(smoothing_passes))):
        smoothed = points.copy()
        smoothed[1:-1] = (points[:-2] + 2.0 * points[1:-1] + points[2:]) / 4.0
        points = smoothed
    chord = np.linspace(points[0], points[-1], len(points))
    return float(np.sqrt(np.mean(np.sum((points - chord) ** 2, axis=1))))


def _legacy_compute_strut_metrics(
    ct_path: str | Path,
    localized_graph_path: str | Path,
    output_metrics_path: str | Path,
    output_profiles_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    config: dict[str, Any] | None = None,
    registration_qa_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute one ID-keyed record for every graph strut."""

    merged = _metrics_config(config)
    volume = load_volume(ct_path)
    graph = load_lattice_json(localized_graph_path)
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    for edge_id, endpoint_ids, start, end in zip(
        graph.edge_ids, graph.edge_node_ids, starts, ends
    ):
        sample = sample_corridor(
            volume.array,
            start,
            end,
            threshold=threshold,
            axial_samples=int(merged["axial_samples"]),
            radius_voxels=float(merged["corridor_radius_voxels"]),
            angular_samples=int(merged["angular_samples"]),
            axial_padding_fraction=float(merged["axial_padding_fraction"]),
        )
        foreground = sample["foreground"]
        valid = sample["valid"]
        radial_valid = np.maximum(valid.sum(axis=1), 1)
        occupancy_profile = foreground.sum(axis=1) / radial_valid
        design_span = (sample["axial_t"] >= 0.0) & (sample["axial_t"] <= 1.0)
        design_occupancy = occupancy_profile[design_span]
        design_foreground = foreground[design_span]
        design_valid = valid[design_span]
        supported = design_occupancy >= float(
            merged["minimum_axial_foreground_fraction"]
        )
        maximum_gap = longest_false_run(supported)
        endpoint_count = min(
            int(merged["endpoint_axial_sample_count"]),
            max(1, len(occupancy_profile) // 2),
        )
        component_count, largest_fraction, edt_radius = _local_connectivity_and_radius(
            volume.array,
            start,
            end,
            threshold=threshold,
            radius_voxels=float(merged["corridor_radius_voxels"]),
            junction_mask_radius_voxels=float(merged["junction_mask_radius_voxels"]),
            axial_samples=int(merged["axial_samples"]),
        )
        curvature = _centerline_curvature(
            sample,
            start,
            end,
            int(merged["centerline_smoothing_passes"]),
        )
        in_bounds_fraction = float(np.mean(valid))
        row = {
            "strut_id": int(edge_id),
            "junction0_id": int(endpoint_ids[0]),
            "junction1_id": int(endpoint_ids[1]),
            "length_voxels": float(np.linalg.norm(end - start)),
            "corridor_foreground_fraction": (
                float(design_foreground.sum() / design_valid.sum())
                if design_valid.sum()
                else 0.0
            ),
            "maximum_axial_gap_samples": maximum_gap,
            "maximum_axial_gap_fraction": maximum_gap / len(supported),
            "endpoint0_support_fraction": float(
                np.mean(design_occupancy[:endpoint_count])
            ),
            "endpoint1_support_fraction": float(
                np.mean(design_occupancy[-endpoint_count:])
            ),
            "interior_component_count": component_count,
            "largest_component_fraction": largest_fraction,
            "edt_radius_median_voxels": edt_radius,
            "centerline_curvature_rms_voxels": curvature,
            "roi_in_bounds_fraction": in_bounds_fraction,
            "roi_valid": in_bounds_fraction >= 1.0,
        }
        rows.append(row)
        profiles.append(
            {
                "strut_id": int(edge_id),
                "axial_t": sample["axial_t"].tolist(),
                "occupancy_profile": occupancy_profile.tolist(),
                "support_profile": supported.tolist(),
            }
        )

    metrics_artifact = _write_metrics_csv(
        output_metrics_path, rows, overwrite=overwrite
    )
    profile_payload = {
        "schema_version": STRUT_METRICS_SCHEMA_VERSION,
        "axis_mapping": AXIS_MAPPING,
        "threshold": float(threshold),
        "profiles": profiles,
    }
    profiles_artifact = write_json_atomic(
        output_profiles_path, profile_payload, overwrite=overwrite
    )
    roi_valid_fraction = float(np.mean([row["roi_valid"] for row in rows]))
    ids = [row["strut_id"] for row in rows]
    gates = {
        "one_row_per_graph_strut": len(rows) == graph.counts["edges"],
        "strut_ids_unique": len(ids) == len(set(ids)),
        "roi_valid_fraction_sufficient": bool(
            roi_valid_fraction >= float(merged["minimum_valid_roi_fraction"])
        ),
    }
    if not all(gates.values()):
        gate = "halt"
    elif roi_valid_fraction < 1.0:
        gate = "manual_review"
    else:
        gate = "pass"
    qa_hash = sha256_file(registration_qa_path) if registration_qa_path else None
    report = {
        "schema_version": STRUT_METRICS_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate != "halt",
        "registration_mode": registration_mode,
        "threshold": float(threshold),
        "axis_mapping": AXIS_MAPPING,
        "counts": {
            **graph.counts,
            "metric_rows": len(rows),
            "valid_rois": int(sum(bool(row["roi_valid"]) for row in rows)),
        },
        "summary": {
            "median_corridor_foreground_fraction": float(
                np.median([row["corridor_foreground_fraction"] for row in rows])
            ),
            "median_edt_radius_voxels": float(
                np.median([row["edt_radius_median_voxels"] for row in rows])
            ),
            "roi_valid_fraction": roi_valid_fraction,
        },
        "gates": gates,
        "artifacts": {
            "metrics": {
                **metrics_artifact,
                "role": "per_strut_metrics_csv",
                "retention": "committed",
            },
            "profiles": {
                **profiles_artifact,
                "role": "per_strut_axial_profiles",
                "retention": "regenerable",
            },
        },
        "hashes": {
            "ct_sha256": sha256_file(volume.path),
            "localized_graph_sha256": graph.source_sha256,
            "metrics_sha256": metrics_artifact["sha256"],
            "profiles_sha256": profiles_artifact["sha256"],
            **({"registration_qa_sha256": qa_hash} if qa_hash else {}),
        },
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "sealed_labels_read": False,
        },
        "warnings": (
            []
            if gate == "pass"
            else [
                f"{len(rows) - int(sum(row['roi_valid'] for row in rows))} invalid ROIs"
            ]
        ),
    }
    report_artifact = write_json_atomic(output_report_path, report, overwrite=overwrite)
    report["artifacts"]["metrics_report"] = {
        **report_artifact,
        "role": "strut_metrics_report",
        "retention": "committed",
    }
    report["hashes"]["metrics_report_sha256"] = report_artifact["sha256"]
    return report


def _legacy_read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = set(METRIC_FIELDS)
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Metrics CSV does not match production schema: {source}")
    parsed: list[dict[str, Any]] = []
    integer_fields = {
        "strut_id",
        "junction0_id",
        "junction1_id",
        "maximum_axial_gap_samples",
        "interior_component_count",
    }
    for row in rows:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if key in integer_fields:
                result[key] = int(value)
            elif key == "roi_valid":
                result[key] = value.lower() == "true"
            else:
                result[key] = float(value) if value else None
        parsed.append(result)
    return parsed


# The production export is the batched rotated-cuboid implementation.  The
# private legacy functions above remain temporarily readable for historical
# reproducibility, but no MCP or package export can select them.
from .strut_metrics import (  # noqa: E402
    METRIC_FIELDS as BATCHED_METRIC_FIELDS,
    compute_strut_metrics as compute_batched_strut_metrics,
    read_metrics_csv as read_batched_metrics_csv,
)

METRIC_FIELDS = BATCHED_METRIC_FIELDS
compute_strut_metrics = compute_batched_strut_metrics
read_metrics_csv = read_batched_metrics_csv


def _normalized_thresholds(value: dict[str, Any]) -> dict[str, float]:
    """Accept a compact flat policy or equivalent per-class dictionaries."""

    def get(flat: str, group: str, nested: str) -> float:
        selected = value.get(flat)
        if selected is None and isinstance(value.get(group), dict):
            selected = value[group].get(nested)
        if not isinstance(selected, (int, float)) or not math.isfinite(selected):
            raise ValueError(f"Missing finite classification threshold: {flat}")
        return float(selected)

    return {
        "missing_occupancy_max": get(
            "missing_occupancy_max", "missing", "maximum_occupancy"
        ),
        "missing_gap_fraction_min": get(
            "missing_gap_fraction_min", "missing", "minimum_gap_fraction"
        ),
        "broken_gap_fraction_min": get(
            "broken_gap_fraction_min", "broken", "minimum_gap_fraction"
        ),
        "broken_largest_component_fraction_max": get(
            "broken_largest_component_fraction_max",
            "broken",
            "maximum_largest_component_fraction",
        ),
        "thin_radius_max": get("thin_radius_max", "thin", "maximum_radius_voxels"),
        "bent_curvature_min": get(
            "bent_curvature_min", "bent", "minimum_curvature_rms_voxels"
        ),
    }


def classify_struts(
    metrics_path: str | Path,
    thresholds: dict[str, Any] | str | Path,
    output_classifications_path: str | Path,
    output_thresholds_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply frozen thresholds with precedence missing > broken > thin > present."""

    rows = read_metrics_csv(metrics_path)
    raw_thresholds = (
        read_json_object(thresholds)
        if isinstance(thresholds, (str, Path))
        else thresholds
    )
    if not isinstance(raw_thresholds, dict):
        raise ValueError("thresholds must be a JSON object")
    normalized = _normalized_thresholds(raw_thresholds)
    records: list[dict[str, Any]] = []
    counts = {"missing": 0, "broken": 0, "thin": 0, "present": 0}
    for row in rows:
        occupancy = float(row["corridor_foreground_fraction"] or 0.0)
        gap = float(row["maximum_axial_gap_fraction"] or 0.0)
        component = float(row["largest_component_fraction"] or 0.0)
        radius = float(row["edt_radius_median_voxels"] or 0.0)
        curvature = float(row["centerline_curvature_rms_voxels"] or 0.0)
        reasons: list[str] = []
        if (
            occupancy <= normalized["missing_occupancy_max"]
            and gap >= normalized["missing_gap_fraction_min"]
        ):
            label = "missing"
            reasons.extend(
                ["occupancy_below_missing_cutoff", "gap_above_missing_cutoff"]
            )
        elif (
            gap >= normalized["broken_gap_fraction_min"]
            or component <= normalized["broken_largest_component_fraction_max"]
        ):
            label = "broken"
            reasons.append("gap_or_connectivity_crossed_broken_cutoff")
        elif radius <= normalized["thin_radius_max"]:
            label = "thin"
            reasons.append("radius_below_thin_cutoff")
        else:
            label = "present"
            reasons.append("no_primary_defect_cutoff_crossed")
        bent = curvature >= normalized["bent_curvature_min"]
        counts[label] += 1
        records.append(
            {
                "strut_id": int(row["strut_id"]),
                "class": label,
                "bent": bent,
                "reasons": reasons,
            }
        )
    threshold_payload = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "thresholds_verbatim": raw_thresholds,
        "thresholds_normalized": normalized,
        "precedence": ["missing", "broken", "thin", "present"],
        "bent_is_non_competing_attribute": True,
    }
    threshold_artifact = write_json_atomic(
        output_thresholds_path, threshold_payload, overwrite=overwrite
    )
    classification_payload = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "gate": "pass",
        "overall_pass": True,
        "counts": {**counts, "total": len(records)},
        "classifications": records,
        "thresholds_sha256": threshold_artifact["sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "provenance": {
            "precedence": ["missing", "broken", "thin", "present"],
            "sealed_labels_read": False,
        },
    }
    classification_artifact = write_json_atomic(
        output_classifications_path,
        classification_payload,
        overwrite=overwrite,
    )
    return {
        **classification_payload,
        "artifacts": {
            "classifications": {
                **classification_artifact,
                "role": "classified_struts",
                "retention": "committed",
            },
            "thresholds": {
                **threshold_artifact,
                "role": "classification_thresholds",
                "retention": "committed",
            },
        },
        "hashes": {
            "metrics_sha256": classification_payload["metrics_sha256"],
            "thresholds_sha256": threshold_artifact["sha256"],
            "classifications_sha256": classification_artifact["sha256"],
        },
        "warnings": [],
    }
