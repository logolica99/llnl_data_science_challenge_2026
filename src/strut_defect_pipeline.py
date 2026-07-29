"""Deterministic thin/thick/bent CT strut measurement and artifact pipeline.

The registered graph is a soft spatial prior. Actual cross-section material is
tracked and segmented in the CT volume by ``strut_cross_section_viewer``.
This module turns those measurements into file hand-offs suitable for bounded
agents and exposes the same functions to the repository MCP server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from strut_cross_section_viewer import (
    detect_dense_boundary_limits,
    extract_cross_sections,
    load_registered_graph,
    make_basis,
)


DEFAULT_THRESHOLDS = {
    "schema_version": 1,
    "thin_radius_ratio_max": 0.78,
    "thick_radius_ratio_min": 1.30,
    "radius_robust_z_threshold": 3.5,
    "minimum_peer_group_size": 20,
    "minimum_valid_samples": 6,
    "minimum_tracking_coverage": 0.75,
    "minimum_tracking_confidence": 0.60,
    "maximum_junction_contamination_fraction": 0.20,
    "maximum_boundary_interference_fraction": 0.15,
    "maximum_interior_radius_cv": 0.25,
    "bent_centerline_rms_voxels": 0.75,
    "bent_centerline_max_voxels": 1.50,
    "bent_adjacent_deviation_voxels": 0.75,
    "bent_minimum_adjacent_samples": 2,
    "bent_curvature_rms_inverse_voxels": 0.15,
    "bent_priority_relative_margin": 0.10,
}

SECTION_FIELDS = [
    "strut_id",
    "sample_index",
    "axis_fraction",
    "distance_voxels",
    "radius_voxels",
    "radius_mm",
    "area_voxels_squared",
    "center_x_voxels",
    "center_y_voxels",
    "center_z_voxels",
    "tracked_center_u_voxels",
    "tracked_center_v_voxels",
    "sampling_plane_center_x_voxels",
    "sampling_plane_center_y_voxels",
    "sampling_plane_center_z_voxels",
    "local_tangent_x",
    "local_tangent_y",
    "local_tangent_z",
    "tracking_method",
    "centerline_deviation_voxels",
    "centerline_deviation_mm",
    "curvature_inverse_voxels",
    "tracking_confidence",
    "tracking_recovered",
    "valid",
    "exclusion_reason",
    "junction_excluded",
    "junction_contaminated",
    "dense_boundary_interference",
]

SUMMARY_FIELDS = [
    "strut_id",
    "unit_cell_edge_idx",
    "peer_group_id",
    "junction0",
    "junction1",
    "length_voxels",
    "length_mm",
    "midpoint_x_voxels",
    "midpoint_y_voxels",
    "midpoint_z_voxels",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "valid_sample_count",
    "median_radius_voxels",
    "min_radius_voxels",
    "max_radius_voxels",
    "median_radius_mm",
    "interior_radius_cv",
    "tracking_coverage",
    "mean_tracking_confidence",
    "junction_contamination_fraction",
    "dense_boundary_interference_fraction",
    "median_registration_offset_voxels",
    "centerline_deviation_rms_voxels",
    "centerline_deviation_max_voxels",
    "curvature_rms_inverse_voxels",
    "max_turn_angle_degrees",
    "measurement_quality",
]

CLASSIFICATION_FIELDS = [
    "strut_id",
    "classification",
    "is_thin",
    "is_thick",
    "is_bent",
    "decision_status",
    "confidence",
    "peer_group_id",
    "peer_sample_count",
    "median_radius_voxels",
    "peer_median_radius_voxels",
    "peer_robust_sigma_voxels",
    "radius_ratio",
    "radius_robust_z",
    "radius_evidence_strength",
    "bent_evidence_strength",
    "centerline_deviation_rms_voxels",
    "centerline_deviation_max_voxels",
    "curvature_rms_inverse_voxels",
    "tracking_coverage",
    "mean_tracking_confidence",
    "classification_priority_reason",
    "reason",
    "evidence_png",
]


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _as_float(value):
    if value in (None, ""):
        return None
    return float(value)


def _as_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _write_csv(path: Path, fields: list[str], rows: Iterable[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, root: Path):
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _prepare_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use overwrite=True only "
            "for an intentional rerun."
        )
    path.mkdir(parents=True, exist_ok=True)


def _exclusion_reason(section: dict, confidence_threshold: float):
    if section.get("dense_boundary_interference", False):
        return "dense_boundary"
    if section.get("junction_contaminated", False):
        return "junction_contaminated"
    if section.get("junction_excluded", False):
        return "near_junction"
    radius = section.get("equivalent_radius_voxels", float("nan"))
    if not math.isfinite(radius):
        return "tracking_lost"
    if float(section.get("tracking_confidence", 0.0)) < confidence_threshold:
        return "low_tracking_confidence"
    return ""


def _longest_true_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _max_turn_angle(sections: list[dict]):
    points = []
    for section in sections:
        if not section["_valid"]:
            continue
        global_point = np.asarray([
            section.get("tracked_center_x_voxels", float("nan")),
            section.get("tracked_center_y_voxels", float("nan")),
            section.get("tracked_center_z_voxels", float("nan")),
        ], dtype=float)
        if np.all(np.isfinite(global_point)):
            points.append(global_point)
        else:
            points.append(np.asarray([
                section["distance_voxels"],
                section["centroid_u_voxels"],
                section["centroid_v_voxels"],
            ], dtype=float))
    if len(points) < 3:
        return None
    vectors = np.diff(np.asarray(points), axis=0)
    angles = []
    for first, second in zip(vectors[:-1], vectors[1:]):
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator <= 0:
            continue
        cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosine)))
    return max(angles) if angles else None


def compute_strut_metrics(
    input_tiff: str | Path,
    registered_json: str | Path,
    output_dir: str | Path,
    threshold: float,
    *,
    positions: int = 11,
    start_fraction: float = 0.10,
    end_fraction: float = 0.90,
    ignore_edge_sections: int = 1,
    tracking_radius_voxels: float = 6.0,
    extent_pixels: float = 12.0,
    grid_size: int = 49,
    voxel_size_mm: float | None = None,
    strut_ids: Iterable[int] | None = None,
    max_struts: int | None = None,
    confidence_threshold: float = 0.45,
    overwrite: bool = False,
):
    """Measure actual CT radius and centerline profiles for registered struts."""
    started = time.time()
    input_tiff = Path(input_tiff).resolve()
    registered_json = Path(registered_json).resolve()
    output_dir = Path(output_dir).resolve()
    _prepare_output(output_dir, overwrite)
    if not input_tiff.is_file() or not registered_json.is_file():
        raise FileNotFoundError("Both input_tiff and registered_json must exist.")
    if positions < 5 or not (0 <= start_fraction < end_fraction <= 1):
        raise ValueError("Require at least 5 positions and 0 <= start < end <= 1.")
    if voxel_size_mm is not None and voxel_size_mm <= 0:
        raise ValueError("voxel_size_mm must be positive when supplied.")

    data, junctions = load_registered_graph(registered_json)
    volume = tifffile.memmap(input_tiff, mode="r")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D TIFF stack, got shape {volume.shape}.")
    safe_z_min, safe_z_max, boundary_fractions = detect_dense_boundary_limits(
        volume, threshold
    )
    selected_ids = None if strut_ids is None else {int(value) for value in strut_ids}
    scan_struts = [
        item for item in data["struts"]
        if selected_ids is None or int(item["id"]) in selected_ids
    ]
    if selected_ids is not None:
        found = {int(item["id"]) for item in scan_struts}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Unknown strut IDs: {missing[:10]}")
    if max_struts is not None:
        scan_struts = scan_struts[:max_struts]
    sample_positions = np.linspace(start_fraction, end_fraction, positions)

    section_rows = []
    summary_rows = []
    for strut in scan_struts:
        strut_id = int(strut["id"])
        start = junctions[int(strut["junction0"])]
        end = junctions[int(strut["junction1"])]
        direction, u, v, _ = make_basis(start, end)
        sections, length, rms_curvature = extract_cross_sections(
            volume,
            start,
            end,
            threshold,
            sample_positions,
            extent_pixels,
            grid_size,
            ignore_edge_sections,
            tracking_radius_voxels=tracking_radius_voxels,
            valid_z_range=(safe_z_min, safe_z_max),
        )
        radii = []
        deviations = []
        for sample_index, section in enumerate(sections):
            reason = _exclusion_reason(section, confidence_threshold)
            valid = not reason
            section["_valid"] = valid
            radius = _finite(section["equivalent_radius_voxels"])
            deviation = _finite(section["registration_residual_voxels"])
            if valid and radius is not None:
                radii.append(radius)
            if valid and deviation is not None:
                deviations.append(deviation)
            if radius is not None:
                ct_center = np.asarray([
                    section["tracked_center_x_voxels"],
                    section["tracked_center_y_voxels"],
                    section["tracked_center_z_voxels"],
                ], dtype=float)
            else:
                ct_center = np.full(3, np.nan)
            section_rows.append({
                "strut_id": strut_id,
                "sample_index": sample_index,
                "axis_fraction": section["position_fraction"],
                "distance_voxels": section["distance_voxels"],
                "radius_voxels": radius,
                "radius_mm": (
                    radius * voxel_size_mm
                    if radius is not None and voxel_size_mm is not None else None
                ),
                "area_voxels_squared": _finite(section["area_voxels_squared"]),
                "center_x_voxels": _finite(ct_center[0]),
                "center_y_voxels": _finite(ct_center[1]),
                "center_z_voxels": _finite(ct_center[2]),
                "tracked_center_u_voxels": _finite(section["centroid_u_voxels"]),
                "tracked_center_v_voxels": _finite(section["centroid_v_voxels"]),
                "sampling_plane_center_x_voxels": _finite(
                    section["sampling_plane_center_x_voxels"]
                ),
                "sampling_plane_center_y_voxels": _finite(
                    section["sampling_plane_center_y_voxels"]
                ),
                "sampling_plane_center_z_voxels": _finite(
                    section["sampling_plane_center_z_voxels"]
                ),
                "local_tangent_x": _finite(section["local_tangent_x"]),
                "local_tangent_y": _finite(section["local_tangent_y"]),
                "local_tangent_z": _finite(section["local_tangent_z"]),
                "tracking_method": section["tracking_method"],
                "centerline_deviation_voxels": deviation,
                "centerline_deviation_mm": (
                    deviation * voxel_size_mm
                    if deviation is not None and voxel_size_mm is not None else None
                ),
                "curvature_inverse_voxels": _finite(
                    section["curvature_inverse_voxels"]
                ),
                "tracking_confidence": _finite(section["tracking_confidence"]),
                "tracking_recovered": bool(
                    section.get("tracking_recovered", False)
                ),
                "valid": valid,
                "exclusion_reason": reason,
                "junction_excluded": bool(section["junction_excluded"]),
                "junction_contaminated": bool(section["junction_contaminated"]),
                "dense_boundary_interference": bool(
                    section["dense_boundary_interference"]
                ),
            })

        radii_array = np.asarray(radii, dtype=float)
        deviations_array = np.asarray(deviations, dtype=float)
        interior = [
            section for section in sections
            if not section.get("junction_excluded", False)
        ]
        coverage = float(sections[0]["tracking_coverage"])
        mean_confidence = float(sections[0]["mean_tracking_confidence"])
        contamination = (
            float(np.mean([
                section.get("junction_contaminated", False)
                for section in interior
            ]))
            if interior else 0.0
        )
        boundary = float(np.mean([
            section.get("dense_boundary_interference", False)
            for section in sections
        ]))
        radius_cv = (
            float(np.std(radii_array) / max(np.mean(radii_array), 1e-6))
            if radii_array.size >= 2 else None
        )
        minimum_valid = min(6, max(3, positions - 2 * ignore_edge_sections))
        quality = (
            "usable"
            if radii_array.size >= minimum_valid and coverage >= 0.55
            else "insufficient_tracking"
        )
        midpoint = (start + end) / 2.0
        edge_index = int(strut.get("unit_cell_edge_idx", -1))
        summary_rows.append({
            "strut_id": strut_id,
            "unit_cell_edge_idx": edge_index,
            "peer_group_id": f"unit_cell_edge_{edge_index}",
            "junction0": int(strut["junction0"]),
            "junction1": int(strut["junction1"]),
            "length_voxels": length,
            "length_mm": length * voxel_size_mm if voxel_size_mm else None,
            "midpoint_x_voxels": midpoint[0],
            "midpoint_y_voxels": midpoint[1],
            "midpoint_z_voxels": midpoint[2],
            "orientation_x": direction[0],
            "orientation_y": direction[1],
            "orientation_z": direction[2],
            "valid_sample_count": int(radii_array.size),
            "median_radius_voxels": (
                float(np.median(radii_array)) if radii_array.size else None
            ),
            "min_radius_voxels": (
                float(np.min(radii_array)) if radii_array.size else None
            ),
            "max_radius_voxels": (
                float(np.max(radii_array)) if radii_array.size else None
            ),
            "median_radius_mm": (
                float(np.median(radii_array)) * voxel_size_mm
                if radii_array.size and voxel_size_mm else None
            ),
            "interior_radius_cv": radius_cv,
            "tracking_coverage": coverage,
            "mean_tracking_confidence": mean_confidence,
            "junction_contamination_fraction": contamination,
            "dense_boundary_interference_fraction": boundary,
            "median_registration_offset_voxels": _finite(
                sections[0]["median_registration_offset_voxels"]
            ),
            "centerline_deviation_rms_voxels": (
                float(np.sqrt(np.mean(deviations_array ** 2)))
                if deviations_array.size else None
            ),
            "centerline_deviation_max_voxels": (
                float(np.max(deviations_array)) if deviations_array.size else None
            ),
            "curvature_rms_inverse_voxels": _finite(rms_curvature),
            "max_turn_angle_degrees": _max_turn_angle(sections),
            "measurement_quality": quality,
        })

    sections_path = output_dir / "strut_section_measurements.csv"
    summary_path = output_dir / "strut_summary.csv"
    manifest_path = output_dir / "measurement_manifest.json"
    _write_csv(sections_path, SECTION_FIELDS, section_rows)
    _write_csv(summary_path, SUMMARY_FIELDS, summary_rows)
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "stage": "strut_measurement",
        "inputs": {
            "input_tiff": str(input_tiff),
            "input_tiff_sha256": _sha256(input_tiff),
            "registered_json": str(registered_json),
            "registered_json_sha256": _sha256(registered_json),
        },
        "config": {
            "threshold": float(threshold),
            "positions": int(positions),
            "start_fraction": float(start_fraction),
            "end_fraction": float(end_fraction),
            "ignore_edge_sections": int(ignore_edge_sections),
            "tracking_radius_voxels": float(tracking_radius_voxels),
            "extent_pixels": float(extent_pixels),
            "grid_size": int(grid_size),
            "voxel_size_mm": voxel_size_mm,
            "axis_mapping": "[x,y,z] -> volume[z,y,x]",
            "registered_json_role": "soft spatial prior",
            "radius_definition": "sqrt(segmented_cross_section_area/pi)",
            "tracking_method": "3d_centerline_local_tangent",
            "bootstrap_method": "registered_axis_continuity_path",
            "centerline_deviation_definition": (
                "orthogonal distance from tracked CT center to robust "
                "best-fit straight 3D CT line"
            ),
        },
        "volume": {
            "shape_zyx": list(map(int, volume.shape)),
            "dtype": str(volume.dtype),
            "dense_boundary_safe_z_min": safe_z_min,
            "dense_boundary_safe_z_max": safe_z_max,
            "dense_boundary_peak_fraction": float(np.max(boundary_fractions)),
        },
        "counts": {
            "struts": len(summary_rows),
            "section_rows": len(section_rows),
            "valid_sections": sum(bool(row["valid"]) for row in section_rows),
        },
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            "section_measurements": _artifact(sections_path, output_dir),
            "strut_summary": _artifact(summary_path, output_dir),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "status": "ready",
        "manifest_path": str(manifest_path),
        "strut_count": len(summary_rows),
        "section_count": len(section_rows),
        "manifest_sha256": _sha256(manifest_path),
    }


def _load_thresholds(path: str | Path | None):
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path is not None:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(supplied) - set(thresholds))
        if unknown:
            raise ValueError(f"Unknown threshold keys: {unknown}")
        thresholds.update(supplied)
    return thresholds


def _tracking_trustworthy(row, thresholds):
    return (
        row["measurement_quality"] == "usable"
        and row["valid_sample_count"] >= thresholds["minimum_valid_samples"]
        and row["tracking_coverage"] >= thresholds["minimum_tracking_coverage"]
        and row["mean_tracking_confidence"]
        >= thresholds["minimum_tracking_confidence"]
        and row["junction_contamination_fraction"]
        <= thresholds["maximum_junction_contamination_fraction"]
        and row["dense_boundary_interference_fraction"]
        <= thresholds["maximum_boundary_interference_fraction"]
    )


def _radius_trustworthy(row, thresholds):
    return (
        _tracking_trustworthy(row, thresholds)
        and row["interior_radius_cv"] is not None
        and row["interior_radius_cv"]
        <= thresholds["maximum_interior_radius_cv"]
    )


def _typed_summary(row):
    integer_fields = {
        "strut_id", "unit_cell_edge_idx", "junction0", "junction1",
        "valid_sample_count",
    }
    text_fields = {"peer_group_id", "measurement_quality"}
    result = {}
    for key, value in row.items():
        if key in integer_fields:
            result[key] = _as_int(value)
        elif key in text_fields:
            result[key] = value
        else:
            result[key] = _as_float(value)
    return result


def _derive_peer_baselines(rows, thresholds):
    groups = defaultdict(list)
    for row in rows:
        if (
            _radius_trustworthy(row, thresholds)
            and row["median_radius_voxels"] is not None
        ):
            groups[row["peer_group_id"]].append(row["median_radius_voxels"])
    baselines = {}
    minimum = int(thresholds["minimum_peer_group_size"])
    for group_id, values in groups.items():
        values = np.asarray(values, dtype=float)
        if values.size < minimum:
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        initial_sigma = max(1.4826 * mad, 0.10 * median, 0.20)
        core = values[np.abs(values - median) <= 4.5 * initial_sigma]
        if core.size < minimum:
            continue
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median)))
        baselines[group_id] = {
            "median_radius_voxels": median,
            "robust_sigma_voxels": max(1.4826 * mad, 0.10 * median, 0.20),
            "sample_count": int(core.size),
        }
    return baselines


def _measurement_provenance(strut_sections_csv):
    manifest_path = strut_sections_csv.parent / "measurement_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    artifact = manifest.get("artifacts", {}).get("section_measurements", {})
    return {
        "schema_version": 1,
        "source": "thin_thick_bent_pipeline",
        "ct_threshold": _finite(config.get("threshold")),
        "positions": _as_int(config.get("positions")),
        "start_fraction": _finite(config.get("start_fraction")),
        "end_fraction": _finite(config.get("end_fraction")),
        "tracking_radius_voxels": _finite(
            config.get("tracking_radius_voxels")
        ),
        "tracking_method": config.get("tracking_method"),
        "centerline_deviation_definition": config.get(
            "centerline_deviation_definition"
        ),
        "section_measurements_sha256": (
            artifact.get("sha256") or _sha256(strut_sections_csv)
        ),
    }


def _typed_profile_sample(row):
    return {
        "sample_index": _as_int(row.get("sample_index")),
        "fraction": _as_float(row.get("axis_fraction")),
        "distance_voxels": _as_float(row.get("distance_voxels")),
        "radius_voxels": _as_float(row.get("radius_voxels")),
        "radius_mm": _as_float(row.get("radius_mm")),
        "area_voxels_squared": _as_float(row.get("area_voxels_squared")),
        "center_x_voxels": _as_float(row.get("center_x_voxels")),
        "center_y_voxels": _as_float(row.get("center_y_voxels")),
        "center_z_voxels": _as_float(row.get("center_z_voxels")),
        "center_u_voxels": _as_float(row.get("tracked_center_u_voxels")),
        "center_v_voxels": _as_float(row.get("tracked_center_v_voxels")),
        "sampling_plane_center_x_voxels": _as_float(
            row.get("sampling_plane_center_x_voxels")
        ),
        "sampling_plane_center_y_voxels": _as_float(
            row.get("sampling_plane_center_y_voxels")
        ),
        "sampling_plane_center_z_voxels": _as_float(
            row.get("sampling_plane_center_z_voxels")
        ),
        "local_tangent_x": _as_float(row.get("local_tangent_x")),
        "local_tangent_y": _as_float(row.get("local_tangent_y")),
        "local_tangent_z": _as_float(row.get("local_tangent_z")),
        "tracking_method": row.get("tracking_method", ""),
        "deviation_voxels": _as_float(
            row.get("centerline_deviation_voxels")
        ),
        "curvature_inverse_voxels": _as_float(
            row.get("curvature_inverse_voxels")
        ),
        "confidence": _as_float(row.get("tracking_confidence")),
        "tracking_recovered": _as_bool(row.get("tracking_recovered")),
        "valid": _as_bool(row.get("valid")),
        "exclusion_reason": row.get("exclusion_reason", ""),
        "junction_excluded": _as_bool(row.get("junction_excluded")),
        "junction_contaminated": _as_bool(
            row.get("junction_contaminated")
        ),
        "dense_boundary_interference": _as_bool(
            row.get("dense_boundary_interference")
        ),
    }


def classify_struts(
    strut_summary_csv: str | Path,
    strut_sections_csv: str | Path,
    output_dir: str | Path,
    *,
    thresholds_json: str | Path | None = None,
    overwrite: bool = False,
):
    """Apply frozen thin/thick/bent thresholds to deterministic metrics."""
    started = time.time()
    strut_summary_csv = Path(strut_summary_csv).resolve()
    strut_sections_csv = Path(strut_sections_csv).resolve()
    output_dir = Path(output_dir).resolve()
    _prepare_output(output_dir, overwrite)
    thresholds = _load_thresholds(thresholds_json)
    rows = [_typed_summary(row) for row in _read_csv(strut_summary_csv)]
    sections_by_strut = defaultdict(list)
    for row in _read_csv(strut_sections_csv):
        sections_by_strut[int(row["strut_id"])].append(
            _typed_profile_sample(row)
        )
    measurement_provenance = _measurement_provenance(strut_sections_csv)
    baselines = _derive_peer_baselines(rows, thresholds)
    classified = []
    findings = {"thin": [], "thick": [], "bent": []}
    for row in rows:
        baseline = baselines.get(row["peer_group_id"])
        tracking_trustworthy = _tracking_trustworthy(row, thresholds)
        radius_trustworthy = _radius_trustworthy(row, thresholds)
        radius = row["median_radius_voxels"]
        ratio = z_score = None
        is_thin = is_thick = False
        if baseline is not None and radius is not None and radius_trustworthy:
            ratio = radius / max(baseline["median_radius_voxels"], 1e-9)
            z_score = (
                radius - baseline["median_radius_voxels"]
            ) / baseline["robust_sigma_voxels"]
            is_thin = (
                ratio <= thresholds["thin_radius_ratio_max"]
                and z_score <= -thresholds["radius_robust_z_threshold"]
            )
            is_thick = (
                ratio >= thresholds["thick_radius_ratio_min"]
                and z_score >= thresholds["radius_robust_z_threshold"]
            )

        profiles = sorted(
            sections_by_strut.get(row["strut_id"], []),
            key=lambda item: item["fraction"],
        )
        deviation_mask = [
            item["valid"]
            and item["deviation_voxels"] is not None
            and item["deviation_voxels"]
            >= thresholds["bent_adjacent_deviation_voxels"]
            for item in profiles
        ]
        adjacent_run = _longest_true_run(deviation_mask)
        adjacent_bend = (
            adjacent_run
            >= thresholds["bent_minimum_adjacent_samples"]
        )
        rms_deviation = row["centerline_deviation_rms_voxels"]
        max_deviation = row["centerline_deviation_max_voxels"]
        curvature = row["curvature_rms_inverse_voxels"]
        strict_bent = bool(
            tracking_trustworthy
            and adjacent_bend
            and max_deviation is not None
            and max_deviation >= thresholds["bent_centerline_max_voxels"]
            and (
                (
                    rms_deviation is not None
                    and rms_deviation
                    >= thresholds["bent_centerline_rms_voxels"]
                )
                or (
                    curvature is not None
                    and curvature
                    >= thresholds["bent_curvature_rms_inverse_voxels"]
                )
            )
        )

        radius_strength = None
        if is_thin:
            radius_strength = min(
                thresholds["thin_radius_ratio_max"] / max(ratio, 1e-9),
                abs(z_score) / max(
                    thresholds["radius_robust_z_threshold"], 1e-9
                ),
            )
        elif is_thick:
            radius_strength = min(
                ratio / max(thresholds["thick_radius_ratio_min"], 1e-9),
                abs(z_score) / max(
                    thresholds["radius_robust_z_threshold"], 1e-9
                ),
            )
        bend_strength = None
        if (
            tracking_trustworthy
            and adjacent_bend
            and max_deviation is not None
        ):
            shape_support = max(
                (
                    rms_deviation
                    / max(thresholds["bent_centerline_rms_voxels"], 1e-9)
                    if rms_deviation is not None else 0.0
                ),
                (
                    curvature
                    / max(
                        thresholds["bent_curvature_rms_inverse_voxels"], 1e-9
                    )
                    if curvature is not None else 0.0
                ),
            )
            bend_strength = min(
                max_deviation
                / max(thresholds["bent_centerline_max_voxels"], 1e-9),
                shape_support,
                adjacent_run
                / max(thresholds["bent_minimum_adjacent_samples"], 1),
            )

        priority_promoted_bend = bool(
            (is_thin or is_thick)
            and bend_strength is not None
            and radius_strength is not None
            and bend_strength >= radius_strength * (
                1.0 - thresholds["bent_priority_relative_margin"]
            )
        )
        is_bent = strict_bent or priority_promoted_bend
        labels = [
            label for label, flag in (
                ("thin", is_thin), ("thick", is_thick), ("bent", is_bent)
            ) if flag
        ]
        priority_reason = ""
        if labels:
            radius_flag = is_thin or is_thick
            bent_priority = (
                is_bent
                and radius_flag
                and bend_strength is not None
                and radius_strength is not None
                and bend_strength >= radius_strength * (
                    1.0 - thresholds["bent_priority_relative_margin"]
                )
            )
            if bent_priority:
                classification = "bent"
                priority_reason = (
                    "bent selected as primary because its "
                    + (
                        "near-threshold " if priority_promoted_bend
                        and not strict_bent else ""
                    )
                    + "evidence strength "
                    f"({bend_strength:.3f}) is within "
                    f"{100 * thresholds['bent_priority_relative_margin']:.1f}% "
                    "of or exceeds the radius-defect evidence strength "
                    f"({radius_strength:.3f}); independent defect flags retained"
                )
            else:
                classification = "+".join(labels)
            decision_status = "candidate"
        elif not tracking_trustworthy:
            classification = "uncertain"
            decision_status = "insufficient_quality"
        elif not radius_trustworthy:
            classification = "uncertain"
            decision_status = "insufficient_radius_quality"
        elif baseline is None:
            classification = "uncertain"
            decision_status = "insufficient_peers"
        else:
            classification = "normal"
            decision_status = "classified"

        scores = []
        if radius_strength is not None:
            scores.append(min(1.0, radius_strength))
        if bend_strength is not None:
            scores.append(min(1.0, bend_strength))
        confidence = (
            min(1.0, max(scores) * row["mean_tracking_confidence"])
            if scores else row["mean_tracking_confidence"]
        )
        reasons = []
        if is_thin or is_thick:
            reasons.append(
                f"median radius ratio {ratio:.3f} and robust z {z_score:.3f} "
                f"versus {row['peer_group_id']}"
            )
        if is_bent:
            reasons.append(
                f"centerline RMS/max deviation {rms_deviation:.3f}/"
                f"{max_deviation:.3f} voxels with adjacent support"
            )
        if priority_reason:
            reasons.append(priority_reason)
        if decision_status == "insufficient_peers":
            reasons.append("peer group did not meet the minimum baseline size")
        elif decision_status == "insufficient_quality":
            reasons.append("measurement quality did not satisfy classification gates")
        elif decision_status == "insufficient_radius_quality":
            reasons.append(
                "centerline tracking was usable but radius variation did not "
                "satisfy the radius-classification gate"
            )
        elif classification == "normal":
            reasons.append("radius and centerline metrics are within frozen thresholds")

        result = {
            "strut_id": row["strut_id"],
            "classification": classification,
            "is_thin": is_thin,
            "is_thick": is_thick,
            "is_bent": is_bent,
            "decision_status": decision_status,
            "confidence": round(float(confidence), 6),
            "peer_group_id": row["peer_group_id"],
            "peer_sample_count": baseline["sample_count"] if baseline else 0,
            "median_radius_voxels": radius,
            "peer_median_radius_voxels": (
                baseline["median_radius_voxels"] if baseline else None
            ),
            "peer_robust_sigma_voxels": (
                baseline["robust_sigma_voxels"] if baseline else None
            ),
            "radius_ratio": ratio,
            "radius_robust_z": z_score,
            "radius_evidence_strength": radius_strength,
            "bent_evidence_strength": bend_strength,
            "centerline_deviation_rms_voxels": rms_deviation,
            "centerline_deviation_max_voxels": max_deviation,
            "curvature_rms_inverse_voxels": curvature,
            "tracking_coverage": row["tracking_coverage"],
            "mean_tracking_confidence": row["mean_tracking_confidence"],
            "classification_priority_reason": priority_reason,
            "reason": "; ".join(reasons),
            "evidence_png": "",
        }
        classified.append(result)
        for label in labels:
            finding = {
                key: result[key] for key in result if key != "evidence_png"
            }
            finding["measurement_profile"] = {
                **measurement_provenance,
                "length_voxels": row["length_voxels"],
                "tracking_coverage": row["tracking_coverage"],
                "median_radius_voxels": radius,
                "centerline_deviation_rms_voxels": rms_deviation,
                "centerline_deviation_max_voxels": max_deviation,
                "curvature_rms_inverse_voxels": curvature,
                "samples": profiles,
            }
            findings[label].append(finding)

    classified_path = output_dir / "classified_struts.csv"
    thresholds_path = output_dir / "thresholds.json"
    baselines_path = output_dir / "peer_baselines.json"
    log_path = output_dir / "decision_log.md"
    _write_csv(classified_path, CLASSIFICATION_FIELDS, classified)
    _write_json(thresholds_path, thresholds)
    _write_json(baselines_path, {
        "schema_version": 1,
        "groups": baselines,
    })
    for defect_class in ("thin", "thick", "bent"):
        _write_json(output_dir / f"findings_{defect_class}.json", {
            "schema_version": 1,
            "defect_class": defect_class,
            "thresholds_path": "thresholds.json",
            "measurement_provenance": measurement_provenance,
            "findings": findings[defect_class],
        })
    counts = Counter(row["classification"] for row in classified)
    log_path.write_text(
        "# Thin/thick/bent decision log\n\n"
        f"- Input struts: {len(classified)}\n"
        f"- Usable peer groups: {len(baselines)}\n"
        f"- Thin findings: {len(findings['thin'])}\n"
        f"- Thick findings: {len(findings['thick'])}\n"
        f"- Bent findings: {len(findings['bent'])}\n"
        f"- Classification counts: `{dict(counts)}`\n\n"
        "Peer groups are registered `unit_cell_edge_idx` families. Baselines use "
        "only measurements passing the frozen coverage, confidence, junction, "
        "boundary, and radius-variation gates. Thin/thick decisions require both "
        "a median-radius ratio and a robust median/MAD score. Bent decisions use "
        "best-fit CT-centerline residual and curvature with adjacent-section "
        "support; rigid registration offset alone is not bending. Tracking gates "
        "for bending are evaluated separately from radius-variation quality. "
        "When bend and radius-defect evidence coexist, the primary label is bent "
        "when its normalized strength is within the frozen relative-priority "
        "margin; independent flags remain preserved.\n",
        encoding="utf-8",
    )
    return {
        "status": "ready",
        "classified_struts_path": str(classified_path),
        "thresholds_path": str(thresholds_path),
        "peer_baselines_path": str(baselines_path),
        "counts": dict(counts),
        "finding_counts": {key: len(value) for key, value in findings.items()},
        "elapsed_seconds": round(time.time() - started, 3),
    }


def render_strut_evidence(
    classified_struts_csv: str | Path,
    strut_sections_csv: str | Path,
    output_dir: str | Path,
    *,
    thresholds_json: str | Path,
    overwrite: bool = False,
):
    """Render radius evidence for thin/thick and centerline evidence for bent."""
    classified_struts_csv = Path(classified_struts_csv).resolve()
    strut_sections_csv = Path(strut_sections_csv).resolve()
    output_dir = Path(output_dir).resolve()
    _prepare_output(output_dir, overwrite)
    thresholds = _load_thresholds(thresholds_json)
    sections_by_strut = defaultdict(list)
    for row in _read_csv(strut_sections_csv):
        sections_by_strut[int(row["strut_id"])].append(row)
    classified = _read_csv(classified_struts_csv)
    manifest_rows = []
    evidence_by_strut = defaultdict(list)
    evidence_by_class = defaultdict(list)

    def classification_relative_path(path):
        return Path(os.path.relpath(
            path, start=classified_struts_csv.parent
        )).as_posix()

    for row in classified:
        strut_id = int(row["strut_id"])
        flags = {
            "thin": _as_bool(row["is_thin"]),
            "thick": _as_bool(row["is_thick"]),
            "bent": _as_bool(row["is_bent"]),
        }
        profiles = sorted(
            sections_by_strut.get(strut_id, []),
            key=lambda item: float(item["axis_fraction"]),
        )
        x = np.asarray([float(item["axis_fraction"]) for item in profiles])
        valid = np.asarray([_as_bool(item["valid"]) for item in profiles])
        if flags["thin"] or flags["thick"]:
            defect_class = "thin" if flags["thin"] else "thick"
            y = np.asarray([
                float(item["radius_voxels"]) if item["radius_voxels"] else np.nan
                for item in profiles
            ])
            baseline = float(row["peer_median_radius_voxels"])
            sigma = float(row["peer_robust_sigma_voxels"])
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(x, y, color="#2563eb", linewidth=1.8, alpha=0.75)
            ax.scatter(x[valid], y[valid], color="#2563eb", label="valid CT radius")
            ax.scatter(
                x[~valid], y[~valid], marker="x", color="#f59e0b",
                label="excluded / low confidence",
            )
            ax.axhline(baseline, color="#16a34a", label="peer median")
            ax.fill_between(
                [0, 1],
                baseline - thresholds["radius_robust_z_threshold"] * sigma,
                baseline + thresholds["radius_robust_z_threshold"] * sigma,
                color="#16a34a",
                alpha=0.12,
                label="peer robust envelope",
            )
            ax.set(
                title=f"Strut {strut_id}: {row['classification']} radius evidence",
                xlabel="Normalized position along strut",
                ylabel="Equivalent-area radius (voxels)",
                xlim=(0, 1),
            )
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            path = output_dir / defect_class / f"strut_{strut_id}_radius.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            relative = classification_relative_path(path)
            evidence_by_strut[strut_id].append(relative)
            evidence_by_class[(strut_id, defect_class)].append(relative)
            manifest_rows.append({
                "strut_id": strut_id,
                "defect_class": defect_class,
                "evidence_type": "radius_profile",
                **_artifact(path, output_dir),
            })

        if flags["bent"]:
            deviation = np.asarray([
                float(item["centerline_deviation_voxels"])
                if item["centerline_deviation_voxels"] else np.nan
                for item in profiles
            ])
            curvature = np.asarray([
                float(item["curvature_inverse_voxels"])
                if item["curvature_inverse_voxels"] else np.nan
                for item in profiles
            ])
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
            axes[0].plot(x, deviation, marker="o", color="#7c3aed")
            axes[0].axhline(
                thresholds["bent_adjacent_deviation_voxels"],
                color="#dc2626", linestyle="--", label="adjacent-deviation gate",
            )
            axes[0].set_ylabel("Best-fit line deviation (voxels)")
            axes[0].legend(fontsize=8)
            axes[1].plot(x, curvature, marker="o", color="#0f766e")
            axes[1].axhline(
                thresholds["bent_curvature_rms_inverse_voxels"],
                color="#dc2626", linestyle="--", label="RMS-curvature gate",
            )
            axes[1].set(
                xlabel="Normalized position along strut",
                ylabel="Curvature (1/voxel)",
                xlim=(0, 1),
            )
            axes[1].legend(fontsize=8)
            for axis in axes:
                axis.grid(alpha=0.2)
            fig.suptitle(f"Strut {strut_id}: bent centerline evidence")
            fig.tight_layout()
            path = output_dir / "bent" / f"strut_{strut_id}_centerline.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            relative = classification_relative_path(path)
            evidence_by_strut[strut_id].append(relative)
            evidence_by_class[(strut_id, "bent")].append(relative)
            manifest_rows.append({
                "strut_id": strut_id,
                "defect_class": "bent",
                "evidence_type": "centerline_deviation",
                **_artifact(path, output_dir),
            })

    # Update the classification artifact so later stages have direct evidence paths.
    typed_rows = []
    for row in classified:
        strut_id = int(row["strut_id"])
        row["evidence_png"] = ";".join(evidence_by_strut.get(strut_id, []))
        typed_rows.append(row)
    _write_csv(classified_struts_csv, CLASSIFICATION_FIELDS, typed_rows)
    for defect_class in ("thin", "thick", "bent"):
        findings_path = (
            classified_struts_csv.parent / f"findings_{defect_class}.json"
        )
        if not findings_path.is_file():
            continue
        payload = json.loads(findings_path.read_text(encoding="utf-8"))
        for finding in payload.get("findings", []):
            paths = evidence_by_class.get(
                (int(finding["strut_id"]), defect_class), []
            )
            finding["evidence_png"] = ";".join(paths)
        _write_json(findings_path, payload)
    manifest_path = output_dir / "evidence_manifest.json"
    _write_json(manifest_path, {
        "schema_version": 1,
        "status": "ready",
        "evidence": manifest_rows,
    })
    return {
        "status": "ready",
        "evidence_manifest_path": str(manifest_path),
        "plot_count": len(manifest_rows),
        "manifest_sha256": _sha256(manifest_path),
    }


def run_pipeline(
    input_tiff: str | Path,
    registered_json: str | Path,
    output_dir: str | Path,
    threshold: float,
    *,
    thresholds_json: str | Path | None = None,
    overwrite: bool = False,
    **measurement_options,
):
    """Run measurement, frozen classification, rendering, and hand-off creation."""
    started = time.time()
    output_dir = Path(output_dir).resolve()
    _prepare_output(output_dir, overwrite)
    metrics_dir = output_dir / "metrics"
    classification_dir = output_dir / "classification"
    evidence_dir = output_dir / "evidence"
    measurement = compute_strut_metrics(
        input_tiff,
        registered_json,
        metrics_dir,
        threshold,
        overwrite=overwrite,
        **measurement_options,
    )
    classification = classify_struts(
        metrics_dir / "strut_summary.csv",
        metrics_dir / "strut_section_measurements.csv",
        classification_dir,
        thresholds_json=thresholds_json,
        overwrite=overwrite,
    )
    evidence = render_strut_evidence(
        classification_dir / "classified_struts.csv",
        metrics_dir / "strut_section_measurements.csv",
        evidence_dir,
        thresholds_json=classification_dir / "thresholds.json",
        overwrite=overwrite,
    )
    handoff_path = output_dir / "classification_handoff.json"
    required = {
        "measurement_manifest": metrics_dir / "measurement_manifest.json",
        "section_measurements": metrics_dir / "strut_section_measurements.csv",
        "strut_summary": metrics_dir / "strut_summary.csv",
        "classified_struts": classification_dir / "classified_struts.csv",
        "thin_findings": classification_dir / "findings_thin.json",
        "thick_findings": classification_dir / "findings_thick.json",
        "bent_findings": classification_dir / "findings_bent.json",
        "thresholds": classification_dir / "thresholds.json",
        "peer_baselines": classification_dir / "peer_baselines.json",
        "decision_log": classification_dir / "decision_log.md",
        "evidence_manifest": evidence_dir / "evidence_manifest.json",
    }
    handoff = {
        "schema_version": 1,
        "status": "ready",
        "stage": "thin_thick_bent_classification",
        "scope": ["thin", "thick", "bent"],
        "counts": classification["counts"],
        "finding_counts": classification["finding_counts"],
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": {
            name: _artifact(path, output_dir) for name, path in required.items()
        },
        "self_verification": {
            "all_required_artifacts_exist": all(path.is_file() for path in required.values()),
            "measurement_status": measurement["status"],
            "classification_status": classification["status"],
            "evidence_status": evidence["status"],
        },
    }
    _write_json(handoff_path, handoff)
    receipt_path = output_dir / "pipeline_receipt.json"
    _write_json(receipt_path, {
        "status": "ready",
        "handoff": _artifact(handoff_path, output_dir),
        "strut_count": measurement["strut_count"],
        "section_count": measurement["section_count"],
        "plot_count": evidence["plot_count"],
    })
    return {
        "status": "ready",
        "handoff_path": str(handoff_path),
        "receipt_path": str(receipt_path),
        "finding_counts": classification["finding_counts"],
        "plot_count": evidence["plot_count"],
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--thresholds-json", type=Path)
    parser.add_argument("--positions", type=int, default=11)
    parser.add_argument("--start-fraction", type=float, default=0.10)
    parser.add_argument("--end-fraction", type=float, default=0.90)
    parser.add_argument("--ignore-edge-sections", type=int, default=1)
    parser.add_argument("--tracking-radius-voxels", type=float, default=6.0)
    parser.add_argument("--extent-pixels", type=float, default=12.0)
    parser.add_argument("--grid-size", type=int, default=49)
    parser.add_argument("--voxel-size-mm", type=float)
    parser.add_argument("--strut-ids", type=int, nargs="+")
    parser.add_argument("--max-struts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    result = run_pipeline(
        args.input_tiff,
        args.registered_json,
        args.output_dir,
        args.threshold,
        thresholds_json=args.thresholds_json,
        overwrite=args.overwrite,
        positions=args.positions,
        start_fraction=args.start_fraction,
        end_fraction=args.end_fraction,
        ignore_edge_sections=args.ignore_edge_sections,
        tracking_radius_voxels=args.tracking_radius_voxels,
        extent_pixels=args.extent_pixels,
        grid_size=args.grid_size,
        voxel_size_mm=args.voxel_size_mm,
        strut_ids=args.strut_ids,
        max_struts=args.max_struts,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
