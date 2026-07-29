"""Deterministic Stage 3 classification over persisted Stage 2 metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .artifacts import read_json_object, sha256_file, write_json_atomic
from .strut_metrics import read_metrics_csv

CLASSIFICATION_SCHEMA_VERSION = "part2-strut-classification/1.0.0"


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
