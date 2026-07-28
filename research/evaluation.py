"""Eval-side detection metrics for sealed missing-strut labels."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from part2_core.artifacts import read_json_object, sha256_file, write_json_atomic

DETECTION_METRICS_SCHEMA_VERSION = "part2-detection-metrics/1.0.0"
CLASSES = ["missing", "broken", "thin", "present"]


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError(
            "Wilson interval requires 0 <= successes <= total and total > 0"
        )
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def _classification_map(payload: dict[str, Any]) -> dict[int, str]:
    rows = payload.get("classifications")
    if not isinstance(rows, list):
        raise ValueError("Classification artifact has no classifications array")
    result: dict[int, str] = {}
    for row in rows:
        identifier = int(row["strut_id"])
        label = str(row["class"])
        if identifier in result or label not in CLASSES:
            raise ValueError(f"Invalid classification row for strut {identifier}")
        result[identifier] = label
    return result


def _sealed_map(payload: dict[str, Any]) -> dict[int, str]:
    for key in ("strut_ids", "missing_strut_ids", "sealed_strut_ids"):
        if isinstance(payload.get(key), list):
            return {int(identifier): "missing" for identifier in payload[key]}
    rows = payload.get("labels")
    if isinstance(rows, list):
        result = {}
        for row in rows:
            identifier = int(row["strut_id"])
            label = str(row.get("class", "missing"))
            if label not in CLASSES:
                raise ValueError(f"Unsupported sealed class {label!r}")
            result[identifier] = label
        return result
    raise ValueError("Sealed label artifact has no recognized ID/label array")


def compute_detection_metrics(
    classifications_path: str | Path,
    sealed_labels_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Report strict/lenient recall and confusion, deliberately not precision."""

    classification_payload = read_json_object(classifications_path)
    sealed_payload = read_json_object(sealed_labels_path)
    predicted = _classification_map(classification_payload)
    actual = _sealed_map(sealed_payload)
    if not actual:
        raise ValueError("Sealed label set is empty")
    missing_predictions = [predicted.get(identifier) for identifier in actual]
    absent = sorted(identifier for identifier in actual if identifier not in predicted)
    if absent:
        raise ValueError(
            f"Classifications omit {len(absent)} sealed strut IDs; first={absent[0]}"
        )
    strict_count = sum(label == "missing" for label in missing_predictions)
    lenient_count = sum(label in {"missing", "broken"} for label in missing_predictions)
    total = len(actual)
    strict_interval = wilson_interval(strict_count, total)
    lenient_interval = wilson_interval(lenient_count, total)
    confusion = {
        actual_class: {predicted_class: 0 for predicted_class in CLASSES}
        for actual_class in CLASSES
    }
    for identifier, actual_class in actual.items():
        confusion[actual_class][predicted[identifier]] += 1
    report = {
        "schema_version": DETECTION_METRICS_SCHEMA_VERSION,
        "gate": "pass",
        "overall_pass": True,
        "protocol": "one_shot_reporting_not_pass_fail",
        "sealed_strut_count": total,
        "strict_recall": {
            "definition": "predicted missing",
            "detected": strict_count,
            "total": total,
            "value": strict_count / total,
            "wilson_95_ci": list(strict_interval),
        },
        "lenient_recall": {
            "definition": "predicted missing or broken",
            "detected": lenient_count,
            "total": total,
            "value": lenient_count / total,
            "wilson_95_ci": list(lenient_interval),
        },
        "confusion_matrix": {
            "class_order": CLASSES,
            "rows_actual_columns_predicted": confusion,
        },
        "omitted_metrics": {
            "precision": (
                "undefined because detections outside sealed intentional deletions "
                "may be unintentional defects"
            ),
            "f1": "not computed because precision is undefined",
        },
        "artifacts": {},
        "hashes": {
            "classifications_sha256": sha256_file(classifications_path),
            "sealed_labels_sha256": sha256_file(sealed_labels_path),
        },
        "provenance": {
            "eval_side": True,
            "sealed_labels_read": True,
        },
        "warnings": [],
    }
    artifact = write_json_atomic(output_path, report, overwrite=overwrite)
    report["artifacts"]["detection_metrics"] = {
        **artifact,
        "role": "sealed_detection_metrics",
        "retention": "committed",
    }
    report["hashes"]["detection_metrics_sha256"] = artifact["sha256"]
    return report
