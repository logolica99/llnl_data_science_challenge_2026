"""Read-only, recompute-free strut report assembly from frozen artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import read_json_object, sha256_file
from .strut_metrics import read_metrics_csv

STRUT_REPORT_SCHEMA_VERSION = "part2-strut-report/1.0.0"


def _find_classification(
    payload: dict[str, Any],
    strut_id: int,
) -> dict[str, Any]:
    for row in payload.get("classifications", []):
        if int(row.get("strut_id", -1)) == strut_id:
            return row
    raise KeyError(f"Strut ID {strut_id} is absent from classification artifact")


def _find_evidence(
    evidence_manifest_path: str | Path | None,
    strut_id: int,
) -> tuple[dict[str, Any], str | None]:
    if evidence_manifest_path is None:
        return {}, None
    payload = read_json_object(evidence_manifest_path)
    if int(payload.get("strut_id", -1)) != strut_id:
        raise ValueError(
            f"Evidence manifest is for strut {payload.get('strut_id')}, not {strut_id}"
        )
    return payload.get("artifacts", {}), sha256_file(evidence_manifest_path)


def get_strut_report(
    strut_id: int,
    metrics_path: str | Path,
    classifications_path: str | Path,
    thresholds_path: str | Path,
    *,
    evidence_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return one compact artifact-backed record without numerical recomputation."""

    metrics = [
        row
        for row in read_metrics_csv(metrics_path)
        if int(row["strut_id"]) == int(strut_id)
    ]
    if len(metrics) != 1:
        raise KeyError(f"Expected exactly one metrics row for strut ID {strut_id}")
    classifications = read_json_object(classifications_path)
    classification = _find_classification(classifications, int(strut_id))
    thresholds = read_json_object(thresholds_path)
    evidence, evidence_hash = _find_evidence(evidence_manifest_path, int(strut_id))
    attribution = classification.get("attribution", "not_attributed")
    hashes = {
        "metrics_sha256": sha256_file(metrics_path),
        "classifications_sha256": sha256_file(classifications_path),
        "thresholds_sha256": sha256_file(thresholds_path),
    }
    if evidence_hash:
        hashes["evidence_manifest_sha256"] = evidence_hash
    return {
        "schema_version": STRUT_REPORT_SCHEMA_VERSION,
        "gate": "pass",
        "strut_id": int(strut_id),
        "class": classification["class"],
        "bent": bool(classification.get("bent", False)),
        "attribution": attribution,
        "reasons": classification.get("reasons", []),
        "metrics": metrics[0],
        "thresholds": thresholds.get(
            "thresholds_normalized",
            thresholds.get("thresholds_verbatim", thresholds),
        ),
        "evidence": evidence,
        "hashes": hashes,
        "provenance": {
            "artifact_backed": True,
            "metrics_recomputed": False,
            "sealed_labels_read": False,
        },
        "warnings": (
            [] if evidence else ["no evidence manifest was supplied for this strut"]
        ),
    }
