"""Deterministic Stage 3 classification over frozen Stage 2 measurements.

This module never samples CT data, recomputes registration, or changes the
Stage 2 connectivity decision.  It transfers Claire's missing/broken profile
rules into a provenance-bound specialist interface and leaves thin/bent as
explicit teammate-owned extension points.
"""

from __future__ import annotations

from collections import Counter
import csv
import io
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from jsonschema import Draft202012Validator

from .artifacts import (
    read_json_object,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)
from .lattice import load_lattice_json
from .strut_metrics import read_metrics_csv


STAGE3_CONFIG_SCHEMA_VERSION = "part2-defect-analysis-config/1.0.0"
SPECIALIST_FINDINGS_SCHEMA_VERSION = "part2-specialist-findings/1.0.0"
CLASSIFICATION_SCHEMA_VERSION = "part2-strut-classification/2.0.0"
CLASSIFIER_VERIFIER_SCHEMA_VERSION = "classifier-verifier-report/1.0.0"
PRECEDENCE = ["missing", "broken", "thin", "present"]
DEFECT_KINDS = ("missing", "broken", "thin", "bent")
VALIDATION_EXPORT_SCHEMA_VERSION = "part2-stage3-validation-export/1.0.0"

DEFAULT_STAGE3_CONFIG: dict[str, Any] = {
    "schema_version": STAGE3_CONFIG_SCHEMA_VERSION,
    "development_mode": True,
    "label_blind": True,
    "policy_source": "frozen_prior_connectivity_method",
    "precedence": PRECEDENCE,
    "bent_is_non_competing_attribute": True,
    # Missing/broken rules match Claire's recovered standalone 96/78 workflow:
    # notes/STANDALONE_CONNECTIVITY_96_78_RECOVERY.md and
    # scripts/classify_{missing_broken,material_loss}_struts.py on
    # codex/recover-standalone-connectivity-96-78.
    "missing": {
        "implementation_status": "complete",
        "central_start_fraction": 0.20,
        "central_end_fraction": 0.80,
        "present_fraction": 0.05,
        "maximum_central_present_slice_fraction": 0.10,
        "smoothing_window_samples": 3,
        "require_primary_disconnection": True,
    },
    "broken": {
        "implementation_status": "complete",
        "central_reference_quantile": 0.90,
        "deficit_ratio": 0.50,
        "minimum_deficit_fraction": 0.15,
        "minimum_deficit_run_samples": 3,
        "minimum_collar_foreground_fraction": 0.05,
        "minimum_shared_component_voxels": 500,
        "connected_material_loss_is_broken": True,
        "unresolved_disconnection_outcome": "review",
    },
    "thin": {
        "implementation_status": "deferred",
        "owner": "teammate",
        "policy": {},
    },
    "bent": {
        "implementation_status": "deferred",
        "owner": "teammate",
        "policy": {},
    },
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_schema(value: Mapping[str, Any], filename: str, label: str) -> None:
    schema = read_json_object(_repository_root() / "analysis" / "schema" / filename)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise ValueError(f"{label} is schema-incompatible: {detail}")


def normalize_stage3_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return and validate the frozen Stage 3 policy fragment."""

    if not isinstance(config, Mapping):
        raise ValueError("Stage 3 analysis config must be an object")
    root = dict(config)
    fragment = root.get("stage_3_defect_analysis", root)
    if not isinstance(fragment, Mapping):
        raise ValueError("stage_3_defect_analysis must be an object")
    wrapped = {"stage_3_defect_analysis": dict(fragment)}
    _validate_schema(
        wrapped,
        "defect_analysis_input.schema.json",
        "Frozen Stage 3 analysis config",
    )
    return dict(fragment)


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"Metric {field} must be a boolean")


def _as_finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Metric {field} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"Metric {field} must be finite")
    return result


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _profile_map(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = read_json_object(path)
    if payload.get("measurement_only") is not True or payload.get(
        "classification_performed"
    ) is not False:
        raise ValueError("Stage 3 requires measurement-only Stage 2 profiles")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Stage 2 profiles artifact is empty or malformed")
    result: dict[int, dict[str, Any]] = {}
    for profile in profiles:
        if not isinstance(profile, dict) or isinstance(profile.get("strut_id"), bool):
            raise ValueError("Stage 2 profile contains an invalid strut ID")
        identifier = int(profile["strut_id"])
        if identifier in result:
            raise ValueError(f"Duplicate Stage 2 profile for strut {identifier}")
        result[identifier] = profile
    return result


def _metric_map(path: str | Path) -> dict[int, dict[str, Any]]:
    rows = read_metrics_csv(path)
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        identifier = int(row["strut_id"])
        if identifier in result:
            raise ValueError(f"Duplicate Stage 2 metric row for strut {identifier}")
        result[identifier] = row
    if not result:
        raise ValueError("Stage 2 metrics artifact is empty")
    return result


def _input_maps(
    metrics_path: str | Path, profiles_path: str | Path
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    metrics = _metric_map(metrics_path)
    profiles = _profile_map(profiles_path)
    if set(metrics) != set(profiles):
        raise ValueError("Stage 2 metrics and profile strut IDs do not match")
    return metrics, profiles


def _profile_features(
    metric: Mapping[str, Any],
    profile: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    axial_t = np.asarray(profile.get("axial_t"), dtype=np.float64)
    fractions = np.asarray(
        profile.get("foreground_fraction", profile.get("occupancy_profile")),
        dtype=np.float64,
    )
    if (
        axial_t.ndim != 1
        or fractions.ndim != 1
        or axial_t.size != fractions.size
        or axial_t.size == 0
        or not np.all(np.isfinite(axial_t))
        or not np.all(np.isfinite(fractions))
    ):
        raise ValueError(f"Strut {metric['strut_id']} has an invalid axial profile")
    missing_policy = policy["missing"]
    broken_policy = policy["broken"]
    central = fractions[
        (axial_t >= float(missing_policy["central_start_fraction"]))
        & (axial_t <= float(missing_policy["central_end_fraction"]))
    ]
    window = int(missing_policy["smoothing_window_samples"])
    if central.size == 0:
        raise ValueError(
            f"Strut {metric['strut_id']} has no central profile samples"
        )
    if central.size < window:
        raise ValueError(
            f"Strut {metric['strut_id']} has fewer than {window} central profile samples"
        )
    smoothed = np.convolve(
        central, np.ones(window, dtype=np.float64) / float(window), mode="valid"
    )
    present = central >= float(missing_policy["present_fraction"])
    central_present_slice_fraction = float(np.mean(present))
    side_width = max(1, int(present.size) // 3)
    reference = float(
        np.quantile(central, float(broken_policy["central_reference_quantile"]))
    )
    deficit_cutoff = float(broken_policy["deficit_ratio"]) * reference
    deficient = central < deficit_cutoff
    material_loss_fraction = (
        float(np.clip(1.0 - central / reference, 0.0, 1.0).mean())
        if reference > 0.0
        else 1.0
    )
    connected = _as_bool(
        metric["same_material_component_connects_a_to_b"],
        "same_material_component_connects_a_to_b",
    )
    collar = min(
        _as_finite_float(
            metric["a_collar_foreground_fraction"],
            "a_collar_foreground_fraction",
        ),
        _as_finite_float(
            metric["b_collar_foreground_fraction"],
            "b_collar_foreground_fraction",
        ),
    )
    endpoint0_component_voxels = int(
        float(metric["endpoint0_to_collar_component_voxel_count_in_corridor"])
    )
    endpoint1_component_voxels = int(
        float(metric["endpoint1_to_collar_component_voxel_count_in_corridor"])
    )
    minimum_endpoint_component_voxels = min(
        endpoint0_component_voxels, endpoint1_component_voxels
    )
    stage2_endpoint_segments_observed = _as_bool(
        metric["both_endpoint_segments_observed"],
        "both_endpoint_segments_observed",
    )
    endpoint_segments_observed = (
        stage2_endpoint_segments_observed
        and collar >= float(broken_policy["minimum_collar_foreground_fraction"])
        and minimum_endpoint_component_voxels
        >= int(broken_policy["minimum_shared_component_voxels"])
    )
    return {
        "same_material_component_connects_a_to_b": connected,
        "central_material_slice_fraction": central_present_slice_fraction,
        "a_side_material_slice_fraction": float(np.mean(present[:side_width])),
        "b_side_material_slice_fraction": float(np.mean(present[-side_width:])),
        "longest_empty_run_slices": _longest_true_run(~present),
        "central_mean_foreground_fraction": float(np.mean(central)),
        "central_reference_foreground_fraction_p90": reference,
        "central_minimum_foreground_fraction": float(np.min(central)),
        "central_minimum_smoothed_foreground_fraction": float(np.min(smoothed)),
        "central_minimum_relative_to_reference": (
            float(np.min(smoothed) / reference) if reference > 0.0 else None
        ),
        "central_material_loss_fraction": material_loss_fraction,
        "central_deficit_cutoff_foreground_fraction": deficit_cutoff,
        "central_deficit_fraction": float(np.mean(deficient)),
        "longest_deficit_run_samples": _longest_true_run(deficient),
        "minimum_endpoint_collar_foreground_fraction": collar,
        "endpoint0_to_collar_component_voxel_count_in_corridor": (
            endpoint0_component_voxels
        ),
        "endpoint1_to_collar_component_voxel_count_in_corridor": (
            endpoint1_component_voxels
        ),
        "minimum_endpoint_to_collar_component_voxel_count_in_corridor": (
            minimum_endpoint_component_voxels
        ),
        "both_endpoint_segments_observed": endpoint_segments_observed,
    }


def _missing_positive(
    features: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    """Claire recovered missing rule: disconnected + sparse central material."""

    if bool(policy["missing"].get("require_primary_disconnection", True)) and bool(
        features["same_material_component_connects_a_to_b"]
    ):
        return False
    return float(features["central_material_slice_fraction"]) <= float(
        policy["missing"]["maximum_central_present_slice_fraction"]
    )


def _finding_row(
    strut_id: int,
    disposition: str,
    reasons: Iterable[str],
    features: Mapping[str, Any],
) -> dict[str, Any]:
    reason_list = list(reasons)
    return {
        "strut_id": int(strut_id),
        "disposition": disposition,
        "reasons": reason_list or ["no_specialist_rule_triggered"],
        "features": dict(features),
        "evidence_required": disposition in {"positive", "review"},
        "evidence_refs": ["per_strut_metrics", "per_strut_profiles"],
    }


def _coverage(rows: list[dict[str, Any]], total: int) -> dict[str, int]:
    counts = Counter(row["disposition"] for row in rows)
    return {
        "nominal_strut_count": total,
        "evaluated_count": total - counts["deferred"],
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "review_count": counts["review"],
        "deferred_count": counts["deferred"],
    }


def analyze_strut_specialist(
    metrics_path: str | Path,
    profiles_path: str | Path,
    analysis_config: Mapping[str, Any] | str | Path,
    output_path: str | Path,
    *,
    specimen_id: str,
    defect_kind: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one schema-identical specialist finding document."""

    if defect_kind not in DEFECT_KINDS:
        raise ValueError(f"Unsupported Stage 3 defect kind: {defect_kind}")
    raw_config = (
        read_json_object(analysis_config)
        if isinstance(analysis_config, (str, Path))
        else dict(analysis_config)
    )
    policy = normalize_stage3_config(raw_config)
    metrics, profiles = _input_maps(metrics_path, profiles_path)
    rows: list[dict[str, Any]] = []
    status = str(policy[defect_kind]["implementation_status"])
    for strut_id in sorted(metrics):
        if status == "deferred":
            rows.append(
                _finding_row(
                    strut_id,
                    "deferred",
                    [f"{defect_kind}_specialist_owned_by_teammate"],
                    {},
                )
            )
            continue
        features = _profile_features(metrics[strut_id], profiles[strut_id], policy)
        connected = bool(features["same_material_component_connects_a_to_b"])
        if defect_kind == "missing":
            positive = _missing_positive(features, policy)
            disposition = "positive" if positive else "negative"
            reasons = (
                [
                    "primary_disconnected",
                    "central_present_slice_fraction_at_most_10_percent",
                ]
                if positive
                else ["missing_rule_not_satisfied"]
            )
        elif defect_kind == "broken":
            missing_positive = _missing_positive(features, policy)
            material_loss = (
                float(features["central_deficit_fraction"])
                >= float(policy["broken"]["minimum_deficit_fraction"])
                or int(features["longest_deficit_run_samples"])
                >= int(policy["broken"]["minimum_deficit_run_samples"])
            )
            endpoint_evidence = bool(features["both_endpoint_segments_observed"])
            allow_connected_bite = bool(
                policy["broken"].get("connected_material_loss_is_broken", True)
            )
            unresolved_outcome = str(
                policy["broken"].get("unresolved_disconnection_outcome", "review")
            )
            if missing_positive:
                disposition = "negative"
                reasons = ["missing_precedence_excludes_broken"]
            elif material_loss and endpoint_evidence and (allow_connected_bite or not connected):
                disposition = "positive"
                reasons = [
                    "central_material_loss_rule_satisfied",
                    "endpoint_material_support_observed",
                    (
                        "connected_bite_case"
                        if connected
                        else "disconnected_fragment_case"
                    ),
                ]
            elif not connected:
                disposition = unresolved_outcome if unresolved_outcome in {
                    "review",
                    "negative",
                    "positive",
                } else "review"
                reasons = ["primary_disconnected_without_sufficient_broken_evidence"]
            else:
                disposition = "negative"
                reasons = ["broken_rule_not_satisfied"]
        else:
            raise ValueError(
                f"{defect_kind} is marked complete but its teammate implementation is absent"
            )
        rows.append(_finding_row(strut_id, disposition, reasons, features))

    specialist = (
        "missing_strut_agent"
        if defect_kind == "missing"
        else "broken_strut_agent"
        if defect_kind == "broken"
        else "thin_strut_agent"
    )
    payload = {
        "schema_version": SPECIALIST_FINDINGS_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "stage_number": 3,
        "specialist": specialist,
        "defect_kind": defect_kind,
        "status": status,
        "policy_sha256": sha256_json(policy),
        "input_hashes": {
            "per_strut_metrics_sha256": sha256_file(metrics_path),
            "per_strut_profiles_sha256": sha256_file(profiles_path),
        },
        "coverage": _coverage(rows, len(metrics)),
        "findings": rows,
        "provenance": {
            "rule_version": "claire-standalone-connectivity-96-78/1.1.0",
            "metrics_recomputed": False,
            "registration_recomputed": False,
            "training_labels_read": False,
            "evaluation_labels_read": False,
            "intentional_deletion_labels_read": False,
        },
    }
    _validate_schema(
        payload, "specialist_findings.schema.json", f"{defect_kind} findings"
    )
    artifact = write_json_atomic(output_path, payload, overwrite=overwrite)
    return {
        **payload,
        "gate": "pass" if status == "complete" else "manual_review",
        "artifacts": {
            "findings": {
                **artifact,
                "role": f"findings_{defect_kind}",
                "retention": "committed",
            }
        },
        "hashes": {f"findings_{defect_kind}_sha256": artifact["sha256"]},
        "warnings": (
            []
            if status == "complete"
            else [f"{defect_kind} remains deferred to the teammate specialist"]
        ),
    }


def _load_findings(
    paths: Mapping[str, str | Path],
    *,
    specimen_id: str,
    expected_ids: set[int],
    metrics_sha256: str,
    profiles_sha256: str,
    policy_sha256: str,
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, dict[str, Any]]]:
    maps: dict[str, dict[int, dict[str, Any]]] = {}
    documents: dict[str, dict[str, Any]] = {}
    if set(paths) != set(DEFECT_KINDS):
        raise ValueError("Merge requires missing, broken, thin, and bent findings")
    for kind in DEFECT_KINDS:
        document = read_json_object(paths[kind])
        _validate_schema(
            document, "specialist_findings.schema.json", f"{kind} findings"
        )
        if (
            document.get("defect_kind") != kind
            or document.get("specimen_id") != specimen_id
            or document.get("policy_sha256") != policy_sha256
            or document.get("input_hashes")
            != {
                "per_strut_metrics_sha256": metrics_sha256,
                "per_strut_profiles_sha256": profiles_sha256,
            }
        ):
            raise ValueError(f"{kind} findings are stale or bound to another specimen")
        rows = {int(row["strut_id"]): row for row in document["findings"]}
        if len(rows) != len(document["findings"]) or set(rows) != expected_ids:
            raise ValueError(f"{kind} findings do not cover every nominal strut exactly once")
        maps[kind] = rows
        documents[kind] = document
    return maps, documents


def _decision_log_text(
    *,
    specimen_id: str,
    gate: str,
    development_mode: bool,
    incomplete: Iterable[str],
    review_ids: Iterable[int],
) -> str:
    incomplete_list = list(incomplete)
    review_list = list(review_ids)
    return "\n".join(
        [
            "# Stage 3 classification decision log",
            "",
            f"- Specimen: `{specimen_id}`",
            f"- Gate: `{gate}`",
            f"- Development mode: `{str(development_mode).lower()}`",
            f"- Precedence: `{' > '.join(PRECEDENCE)}`",
            "- Missing: primary A-to-B component is disconnected and at most 10% of central (20%-80%) axial slices are material-bearing (foreground fraction ≥ 0.05).",
            "- Broken: missing is false, endpoint material is observed, and either at least 15% of central slices are below 50% of central P90 or the deficient run is at least three slices.",
            "- Connected bite cases may be broken; unresolved disconnections require review.",
            "- Bent is a separate non-competing attribute.",
            f"- Deferred specialist implementations: `{', '.join(incomplete_list) if incomplete_list else 'none'}`",
            f"- Specialist-review struts: `{len(review_list)}`",
            "- Training, evaluation, and intentional-deletion labels accessed: `false`",
            "",
        ]
    )


def merge_strut_classifications(
    metrics_path: str | Path,
    profiles_path: str | Path,
    analysis_config: Mapping[str, Any] | str | Path,
    findings_paths: Mapping[str, str | Path],
    output_classifications_path: str | Path,
    output_thresholds_path: str | Path,
    output_decision_log_path: str | Path,
    *,
    specimen_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge complete or deferred specialist findings with fixed precedence."""

    raw_config = (
        read_json_object(analysis_config)
        if isinstance(analysis_config, (str, Path))
        else dict(analysis_config)
    )
    policy = normalize_stage3_config(raw_config)
    metrics, _profiles = _input_maps(metrics_path, profiles_path)
    metrics_sha256 = sha256_file(metrics_path)
    profiles_sha256 = sha256_file(profiles_path)
    policy_sha256 = sha256_json(policy)
    findings, documents = _load_findings(
        findings_paths,
        specimen_id=specimen_id,
        expected_ids=set(metrics),
        metrics_sha256=metrics_sha256,
        profiles_sha256=profiles_sha256,
        policy_sha256=policy_sha256,
    )
    incomplete = [
        kind for kind, document in documents.items() if document["status"] != "complete"
    ]
    classifications: list[dict[str, Any]] = []
    counts = Counter()
    review_ids: list[int] = []
    for strut_id in sorted(metrics):
        dispositions = {
            kind: findings[kind][strut_id]["disposition"] for kind in DEFECT_KINDS
        }
        specialist_review = any(
            value == "review" for value in dispositions.values()
        )
        reasons: list[str] = []
        if dispositions["missing"] == "positive":
            label = "missing"
            reasons.extend(findings["missing"][strut_id]["reasons"])
        elif dispositions["broken"] == "positive":
            label = "broken"
            reasons.extend(findings["broken"][strut_id]["reasons"])
        elif dispositions["thin"] == "positive":
            label = "thin"
            reasons.extend(findings["thin"][strut_id]["reasons"])
        elif specialist_review:
            label = "deferred"
            review_ids.append(strut_id)
            reasons.append("specialist_review_required")
        elif incomplete:
            label = "deferred"
            reasons.append("specialist_implementation_deferred")
        else:
            label = "present"
            reasons.append("no_primary_defect_specialist_positive")
        bent: bool | None = (
            None
            if documents["bent"]["status"] != "complete"
            else dispositions["bent"] == "positive"
        )
        evidence_required = (
            label in {"missing", "broken", "thin"}
            or bent is True
            or specialist_review
        )
        counts[label] += 1
        classifications.append(
            {
                "strut_id": strut_id,
                "class": label,
                "bent": bent,
                "reasons": reasons,
                "evidence_required": evidence_required,
            }
        )

    development_mode = bool(policy["development_mode"])
    gate = "manual_review" if incomplete or review_ids or development_mode else "pass"
    thresholds_payload = {
        "schema_version": STAGE3_CONFIG_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "stage_number": 3,
        "gate": gate,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "precedence": PRECEDENCE,
        "bent_is_non_competing_attribute": True,
        "label_access": {
            "training": False,
            "evaluation": False,
            "intentional_deletion": False,
        },
    }
    thresholds_artifact = write_json_atomic(
        output_thresholds_path, thresholds_payload, overwrite=overwrite
    )
    finding_hashes = {
        kind: sha256_file(path) for kind, path in sorted(findings_paths.items())
    }
    classification_payload = {
        "schema_version": CLASSIFICATION_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "stage_number": 3,
        "gate": gate,
        "overall_pass": gate == "pass",
        "development_mode": development_mode,
        "counts": {
            "missing": counts["missing"],
            "broken": counts["broken"],
            "thin": counts["thin"],
            "present": counts["present"],
            "deferred": counts["deferred"],
            "total": len(classifications),
        },
        "classifications": classifications,
        "thresholds_sha256": thresholds_artifact["sha256"],
        "metrics_sha256": metrics_sha256,
        "profiles_sha256": profiles_sha256,
        "specialist_findings_sha256": finding_hashes,
        "provenance": {
            "precedence": PRECEDENCE,
            "bent_is_non_competing_attribute": True,
            "metrics_recomputed": False,
            "label_access": {
                "training": False,
                "evaluation": False,
                "intentional_deletion": False,
            },
        },
    }
    _validate_schema(
        classification_payload,
        "classified_struts.schema.json",
        "Merged Stage 3 classifications",
    )
    classifications_artifact = write_json_atomic(
        output_classifications_path, classification_payload, overwrite=overwrite
    )
    decision_log = _decision_log_text(
        specimen_id=specimen_id,
        gate=gate,
        development_mode=development_mode,
        incomplete=incomplete,
        review_ids=review_ids,
    )
    decision_artifact = write_text_atomic(
        output_decision_log_path, decision_log, overwrite=overwrite
    )
    return {
        **classification_payload,
        "artifacts": {
            "classifications": {
                **classifications_artifact,
                "role": "classified_struts",
                "retention": "committed",
            },
            "thresholds": {
                **thresholds_artifact,
                "role": "classification_thresholds",
                "retention": "committed",
            },
            "decision_log": {
                **decision_artifact,
                "role": "classification_decision_log",
                "retention": "committed",
            },
        },
        "hashes": {
            "metrics_sha256": metrics_sha256,
            "profiles_sha256": profiles_sha256,
            "thresholds_sha256": thresholds_artifact["sha256"],
            "classifications_sha256": classifications_artifact["sha256"],
            "decision_log_sha256": decision_artifact["sha256"],
            "specialist_findings_sha256": finding_hashes,
        },
        "warnings": [
            *(f"Deferred specialist implementation: {kind}" for kind in incomplete),
            *(f"Specialist review required for strut {strut_id}" for strut_id in review_ids[:20]),
        ],
    }


def verify_strut_classifications(
    metrics_path: str | Path,
    profiles_path: str | Path,
    findings_paths: Mapping[str, str | Path],
    classifications_path: str | Path,
    thresholds_path: str | Path,
    decision_log_path: str | Path,
    evidence_manifest_paths: Iterable[str | Path],
    output_path: str | Path,
    *,
    analysis_config: str | Path,
    localized_graph_path: str | Path,
    ct_path: str | Path,
    specimen_id: str,
    attempt: int,
    run_token: str,
    config_sha256: str,
    contract_sha256: str,
    predecessor_receipt_sha256: str,
    input_handoff_sha256: str,
) -> dict[str, Any]:
    """Independently verify a complete production Stage 3 bundle."""

    metrics, _profiles = _input_maps(metrics_path, profiles_path)
    metrics_sha256 = sha256_file(metrics_path)
    profiles_sha256 = sha256_file(profiles_path)
    expected_ct_sha256 = sha256_file(ct_path)
    expected_graph_sha256 = sha256_file(localized_graph_path)
    raw_config = read_json_object(analysis_config)
    expected_policy = normalize_stage3_config(raw_config)
    expected_threshold = raw_config.get("threshold")
    if isinstance(expected_threshold, bool) or not isinstance(
        expected_threshold, (int, float)
    ):
        raise ValueError("Frozen analysis config lacks a numeric Otsu threshold")
    classifications = read_json_object(classifications_path)
    _validate_schema(
        classifications,
        "classified_struts.schema.json",
        "Stage 3 classifications",
    )
    if (
        classifications.get("specimen_id") != specimen_id
        or classifications.get("metrics_sha256") != metrics_sha256
        or classifications.get("profiles_sha256") != profiles_sha256
        or classifications.get("gate") != "pass"
        or classifications.get("overall_pass") is not True
        or classifications.get("development_mode") is not False
    ):
        raise ValueError("Classifier verifier cannot pass a partial development bundle")
    rows = classifications["classifications"]
    ids = [int(row["strut_id"]) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(metrics):
        raise ValueError("Classifications do not cover every nominal strut exactly once")
    thresholds = read_json_object(thresholds_path)
    required_threshold_fields = {
        "schema_version",
        "specimen_id",
        "stage_number",
        "gate",
        "policy",
        "policy_sha256",
        "precedence",
        "bent_is_non_competing_attribute",
        "label_access",
    }
    if set(thresholds) != required_threshold_fields:
        raise ValueError("Stage 3 thresholds have undeclared or missing fields")
    policy = normalize_stage3_config(thresholds["policy"])
    policy_sha256 = sha256_json(policy)
    if (
        thresholds.get("schema_version") != STAGE3_CONFIG_SCHEMA_VERSION
        or thresholds.get("specimen_id") != specimen_id
        or thresholds.get("stage_number") != 3
        or thresholds.get("gate") != "pass"
        or thresholds.get("policy_sha256") != policy_sha256
        or policy != expected_policy
        or thresholds.get("precedence") != PRECEDENCE
        or thresholds.get("bent_is_non_competing_attribute") is not True
        or any(thresholds.get("label_access", {}).values())
        or classifications.get("thresholds_sha256") != sha256_file(thresholds_path)
    ):
        raise ValueError("Stage 3 thresholds are stale or violate the frozen policy")
    finding_maps, finding_documents = _load_findings(
        findings_paths,
        specimen_id=specimen_id,
        expected_ids=set(metrics),
        metrics_sha256=metrics_sha256,
        profiles_sha256=profiles_sha256,
        policy_sha256=policy_sha256,
    )
    finding_hashes = {
        f"findings_{kind}": sha256_file(findings_paths[kind])
        for kind in DEFECT_KINDS
    }
    classification_finding_hashes = {
        kind: finding_hashes[f"findings_{kind}"] for kind in DEFECT_KINDS
    }
    if classifications.get("specialist_findings_sha256") != classification_finding_hashes:
        raise ValueError("Classifications are not bound to the supplied specialist findings")
    for kind, document in finding_documents.items():
        if document.get("status") != "complete":
            raise ValueError(f"Classifier verifier cannot pass deferred {kind} findings")
        if document.get("coverage") != _coverage(document["findings"], len(metrics)):
            raise ValueError(f"{kind} findings coverage summary is incorrect")

    expected_rows: list[dict[str, Any]] = []
    expected_counts = Counter()
    for strut_id in sorted(metrics):
        dispositions = {
            kind: finding_maps[kind][strut_id]["disposition"]
            for kind in DEFECT_KINDS
        }
        if dispositions["missing"] == "positive":
            label = "missing"
            reasons = finding_maps["missing"][strut_id]["reasons"]
        elif dispositions["broken"] == "positive":
            label = "broken"
            reasons = finding_maps["broken"][strut_id]["reasons"]
        elif dispositions["thin"] == "positive":
            label = "thin"
            reasons = finding_maps["thin"][strut_id]["reasons"]
        elif any(value in {"review", "deferred"} for value in dispositions.values()):
            raise ValueError(f"Production classification contains unresolved strut {strut_id}")
        else:
            label = "present"
            reasons = ["no_primary_defect_specialist_positive"]
        bent = dispositions["bent"] == "positive"
        expected_counts[label] += 1
        expected_rows.append(
            {
                "strut_id": strut_id,
                "class": label,
                "bent": bent,
                "reasons": list(reasons),
                "evidence_required": label != "present" or bent,
            }
        )
    if rows != expected_rows:
        raise ValueError("Classifications do not match independently recomputed precedence")
    expected_count_document = {
        "missing": expected_counts["missing"],
        "broken": expected_counts["broken"],
        "thin": expected_counts["thin"],
        "present": expected_counts["present"],
        "deferred": 0,
        "total": len(expected_rows),
    }
    if classifications.get("counts") != expected_count_document:
        raise ValueError("Classification counts do not match the classification rows")
    expected_rows_by_id = {int(row["strut_id"]): row for row in expected_rows}
    classifications_sha256 = sha256_file(classifications_path)
    thresholds_sha256 = sha256_file(thresholds_path)
    expected_log = _decision_log_text(
        specimen_id=specimen_id,
        gate="pass",
        development_mode=False,
        incomplete=[],
        review_ids=[],
    )
    if Path(decision_log_path).read_text(encoding="utf-8") != expected_log:
        raise ValueError("Decision log does not match the deterministic merge execution")
    manifests = []
    evidenced_ids: set[int] = set()
    for path in evidence_manifest_paths:
        evidence_path = Path(path).expanduser().resolve()
        document = read_json_object(evidence_path)
        strut_id = int(document.get("strut_id", -1))
        if strut_id in evidenced_ids:
            raise ValueError(f"Duplicate evidence packet for strut {strut_id}")
        classification = expected_rows_by_id.get(strut_id)
        expected_evidence_path = (
            _repository_root()
            / "analysis"
            / specimen_id
            / "evidence"
            / f"strut_{strut_id}"
            / "manifest.json"
        ).resolve()
        if (
            evidence_path != expected_evidence_path
            or document.get("schema_version") != "part2-strut-evidence/1.0.0"
            or document.get("gate") != "pass"
            or document.get("specimen_id") != specimen_id
            or classification is None
            or document.get("classification") != classification
            or document.get("stage_2_metrics") != metrics[strut_id]
            or document.get("hashes", {}).get("metrics_sha256") != metrics_sha256
            or document.get("hashes", {}).get("profiles_sha256") != profiles_sha256
            or document.get("hashes", {}).get("classifications_sha256")
            != classifications_sha256
            or document.get("hashes", {}).get("thresholds_sha256")
            != thresholds_sha256
            or document.get("hashes", {}).get("ct_sha256") != expected_ct_sha256
            or document.get("hashes", {}).get("localized_graph_sha256")
            != expected_graph_sha256
            or document.get("threshold") != float(expected_threshold)
            or document.get("classification_policy") != policy
            or document.get("provenance", {}).get("metrics_recomputed") is not False
            or document.get("provenance", {}).get("classification_recomputed") is not False
            or document.get("provenance", {}).get("sealed_labels_read") is not False
        ):
            raise ValueError(f"Evidence packet for strut {strut_id} is stale or malformed")
        for artifact in document.get("artifacts", {}).values():
            artifact_path = Path(artifact.get("path", "")).expanduser().resolve()
            try:
                artifact_path.relative_to(evidence_path.parent)
            except ValueError as exc:
                raise ValueError(
                    f"Evidence artifact for strut {strut_id} escapes its packet directory"
                ) from exc
            if not artifact_path.is_file() or artifact.get("sha256") != sha256_file(artifact_path):
                raise ValueError(f"Evidence artifact hash mismatch for strut {strut_id}")
        evidenced_ids.add(strut_id)
        try:
            relative_path = evidence_path.relative_to(_repository_root()).as_posix()
        except ValueError as exc:
            raise ValueError("Evidence manifests must remain inside the repository") from exc
        manifests.append({"path": relative_path, "sha256": sha256_file(evidence_path)})
    required_evidence = {
        int(row["strut_id"]) for row in rows if bool(row["evidence_required"])
    }
    if evidenced_ids != required_evidence:
        raise ValueError("Evidence packets do not exactly cover all non-present/bent calls")
    evidence_set_sha256 = sha256_json(
        sorted(manifests, key=lambda item: (item["path"], item["sha256"]))
    )
    report = {
        "schema_version": CLASSIFIER_VERIFIER_SCHEMA_VERSION,
        "owner": "classifier_verifier",
        "gate": "pass",
        "specimen_id": specimen_id,
        "stage_number": 3,
        "attempt": int(attempt),
        "run_token": run_token,
        "config_sha256": config_sha256,
        "contract_sha256": contract_sha256,
        "predecessor_receipt_sha256": predecessor_receipt_sha256,
        "input_handoff_sha256": input_handoff_sha256,
        "participated_in_classification": False,
        "label_access": {
            "development_split_read": False,
            "sealed_split_read": False,
        },
        "bindings": {
            "classified_struts_sha256": sha256_file(classifications_path),
            "thresholds_sha256": sha256_file(thresholds_path),
            "decision_log_sha256": sha256_file(decision_log_path),
            "evidence_set_sha256": evidence_set_sha256,
            "per_strut_metrics_sha256": metrics_sha256,
            "specialist_findings_sha256": finding_hashes,
        },
        "self_verification": {
            "every_strut_labeled_once": True,
            "fixed_precedence_respected": True,
            "bent_kept_separate": True,
            "every_adjudication_logged": True,
            "evidence_support_checked": True,
            "cutoffs_audited": True,
            "decision_log_matches_execution": True,
            "development_split_not_accessed": True,
            "sealed_split_not_accessed": True,
        },
    }
    artifact = write_json_atomic(output_path, report)
    return {
        **report,
        "artifacts": {
            "verifier_report": {
                **artifact,
                "role": "classifier_verifier_report",
                "retention": "committed",
            }
        },
        "hashes": {
            "verifier_report_sha256": artifact["sha256"],
            "evidence_set_sha256": evidence_set_sha256,
        },
        "warnings": [],
    }


_VALIDATION_EXPORT_FIELDS = [
    "strut_id",
    "defect_class",
    "touches_excluded_nominal_plane",
    "excluded_nominal_axis",
    "excluded_nominal_value",
    "x0_nominal",
    "y0_nominal",
    "z0_nominal",
    "x1_nominal",
    "y1_nominal",
    "z1_nominal",
    "same_material_component_connects_a_to_b",
    "a_collar_foreground_fraction",
    "b_collar_foreground_fraction",
    "endpoint0_to_collar_component_voxel_count_in_corridor",
    "endpoint1_to_collar_component_voxel_count_in_corridor",
    "minimum_endpoint_to_collar_component_voxel_count_in_corridor",
    "central_material_slice_fraction",
    "a_side_material_slice_fraction",
    "b_side_material_slice_fraction",
    "longest_empty_run_slices",
    "central_mean_foreground_fraction",
    "central_reference_foreground_fraction_p90",
    "central_minimum_foreground_fraction",
    "central_minimum_smoothed_foreground_fraction",
    "central_minimum_relative_to_reference",
    "central_material_loss_fraction",
    "central_deficit_cutoff_foreground_fraction",
    "central_deficit_fraction",
    "longest_deficit_run_samples",
    "minimum_endpoint_collar_foreground_fraction",
    "both_endpoint_segments_observed",
    "reasons",
]


def _write_validation_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=_VALIDATION_EXPORT_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return write_text_atomic(path, stream.getvalue(), overwrite=overwrite)


def export_stage3_validation_csvs(
    classifications_path: str | Path,
    missing_findings_path: str | Path,
    broken_findings_path: str | Path,
    metrics_path: str | Path,
    nominal_graph_path: str | Path,
    output_directory: str | Path,
    *,
    excluded_nominal_axis: str = "y",
    excluded_nominal_value: float = 18.0,
    coordinate_tolerance: float = 1e-9,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export non-authoritative CSVs for Stage 3 development validation.

    This operation never changes a classification or a production artifact.
    The filtered views exclude missing/broken struts when either nominal
    endpoint touches the explicitly supplied plane (specimen crop artifact).
    """

    axis_by_name = {"x": 0, "y": 1, "z": 2}
    if excluded_nominal_axis not in axis_by_name:
        raise ValueError("excluded_nominal_axis must be one of x, y, or z")
    if not math.isfinite(float(excluded_nominal_value)):
        raise ValueError("excluded_nominal_value must be finite")
    if not math.isfinite(float(coordinate_tolerance)) or coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be finite and nonnegative")

    classifications = read_json_object(classifications_path)
    _validate_schema(
        classifications,
        "classified_struts.schema.json",
        "Stage 3 classifications",
    )
    specimen_id = str(classifications["specimen_id"])
    metrics = _metric_map(metrics_path)
    metrics_sha256 = sha256_file(metrics_path)
    if classifications.get("metrics_sha256") != metrics_sha256:
        raise ValueError("Classifications are not bound to the supplied metrics CSV")

    findings: dict[str, dict[int, dict[str, Any]]] = {}
    findings_hashes: dict[str, str] = {}
    for kind, path in (
        ("missing", missing_findings_path),
        ("broken", broken_findings_path),
    ):
        document = read_json_object(path)
        _validate_schema(
            document,
            "specialist_findings.schema.json",
            f"{kind} findings",
        )
        digest = sha256_file(path)
        if (
            document.get("specimen_id") != specimen_id
            or document.get("defect_kind") != kind
            or document.get("status") != "complete"
            or document.get("input_hashes", {}).get("per_strut_metrics_sha256")
            != metrics_sha256
            or classifications.get("specialist_findings_sha256", {}).get(kind)
            != digest
        ):
            raise ValueError(f"{kind} findings are stale or not classification-bound")
        rows = {int(row["strut_id"]): row for row in document["findings"]}
        if len(rows) != len(document["findings"]) or set(rows) != set(metrics):
            raise ValueError(f"{kind} findings do not exactly cover the metrics CSV")
        findings[kind] = rows
        findings_hashes[kind] = digest

    classification_rows = {
        int(row["strut_id"]): row for row in classifications["classifications"]
    }
    if (
        len(classification_rows) != len(classifications["classifications"])
        or set(classification_rows) != set(metrics)
    ):
        raise ValueError("Classifications do not exactly cover the metrics CSV")

    graph = load_lattice_json(nominal_graph_path)
    if set(int(value) for value in graph.edge_ids) != set(classification_rows):
        raise ValueError("Nominal graph strut IDs do not match Stage 3 classifications")
    edge_rows = {
        int(edge_id): row for row, edge_id in enumerate(graph.edge_ids.tolist())
    }
    axis_index = axis_by_name[excluded_nominal_axis]

    def export_row(strut_id: int, defect_class: str) -> dict[str, Any]:
        edge_row = edge_rows[strut_id]
        endpoint_rows = graph.edge_node_rows[edge_row]
        first = graph.node_positions_xyz[int(endpoint_rows[0])]
        second = graph.node_positions_xyz[int(endpoint_rows[1])]
        touches = bool(
            np.isclose(
                first[axis_index],
                excluded_nominal_value,
                rtol=0.0,
                atol=coordinate_tolerance,
            )
            or np.isclose(
                second[axis_index],
                excluded_nominal_value,
                rtol=0.0,
                atol=coordinate_tolerance,
            )
        )
        metric = metrics[strut_id]
        finding = findings[defect_class][strut_id]
        features = finding["features"]
        return {
            "strut_id": strut_id,
            "defect_class": defect_class,
            "touches_excluded_nominal_plane": touches,
            "excluded_nominal_axis": excluded_nominal_axis,
            "excluded_nominal_value": float(excluded_nominal_value),
            "x0_nominal": float(first[0]),
            "y0_nominal": float(first[1]),
            "z0_nominal": float(first[2]),
            "x1_nominal": float(second[0]),
            "y1_nominal": float(second[1]),
            "z1_nominal": float(second[2]),
            "same_material_component_connects_a_to_b": metric[
                "same_material_component_connects_a_to_b"
            ],
            "a_collar_foreground_fraction": metric[
                "a_collar_foreground_fraction"
            ],
            "b_collar_foreground_fraction": metric[
                "b_collar_foreground_fraction"
            ],
            "endpoint0_to_collar_component_voxel_count_in_corridor": metric[
                "endpoint0_to_collar_component_voxel_count_in_corridor"
            ],
            "endpoint1_to_collar_component_voxel_count_in_corridor": metric[
                "endpoint1_to_collar_component_voxel_count_in_corridor"
            ],
            "central_material_slice_fraction": features.get(
                "central_material_slice_fraction"
            ),
            "a_side_material_slice_fraction": features.get(
                "a_side_material_slice_fraction"
            ),
            "b_side_material_slice_fraction": features.get(
                "b_side_material_slice_fraction"
            ),
            "longest_empty_run_slices": features.get("longest_empty_run_slices"),
            "central_mean_foreground_fraction": features.get(
                "central_mean_foreground_fraction"
            ),
            "central_reference_foreground_fraction_p90": features.get(
                "central_reference_foreground_fraction_p90"
            ),
            "central_minimum_foreground_fraction": features.get(
                "central_minimum_foreground_fraction"
            ),
            "central_minimum_smoothed_foreground_fraction": features.get(
                "central_minimum_smoothed_foreground_fraction"
            ),
            "central_minimum_relative_to_reference": features.get(
                "central_minimum_relative_to_reference"
            ),
            "central_material_loss_fraction": features.get(
                "central_material_loss_fraction"
            ),
            "central_deficit_cutoff_foreground_fraction": features.get(
                "central_deficit_cutoff_foreground_fraction"
            ),
            "central_deficit_fraction": features.get("central_deficit_fraction"),
            "longest_deficit_run_samples": features.get(
                "longest_deficit_run_samples"
            ),
            "minimum_endpoint_collar_foreground_fraction": features.get(
                "minimum_endpoint_collar_foreground_fraction"
            ),
            "both_endpoint_segments_observed": features.get(
                "both_endpoint_segments_observed"
            ),
            "minimum_endpoint_to_collar_component_voxel_count_in_corridor": features.get(
                "minimum_endpoint_to_collar_component_voxel_count_in_corridor"
            ),
            "reasons": "|".join(str(value) for value in finding["reasons"]),
        }

    missing_rows = [
        export_row(strut_id, "missing")
        for strut_id, row in sorted(classification_rows.items())
        if row["class"] == "missing"
    ]
    broken_rows = [
        export_row(strut_id, "broken")
        for strut_id, row in sorted(classification_rows.items())
        if row["class"] == "broken"
    ]
    filtered_missing_rows = [
        row for row in missing_rows if not row["touches_excluded_nominal_plane"]
    ]
    # Claire's final broken viewer list kept material-loss cases from the
    # failed-connectivity (disconnected) candidate set, not connected bites.
    filtered_broken_rows = [
        row
        for row in broken_rows
        if not row["touches_excluded_nominal_plane"]
        and not bool(row["same_material_component_connects_a_to_b"])
    ]
    connected_bite_broken = sum(
        1
        for row in broken_rows
        if bool(row["same_material_component_connects_a_to_b"])
    )

    output = Path(output_directory).expanduser().resolve()
    missing_artifact = _write_validation_csv(
        output / "missing_struts.csv", missing_rows, overwrite=overwrite
    )
    broken_artifact = _write_validation_csv(
        output / "broken_struts.csv", broken_rows, overwrite=overwrite
    )
    filtered_missing_artifact = _write_validation_csv(
        output / "missing_struts_viewer_filtered.csv",
        filtered_missing_rows,
        overwrite=overwrite,
    )
    filtered_broken_artifact = _write_validation_csv(
        output / "broken_struts_viewer_filtered.csv",
        filtered_broken_rows,
        overwrite=overwrite,
    )
    manifest = {
        "schema_version": VALIDATION_EXPORT_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "non_authoritative": True,
        "production_receipt_artifact": False,
        "classification_gate": classifications["gate"],
        "development_mode": classifications["development_mode"],
        "sources": {
            "classified_struts_sha256": sha256_file(classifications_path),
            "per_strut_metrics_sha256": metrics_sha256,
            "findings_missing_sha256": findings_hashes["missing"],
            "findings_broken_sha256": findings_hashes["broken"],
            "nominal_graph_sha256": graph.source_sha256,
        },
        "filter": {
            "applies_to": [
                "missing_struts_viewer_filtered.csv",
                "broken_struts_viewer_filtered.csv",
            ],
            "coordinate_source": "nominal_graph.junctions[].position",
            "axis": excluded_nominal_axis,
            "value": float(excluded_nominal_value),
            "coordinate_tolerance": float(coordinate_tolerance),
            "exclusion_rule": (
                "missing: either nominal endpoint touches the plane; "
                "broken: plane touch OR primary A-B still connected (connected bite)"
            ),
            "scientific_classification_changed": False,
            "intended_use": (
                "Specimen crop-face + Claire-like disconnected-broken viewer/report "
                "filter; not a scientific relabel of Stage 3 findings"
            ),
        },
        "counts": {
            "missing_all": len(missing_rows),
            "broken_all": len(broken_rows),
            "broken_connected_bite": connected_bite_broken,
            "missing_touching_excluded_plane": sum(
                1 for row in missing_rows if row["touches_excluded_nominal_plane"]
            ),
            "broken_touching_excluded_plane": sum(
                1 for row in broken_rows if row["touches_excluded_nominal_plane"]
            ),
            "missing_viewer_filtered": len(filtered_missing_rows),
            "broken_viewer_filtered": len(filtered_broken_rows),
        },
        "outputs": {
            "missing_all": {
                "filename": "missing_struts.csv",
                "sha256": missing_artifact["sha256"],
            },
            "broken_all": {
                "filename": "broken_struts.csv",
                "sha256": broken_artifact["sha256"],
            },
            "missing_viewer_filtered": {
                "filename": "missing_struts_viewer_filtered.csv",
                "sha256": filtered_missing_artifact["sha256"],
            },
            "broken_viewer_filtered": {
                "filename": "broken_struts_viewer_filtered.csv",
                "sha256": filtered_broken_artifact["sha256"],
            },
        },
    }
    manifest_artifact = write_json_atomic(
        output / "stage3_validation_export_manifest.json",
        manifest,
        overwrite=overwrite,
    )
    return {
        "gate": "pass",
        "specimen_id": specimen_id,
        "non_authoritative": True,
        "counts": manifest["counts"],
        "filter": manifest["filter"],
        "artifacts": {
            "missing_struts_csv": {
                **missing_artifact,
                "role": "stage3_validation_missing_csv",
                "retention": "regenerable",
            },
            "broken_struts_csv": {
                **broken_artifact,
                "role": "stage3_validation_broken_csv",
                "retention": "regenerable",
            },
            "missing_struts_viewer_filtered_csv": {
                **filtered_missing_artifact,
                "role": "stage3_validation_missing_filtered_csv",
                "retention": "regenerable",
            },
            "broken_struts_viewer_filtered_csv": {
                **filtered_broken_artifact,
                "role": "stage3_validation_broken_filtered_csv",
                "retention": "regenerable",
            },
            "validation_manifest": {
                **manifest_artifact,
                "role": "stage3_validation_export_manifest",
                "retention": "regenerable",
            },
        },
        "hashes": {
            "missing_struts_csv_sha256": missing_artifact["sha256"],
            "broken_struts_csv_sha256": broken_artifact["sha256"],
            "missing_struts_viewer_filtered_csv_sha256": filtered_missing_artifact[
                "sha256"
            ],
            "broken_struts_viewer_filtered_csv_sha256": filtered_broken_artifact[
                "sha256"
            ],
            "validation_manifest_sha256": manifest_artifact["sha256"],
        },
        "warnings": [
            "Validation CSVs are not production Stage 3 receipt artifacts",
            (
                f"The {excluded_nominal_axis}={float(excluded_nominal_value):g} "
                "exclusion changes only the viewer-filtered CSVs and report remap"
            ),
        ],
    }


def prepare_hackathon_report_classifications(
    classifications_path: str | Path,
    output_path: str | Path,
    *,
    nominal_graph_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    excluded_nominal_axis: str = "y",
    excluded_nominal_value: float = 18.0,
    coordinate_tolerance: float = 1e-9,
    require_disconnected_for_broken: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build report-ready classifications with deferred+crop remaps.

    Scientific ``classified_struts.json`` is left unchanged. This artifact is
    for hackathon spatial stats / NDE report only:

    1. ``deferred`` → ``present`` (thin/bent incomplete)
    2. ``missing``/``broken`` touching the excluded nominal plane → ``present``
       (specimen high-Y / XZ crop-face human artifact)
    3. ``broken`` that remain A-B connected (connected bites) → ``present``
       when ``require_disconnected_for_broken`` (Claire final-78 scope)
    """

    axis_by_name = {"x": 0, "y": 1, "z": 2}
    if excluded_nominal_axis not in axis_by_name:
        raise ValueError("excluded_nominal_axis must be one of x, y, or z")
    if not math.isfinite(float(excluded_nominal_value)):
        raise ValueError("excluded_nominal_value must be finite")
    if not math.isfinite(float(coordinate_tolerance)) or coordinate_tolerance < 0:
        raise ValueError("coordinate_tolerance must be finite and nonnegative")

    classifications = read_json_object(classifications_path)
    plane_touching: set[int] = set()
    if nominal_graph_path is not None:
        graph = load_lattice_json(nominal_graph_path)
        axis_index = axis_by_name[excluded_nominal_axis]
        edge_rows = {
            int(edge_id): row for row, edge_id in enumerate(graph.edge_ids.tolist())
        }
        for strut_id, edge_row in edge_rows.items():
            endpoints = graph.edge_node_rows[edge_row]
            first = graph.node_positions_xyz[int(endpoints[0])]
            second = graph.node_positions_xyz[int(endpoints[1])]
            if np.isclose(
                first[axis_index],
                excluded_nominal_value,
                rtol=0.0,
                atol=coordinate_tolerance,
            ) or np.isclose(
                second[axis_index],
                excluded_nominal_value,
                rtol=0.0,
                atol=coordinate_tolerance,
            ):
                plane_touching.add(strut_id)

    connected_by_strut: dict[int, bool] = {}
    if metrics_path is not None and require_disconnected_for_broken:
        for strut_id, metric in _metric_map(metrics_path).items():
            connected_by_strut[int(strut_id)] = bool(
                metric["same_material_component_connects_a_to_b"]
            )

    remapped: list[dict[str, Any]] = []
    deferred_count = 0
    crop_excluded = {"missing": 0, "broken": 0}
    connected_bite_excluded = 0
    for row in classifications.get("classifications", []):
        item = dict(row)
        strut_id = int(item["strut_id"])
        label = item.get("class")
        if label == "deferred":
            deferred_count += 1
            item["class"] = "present"
            item["reasons"] = list(item.get("reasons") or []) + [
                "hackathon_remap_deferred_to_present_for_report"
            ]
        elif label in {"missing", "broken"} and strut_id in plane_touching:
            crop_excluded[label] += 1
            item["class"] = "present"
            item["reasons"] = list(item.get("reasons") or []) + [
                (
                    "hackathon_exclude_crop_plane_"
                    f"{excluded_nominal_axis}_{float(excluded_nominal_value):g}"
                )
            ]
        elif (
            label == "broken"
            and require_disconnected_for_broken
            and connected_by_strut.get(strut_id, False)
        ):
            connected_bite_excluded += 1
            item["class"] = "present"
            item["reasons"] = list(item.get("reasons") or []) + [
                "hackathon_exclude_connected_bite_broken_for_report"
            ]
        remapped.append(item)

    counts = Counter(str(row["class"]) for row in remapped)
    payload = dict(classifications)
    payload["classifications"] = remapped
    payload["counts"] = {
        key: int(counts.get(key, 0))
        for key in ("missing", "broken", "thin", "present")
    }
    payload["hackathon_report_remap"] = {
        "deferred_to_present": True,
        "deferred_count": deferred_count,
        "connected_bite_broken_to_present": {
            "enabled": require_disconnected_for_broken and metrics_path is not None,
            "excluded_count": connected_bite_excluded,
            "rule": "Claire final-78: report broken requires A-B disconnection",
        },
        "crop_plane_exclusion": {
            "enabled": nominal_graph_path is not None,
            "axis": excluded_nominal_axis,
            "value": float(excluded_nominal_value),
            "coordinate_tolerance": float(coordinate_tolerance),
            "excluded_missing": crop_excluded["missing"],
            "excluded_broken": crop_excluded["broken"],
        },
        "source_classifications": str(classifications_path),
        "scientific_classification_changed": False,
    }
    artifact = write_json_atomic(output_path, payload, overwrite=overwrite)
    return {
        "gate": "pass",
        "counts": payload["counts"],
        "deferred_remapped": deferred_count,
        "crop_excluded_missing": crop_excluded["missing"],
        "crop_excluded_broken": crop_excluded["broken"],
        "connected_bite_excluded_broken": connected_bite_excluded,
        "artifacts": {
            "classified_struts_report": {
                **artifact,
                "role": "classified_struts_report",
                "retention": "regenerable",
            }
        },
        "hashes": {"classified_struts_report_sha256": artifact["sha256"]},
        "warnings": [
            *(
                [
                    (
                        "Remapped crop-plane missing/broken to present for report: "
                        f"missing={crop_excluded['missing']}, "
                        f"broken={crop_excluded['broken']}"
                    )
                ]
                if crop_excluded["missing"] or crop_excluded["broken"]
                else []
            ),
            *(
                [
                    (
                        "Remapped connected-bite broken to present for report: "
                        f"broken={connected_bite_excluded}"
                    )
                ]
                if connected_bite_excluded
                else []
            ),
        ],
    }
