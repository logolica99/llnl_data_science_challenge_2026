"""Hash-sealed specimen-ingest hand-off and data-prep completion adapter."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from specimen_manifest import (
    ANALYSIS_READY,
    DEFAULT_SCHEMA,
    ManifestValidationError,
    canonical_json_sha256,
    load_json,
    sha256_file,
    validate_manifest,
)
from specimen_ingest import validate_ingest_artifact_bundle


HANDOFF_SCHEMA_VERSION = "data-prep-handoff/1.1.0"
RESULT_SCHEMA_VERSION = "data-prep-result/1.2.0"
COMPLETION_SCHEMA_VERSION = "data-prep-completion/1.2.0"
SEGMENTATION_VERIFICATION_SCHEMA_VERSION = (
    "segmentation-verification-mcp-evidence/1.0.0"
)
REQUIRED_DERIVED = {
    "graph_summary",
    "voxel_spacing",
    "segmentation_result",
    "registration_result",
}
ROI_OUTPUTS = {
    "segmentation",
    "registration",
    "node_localization",
    "coarse_region_screening",
    "padded_roi_definition",
}
DIRECT_METROLOGY_OUTPUTS = {
    "absolute_metrology",
    "direct_dimensional_measurement",
}
DATA_PREP_RESULT_FIELDS = {
    "schema_version",
    "specimen_id",
    "design_id",
    "requested_analysis_scope",
    "registration_mode",
    "input_manifest_sha256",
    "input_manifest_artifact_sha256",
    "analysis_parameters_sha256",
    "authorized_outputs",
    "unauthorized_outputs",
    "roi_gate_results",
    "metrology_gate_status",
    "localization_quality_counts",
    "reason_codes",
    "artifact_bindings",
    "aligned_graph",
    "canonical_mask",
    "derived",
    "self_verification",
}
DATA_PREP_ARTIFACT_BINDING_ROLES = {
    "segmentation_verification_mcp_response",
    "localization_report",
    "registration_qa",
}
DATA_PREP_SELF_VERIFICATION_FIELDS = {
    "exact_otsu_complete",
    "registration_complete",
    "local_recenter_complete",
    "roi_gate_pass",
    "scope_bound_to_hashed_intake",
    "localization_quality_propagated",
    "defect_labels_not_accessed",
}
DATA_PREP_ROI_GATE_FIELD_SETS = (
    {
        "image_support",
        "localization_quality",
        "coarse_region_support",
        "padded_roi_in_bounds",
    },
    {
        "production_image_qa_pass",
        "coarse_capture_pass",
        "localization_binding_pass",
        "localization_quantitative_gate_pass",
        "padded_roi_capture_pass",
        "overall_pass",
    },
)
DATA_PREP_LOCALIZATION_COUNT_FIELD_SETS = (
    {
        "primary",
        "stable_coarse",
        "fallback",
        "ambiguous",
        "rejected",
        "boundary_limited",
    },
    {
        "primary_nodes",
        "stable_coarse_nodes",
        "fallback_nodes",
        "ambiguous_nodes",
        "rejected_or_low_confidence_nodes",
        "boundary_limited_nodes",
        "primary_edges",
        "stable_coarse_edges",
        "fallback_edges",
        "ambiguous_edges",
        "roi_screening_usable_edges",
        "direct_metrology_usable_edges",
    },
)


class DataPrepHandoffError(ValueError):
    """Raised when an intake or data-prep envelope fails its hash contract."""


def _closed_object(value: Any, keys: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataPrepHandoffError(f"{field} schema is open or incomplete")
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unexpected = sorted(set(value) - keys)
        raise DataPrepHandoffError(
            f"{field} schema is open or incomplete; missing={missing}, "
            f"unexpected={unexpected}"
        )
    return value


def validate_data_prep_result_shape(result: Any) -> dict[str, Any]:
    """Validate the closed, non-scientific control surface of a Stage 1 result."""

    result_object = _closed_object(
        result,
        DATA_PREP_RESULT_FIELDS,
        field="data-prep result",
    )
    artifact_bindings = _closed_object(
        result_object["artifact_bindings"],
        DATA_PREP_ARTIFACT_BINDING_ROLES,
        field="data-prep artifact_bindings",
    )
    for role in sorted(DATA_PREP_ARTIFACT_BINDING_ROLES):
        binding = _closed_object(
            artifact_bindings[role],
            {"path", "sha256", "role"},
            field=f"data-prep artifact binding {role}",
        )
        if binding["role"] != role:
            raise DataPrepHandoffError(
                f"Data-prep artifact binding role mismatch for {role}"
            )
    _closed_object(
        result_object["self_verification"],
        DATA_PREP_SELF_VERIFICATION_FIELDS,
        field="data-prep self_verification",
    )
    _closed_object(
        result_object["aligned_graph"],
        {"path", "sha256", "role", "retention"},
        field="data-prep aligned_graph",
    )
    _closed_object(
        result_object["canonical_mask"],
        {"path", "sha256", "role", "retention", "dtype", "shape", "array_axes"},
        field="data-prep canonical_mask",
    )
    _closed_object(
        result_object["derived"],
        REQUIRED_DERIVED,
        field="data-prep derived records",
    )

    roi_gate_results = result_object["roi_gate_results"]
    if (
        not isinstance(roi_gate_results, dict)
        or set(roi_gate_results) not in DATA_PREP_ROI_GATE_FIELD_SETS
    ):
        raise DataPrepHandoffError(
            "Data-prep roi_gate_results schema is open or incompatible"
        )
    quality_counts = result_object["localization_quality_counts"]
    if (
        not isinstance(quality_counts, dict)
        or set(quality_counts) not in DATA_PREP_LOCALIZATION_COUNT_FIELD_SETS
    ):
        raise DataPrepHandoffError(
            "Data-prep localization_quality_counts schema is open or incompatible"
        )
    return result_object


def _validate_canonical_mask(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path,
) -> None:
    """Consume hash-bound MCP evidence for the exact-Otsu canonical mask."""

    canonical_mask = result.get("canonical_mask")
    if not isinstance(canonical_mask, dict):
        raise DataPrepHandoffError(
            "Data-prep result must provide canonical_mask as an artifact object"
        )
    required = {"path", "sha256", "role", "retention", "dtype", "shape", "array_axes"}
    if set(canonical_mask) != required:
        raise DataPrepHandoffError("canonical_mask contract fields are incomplete")
    repository = repository_root.resolve()
    mask_relative = Path(str(canonical_mask.get("path", "")))
    if (
        mask_relative.is_absolute()
        or ".." in mask_relative.parts
        or mask_relative.suffix.lower() != ".npy"
    ):
        raise DataPrepHandoffError(
            "Canonical mask must be a repository-contained NPY artifact"
        )
    mask_path = (repository / mask_relative).resolve()
    if (
        not mask_path.is_relative_to(repository)
        or not mask_path.is_file()
        or sha256_file(mask_path) != canonical_mask.get("sha256")
    ):
        raise DataPrepHandoffError("Canonical mask is missing or has a SHA-256 mismatch")

    expected_shape = manifest["inputs"]["ct_metadata"]["shape"]
    if (
        canonical_mask.get("role") != "canonical_segmentation_mask"
        or canonical_mask.get("dtype") != "uint8"
        or canonical_mask.get("shape") != expected_shape
        or canonical_mask.get("array_axes") != ["z", "y", "x"]
        or canonical_mask.get("retention") not in {"committed", "regenerable"}
    ):
        raise DataPrepHandoffError(
            "Canonical mask path/role/dtype/shape/retention/axes contract mismatch"
        )

    ct_descriptor = manifest["inputs"]["ct"]
    ct_relative = Path(str(ct_descriptor.get("path", "")))
    if ct_relative.is_absolute() or ".." in ct_relative.parts:
        raise DataPrepHandoffError("Manifest CT path escapes repository")
    ct_path = (repository / ct_relative).resolve()
    if (
        not ct_path.is_relative_to(repository)
        or not ct_path.is_file()
        or sha256_file(ct_path) != ct_descriptor.get("sha256")
    ):
        raise DataPrepHandoffError("Manifest CT is missing or has a SHA-256 mismatch")

    segmentation_policy = manifest["analysis_parameters"]["segmentation"]
    segmentation_record = result["derived"]["segmentation_result"]
    segmentation_values = segmentation_record.get("values", {})
    provenance = segmentation_record.get("provenance", {})
    if (
        segmentation_policy.get("method") != "exact_histogram_otsu"
        or segmentation_policy.get("method_version") != "2.0.0"
        or segmentation_policy.get("comparison") != "value >= threshold"
        or segmentation_record.get("method") != "exact_histogram_otsu"
        or segmentation_record.get("method_version") != "2.0.0"
        or provenance.get("config_sha256")
        != manifest["analysis_parameters_sha256"]
        or provenance.get("input_sha256") != [ct_descriptor["sha256"]]
        or segmentation_values.get("overall_pass") is not True
    ):
        raise DataPrepHandoffError(
            "Canonical mask is not bound to the frozen exact-Otsu result"
        )
    artifact_bindings = result.get("artifact_bindings")
    if not isinstance(artifact_bindings, dict):
        raise DataPrepHandoffError("Data-prep result has no artifact_bindings object")
    evidence_binding = _closed_object(
        artifact_bindings.get("segmentation_verification_mcp_response"),
        {"path", "sha256", "role"},
        field="segmentation verification artifact binding",
    )
    expected_evidence_relative = (
        Path("analysis")
        / manifest["specimen_id"]
        / "segmentation"
        / "segmentation_verification_mcp_response.json"
    )
    evidence_relative = Path(str(evidence_binding["path"]))
    if (
        evidence_binding["role"] != "segmentation_verification_mcp_response"
        or evidence_relative != expected_evidence_relative
        or evidence_relative.is_absolute()
        or ".." in evidence_relative.parts
    ):
        raise DataPrepHandoffError(
            "Segmentation verification evidence is not specimen scoped"
        )
    evidence_path = (repository / evidence_relative).resolve()
    if (
        not evidence_path.is_relative_to(repository)
        or not evidence_path.is_file()
        or sha256_file(evidence_path) != evidence_binding["sha256"]
    ):
        raise DataPrepHandoffError(
            "Segmentation verification evidence is missing or has a SHA-256 mismatch"
        )
    evidence = load_json(evidence_path)
    _closed_object(
        evidence,
        {
            "schema_version",
            "response_schema_version",
            "tool",
            "status",
            "gate",
            "summary",
            "specimen_id",
            "design_id",
            "requested_analysis_scope",
            "registration_mode",
            "request",
            "policy",
            "result",
            "bindings",
            "hashes",
            "warnings",
            "error",
        },
        field="segmentation verification evidence",
    )
    requested_scope = manifest["analysis_parameters"]["requested_analysis_scope"]
    registration_mode = manifest["analysis_parameters"]["registration"]["mode"]
    if (
        evidence["schema_version"] != SEGMENTATION_VERIFICATION_SCHEMA_VERSION
        or evidence["response_schema_version"] != "part2-mcp-response/1.0.0"
        or evidence["tool"] != "verify_canonical_segmentation"
        or evidence["status"] != "ok"
        or evidence["gate"] != "pass"
        or evidence["summary"] != "Persisted canonical segmentation verification"
        or evidence["specimen_id"] != manifest["specimen_id"]
        or evidence["design_id"] != manifest["design_id"]
        or evidence["requested_analysis_scope"] != requested_scope
        or evidence["registration_mode"] != registration_mode
        or evidence["warnings"] != []
        or evidence["error"] is not None
    ):
        raise DataPrepHandoffError(
            "Segmentation verification evidence identity or terminal gate is invalid"
        )

    try:
        manifest_relative = manifest_path.resolve().relative_to(repository).as_posix()
    except ValueError as exc:
        raise DataPrepHandoffError("Specimen manifest path escapes repository") from exc
    expected_segmentation_root = Path("analysis") / manifest["specimen_id"] / "segmentation"
    expected_otsu_relative = expected_segmentation_root / "histogram_report.json"
    expected_comparison_relative = expected_segmentation_root / "mask_comparison.json"
    request = _closed_object(
        evidence["request"],
        {
            "specimen_id",
            "design_id",
            "analysis_policy_artifact_filepath",
            "exact_otsu_report_filepath",
            "canonical_mask_filepath",
            "mask_comparison_report_filepath",
            "output_filepath",
            "registration_mode",
            "overwrite",
        },
        field="segmentation verification request",
    )
    expected_request = {
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "analysis_policy_artifact_filepath": manifest_relative,
        "exact_otsu_report_filepath": expected_otsu_relative.as_posix(),
        "canonical_mask_filepath": mask_relative.as_posix(),
        "mask_comparison_report_filepath": expected_comparison_relative.as_posix(),
        "output_filepath": expected_evidence_relative.as_posix(),
        "registration_mode": registration_mode,
        "overwrite": False,
    }
    if request != expected_request:
        raise DataPrepHandoffError(
            "Segmentation verification request is stale or cross-specimen"
        )

    policy = _closed_object(
        evidence["policy"],
        {"analysis_parameters_sha256", "segmentation_policy_sha256"},
        field="segmentation verification policy",
    )
    expected_segmentation_policy_sha256 = canonical_json_sha256(segmentation_policy)
    if policy != {
        "analysis_parameters_sha256": manifest["analysis_parameters_sha256"],
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
    }:
        raise DataPrepHandoffError(
            "Segmentation verification evidence uses a stale frozen policy"
        )

    bindings = _closed_object(
        evidence["bindings"],
        {
            "analysis_policy_artifact",
            "ct_volume",
            "exact_otsu_report",
            "canonical_mask",
            "mask_comparison_report",
        },
        field="segmentation verification bindings",
    )
    expected_binding_fields = {
        "analysis_policy_artifact": {"path", "sha256", "role"},
        "ct_volume": {"path", "sha256", "role"},
        "exact_otsu_report": {"path", "sha256", "role"},
        "canonical_mask": {
            "path",
            "sha256",
            "role",
            "dtype",
            "shape",
            "array_axes",
        },
        "mask_comparison_report": {"path", "sha256", "role"},
    }
    for name, fields in expected_binding_fields.items():
        _closed_object(
            bindings[name],
            fields,
            field=f"segmentation verification binding {name}",
        )
    otsu_path = (repository / expected_otsu_relative).resolve()
    comparison_path = (repository / expected_comparison_relative).resolve()
    for path, label in (
        (otsu_path, "exact Otsu report"),
        (comparison_path, "mask comparison report"),
    ):
        if not path.is_relative_to(repository) or not path.is_file():
            raise DataPrepHandoffError(f"Bound {label} is unavailable")
    expected_bindings = {
        "analysis_policy_artifact": {
            "path": manifest_relative,
            "sha256": result["input_manifest_artifact_sha256"],
            "role": "specimen_manifest",
        },
        "ct_volume": {
            "path": ct_relative.as_posix(),
            "sha256": ct_descriptor["sha256"],
            "role": "ct_volume",
        },
        "exact_otsu_report": {
            "path": expected_otsu_relative.as_posix(),
            "sha256": sha256_file(otsu_path),
            "role": "otsu_report",
        },
        "canonical_mask": {
            "path": mask_relative.as_posix(),
            "sha256": canonical_mask["sha256"],
            "role": "canonical_segmentation_mask",
            "dtype": "uint8",
            "shape": expected_shape,
            "array_axes": ["z", "y", "x"],
        },
        "mask_comparison_report": {
            "path": expected_comparison_relative.as_posix(),
            "sha256": sha256_file(comparison_path),
            "role": "segmentation_mask_comparison",
        },
    }
    if bindings != expected_bindings:
        raise DataPrepHandoffError(
            "Segmentation verification artifact bindings are stale or mismatched"
        )

    evidence_hashes = _closed_object(
        evidence["hashes"],
        {
            "request_sha256",
            "analysis_policy_artifact_sha256",
            "analysis_parameters_sha256",
            "segmentation_policy_sha256",
            "ct_sha256",
            "exact_otsu_report_sha256",
            "canonical_mask_sha256",
            "mask_comparison_report_sha256",
        },
        field="segmentation verification hashes",
    )
    expected_hashes = {
        "request_sha256": canonical_json_sha256(request),
        "analysis_policy_artifact_sha256": result[
            "input_manifest_artifact_sha256"
        ],
        "analysis_parameters_sha256": manifest["analysis_parameters_sha256"],
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
        "ct_sha256": ct_descriptor["sha256"],
        "exact_otsu_report_sha256": expected_bindings["exact_otsu_report"]["sha256"],
        "canonical_mask_sha256": canonical_mask["sha256"],
        "mask_comparison_report_sha256": expected_bindings[
            "mask_comparison_report"
        ]["sha256"],
    }
    if evidence_hashes != expected_hashes:
        raise DataPrepHandoffError(
            "Segmentation verification hash bindings are stale or mismatched"
        )

    verified = _closed_object(
        evidence["result"],
        {
            "threshold",
            "threshold_comparison",
            "shape",
            "dtype",
            "voxel_count",
            "foreground_voxel_count",
            "foreground_fraction",
            "otsu_separability",
            "background_mean",
            "foreground_mean",
            "class_mean_separation_sigma",
            "significant_modes",
            "histogram_sha256",
            "mismatched_voxels",
            "false_positive_voxels",
            "false_negative_voxels",
            "exact_threshold_match",
            "overall_pass",
        },
        field="segmentation verification result",
    )
    if (
        verified["threshold_comparison"] != "value >= threshold"
        or verified["shape"] != expected_shape
        or verified["dtype"] != "uint8"
        or verified["mismatched_voxels"] != 0
        or verified["false_positive_voxels"] != 0
        or verified["false_negative_voxels"] != 0
        or verified["exact_threshold_match"] is not True
        or verified["overall_pass"] is not True
    ):
        raise DataPrepHandoffError(
            "Segmentation verification did not prove an exact canonical mask"
        )

    exact_fields = (
        "threshold",
        "voxel_count",
        "foreground_voxel_count",
        "significant_modes",
        "histogram_sha256",
        "overall_pass",
    )
    float_fields = (
        "foreground_fraction",
        "otsu_separability",
        "background_mean",
        "foreground_mean",
        "class_mean_separation_sigma",
    )
    exact_mismatches = [
        field
        for field in exact_fields
        if segmentation_values.get(field) != verified.get(field)
    ]
    for field in float_fields:
        declared = segmentation_values.get(field)
        replayed = verified.get(field)
        if (
            not isinstance(declared, (int, float))
            or isinstance(declared, bool)
            or not isinstance(replayed, (int, float))
            or isinstance(replayed, bool)
            or not math.isclose(
                float(declared),
                float(replayed),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            exact_mismatches.append(field)
    if exact_mismatches:
        raise DataPrepHandoffError(
            "Canonical mask segmentation result differs from its MCP verification: "
            + ", ".join(exact_mismatches)
        )


def _expected_authorizations(scope: str) -> tuple[set[str], set[str]]:
    if scope == "roi_screening":
        return set(ROI_OUTPUTS), set(DIRECT_METROLOGY_OUTPUTS)
    if scope == "direct_metrology":
        return set(ROI_OUTPUTS | DIRECT_METROLOGY_OUTPUTS), set()
    raise DataPrepHandoffError(f"Unsupported requested analysis scope: {scope!r}")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _atomic_write_if_changed(path: Path, value: Any) -> bool:
    payload = _json_bytes(value)
    if path.is_file() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return True


def _verify_intake_receipt(
    manifest: dict[str, Any], receipt: dict[str, Any]
) -> None:
    if receipt.get("schema_version") != "ingest-receipt/1.3.0":
        raise DataPrepHandoffError("Unsupported ingest receipt schema")
    if receipt.get("specimen_id") != manifest["specimen_id"]:
        raise DataPrepHandoffError("Receipt specimen_id does not match manifest")
    if receipt.get("design_id") != manifest["design_id"]:
        raise DataPrepHandoffError("Receipt design_id does not match manifest")
    requested_scope = manifest["analysis_parameters"]["requested_analysis_scope"]
    if receipt.get("requested_analysis_scope") != requested_scope:
        raise DataPrepHandoffError(
            "Receipt requested_analysis_scope does not match manifest"
        )
    receipt_without_hash = {
        key: value
        for key, value in receipt.items()
        if key != "canonical_receipt_sha256"
    }
    if receipt.get("canonical_receipt_sha256") != canonical_json_sha256(
        receipt_without_hash
    ):
        raise DataPrepHandoffError("Ingest receipt canonical hash is invalid")
    expected_manifest_hash = canonical_json_sha256(manifest)
    if receipt.get("manifest_sha256") != expected_manifest_hash:
        raise DataPrepHandoffError(
            "Receipt manifest_sha256 does not match the current intake manifest"
        )
    receipt_hashes = receipt.get("input_sha256", {})
    for name, artifact in manifest["inputs"].items():
        if name == "ct_metadata":
            continue
        if receipt_hashes.get(name) != artifact["sha256"]:
            raise DataPrepHandoffError(
                f"Receipt input hash does not match manifest input {name}"
            )
    verification = receipt.get("self_verification", {})
    required_checks = (
        "association_explicit",
        "all_paths_repository_relative",
        "all_inputs_hashed",
        "ct_metadata_mcp_integrity_chain_valid",
        "ct_metadata_response_schema_closed",
        "ct_metadata_response_hash_bound",
        "ct_metadata_response_header_only",
        "ct_metadata_response_path_and_ct_hash_match",
        "ct_metadata_mcp_call_receipt_closed",
        "ct_metadata_mcp_call_receipt_hash_bound",
        "ct_metadata_header_facts_bound_to_call_receipt",
        "cad_readable",
        "cad_readable_or_not_supplied",
        "graph_id_reference_integrity",
        "normalized_graph_hash_bound",
        "manifest_schema_valid",
        "segmentation_not_run",
        "registration_not_run",
        "defect_labels_not_derived",
    )
    failed = [name for name in required_checks if verification.get(name) is not True]
    if failed:
        raise DataPrepHandoffError(
            "Ingest receipt failed self-verification: " + ", ".join(failed)
        )


def _build_data_prep_handoff(
    manifest: dict[str, Any],
    receipt: dict[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    state = manifest["lifecycle_state"]
    if state == "ready_for_data_prep":
        status = "ready"
        action = "run_data_prep"
    elif state == "provisional":
        status = "halt"
        action = "resolve_intake_fields"
    elif state == ANALYSIS_READY:
        status = "complete"
        action = "none"
    else:
        raise DataPrepHandoffError(f"Unsupported lifecycle state: {state}")

    allowlisted_inputs = {
        name: {
            "path": artifact["path"],
            "sha256": artifact["sha256"],
            "role": artifact["role"],
        }
        for name, artifact in manifest["inputs"].items()
        if name != "ct_metadata"
    }
    handoff_base = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "requested_analysis_scope": manifest["analysis_parameters"][
            "requested_analysis_scope"
        ],
        "status": status,
        "action": action,
        "lifecycle_state": state,
        "manifest_path": manifest_path.resolve().relative_to(
            repository_root.resolve()
        ).as_posix(),
        "manifest_sha256": canonical_json_sha256(manifest),
        "ingest_receipt_sha256": receipt["canonical_receipt_sha256"],
        "analysis_parameters_sha256": manifest["analysis_parameters_sha256"],
        "registration_mode": manifest["analysis_parameters"]["registration"]["mode"],
        "localization_policy": manifest["analysis_parameters"]["localization_policy"],
        "qa_policy": manifest["analysis_parameters"]["qa_policy"],
        "authorized_outputs": (
            [
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
            ]
            if manifest["analysis_parameters"]["requested_analysis_scope"]
            == "roi_screening"
            else [
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
                "absolute_metrology",
                "direct_dimensional_measurement",
            ]
        ),
        "unauthorized_outputs": (
            ["absolute_metrology", "direct_dimensional_measurement"]
            if manifest["analysis_parameters"]["requested_analysis_scope"]
            == "roi_screening"
            else []
        ),
        "allowlisted_inputs": allowlisted_inputs,
        "unresolved_fields": manifest["unresolved_fields"],
        "forbidden_inputs": [
            "defect labels",
            "dev split",
            "sealed split",
            "ground-truth segmentation",
        ],
        "required_outputs": [
            "registered and independently localized nominal graph",
            "exact-histogram Otsu result",
            "canonical uint8 ZYX mask contract",
            "bounded segmentation-mask comparison",
            "registration QA",
            "local node recentering",
            "ROI capture gate",
            "metrology gate",
            "data-prep completion receipt",
        ],
        "maximum_agent_retries": manifest["analysis_parameters"]["budgets"][
            "maximum_agent_retries"
        ],
    }
    return {
        **handoff_base,
        "canonical_handoff_sha256": canonical_json_sha256(handoff_base),
    }


def _validated_ingest_bundle(
    manifest_path: Path,
    receipt_path: Path,
    *,
    repository_root: Path,
    schema_path: Path,
    require_ready: bool,
    expected_specimen_id: str | None = None,
    expected_design_id: str | None = None,
    expected_analysis_scope: str | None = None,
    expected_registration_mode: str | None = None,
) -> dict[str, Any]:
    config_directory = manifest_path.resolve().parent
    try:
        bundle = validate_ingest_artifact_bundle(
            repository_root=repository_root,
            manifest_path=manifest_path,
            request_path=config_directory / "ingest_request.json",
            receipt_path=receipt_path,
            ct_metadata_response_path=(
                config_directory / "ct_metadata_response.json"
            ),
            ct_metadata_call_receipt_path=(
                config_directory / "ct_metadata_mcp_call_receipt.json"
            ),
            schema_path=schema_path,
            expected_specimen_id=expected_specimen_id,
            expected_design_id=expected_design_id,
            expected_analysis_scope=expected_analysis_scope,
            expected_registration_mode=expected_registration_mode,
            require_ready=require_ready,
        )
    except (OSError, ValueError) as exc:
        raise DataPrepHandoffError(
            f"Stage 0 intake artifact bundle failed validation: {exc}"
        ) from exc
    _verify_intake_receipt(bundle["manifest"], bundle["receipt"])
    return bundle


def create_data_prep_handoff(
    manifest_path: Path,
    receipt_path: Path,
    *,
    repository_root: Path,
    output_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Verify intake artifacts and emit the deterministic next-stage envelope."""

    bundle = _validated_ingest_bundle(
        manifest_path,
        receipt_path,
        repository_root=repository_root,
        schema_path=schema_path,
        require_ready=False,
    )
    handoff = _build_data_prep_handoff(
        bundle["manifest"],
        bundle["receipt"],
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    destination = output_path or (
        manifest_path.parent / "data_prep_handoff.json"
    )
    changed = _atomic_write_if_changed(destination, handoff)
    return {
        "handoff": handoff,
        "path": str(destination),
        "changed": changed,
    }


def validate_stage0_artifact_bundle(
    *,
    repository_root: Path,
    manifest_path: Path,
    request_path: Path,
    receipt_path: Path,
    ct_metadata_response_path: Path,
    ct_metadata_call_receipt_path: Path,
    handoff_path: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    expected_specimen_id: str,
    expected_design_id: str,
    expected_analysis_scope: str,
    expected_registration_mode: str,
) -> dict[str, Any]:
    """Semantically validate all Stage 0 pass artifacts without modifying them."""

    try:
        bundle = validate_ingest_artifact_bundle(
            repository_root=repository_root,
            manifest_path=manifest_path,
            request_path=request_path,
            receipt_path=receipt_path,
            ct_metadata_response_path=ct_metadata_response_path,
            ct_metadata_call_receipt_path=ct_metadata_call_receipt_path,
            schema_path=schema_path,
            expected_specimen_id=expected_specimen_id,
            expected_design_id=expected_design_id,
            expected_analysis_scope=expected_analysis_scope,
            expected_registration_mode=expected_registration_mode,
            require_ready=True,
        )
    except (OSError, ValueError) as exc:
        raise DataPrepHandoffError(
            f"Stage 0 intake artifact bundle failed validation: {exc}"
        ) from exc
    _verify_intake_receipt(bundle["manifest"], bundle["receipt"])
    expected_handoff = _build_data_prep_handoff(
        bundle["manifest"],
        bundle["receipt"],
        manifest_path=manifest_path,
        repository_root=repository_root,
    )
    try:
        actual_handoff = load_json(handoff_path)
    except (OSError, ValueError) as exc:
        raise DataPrepHandoffError("Stage 0 data-prep handoff is unreadable") from exc
    if actual_handoff != expected_handoff:
        raise DataPrepHandoffError(
            "Stage 0 data-prep handoff is stale, open, or inconsistent"
        )
    return {**bundle, "data_prep_handoff": actual_handoff}


def _validate_data_prep_result(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    manifest_path: Path,
    repository_root: Path,
) -> None:
    validate_data_prep_result_shape(result)
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise DataPrepHandoffError("Unsupported data-prep result schema")
    if result.get("specimen_id") != manifest["specimen_id"]:
        raise DataPrepHandoffError("Data-prep result specimen_id mismatch")
    if result.get("design_id") != manifest["design_id"]:
        raise DataPrepHandoffError("Data-prep result design_id mismatch")
    if result.get("input_manifest_sha256") != canonical_json_sha256(manifest):
        raise DataPrepHandoffError("Data-prep result uses a stale intake manifest")
    if result.get("input_manifest_artifact_sha256") != sha256_file(manifest_path):
        raise DataPrepHandoffError(
            "Data-prep result uses a stale intake manifest artifact"
        )
    if (
        result.get("analysis_parameters_sha256")
        != manifest["analysis_parameters_sha256"]
    ):
        raise DataPrepHandoffError("Data-prep result uses stale analysis parameters")
    declared_mode = manifest["analysis_parameters"]["registration"]["mode"]
    if result.get("registration_mode", declared_mode) != declared_mode:
        raise DataPrepHandoffError("Data-prep result registration_mode mismatch")
    requested_scope = manifest["analysis_parameters"]["requested_analysis_scope"]
    if result.get("requested_analysis_scope") != requested_scope:
        raise DataPrepHandoffError(
            "Data-prep result requested_analysis_scope mismatch"
        )
    verification = result["self_verification"]
    required_checks = tuple(sorted(DATA_PREP_SELF_VERIFICATION_FIELDS))
    failed = [name for name in required_checks if verification.get(name) is not True]
    if failed:
        raise DataPrepHandoffError(
            "Data-prep result failed self-verification: " + ", ".join(failed)
        )

    authorized = result.get("authorized_outputs")
    unauthorized = result.get("unauthorized_outputs")
    if (
        not isinstance(authorized, list)
        or not all(isinstance(value, str) and value for value in authorized)
        or len(authorized) != len(set(authorized))
        or not isinstance(unauthorized, list)
        or not all(isinstance(value, str) and value for value in unauthorized)
        or len(unauthorized) != len(set(unauthorized))
        or set(authorized) & set(unauthorized)
    ):
        raise DataPrepHandoffError(
            "Data-prep authorization lists must be disjoint unique string arrays"
        )
    expected_authorized, expected_unauthorized = _expected_authorizations(
        requested_scope
    )
    if set(authorized) != expected_authorized or set(unauthorized) != expected_unauthorized:
        raise DataPrepHandoffError(
            "Data-prep authorization lists do not exactly match the requested scope"
        )
    roi_gate_results = result.get("roi_gate_results")
    if (
        not isinstance(roi_gate_results, dict)
        or not roi_gate_results
        or not all(value is True for value in roi_gate_results.values())
    ):
        raise DataPrepHandoffError("Data-prep ROI gates did not all pass")
    metrology_status = result.get("metrology_gate_status")
    expected_status = "not_authorized" if requested_scope == "roi_screening" else "pass"
    if metrology_status != expected_status:
        raise DataPrepHandoffError(
            "Data-prep metrology_gate_status is inconsistent with the requested scope"
        )
    quality_counts = result.get("localization_quality_counts")
    if (
        not isinstance(quality_counts, dict)
        or not quality_counts
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in quality_counts.values()
        )
    ):
        raise DataPrepHandoffError(
            "Data-prep localization_quality_counts are missing or malformed"
        )
    reason_codes = result.get("reason_codes")
    expected_reason_codes = {
        "ROI_GATES_PASS",
        (
            "METROLOGY_NOT_AUTHORIZED"
            if requested_scope == "roi_screening"
            else "METROLOGY_GATES_PASS"
        ),
    }
    if (
        not isinstance(reason_codes, list)
        or len(reason_codes) != len(expected_reason_codes)
        or set(reason_codes) != expected_reason_codes
    ):
        raise DataPrepHandoffError(
            "Data-prep reason_codes do not exactly match the requested scope"
        )

    artifact_bindings = result["artifact_bindings"]
    bound_documents: dict[str, dict[str, Any]] = {}
    for role in ("localization_report", "registration_qa"):
        binding = artifact_bindings.get(role)
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "role"}:
            raise DataPrepHandoffError(f"Missing closed artifact binding for {role}")
        if binding["role"] != role:
            raise DataPrepHandoffError(f"Artifact binding role mismatch for {role}")
        binding_relative = Path(str(binding["path"]))
        if binding_relative.is_absolute() or ".." in binding_relative.parts:
            raise DataPrepHandoffError(f"Artifact binding path escapes repository: {role}")
        expected_relative = (
            Path("analysis")
            / manifest["specimen_id"]
            / ("registration" if role == "localization_report" else "qa")
            / (
                "localization_report.json"
                if role == "localization_report"
                else "registration_qa.json"
            )
        )
        if binding_relative != expected_relative:
            raise DataPrepHandoffError(
                f"Artifact binding is outside the specimen-scoped Stage 1 path: {role}"
            )
        resolved_repository_root = repository_root.resolve()
        binding_path = (resolved_repository_root / binding_relative).resolve()
        if (
            not binding_path.is_relative_to(resolved_repository_root)
            or not binding_path.is_file()
            or sha256_file(binding_path) != binding.get("sha256")
        ):
            raise DataPrepHandoffError(f"Artifact binding is missing or stale: {role}")
        document = load_json(binding_path)
        bound_documents[role] = document
        expected_schema = (
            "part2-node-localization/1.2.0"
            if role == "localization_report"
            else "part2-registration-qa/1.2.0"
        )
        if document.get("schema_version") != expected_schema:
            raise DataPrepHandoffError(f"Artifact binding schema is incompatible: {role}")
        for field, expected_value in (
            ("specimen_id", manifest["specimen_id"]),
            ("design_id", manifest["design_id"]),
            ("requested_analysis_scope", requested_scope),
        ):
            if document.get(field) != expected_value:
                raise DataPrepHandoffError(
                    f"Artifact binding {field} mismatch: {role}"
                )
        if document.get("gate") != "pass" or document.get("overall_pass") is not True:
            raise DataPrepHandoffError(f"Artifact binding did not pass: {role}")
        if role == "registration_qa":
            if (
                set(document.get("authorized_outputs", [])) != expected_authorized
                or set(document.get("unauthorized_outputs", []))
                != expected_unauthorized
                or document.get("roi_gate_results") != roi_gate_results
                or document.get("localization_quality_counts") != quality_counts
                or document.get("reason_codes") != reason_codes
                or document.get("metrology", {}).get("status") != metrology_status
            ):
                raise DataPrepHandoffError(
                    "Data-prep result does not exactly reproduce bound registration QA policy fields"
                )

    localization_document = bound_documents["localization_report"]
    qa_document = bound_documents["registration_qa"]
    expected_ct_sha256 = manifest["inputs"]["ct"]["sha256"]
    expected_analysis_parameters_sha256 = manifest["analysis_parameters_sha256"]
    expected_localization_policy = manifest["analysis_parameters"][
        "localization_policy"
    ]
    expected_localization_policy_sha256 = canonical_json_sha256(
        expected_localization_policy
    )
    expected_localization_config_sha256 = canonical_json_sha256(
        {
            key: value
            for key, value in expected_localization_policy.items()
            if key != "schema_version"
        }
    )
    expected_qa_policy_sha256 = canonical_json_sha256(
        manifest["analysis_parameters"]["qa_policy"]
    )
    expected_qa_config_sha256 = canonical_json_sha256(
        {
            key: value
            for key, value in manifest["analysis_parameters"]["qa_policy"].items()
            if key != "schema_version"
        }
    )
    expected_segmentation_policy_sha256 = canonical_json_sha256(
        manifest["analysis_parameters"]["segmentation"]
    )
    expected_scope_artifact_sha256 = sha256_file(manifest_path)
    localization_hashes = localization_document.get("hashes", {})
    qa_hashes = qa_document.get("hashes", {})
    if not isinstance(localization_hashes, dict) or not isinstance(qa_hashes, dict):
        raise DataPrepHandoffError("Stage 1 reports must expose closed hash bindings")
    expected_localization_hash_keys = {
        "ct_sha256",
        "input_registered_graph_sha256",
        "localized_graph_sha256",
        "registration_report_sha256",
        "analysis_policy_artifact_sha256",
        "analysis_parameters_sha256",
        "localization_policy_sha256",
        "segmentation_policy_sha256",
    }
    expected_qa_hash_keys = {
        "ct_sha256",
        "localized_graph_sha256",
        "registration_report_sha256",
        "analysis_scope_artifact_sha256",
        "analysis_parameters_sha256",
        "localization_policy_sha256",
        "qa_policy_sha256",
        "segmentation_policy_sha256",
        "localization_report_sha256",
    }
    if (
        set(localization_hashes) != expected_localization_hash_keys
        or set(qa_hashes) != expected_qa_hash_keys
    ):
        raise DataPrepHandoffError("Stage 1 reports must use closed hash schemas")
    expected_localization_hashes = {
        "ct_sha256": expected_ct_sha256,
        "analysis_policy_artifact_sha256": expected_scope_artifact_sha256,
        "analysis_parameters_sha256": expected_analysis_parameters_sha256,
        "localization_policy_sha256": expected_localization_policy_sha256,
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
    }
    for name, expected_value in expected_localization_hashes.items():
        if localization_hashes.get(name) != expected_value:
            raise DataPrepHandoffError(
                f"Localization report {name} does not match the hashed specimen manifest"
            )
    localized_graph_sha256 = localization_hashes.get("localized_graph_sha256")
    if not isinstance(localized_graph_sha256, str) or len(localized_graph_sha256) != 64:
        raise DataPrepHandoffError(
            "Localization report localized_graph_sha256 is missing"
        )
    if localization_document.get("registration_mode") != declared_mode:
        raise DataPrepHandoffError("Localization report registration_mode mismatch")
    if localization_document.get("threshold") != result["derived"][
        "segmentation_result"
    ]["values"]["threshold"]:
        raise DataPrepHandoffError(
            "Localization threshold differs from the bound segmentation result"
        )
    expected_quantitative_policy = {
        key: expected_localization_policy[key]
        for key in (
            "minimum_primary_or_stable_coarse_fraction",
            "maximum_fallback_fraction",
            "maximum_ambiguous_fraction",
            "maximum_rejected_fraction",
            "maximum_boundary_limited_fraction",
        )
    }
    if (
        localization_document.get("quantitative_policy")
        != expected_quantitative_policy
        or localization_document.get("provenance", {}).get("policy_binding")
        != "hashed_analysis_parameters"
        or localization_document.get("provenance", {}).get("config_sha256")
        != expected_localization_config_sha256
    ):
        raise DataPrepHandoffError(
            "Localization report uses stale or looser localization policy"
        )
    localization_segmentation = localization_document.get(
        "segmentation_binding", {}
    )
    expected_localization_segmentation = {
        "method": "exact_histogram_otsu",
        "method_version": "2.0.0",
        "threshold": result["derived"]["segmentation_result"]["values"][
            "threshold"
        ],
        "threshold_comparison": "value >= threshold",
        "ct_sha256": expected_ct_sha256,
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
        "overall_pass": True,
    }
    if (
        localization_segmentation != expected_localization_segmentation
    ):
        raise DataPrepHandoffError(
            "Localization report is not bound to the exact segmentation result"
        )
    expected_localization_policy_source = {
        "source_artifact_path": str(manifest_path.resolve()),
        "source_artifact_sha256": expected_scope_artifact_sha256,
        "analysis_parameters_sha256": expected_analysis_parameters_sha256,
        "localization_policy_sha256": expected_localization_policy_sha256,
        "requested_analysis_scope": requested_scope,
        "registration_mode": declared_mode,
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "declared_ct_sha256": expected_ct_sha256,
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
    }
    if (
        localization_document.get("analysis_policy_source")
        != expected_localization_policy_source
    ):
        raise DataPrepHandoffError(
            "Localization report policy source is not the current specimen manifest"
        )

    expected_qa_hashes = {
        "ct_sha256": expected_ct_sha256,
        "localized_graph_sha256": localized_graph_sha256,
        "analysis_scope_artifact_sha256": expected_scope_artifact_sha256,
        "analysis_parameters_sha256": expected_analysis_parameters_sha256,
        "localization_policy_sha256": expected_localization_policy_sha256,
        "qa_policy_sha256": expected_qa_policy_sha256,
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
        "localization_report_sha256": artifact_bindings["localization_report"][
            "sha256"
        ],
    }
    for name, expected_value in expected_qa_hashes.items():
        if qa_hashes.get(name) != expected_value:
            raise DataPrepHandoffError(
                f"Registration QA {name} does not match its bound Stage 1 inputs"
            )
    if (
        qa_document.get("registration_mode") != declared_mode
        or qa_document.get("threshold")
        != result["derived"]["segmentation_result"]["values"]["threshold"]
        or qa_document.get("provenance", {}).get("policy_binding")
        != "hashed_analysis_parameters"
        or qa_document.get("provenance", {}).get("config_sha256")
        != expected_qa_config_sha256
    ):
        raise DataPrepHandoffError(
            "Registration QA mode, threshold, or policy binding is stale"
        )
    qa_segmentation = qa_document.get("segmentation_binding", {})
    expected_qa_segmentation = {
        **expected_localization_segmentation,
        "gates": {
            "exact_otsu_replay_passed": True,
            "threshold_matches_exact_otsu": True,
        },
    }
    if (
        qa_segmentation != expected_qa_segmentation
    ):
        raise DataPrepHandoffError(
            "Registration QA is not bound to the exact segmentation result"
        )
    localization_binding = qa_document.get("localization_binding")
    if (
        not isinstance(localization_binding, dict)
        or localization_binding.get("overall_pass") is not True
        or not isinstance(localization_binding.get("gates"), dict)
        or not localization_binding["gates"]
        or not all(value is True for value in localization_binding["gates"].values())
        or localization_binding.get("artifact", {}).get("sha256")
        != artifact_bindings["localization_report"]["sha256"]
    ):
        raise DataPrepHandoffError(
            "Registration QA localization binding is missing or did not pass"
        )
    expected_binding_values = {
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "requested_analysis_scope": requested_scope,
        "registration_mode": declared_mode,
        "ct_sha256": expected_ct_sha256,
        "localized_graph_sha256": localized_graph_sha256,
        "analysis_parameters_sha256": expected_analysis_parameters_sha256,
        "localization_policy_sha256": expected_localization_policy_sha256,
        "analysis_policy_artifact_sha256": expected_scope_artifact_sha256,
    }
    if (
        localization_binding.get("expected") != expected_binding_values
        or localization_binding.get("observed") != expected_binding_values
    ):
        raise DataPrepHandoffError(
            "Registration QA localization binding is stale or cross-input"
        )
    bound_localization_path = (
        repository_root.resolve()
        / Path(artifact_bindings["localization_report"]["path"])
    ).resolve()
    if (
        Path(str(localization_binding.get("artifact", {}).get("path", "")))
        .expanduser()
        .resolve()
        != bound_localization_path
    ):
        raise DataPrepHandoffError(
            "Registration QA localization artifact path differs from the bound report"
        )
    expected_nested_paths = {
        "localized_graph": (
            Path("analysis")
            / manifest["specimen_id"]
            / "registration"
            / "localized_graph.json"
        ),
        "registration_report": (
            Path("analysis")
            / manifest["specimen_id"]
            / "registration"
            / "registration_report.json"
        ),
    }
    localization_artifacts = localization_document.get("artifacts", {})
    for role, expected_relative in expected_nested_paths.items():
        descriptor = localization_artifacts.get(role)
        if (
            not isinstance(descriptor, dict)
            or set(descriptor) != {"path", "sha256", "role", "retention"}
        ):
            raise DataPrepHandoffError(
                f"Localization report has no closed {role} artifact binding"
            )
        resolved_nested = Path(str(descriptor["path"])).expanduser().resolve()
        expected_nested = (repository_root.resolve() / expected_relative).resolve()
        if (
            resolved_nested != expected_nested
            or not resolved_nested.is_relative_to(repository_root.resolve())
            or not resolved_nested.is_file()
            or sha256_file(resolved_nested) != descriptor["sha256"]
        ):
            raise DataPrepHandoffError(
                f"Localization report {role} artifact is stale or outside the run"
            )
    if (
        localization_artifacts["localized_graph"]["sha256"]
        != localized_graph_sha256
        or localization_artifacts["localized_graph"]["role"]
        != "independently_localized_lattice_graph"
        or localization_artifacts["registration_report"]["role"]
        != "registration_report"
    ):
        raise DataPrepHandoffError(
            "Localization nested artifact role/hash bindings are inconsistent"
        )
    registration_descriptor = localization_artifacts["registration_report"]
    if (
        registration_descriptor["sha256"]
        != localization_hashes["registration_report_sha256"]
        or qa_hashes["registration_report_sha256"]
        != registration_descriptor["sha256"]
    ):
        raise DataPrepHandoffError(
            "Registration report hash is inconsistent across Stage 1 bindings"
        )
    registration_document = load_json(
        Path(str(registration_descriptor["path"])).expanduser().resolve()
    )
    if (
        registration_document.get("schema_version")
        != "part2-registration/1.0.0"
        or registration_document.get("mode") != declared_mode
        or registration_document.get("gate") != "pass"
        or registration_document.get("hashes", {}).get("ct_sha256")
        != expected_ct_sha256
        or registration_document.get("hashes", {}).get(
            "registered_graph_sha256"
        )
        != localization_hashes["input_registered_graph_sha256"]
    ):
        raise DataPrepHandoffError(
            "Registration report is not bound to the localization inputs"
        )
    scope_source = qa_document.get("scope_source", {})
    expected_scope_source = {
        "requested_analysis_scope": requested_scope,
        "registration_mode": declared_mode,
        "source_artifact_kind": "specimen_manifest",
        "source_artifact_path": str(manifest_path.resolve()),
        "source_artifact_sha256": expected_scope_artifact_sha256,
        "analysis_parameters_sha256": expected_analysis_parameters_sha256,
        "localization_policy_sha256": expected_localization_policy_sha256,
        "qa_policy_sha256": expected_qa_policy_sha256,
        "segmentation_policy_sha256": expected_segmentation_policy_sha256,
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "declared_ct_sha256": expected_ct_sha256,
    }
    if (
        not isinstance(scope_source, dict)
        or scope_source != expected_scope_source
    ):
        raise DataPrepHandoffError(
            "Registration QA scope source is not the current specimen manifest"
        )

    registration_values = result["derived"]["registration_result"]["values"]
    required_registration_values = {
        "specimen_id": manifest["specimen_id"],
        "design_id": manifest["design_id"],
        "requested_analysis_scope": requested_scope,
        "authorized_outputs": authorized,
        "unauthorized_outputs": unauthorized,
        "roi_gate_results": roi_gate_results,
        "metrology_gate_status": metrology_status,
        "localization_quality_counts": quality_counts,
        "reason_codes": reason_codes,
    }
    for field, expected_value in required_registration_values.items():
        if registration_values.get(field) != expected_value:
            raise DataPrepHandoffError(
                f"derived.registration_result.{field} differs from the bound Stage 1 result"
            )

    aligned = result.get("aligned_graph")
    if not isinstance(aligned, dict):
        raise DataPrepHandoffError("Data-prep result must identify the aligned graph")
    relative = Path(aligned.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise DataPrepHandoffError("Aligned graph path escapes repository")
    resolved_repository_root = repository_root.resolve()
    resolved = (resolved_repository_root / relative).resolve()
    if not resolved.is_relative_to(resolved_repository_root) or not resolved.is_file():
        raise DataPrepHandoffError(f"Aligned graph is unavailable: {relative}")
    if sha256_file(resolved) != aligned.get("sha256"):
        raise DataPrepHandoffError("Aligned graph SHA-256 mismatch")
    mode = manifest["analysis_parameters"]["registration"]["mode"]
    expected_role = (
        "aligned_graph" if mode == "challenge_aligned_json" else "derived_aligned_graph"
    )
    if aligned.get("role") != expected_role:
        raise DataPrepHandoffError(
            f"Aligned graph role must be {expected_role!r} in {mode}"
        )

    _validate_canonical_mask(
        manifest,
        result,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )


def apply_data_prep_result(
    manifest_path: Path,
    result_path: Path,
    *,
    repository_root: Path,
    completion_receipt_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Atomically advance a ready intake manifest after deterministic Stage 1."""
    validate_manifest(
        manifest_path,
        schema_path=schema_path,
        repository_root=repository_root,
    )
    manifest = load_json(manifest_path)
    result = load_json(result_path)
    destination = completion_receipt_path or (
        manifest_path.parent / "data_prep_completion_receipt.json"
    )
    if manifest["lifecycle_state"] == ANALYSIS_READY:
        if not destination.is_file():
            raise DataPrepHandoffError(
                "Analysis-ready replay is missing its completion receipt"
            )
        completion = load_json(destination)
        completion_base = {
            key: value
            for key, value in completion.items()
            if key != "canonical_completion_sha256"
        }
        if (
            completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
            or completion.get("canonical_completion_sha256")
            != canonical_json_sha256(completion_base)
            or completion.get("analysis_ready_manifest_sha256")
            != canonical_json_sha256(manifest)
            or completion.get("data_prep_result_sha256")
            != canonical_json_sha256(result)
            or completion.get("prior_manifest_artifact_sha256")
            != result.get("input_manifest_artifact_sha256")
            or completion.get("specimen_id") != manifest["specimen_id"]
            or completion.get("design_id") != manifest["design_id"]
            or completion.get("analysis_parameters_sha256")
            != manifest["analysis_parameters_sha256"]
            or completion.get("canonical_mask") != result.get("canonical_mask")
            or manifest["inputs"].get("canonical_mask")
            != result.get("canonical_mask")
        ):
            raise DataPrepHandoffError(
                "Analysis-ready replay differs from the sealed completion"
            )
        _validate_canonical_mask(
            manifest,
            result,
            manifest_path=manifest_path,
            repository_root=repository_root,
        )
        return {
            "manifest_path": str(manifest_path),
            "completion_receipt_path": str(destination),
            "analysis_ready_manifest_sha256": canonical_json_sha256(manifest),
            "canonical_completion_sha256": completion[
                "canonical_completion_sha256"
            ],
            "changed": {"manifest": False, "completion_receipt": False},
        }
    if manifest["lifecycle_state"] != "ready_for_data_prep":
        raise DataPrepHandoffError(
            "Data prep can only finalize a ready_for_data_prep manifest"
        )
    _validate_data_prep_result(
        manifest,
        result,
        manifest_path=manifest_path,
        repository_root=repository_root,
    )

    prior_manifest_hash = canonical_json_sha256(manifest)
    finalized = json.loads(json.dumps(manifest))
    finalized["inputs"]["aligned_graph"] = result["aligned_graph"]
    finalized["inputs"]["canonical_mask"] = result["canonical_mask"]
    finalized["derived"] = result["derived"]
    finalized["lifecycle_state"] = ANALYSIS_READY
    finalized["unresolved_fields"] = []

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=manifest_path.parent, suffix=".json", delete=False
    ) as stream:
        temporary_manifest = Path(stream.name)
        stream.write(_json_bytes(finalized))
    try:
        validate_manifest(
            temporary_manifest,
            schema_path=schema_path,
            repository_root=repository_root,
            required_lifecycle=ANALYSIS_READY,
        )
    except (ManifestValidationError, OSError) as exc:
        raise DataPrepHandoffError(
            f"Finalized manifest failed readiness validation: {exc}"
        ) from exc
    finally:
        temporary_manifest.unlink(missing_ok=True)

    finalized_hash = canonical_json_sha256(finalized)
    result_hash = canonical_json_sha256(result)
    completion_base = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "specimen_id": finalized["specimen_id"],
        "design_id": finalized["design_id"],
        "requested_analysis_scope": result["requested_analysis_scope"],
        "authorized_outputs": result["authorized_outputs"],
        "unauthorized_outputs": result["unauthorized_outputs"],
        "roi_gate_results": result["roi_gate_results"],
        "metrology_gate_status": result["metrology_gate_status"],
        "localization_quality_counts": result["localization_quality_counts"],
        "reason_codes": result["reason_codes"],
        "artifact_bindings": result["artifact_bindings"],
        "prior_manifest_sha256": prior_manifest_hash,
        "prior_manifest_artifact_sha256": result[
            "input_manifest_artifact_sha256"
        ],
        "analysis_ready_manifest_sha256": finalized_hash,
        "data_prep_result_sha256": result_hash,
        "analysis_parameters_sha256": finalized["analysis_parameters_sha256"],
        "lifecycle_state": ANALYSIS_READY,
        "registration_mode": finalized["analysis_parameters"]["registration"]["mode"],
        "self_verification": result["self_verification"],
        "canonical_mask": result["canonical_mask"],
    }
    completion = {
        **completion_base,
        "canonical_completion_sha256": canonical_json_sha256(completion_base),
    }
    # Publish the receipt first. A crash can leave a harmless receipt whose
    # target hash is absent, but never an analysis-ready manifest without its
    # completion receipt.
    receipt_changed = _atomic_write_if_changed(destination, completion)
    manifest_changed = _atomic_write_if_changed(manifest_path, finalized)
    return {
        "manifest_path": str(manifest_path),
        "completion_receipt_path": str(destination),
        "analysis_ready_manifest_sha256": finalized_hash,
        "canonical_completion_sha256": completion[
            "canonical_completion_sha256"
        ],
        "changed": {
            "manifest": manifest_changed,
            "completion_receipt": receipt_changed,
        },
    }
