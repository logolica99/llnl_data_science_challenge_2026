"""Stage 2 strict batched strut-metrics MCP adapter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from part2_core import compute_strut_metrics as _compute_strut_metrics
from specimen_manifest import (
    canonical_json_sha256 as _canonical_json_sha256,
    load_json as _load_json,
)

from . import common
from .common import (
    MCPResponseEnvelope,
    _core_response,
    _repository_output_directory,
    _repository_path,
    _run_structured_tool,
    _sha256_file,
)
from .registry import mcp

_STAGE2_METRIC_INPUT_ROLES = frozenset(
    {
        "analysis_ready_specimen_manifest",
        "data_prep_completion_receipt",
        "analysis_config",
        "ct_volume",
        "localized_graph",
        "registration_qa",
        "otsu_report",
        "canonical_segmentation_mask",
        "segmentation_mask_comparison",
    }
)
_FORBIDDEN_STAGE2_PATH_TOKENS = frozenset(
    {
        "label",
        "labels",
        "development",
        "dev_split",
        "sealed",
        "ground_truth",
        "classification",
    }
)


def _stage2_json(path: Path, role: str) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Stage 2 {role} must be a JSON object")
    return value


def _stage2_artifact_record(
    records: dict[str, dict[str, Any]],
    role: str,
    expected_relative: str,
    expected_path: Path,
) -> dict[str, Any]:
    record = records.get(role)
    if record is None:
        raise ValueError(f"Stage 2 handoff is missing required artifact role {role}")
    if record.get("path") != expected_relative:
        raise ValueError(f"Stage 2 handoff path mismatch for {role}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or digest != _sha256_file(expected_path):
        raise ValueError(f"Stage 2 handoff hash mismatch for {role}")
    return record


def _validate_stage2_metric_bundle(
    *,
    handoff_path: Path,
    handoff_relative: str,
    paths: dict[str, tuple[Path, str]],
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """Fail closed unless the MCP request reproduces one canonical Stage 2 handoff."""

    handoff = _stage2_json(handoff_path, "handoff")
    expected_handoff_fields = {
        "schema_version",
        "specimen_id",
        "stage_number",
        "stage",
        "owner",
        "attempt",
        "run_token",
        "created_at",
        "registration_mode",
        "config_sha256",
        "contract_version",
        "contract_sha256",
        "predecessor_receipt_sha256",
        "input_artifacts",
        "forbidden_operations",
        "canonical_handoff_sha256",
    }
    if set(handoff) != expected_handoff_fields:
        raise ValueError("Stage 2 handoff is open-ended or schema-incompatible")
    handoff_base = {
        key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"
    }
    if (
        handoff.get("schema_version") != "part2-stage-handoff/1.0.0"
        or handoff.get("stage_number") != 2
        or handoff.get("stage") != "strut_metrics"
        or handoff.get("owner") != "strut_metrics"
        or handoff.get("registration_mode") != "autonomous_v2"
        or handoff.get("canonical_handoff_sha256")
        != _canonical_json_sha256(handoff_base)
    ):
        raise ValueError("Stage 2 handoff identity or canonical hash is invalid")
    contract_path = common.REPOSITORY_ROOT / "analysis" / "contracts" / "strut_metrics.json"
    contract = _stage2_json(contract_path, "contract")
    if (
        handoff.get("contract_version") != contract.get("schema_version")
        or handoff.get("contract_sha256") != _sha256_file(contract_path)
    ):
        raise ValueError("Stage 2 handoff is bound to a stale strut-metrics contract")
    if handoff.get("forbidden_operations") != contract.get("forbidden_operations"):
        raise ValueError("Stage 2 handoff forbidden operations differ from the live contract")
    predecessor = handoff.get("predecessor_receipt_sha256")
    if (
        not isinstance(predecessor, str)
        or len(predecessor) != 64
        or any(character not in "0123456789abcdef" for character in predecessor)
    ):
        raise ValueError("Stage 2 handoff lacks the verified Stage 1 receipt hash")
    artifacts = handoff.get("input_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Stage 2 handoff input_artifacts must be an array")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not {"role", "path", "sha256"} <= set(record):
            raise ValueError("Stage 2 handoff contains a malformed artifact record")
        role = record.get("role")
        artifact_path = record.get("path")
        if not isinstance(role, str) or role in records or not isinstance(artifact_path, str):
            raise ValueError("Stage 2 handoff contains duplicate or invalid artifact roles")
        lowered_parts = {part.lower() for part in Path(artifact_path).parts}
        if (
            role not in _STAGE2_METRIC_INPUT_ROLES
            or lowered_parts & _FORBIDDEN_STAGE2_PATH_TOKENS
            or any(token in role.lower() for token in _FORBIDDEN_STAGE2_PATH_TOKENS)
        ):
            raise ValueError(f"Stage 2 handoff exposes forbidden or unauthorized artifact {role}")
        records[role] = record
    if set(records) != _STAGE2_METRIC_INPUT_ROLES:
        missing = sorted(_STAGE2_METRIC_INPUT_ROLES - set(records))
        extra = sorted(set(records) - _STAGE2_METRIC_INPUT_ROLES)
        raise ValueError(f"Stage 2 handoff artifact roles mismatch; missing={missing}, extra={extra}")
    for role, (path, relative) in paths.items():
        _stage2_artifact_record(records, role, relative, path)
    control_config_hash = handoff.get("config_sha256")
    if (
        not isinstance(control_config_hash, str)
        or len(control_config_hash) != 64
        or any(character not in "0123456789abcdef" for character in control_config_hash)
    ):
        raise ValueError("Stage 2 handoff has an invalid frozen control-config hash")

    specimen_id = handoff.get("specimen_id")
    if not isinstance(specimen_id, str) or not specimen_id:
        raise ValueError("Stage 2 handoff specimen_id is invalid")
    expected_handoff = (
        Path("analysis")
        / specimen_id
        / "handoffs"
        / f"stage_2_strut_metrics_attempt_{handoff['attempt']}.json"
    ).as_posix()
    if handoff_relative != expected_handoff:
        raise ValueError("Stage 2 handoff is outside its canonical specimen path")

    config = _stage2_json(paths["analysis_config"][0], "analysis config")
    schema_path = common.REPOSITORY_ROOT / "analysis" / "schema" / "strut_metrics_input.schema.json"
    schema = _stage2_json(schema_path, "configuration schema")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise ValueError(f"Frozen Stage 2 analysis config is invalid: {detail}")
    fragment = config["stage_2_strut_metrics"]

    manifest = _stage2_json(
        paths["analysis_ready_specimen_manifest"][0], "analysis-ready specimen manifest"
    )
    if manifest.get("lifecycle_state") != "analysis_ready" or manifest.get("specimen_id") != specimen_id:
        raise ValueError("Stage 2 requires the matching analysis-ready specimen manifest")
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("Analysis-ready manifest has no closed inputs object")
    ct_binding = manifest_inputs.get("ct")
    mask_binding = manifest_inputs.get("canonical_mask")
    if not isinstance(ct_binding, dict) or not isinstance(mask_binding, dict):
        raise ValueError("Analysis-ready manifest does not bind CT and canonical mask")
    if (
        ct_binding.get("path") != paths["ct_volume"][1]
        or ct_binding.get("sha256") != records["ct_volume"]["sha256"]
        or ct_binding.get("role") != "ct_volume"
        or mask_binding.get("path") != paths["canonical_segmentation_mask"][1]
        or mask_binding.get("sha256") != records["canonical_segmentation_mask"]["sha256"]
        or mask_binding.get("role") != "canonical_segmentation_mask"
        or mask_binding.get("dtype") != "uint8"
        or mask_binding.get("array_axes") != ["z", "y", "x"]
        or mask_binding.get("retention") != "committed"
    ):
        raise ValueError("Manifest CT or canonical-mask binding differs from Stage 2 handoff")
    ct_metadata = manifest_inputs.get("ct_metadata")
    if not isinstance(ct_metadata, dict) or mask_binding.get("shape") != ct_metadata.get("shape"):
        raise ValueError("Canonical mask shape binding differs from CT metadata")
    analysis_parameters = manifest.get("analysis_parameters")
    if not isinstance(analysis_parameters, dict):
        raise ValueError("Analysis-ready manifest has no frozen analysis_parameters")
    requested_scope = analysis_parameters.get("requested_analysis_scope")
    if requested_scope not in {"roi_screening", "direct_metrology"}:
        raise ValueError("Analysis-ready manifest requested_analysis_scope is invalid")
    if analysis_parameters.get("registration", {}).get("mode") != handoff.get(
        "registration_mode"
    ):
        raise ValueError("Analysis-ready manifest registration mode differs from the handoff")
    derived = manifest.get("derived")
    registration_result = (
        derived.get("registration_result") if isinstance(derived, dict) else None
    )
    registration_values = (
        registration_result.get("values")
        if isinstance(registration_result, dict)
        else None
    )
    if not isinstance(registration_values, dict):
        raise ValueError("Analysis-ready manifest has no finalized registration result")

    completion = _stage2_json(
        paths["data_prep_completion_receipt"][0], "data-prep completion receipt"
    )
    completion_base = {
        key: value for key, value in completion.items() if key != "canonical_completion_sha256"
    }
    if (
        completion.get("schema_version") != "data-prep-completion/1.2.0"
        or completion.get("specimen_id") != specimen_id
        or completion.get("design_id") != manifest.get("design_id")
        or completion.get("lifecycle_state") != "analysis_ready"
        or completion.get("canonical_completion_sha256")
        != _canonical_json_sha256(completion_base)
        or completion.get("analysis_ready_manifest_sha256")
        != _canonical_json_sha256(manifest)
        or completion.get("canonical_mask") != mask_binding
        or completion.get("registration_mode") != handoff.get("registration_mode")
        or completion.get("requested_analysis_scope") != requested_scope
    ):
        raise ValueError("Data-prep completion receipt is stale or incompatible")
    authorized_outputs = completion.get("authorized_outputs")
    unauthorized_outputs = completion.get("unauthorized_outputs")
    if (
        not isinstance(authorized_outputs, list)
        or not isinstance(unauthorized_outputs, list)
        or any(not isinstance(value, str) for value in authorized_outputs)
        or any(not isinstance(value, str) for value in unauthorized_outputs)
        or set(authorized_outputs) & set(unauthorized_outputs)
    ):
        raise ValueError("Data-prep output authorization lists are malformed or overlap")
    if (
        registration_values.get("requested_analysis_scope") != requested_scope
        or registration_values.get("authorized_outputs") != authorized_outputs
        or registration_values.get("unauthorized_outputs") != unauthorized_outputs
    ):
        raise ValueError(
            "Finalized manifest registration scope or output authorizations differ "
            "from the data-prep completion receipt"
        )
    metrology_outputs = {"absolute_metrology", "direct_dimensional_measurement"}
    if requested_scope == "roi_screening" and (
        set(authorized_outputs) & metrology_outputs
        or not metrology_outputs <= set(unauthorized_outputs)
    ):
        raise ValueError("ROI-screening Stage 1 receipt incorrectly authorizes metrology")
    if requested_scope == "direct_metrology" and not metrology_outputs <= set(
        authorized_outputs
    ):
        raise ValueError("Direct-metrology Stage 1 receipt lacks metrology authorization")

    qa = _stage2_json(paths["registration_qa"][0], "registration QA")
    if (
        qa.get("specimen_id") != specimen_id
        or qa.get("gate") != "pass"
        or qa.get("overall_pass") is not True
        or qa.get("registration_mode") != handoff.get("registration_mode")
    ):
        raise ValueError("Stage 1 registration QA did not pass for this specimen")

    otsu = _stage2_json(paths["otsu_report"][0], "exact Otsu report")
    threshold = otsu.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or otsu.get("overall_pass") is not True
        or not math.isclose(
            float(fragment["otsu_threshold"]), float(threshold), rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("Frozen Stage 2 threshold does not match the passing Stage 1 Otsu result")

    comparison = _stage2_json(
        paths["segmentation_mask_comparison"][0], "segmentation mask comparison"
    )
    candidates = comparison.get("candidates")
    matching_candidates = [
        candidate
        for candidate in candidates or []
        if isinstance(candidate, dict)
        and candidate.get("path") == paths["canonical_segmentation_mask"][1]
        and candidate.get("sha256") == records["canonical_segmentation_mask"]["sha256"]
        and candidate.get("exact_threshold_match") is True
        and math.isclose(float(candidate.get("threshold", math.nan)), float(threshold), abs_tol=1e-12)
    ]
    if comparison.get("overall_pass") is not True or len(matching_candidates) != 1:
        raise ValueError("Canonical mask lacks an exact passing Stage 1 comparison binding")

    input_provenance = {
        "stage_2_handoff": {
            "path": handoff_relative,
            "sha256": _sha256_file(handoff_path),
            "canonical_sha256": handoff["canonical_handoff_sha256"],
        },
        "predecessor_stage_1_receipt_sha256": predecessor,
        "contract_sha256": handoff["contract_sha256"],
        "config_sha256": handoff["config_sha256"],
        "requested_analysis_scope": requested_scope,
        "authorized_outputs": sorted(authorized_outputs),
        "unauthorized_outputs": sorted(unauthorized_outputs),
        "input_artifacts": {
            role: {"path": record["path"], "sha256": record["sha256"]}
            for role, record in sorted(records.items())
        },
    }
    return handoff, config, float(threshold), input_provenance


@mcp.tool()
def compute_strut_metrics(
    stage_2_handoff_filepath: str,
    analysis_ready_specimen_manifest_filepath: str,
    data_prep_completion_receipt_filepath: str,
    analysis_config_filepath: str,
    ct_filepath: str,
    localized_graph_filepath: str,
    registration_qa_filepath: str,
    otsu_report_filepath: str,
    canonical_segmentation_mask_filepath: str,
    segmentation_mask_comparison_filepath: str,
    output_directory: str,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Compute strict Stage 2 label-blind evidence from one sealed Stage 1 handoff."""

    def operation() -> dict[str, Any]:
        handoff, handoff_relative = _repository_path(
            stage_2_handoff_filepath, must_exist=True, expected_suffixes={".json"}
        )
        specifications = {
            "analysis_ready_specimen_manifest": (
                analysis_ready_specimen_manifest_filepath,
                {".json"},
            ),
            "data_prep_completion_receipt": (
                data_prep_completion_receipt_filepath,
                {".json"},
            ),
            "analysis_config": (analysis_config_filepath, {".json"}),
            "ct_volume": (ct_filepath, {".npy", ".tif", ".tiff"}),
            "localized_graph": (localized_graph_filepath, {".json"}),
            "registration_qa": (registration_qa_filepath, {".json"}),
            "otsu_report": (otsu_report_filepath, {".json"}),
            "canonical_segmentation_mask": (
                canonical_segmentation_mask_filepath,
                {".npy"},
            ),
            "segmentation_mask_comparison": (
                segmentation_mask_comparison_filepath,
                {".json"},
            ),
        }
        paths = {
            role: _repository_path(filepath, must_exist=True, expected_suffixes=suffixes)
            for role, (filepath, suffixes) in specifications.items()
        }
        handoff_document, config, threshold, input_provenance = _validate_stage2_metric_bundle(
            handoff_path=handoff,
            handoff_relative=handoff_relative,
            paths=paths,
        )
        output, output_relative = _repository_output_directory(output_directory)
        expected_output = (
            Path("analysis") / str(handoff_document["specimen_id"]) / "struts"
        ).as_posix()
        if output_relative != expected_output:
            raise ValueError("Stage 2 output_directory is not the canonical specimen struts path")
        if overwrite:
            raise ValueError(
                "Strict Stage 2 artifacts are immutable; use exact idempotent replay"
            )
        payload = _compute_strut_metrics(
            paths["ct_volume"][0],
            paths["localized_graph"][0],
            output / "per_strut_metrics.csv",
            output / "per_strut_profiles.json",
            output / "metrics_report.json",
            threshold=threshold,
            registration_mode=str(handoff_document["registration_mode"]),
            config=config,
            registration_qa_path=paths["registration_qa"][0],
            canonical_mask_path=paths["canonical_segmentation_mask"][0],
            output_calibration_path=output / "corridor_calibration.json",
            input_provenance=input_provenance,
            overwrite=overwrite,
        )
        return _core_response(
            "compute_strut_metrics",
            f"Measured {payload['counts']['metric_rows']} nominal struts with gate {payload['gate']}",
            payload,
        )

    return _run_structured_tool("compute_strut_metrics", operation)
