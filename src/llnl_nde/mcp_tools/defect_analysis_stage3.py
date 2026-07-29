"""Strict Stage 3 specialist classification and evidence MCP adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from llnl_nde.core import (
    analyze_strut_specialist as _analyze_strut_specialist,
    merge_strut_classifications as _merge_strut_classifications,
    normalize_stage3_config as _normalize_stage3_config,
    render_strut_evidence as _render_strut_evidence,
    verify_strut_classifications as _verify_strut_classifications,
)
from llnl_nde.core.artifacts import read_json_object, sha256_json

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


def _stage3_json(path: Path, role: str) -> dict[str, Any]:
    value = read_json_object(path)
    if not isinstance(value, dict):
        raise ValueError(f"Stage 3 {role} must be a JSON object")
    return value


_STAGE3_BASE_INPUT_ROLES = frozenset(
    {
        "analysis_config",
        "corridor_calibration",
        "per_strut_metrics",
        "per_strut_profiles",
        "localized_graph",
        "ct_volume",
    }
)
_FORBIDDEN_STAGE3_PATH_TOKENS = frozenset(
    {
        "label",
        "labels",
        "development_split",
        "dev_split",
        "sealed",
        "ground_truth",
        "intentional_deletion",
    }
)


def _validate_stage3_bundle(
    *,
    handoff_path: Path,
    handoff_relative: str,
    paths: dict[str, tuple[Path, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless one request reproduces the Stage 3 handoff exactly."""

    handoff = _stage3_json(handoff_path, "Stage 3 handoff")
    expected_fields = {
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
    if set(handoff) != expected_fields:
        raise ValueError("Stage 3 handoff is open-ended or schema-incompatible")
    handoff_base = {
        key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"
    }
    if (
        handoff.get("schema_version") != "part2-stage-handoff/1.0.0"
        or handoff.get("stage_number") != 3
        or handoff.get("stage") != "defect_analysis"
        or handoff.get("owner") != "defect_lead"
        or handoff.get("registration_mode") != "autonomous_v2"
        or handoff.get("canonical_handoff_sha256")
        != sha256_json(handoff_base)
    ):
        raise ValueError("Stage 3 handoff identity or canonical hash is invalid")
    contract_path = common.REPOSITORY_ROOT / "analysis" / "contracts" / "defect_analysis.json"
    contract = _stage3_json(contract_path, "Stage 3 contract")
    if (
        handoff.get("contract_version") != contract.get("schema_version")
        or handoff.get("contract_sha256") != _sha256_file(contract_path)
        or handoff.get("forbidden_operations") != contract.get("forbidden_operations")
    ):
        raise ValueError("Stage 3 handoff is bound to a stale defect-analysis contract")
    predecessor = handoff.get("predecessor_receipt_sha256")
    if (
        not isinstance(predecessor, str)
        or len(predecessor) != 64
        or any(character not in "0123456789abcdef" for character in predecessor)
    ):
        raise ValueError("Stage 3 handoff lacks the verified Stage 2 receipt hash")
    specimen_id = handoff.get("specimen_id")
    if not isinstance(specimen_id, str) or not specimen_id:
        raise ValueError("Stage 3 handoff specimen_id is invalid")
    expected_handoff = (
        Path("analysis")
        / specimen_id
        / "handoffs"
        / f"stage_3_defect_analysis_attempt_{handoff['attempt']}.json"
    ).as_posix()
    if handoff_relative != expected_handoff:
        raise ValueError("Stage 3 handoff is outside its canonical specimen path")
    artifacts = handoff.get("input_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Stage 3 handoff input_artifacts must be an array")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not {"role", "path", "sha256"} <= set(record):
            raise ValueError("Stage 3 handoff contains a malformed artifact record")
        role = record.get("role")
        artifact_path = record.get("path")
        if not isinstance(role, str) or role in records or not isinstance(artifact_path, str):
            raise ValueError("Stage 3 handoff contains duplicate or invalid artifact roles")
        lowered = {part.lower() for part in Path(artifact_path).parts}
        if (
            role not in _STAGE3_BASE_INPUT_ROLES
            or lowered & _FORBIDDEN_STAGE3_PATH_TOKENS
            or any(token in role.lower() for token in _FORBIDDEN_STAGE3_PATH_TOKENS)
        ):
            raise ValueError(f"Stage 3 handoff exposes forbidden artifact {role}")
        records[role] = record
    if set(records) != _STAGE3_BASE_INPUT_ROLES:
        raise ValueError("Stage 3 handoff artifact roles do not match the contract")
    for role, (path, relative) in paths.items():
        record = records[role]
        if record.get("path") != relative or record.get("sha256") != _sha256_file(path):
            raise ValueError(f"Stage 3 handoff path/hash mismatch for {role}")
    config = _stage3_json(paths["analysis_config"][0], "Stage 3 analysis config")
    _normalize_stage3_config(config)
    profiles = _stage3_json(paths["per_strut_profiles"][0], "Stage 2 profiles")
    if profiles.get("measurement_only") is not True or profiles.get(
        "classification_performed"
    ) is not False:
        raise ValueError("Stage 3 profiles are not the immutable measurement-only bundle")
    calibration = _stage3_json(
        paths["corridor_calibration"][0], "Stage 2 corridor calibration"
    )
    if calibration.get("label_blind") is not True:
        raise ValueError("Stage 3 requires label-blind Stage 2 corridor calibration")
    return handoff, config


@mcp.tool()
def classify_struts(
    stage_3_handoff_filepath: str,
    analysis_config_filepath: str,
    corridor_calibration_filepath: str,
    metrics_filepath: str,
    profiles_filepath: str,
    localized_graph_filepath: str,
    ct_filepath: str,
    output_directory: str,
    operation: Literal[
        "analyze_missing",
        "analyze_broken",
        "analyze_thin",
        "analyze_bent",
        "merge",
        "verify",
    ],
    evidence_manifest_filepaths: list[str] | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Run one bounded Stage 3 specialist, merge, or independent verification."""

    def run_operation() -> dict[str, Any]:
        handoff, handoff_relative = _repository_path(
            stage_3_handoff_filepath, must_exist=True, expected_suffixes={".json"}
        )
        specifications = {
            "analysis_config": (analysis_config_filepath, {".json"}),
            "corridor_calibration": (corridor_calibration_filepath, {".json"}),
            "per_strut_metrics": (metrics_filepath, {".csv"}),
            "per_strut_profiles": (profiles_filepath, {".json"}),
            "localized_graph": (localized_graph_filepath, {".json"}),
            "ct_volume": (ct_filepath, {".npy", ".tif", ".tiff"}),
        }
        paths = {
            role: _repository_path(filepath, must_exist=True, expected_suffixes=suffixes)
            for role, (filepath, suffixes) in specifications.items()
        }
        handoff_document, config = _validate_stage3_bundle(
            handoff_path=handoff,
            handoff_relative=handoff_relative,
            paths=paths,
        )
        output, output_relative = _repository_output_directory(output_directory)
        expected_output = (
            Path("analysis") / str(handoff_document["specimen_id"]) / "struts"
        ).as_posix()
        if output_relative != expected_output:
            raise ValueError("Stage 3 output_directory is not the canonical specimen struts path")
        if overwrite:
            raise ValueError("Strict Stage 3 artifacts are immutable; overwrite is forbidden")
        specimen_id = str(handoff_document["specimen_id"])
        findings_paths = {
            kind: output / f"findings_{kind}.json" for kind in ("missing", "broken", "thin", "bent")
        }
        if operation.startswith("analyze_"):
            defect_kind = operation.removeprefix("analyze_")
            payload = _analyze_strut_specialist(
                paths["per_strut_metrics"][0],
                paths["per_strut_profiles"][0],
                config,
                findings_paths[defect_kind],
                specimen_id=specimen_id,
                defect_kind=defect_kind,
            )
            return _core_response(
                "classify_struts",
                f"Stage 3 {defect_kind} specialist returned {payload['gate']}",
                payload,
            )
        if operation == "merge":
            payload = _merge_strut_classifications(
                paths["per_strut_metrics"][0],
                paths["per_strut_profiles"][0],
                config,
                findings_paths,
                output / "classified_struts.json",
                output / "thresholds.json",
                output / "decision_log.md",
                specimen_id=specimen_id,
            )
            return _core_response(
                "classify_struts",
                f"Merged {payload['counts']['total']} Stage 3 struts with gate {payload['gate']}",
                payload,
            )
        if operation == "verify":
            evidence_paths = [
                _repository_path(path, must_exist=True, expected_suffixes={".json"})[0]
                for path in (evidence_manifest_filepaths or [])
            ]
            payload = _verify_strut_classifications(
                paths["per_strut_metrics"][0],
                paths["per_strut_profiles"][0],
                findings_paths,
                output / "classified_struts.json",
                output / "thresholds.json",
                output / "decision_log.md",
                evidence_paths,
                output / "verifier_report.json",
                analysis_config=paths["analysis_config"][0],
                localized_graph_path=paths["localized_graph"][0],
                ct_path=paths["ct_volume"][0],
                specimen_id=specimen_id,
                attempt=int(handoff_document["attempt"]),
                run_token=str(handoff_document["run_token"]),
                config_sha256=str(handoff_document["config_sha256"]),
                contract_sha256=str(handoff_document["contract_sha256"]),
                predecessor_receipt_sha256=str(
                    handoff_document["predecessor_receipt_sha256"]
                ),
                input_handoff_sha256=str(handoff_document["canonical_handoff_sha256"]),
            )
            return _core_response(
                "classify_struts",
                "Independent classifier verification passed",
                payload,
            )
        raise ValueError(f"Unsupported Stage 3 operation: {operation}")

    return _run_structured_tool("classify_struts", run_operation)


@mcp.tool()
def render_strut_evidence(
    stage_3_handoff_filepath: str,
    analysis_config_filepath: str,
    corridor_calibration_filepath: str,
    ct_filepath: str,
    localized_graph_filepath: str,
    metrics_filepath: str,
    profiles_filepath: str,
    classifications_filepath: str,
    thresholds_filepath: str,
    output_directory: str,
    strut_id: int,
    crop_margin_voxels: int = 8,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Render hash-bound local-frame evidence for one non-present Stage 3 call."""

    def operation() -> dict[str, Any]:
        handoff, handoff_relative = _repository_path(
            stage_3_handoff_filepath, must_exist=True, expected_suffixes={".json"}
        )
        specifications = {
            "analysis_config": (analysis_config_filepath, {".json"}),
            "corridor_calibration": (corridor_calibration_filepath, {".json"}),
            "per_strut_metrics": (metrics_filepath, {".csv"}),
            "per_strut_profiles": (profiles_filepath, {".json"}),
            "localized_graph": (localized_graph_filepath, {".json"}),
            "ct_volume": (ct_filepath, {".npy", ".tif", ".tiff"}),
        }
        paths = {
            role: _repository_path(filepath, must_exist=True, expected_suffixes=suffixes)
            for role, (filepath, suffixes) in specifications.items()
        }
        handoff_document, config = _validate_stage3_bundle(
            handoff_path=handoff,
            handoff_relative=handoff_relative,
            paths=paths,
        )
        classifications, classifications_relative = _repository_path(
            classifications_filepath, must_exist=True, expected_suffixes={".json"}
        )
        thresholds, thresholds_relative = _repository_path(
            thresholds_filepath, must_exist=True, expected_suffixes={".json"}
        )
        expected_struts = Path("analysis") / str(handoff_document["specimen_id"]) / "struts"
        if classifications_relative != (expected_struts / "classified_struts.json").as_posix():
            raise ValueError("Stage 3 evidence classification path is not canonical")
        if thresholds_relative != (expected_struts / "thresholds.json").as_posix():
            raise ValueError("Stage 3 evidence thresholds path is not canonical")
        output, output_relative = _repository_output_directory(output_directory)
        expected_output = (
            Path("analysis") / str(handoff_document["specimen_id"]) / "evidence"
        ).as_posix()
        if output_relative != expected_output:
            raise ValueError("Stage 3 evidence output is outside the canonical specimen path")
        if overwrite:
            raise ValueError("Strict Stage 3 evidence is immutable; overwrite is forbidden")
        threshold = config.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("Stage 3 analysis config lacks the frozen Stage 1 Otsu threshold")
        payload = _render_strut_evidence(
            paths["ct_volume"][0],
            paths["localized_graph"][0],
            paths["per_strut_profiles"][0],
            output,
            strut_id=strut_id,
            threshold=float(threshold),
            crop_margin_voxels=crop_margin_voxels,
            metrics_path=paths["per_strut_metrics"][0],
            classifications_path=classifications,
            thresholds_path=thresholds,
            specimen_id=str(handoff_document["specimen_id"]),
            overwrite=overwrite,
        )
        return _core_response(
            "render_strut_evidence",
            f"Rendered evidence packet for strut {strut_id}",
            payload,
        )

    return _run_structured_tool("render_strut_evidence", operation)
