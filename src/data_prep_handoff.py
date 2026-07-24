"""Hash-sealed specimen-ingest hand-off and data-prep completion adapter."""

from __future__ import annotations

import json
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


HANDOFF_SCHEMA_VERSION = "data-prep-handoff/1.0.0"
RESULT_SCHEMA_VERSION = "data-prep-result/1.0.0"
COMPLETION_SCHEMA_VERSION = "data-prep-completion/1.0.0"
REQUIRED_DERIVED = {
    "graph_summary",
    "voxel_spacing",
    "segmentation_result",
    "registration_result",
}


class DataPrepHandoffError(ValueError):
    """Raised when an intake or data-prep envelope fails its hash contract."""


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
    if receipt.get("schema_version") != "ingest-receipt/1.0.0":
        raise DataPrepHandoffError("Unsupported ingest receipt schema")
    if receipt.get("specimen_id") != manifest["specimen_id"]:
        raise DataPrepHandoffError("Receipt specimen_id does not match manifest")
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
        "cad_readable",
        "graph_id_reference_integrity",
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


def create_data_prep_handoff(
    manifest_path: Path,
    receipt_path: Path,
    *,
    repository_root: Path,
    output_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Verify intake artifacts and emit the deterministic next-stage envelope."""
    validate_manifest(
        manifest_path,
        schema_path=schema_path,
        repository_root=repository_root,
    )
    manifest = load_json(manifest_path)
    receipt = load_json(receipt_path)
    _verify_intake_receipt(manifest, receipt)

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
        "allowlisted_inputs": allowlisted_inputs,
        "unresolved_fields": manifest["unresolved_fields"],
        "forbidden_inputs": [
            "defect labels",
            "dev split",
            "sealed split",
            "ground-truth segmentation",
        ],
        "required_outputs": [
            "aligned graph",
            "exact-histogram Otsu result",
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
    handoff = {
        **handoff_base,
        "canonical_handoff_sha256": canonical_json_sha256(handoff_base),
    }
    destination = output_path or (
        manifest_path.parent / "data_prep_handoff.json"
    )
    changed = _atomic_write_if_changed(destination, handoff)
    return {
        "handoff": handoff,
        "path": str(destination),
        "changed": changed,
    }


def _validate_data_prep_result(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    repository_root: Path,
) -> None:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise DataPrepHandoffError("Unsupported data-prep result schema")
    if result.get("specimen_id") != manifest["specimen_id"]:
        raise DataPrepHandoffError("Data-prep result specimen_id mismatch")
    if result.get("input_manifest_sha256") != canonical_json_sha256(manifest):
        raise DataPrepHandoffError("Data-prep result uses a stale intake manifest")
    if (
        result.get("analysis_parameters_sha256")
        != manifest["analysis_parameters_sha256"]
    ):
        raise DataPrepHandoffError("Data-prep result uses stale analysis parameters")
    if set(result.get("derived", {})) != REQUIRED_DERIVED:
        raise DataPrepHandoffError(
            "Data-prep result must provide exactly: "
            + ", ".join(sorted(REQUIRED_DERIVED))
        )
    verification = result.get("self_verification", {})
    required_checks = (
        "exact_otsu_complete",
        "registration_complete",
        "local_recenter_complete",
        "roi_gate_pass",
        "metrology_gate_pass",
        "defect_labels_not_accessed",
    )
    failed = [name for name in required_checks if verification.get(name) is not True]
    if failed:
        raise DataPrepHandoffError(
            "Data-prep result failed self-verification: " + ", ".join(failed)
        )

    aligned = result.get("aligned_graph")
    if not isinstance(aligned, dict):
        raise DataPrepHandoffError("Data-prep result must identify the aligned graph")
    relative = Path(aligned.get("path", ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise DataPrepHandoffError("Aligned graph path escapes repository")
    resolved = repository_root.resolve() / relative
    if not resolved.is_file():
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


def apply_data_prep_result(
    manifest_path: Path,
    result_path: Path,
    *,
    repository_root: Path,
    completion_receipt_path: Path | None = None,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Atomically advance a ready intake manifest after deterministic Stage 2."""
    validate_manifest(
        manifest_path,
        schema_path=schema_path,
        repository_root=repository_root,
    )
    manifest = load_json(manifest_path)
    if manifest["lifecycle_state"] != "ready_for_data_prep":
        raise DataPrepHandoffError(
            "Data prep can only finalize a ready_for_data_prep manifest"
        )
    result = load_json(result_path)
    _validate_data_prep_result(manifest, result, repository_root=repository_root)

    prior_manifest_hash = canonical_json_sha256(manifest)
    finalized = json.loads(json.dumps(manifest))
    finalized["inputs"]["aligned_graph"] = result["aligned_graph"]
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
        "prior_manifest_sha256": prior_manifest_hash,
        "analysis_ready_manifest_sha256": finalized_hash,
        "data_prep_result_sha256": result_hash,
        "analysis_parameters_sha256": finalized["analysis_parameters_sha256"],
        "lifecycle_state": ANALYSIS_READY,
        "self_verification": result["self_verification"],
    }
    completion = {
        **completion_base,
        "canonical_completion_sha256": canonical_json_sha256(completion_base),
    }
    destination = completion_receipt_path or (
        manifest_path.parent / "data_prep_completion_receipt.json"
    )
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
