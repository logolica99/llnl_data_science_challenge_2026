"""Deterministic control-plane state for the Part 2 NDE pipeline.

This module owns sequencing, hashes, access policy, and immutable receipts.  It
does not import or execute CT, registration, ROI, classification, rendering, or
evaluation algorithms.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence


PIPELINE_SCHEMA_VERSION = "part2-pipeline-manifest/1.0.0"
HANDOFF_SCHEMA_VERSION = "part2-stage-handoff/1.0.0"
RECEIPT_SCHEMA_VERSION = "part2-stage-receipt/1.0.0"
REGISTRATION_FREEZE_SCHEMA_VERSION = "part2-registration-freeze/1.0.0"
POST_FREEZE_HANDOFF_SCHEMA_VERSION = "part2-post-freeze-handoff/1.0.0"
CONTRACT_SCHEMA_VERSION = "agent-stage-contract/1.0.0"
MCP_RESPONSE_SCHEMA_VERSION = "part2-mcp-response/1.0.0"
STAGE_NUMBERS = tuple(range(7))
STAGE_NAMES = (
    "specimen_ingest",
    "design_diff",
    "data_prep",
    "strut_metrics",
    "defect_analysis",
    "sealed_evaluation",
    "nde_report",
)
REGISTRATION_MODES = {"challenge_aligned_json", "autonomous_v2"}
TERMINAL_STATES = {"pass", "manual_review", "halt"}
CONTROL_STATES = {"locked", "ready", "running", *TERMINAL_STATES}
SENSITIVE_KINDS = {
    "development_labels",
    "sealed_labels",
    "defect_labels",
    "aligned_graph",
}
RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "contract_version",
        "contract_sha256",
        "specimen_id",
        "stage_number",
        "stage",
        "owner",
        "attempt",
        "run_token",
        "terminal_state",
        "failure_kind",
        "completed_at",
        "config_sha256",
        "registration_mode",
        "predecessor_receipt_sha256",
        "input_handoff",
        "scoped_handoffs",
        "supplemental_handoffs",
        "registration_freeze",
        "output_artifacts",
        "assertions",
        "error",
        "canonical_receipt_sha256",
    }
)
RECEIPT_FAILURE_KINDS = frozenset(
    {
        "agent",
        "judgment",
        "deterministic_gate",
        "policy",
        "dependency",
        "schema",
        "artifact",
    }
)
RECEIPT_ERROR_CODES = frozenset(
    {
        "agent_failure",
        "manual_review_required",
        "deterministic_gate_failure",
        "policy_violation",
        "schema_incompatible",
        "artifact_verification_failure",
        "attempts_exhausted",
    }
)

RepositoryPath = str | Path
Clock = Callable[[], str]
UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z"
)


class OrchestrationError(ValueError):
    """Base class for fail-closed control-plane errors."""


class ManifestValidationError(OrchestrationError):
    """Raised when pipeline state or a frozen dependency is invalid."""


class IllegalTransitionError(OrchestrationError):
    """Raised when a caller attempts an undeclared state transition."""


class ArtifactVerificationError(OrchestrationError):
    """Raised when an artifact is missing, stale, aliased, or tampered."""


class ReceiptValidationError(OrchestrationError):
    """Raised when a completion or checkpoint receipt is not current."""


class AccessPolicyError(OrchestrationError):
    """Raised when a stage attempts to consume a forbidden input."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: RepositoryPath, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _read_object(path: RepositoryPath) -> dict[str, Any]:
    resolved = Path(path)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"Unreadable JSON object {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestValidationError(f"Expected a JSON object: {resolved}")
    return value


def _atomic_write_if_changed(path: Path, value: Any) -> bool:
    payload = _json_bytes(value)
    if path.is_file() and path.read_bytes() == payload:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    """Serialize transitions, including the Stage 5 one-shot reservation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_timestamp_text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 27
        or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None
    ):
        raise OrchestrationError(
            "Timestamp must be a bounded RFC3339 UTC value ending in Z"
        )
    from datetime import datetime

    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise OrchestrationError("Timestamp is not a valid UTC date-time") from exc
    return value


def _timestamp(value: str | None, clock: Clock | None) -> str:
    if value is None:
        if clock is None:
            from datetime import datetime, timezone

            value = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        else:
            value = clock()
    return _validate_timestamp_text(value)


def _validate_specimen_id(specimen_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", specimen_id):
        raise OrchestrationError(
            "specimen_id must contain only letters, digits, dot, underscore, or hyphen"
        )


def _relative_existing_file(
    repository_root: Path,
    path: RepositoryPath,
    *,
    reject_alias: bool,
) -> tuple[Path, str]:
    root = repository_root.resolve()
    candidate = Path(path)
    lexical = candidate if candidate.is_absolute() else root / candidate
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ArtifactVerificationError(f"Path escapes repository root: {path}") from exc
    if not resolved.is_file():
        raise ArtifactVerificationError(f"Artifact is unavailable: {relative}")
    if reject_alias:
        if candidate.is_absolute():
            supplied = candidate.resolve().relative_to(root).as_posix()
        else:
            supplied = candidate.as_posix()
        if supplied != relative:
            raise ArtifactVerificationError(
                f"Artifact aliases are forbidden; use canonical path {relative!r}"
            )
        if ".." in candidate.parts or "." in candidate.parts:
            raise ArtifactVerificationError(f"Non-canonical artifact path: {path}")
    return resolved, relative


def _repository_output_location(
    repository_root: Path,
    supplied: RepositoryPath | None,
    default: Path,
    *,
    required_directory: Path,
    required_name: str | None = None,
) -> tuple[Path, Path]:
    """Resolve a new control artifact without permitting aliases or traversal."""

    root = repository_root.resolve()
    candidate = Path(supplied) if supplied is not None else default
    if any(part in {".", ".."} for part in candidate.parts):
        raise ArtifactVerificationError(
            f"Non-canonical control artifact path: {candidate}"
        )
    lexical = candidate if candidate.is_absolute() else root / candidate
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactVerificationError(
            f"Control artifact path escapes repository root: {candidate}"
        ) from exc
    expected_directory = (root / required_directory).resolve()
    if resolved.parent != expected_directory:
        raise ArtifactVerificationError(
            f"Control artifact must be directly under {required_directory.as_posix()}"
        )
    if required_name is not None and resolved.name != required_name:
        raise ArtifactVerificationError(
            f"Control artifact filename must be {required_name!r}"
        )
    if resolved.exists() and not resolved.is_file():
        raise ArtifactVerificationError(
            f"Control artifact destination is not a file: {relative.as_posix()}"
        )
    return relative, resolved


def _pipeline_manifest_location(
    repository_root: Path, manifest_path: RepositoryPath
) -> Path:
    """Validate the canonical manifest location before creating a lock file."""

    root = repository_root.resolve()
    candidate = Path(manifest_path)
    if any(part in {".", ".."} for part in candidate.parts):
        raise ArtifactVerificationError(f"Non-canonical manifest path: {candidate}")
    lexical = candidate if candidate.is_absolute() else root / candidate
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactVerificationError("Pipeline manifest escapes repository") from exc
    if (
        len(relative.parts) != 3
        or relative.parts[0] != "analysis"
        or relative.parts[2] != "manifest.json"
    ):
        raise ArtifactVerificationError(
            "Pipeline manifest must be analysis/<specimen_id>/manifest.json"
        )
    _validate_specimen_id(relative.parts[1])
    if not resolved.is_file():
        raise ArtifactVerificationError(
            f"Pipeline manifest is unavailable: {relative.as_posix()}"
        )
    return resolved


def artifact_record(
    repository_root: RepositoryPath,
    path: RepositoryPath,
    *,
    role: str,
    consumer: str | None = None,
    producer: str | None = None,
    phase: str = "input",
    sensitivity: str | None = None,
    replaces_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a compact hash record for one existing repository artifact."""

    root = Path(repository_root).resolve()
    resolved, relative = _relative_existing_file(root, path, reject_alias=False)
    record: dict[str, Any] = {
        "path": relative,
        "sha256": sha256_file(resolved),
        "role": role,
        "phase": phase,
    }
    if consumer is not None:
        record["consumer"] = consumer
    if producer is not None:
        record["producer"] = producer
    if sensitivity is not None:
        if sensitivity not in SENSITIVE_KINDS:
            raise OrchestrationError(f"Unsupported sensitivity: {sensitivity}")
        record["sensitivity"] = sensitivity
    if replaces_sha256 is not None:
        record["replaces_sha256"] = replaces_sha256
    return record


def _normalize_artifact(
    repository_root: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {"path", "sha256", "role"}
    missing = sorted(required - set(value))
    if missing:
        raise ArtifactVerificationError(
            "Artifact record is missing: " + ", ".join(missing)
        )
    if not isinstance(value["path"], str) or Path(value["path"]).is_absolute():
        raise ArtifactVerificationError("Artifact paths must be repository-relative")
    resolved, relative = _relative_existing_file(
        repository_root, value["path"], reject_alias=True
    )
    digest = sha256_file(resolved)
    if value["sha256"] != digest:
        raise ArtifactVerificationError(
            f"Artifact SHA-256 mismatch for {relative}: expected {value['sha256']}, got {digest}"
        )
    role = value["role"]
    if not isinstance(role, str) or not role:
        raise ArtifactVerificationError("Artifact role must be a non-empty string")
    result: dict[str, Any] = {
        "path": relative,
        "sha256": digest,
        "role": role,
        "phase": str(value.get("phase", "input")),
    }
    for key in ("consumer", "producer", "sensitivity", "replaces_sha256"):
        if key in value:
            result[key] = value[key]
    if result.get("sensitivity") not in (None, *sorted(SENSITIVE_KINDS)):
        raise ArtifactVerificationError(
            f"Unsupported artifact sensitivity {result.get('sensitivity')!r}"
        )
    return result


def _render_pattern(pattern: str, manifest: Mapping[str, Any]) -> str:
    rendered = (
        pattern.replace("<specimen_id>", str(manifest["specimen_id"]))
        .replace("<stage_number>", "*")
        .replace("<stage>", "*")
        .replace("<attempt>", "*")
        .replace("<timestamp>", "*")
        .replace("<strut_id>", "*")
    )
    # Contract prose placeholders such as <manifest-declared-ct-volume> are
    # bounded by role, consumer, phase, hash, and repository containment.
    return re.sub(r"<[^>]+>", "*", rendered)


def _rule_matches(
    rule: Mapping[str, Any],
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> bool:
    role = rule.get("role")
    if role is not None and not fnmatch.fnmatchcase(str(artifact["role"]), str(role)):
        return False
    path = rule.get("path")
    if path is not None and not fnmatch.fnmatchcase(
        str(artifact["path"]), _render_pattern(str(path), manifest)
    ):
        return False
    consumers = rule.get("consumers")
    if consumers is not None and artifact.get("consumer") not in consumers:
        return False
    producers = rule.get("producers")
    if producers is not None and artifact.get("producer") not in producers:
        return False
    phases = rule.get("phases")
    if phases is not None and artifact.get("phase", "input") not in phases:
        return False
    return True


def _contract_artifact_policy(
    contract: Mapping[str, Any],
    direction: Literal["input", "output"],
) -> dict[str, Any]:
    value = contract.get(f"{direction}_artifacts", {})
    if not isinstance(value, dict):
        raise ManifestValidationError(
            f"{contract.get('stage')}: {direction}_artifacts must be an object"
        )
    return value


def _require_manifest_declared_source(
    manifest: Mapping[str, Any], artifact: Mapping[str, Any]
) -> None:
    """Bind downstream source placeholders to the passed Stage 0 handoff."""

    role = str(artifact["role"])
    if role == "autonomous_validation_reference":
        # This source is introduced only by the dedicated post-freeze
        # authorization function, never by a normal Stage 2 start.
        return
    source_role = {
        "challenge_aligned_json": "challenge_aligned_graph",
        "aligned_graph_reference": "challenge_aligned_graph",
    }.get(role, role)
    stage_zero = manifest.get("stages", {}).get("0", {})
    passed_attempt = next(
        (
            attempt
            for attempt in reversed(stage_zero.get("attempts", []))
            if attempt.get("effective_terminal_state") == "pass"
        ),
        None,
    )
    if passed_attempt is None:
        raise AccessPolicyError(
            f"Manifest-declared source {source_role!r} is unavailable before Stage 0 pass"
        )
    declared = next(
        (
            value
            for value in passed_attempt.get("input_artifacts", [])
            if value.get("role") == source_role
        ),
        None,
    )
    if declared is None:
        raise AccessPolicyError(
            f"Stage 0 did not declare an authorized {source_role!r} source"
        )
    if (
        artifact.get("path") != declared.get("path")
        or artifact.get("sha256") != declared.get("sha256")
    ):
        raise AccessPolicyError(
            f"{role!r} does not match the path and hash frozen by Stage 0"
        )


def _validate_artifact_allowlist(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
    *,
    direction: Literal["input", "output"],
    require_all: bool,
) -> None:
    policy = _contract_artifact_policy(contract, direction)
    allowed = policy.get("allowed", [])
    forbidden = policy.get("forbidden", [])
    if not isinstance(allowed, list) or not isinstance(forbidden, list):
        raise ManifestValidationError("Artifact allowlists must be arrays")
    for artifact in artifacts:
        if any(_rule_matches(rule, artifact, manifest) for rule in forbidden):
            raise AccessPolicyError(
                f"{contract['stage']} forbids {direction} artifact "
                f"{artifact['role']} at {artifact['path']}"
            )
        if not any(_rule_matches(rule, artifact, manifest) for rule in allowed):
            raise AccessPolicyError(
                f"{contract['stage']} does not allow {direction} artifact "
                f"{artifact['role']} at {artifact['path']}"
            )
        matching_rules = [
            rule for rule in allowed if _rule_matches(rule, artifact, manifest)
        ]
        if direction == "input" and any(
            str(rule.get("path", "")).startswith("<manifest-declared-")
            for rule in matching_rules
        ):
            _require_manifest_declared_source(manifest, artifact)
    role_counts: dict[str, int] = {}
    for artifact in artifacts:
        role = str(artifact["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    repeated = []
    for role, count in role_counts.items():
        if count <= 1:
            continue
        matching = [
            rule
            for rule in allowed
            if fnmatch.fnmatchcase(role, str(rule.get("role", "")))
        ]
        if not any(rule.get("allow_multiple") is True for rule in matching):
            repeated.append(role)
    if repeated:
        raise AccessPolicyError(
            f"{contract['stage']} received duplicate {direction} roles without "
            "allow_multiple: " + ", ".join(sorted(repeated))
        )
    if require_all:
        roles = [str(artifact["role"]) for artifact in artifacts]
        missing = [
            role
            for role in policy.get("required_roles", [])
            if role != "stage_handoff"
            if not any(fnmatch.fnmatchcase(candidate, str(role)) for candidate in roles)
        ]
        if missing:
            raise ReceiptValidationError(
                f"{contract['stage']} is missing required {direction} roles: "
                + ", ".join(missing)
            )


def _validate_contract_document(stage_number: int, contract: Mapping[str, Any]) -> None:
    dependencies = contract.get("required_dependencies")
    if not isinstance(dependencies, Mapping):
        raise ManifestValidationError(
            f"Stage {stage_number} required_dependencies must be an object"
        )
    agents = dependencies.get("agents")
    tools = dependencies.get("mcp_tools")
    if not isinstance(agents, list) or not agents:
        raise ManifestValidationError(
            f"Stage {stage_number} must declare at least one bounded agent"
        )
    if not isinstance(tools, list) or not tools:
        raise ManifestValidationError(
            f"Stage {stage_number} must declare segmentation-tools dependencies"
        )
    for agent in agents:
        if (
            not isinstance(agent, Mapping)
            or not isinstance(agent.get("name"), str)
            or not agent.get("name")
            or agent.get("contract_version") != CONTRACT_SCHEMA_VERSION
        ):
            raise ManifestValidationError(
                f"Stage {stage_number} has an incompatible agent dependency"
            )
    for tool in tools:
        if (
            not isinstance(tool, Mapping)
            or tool.get("server") != "segmentation-tools"
            or not isinstance(tool.get("name"), str)
            or not tool.get("name")
            or not isinstance(tool.get("response_schema_version"), str)
            or not tool.get("response_schema_version")
        ):
            raise ManifestValidationError(
                f"Stage {stage_number} has an incompatible MCP dependency"
            )
    for direction in ("input", "output"):
        policy = contract.get(f"{direction}_artifacts")
        if not isinstance(policy, Mapping):
            raise ManifestValidationError(
                f"Stage {stage_number} {direction}_artifacts must be an object"
            )
        allowed = policy.get("allowed")
        required_roles = policy.get("required_roles")
        forbidden = policy.get("forbidden", [])
        if (
            not isinstance(allowed, list)
            or not isinstance(required_roles, list)
            or not isinstance(forbidden, list)
            or not all(isinstance(role, str) and role for role in required_roles)
        ):
            raise ManifestValidationError(
                f"Stage {stage_number} {direction} artifact policy is incompatible"
            )
        for rule in [*allowed, *forbidden]:
            if (
                not isinstance(rule, Mapping)
                or not isinstance(rule.get("role"), str)
                or not rule.get("role")
                or not isinstance(rule.get("path"), str)
                or not rule.get("path")
            ):
                raise ManifestValidationError(
                    f"Stage {stage_number} has an invalid {direction} artifact rule"
                )
        missing_rules = [
            role
            for role in required_roles
            if not any(
                fnmatch.fnmatchcase(role, str(rule["role"])) for rule in allowed
            )
        ]
        if missing_rules:
            raise ManifestValidationError(
                f"Stage {stage_number} required {direction} roles lack allow rules: "
                + ", ".join(missing_rules)
            )
    if stage_number == 4:
        schemas = contract.get("output_document_schemas")
        if not isinstance(schemas, Mapping) or set(schemas) != {
            "missing_calibration_attestation",
            "classifier_verifier_report",
        }:
            raise ManifestValidationError(
                "Stage 4 must declare both closed output document schemas"
            )
        attestation = schemas["missing_calibration_attestation"]
        verifier = schemas["classifier_verifier_report"]
        if (
            not isinstance(attestation, Mapping)
            or attestation.get("schema_version")
            != "missing-calibration-attestation/1.0.0"
            or attestation.get("additional_properties") is not False
            or set(attestation.get("required_fields", []))
            != {
                "schema_version",
                "owner",
                "gate",
                "specimen_id",
                "stage_number",
                "attempt",
                "run_token",
                "scoped_handoff_sha256",
                "per_strut_metrics_sha256",
                "findings_missing_sha256",
                "development_split_accessed",
                "raw_development_labels_included",
                "calibration_summary",
            }
            or not isinstance(verifier, Mapping)
            or verifier.get("schema_version")
            != "classifier-verifier-report/1.0.0"
            or verifier.get("additional_properties") is not False
            or set(verifier.get("binding_fields", []))
            != {
                "classified_struts_sha256",
                "thresholds_sha256",
                "decision_log_sha256",
                "evidence_set_sha256",
                "per_strut_metrics_sha256",
                "specialist_findings_sha256",
            }
        ):
            raise ManifestValidationError(
                "Stage 4 output document schemas are incompatible"
            )


def _load_contracts(
    repository_root: Path,
    contracts_directory: RepositoryPath,
) -> list[dict[str, Any]]:
    root = repository_root.resolve()
    directory = Path(contracts_directory)
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ManifestValidationError("Contracts directory escapes repository") from exc
    contracts: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        value = _read_object(path)
        if "stage_number" not in value:
            continue
        stage_number = value["stage_number"]
        if not isinstance(stage_number, int) or stage_number not in STAGE_NUMBERS:
            continue
        if stage_number in contracts:
            raise ManifestValidationError(f"Duplicate contract for Stage {stage_number}")
        if value.get("schema_version") != CONTRACT_SCHEMA_VERSION:
            raise ManifestValidationError(
                f"Stage {stage_number} contract schema is incompatible"
            )
        for key in (
            "stage",
            "owner",
            "execution_kind",
            "maximum_attempts",
            "required_dependencies",
            "input_artifacts",
            "output_artifacts",
        ):
            if key not in value:
                raise ManifestValidationError(
                    f"Stage {stage_number} contract is missing {key}"
                )
        _validate_contract_document(stage_number, value)
        relative = path.relative_to(root).as_posix()
        contracts[stage_number] = {
            "document": value,
            "path": relative,
            "sha256": sha256_file(path),
        }
    if set(contracts) != set(STAGE_NUMBERS):
        missing = sorted(set(STAGE_NUMBERS) - set(contracts))
        raise ManifestValidationError(
            "Missing stage contracts: " + ", ".join(str(value) for value in missing)
        )
    result = [contracts[number] for number in STAGE_NUMBERS]
    for number, item in enumerate(result):
        contract = item["document"]
        if contract.get("stage") != STAGE_NAMES[number]:
            raise ManifestValidationError(
                f"Stage {number} must be named {STAGE_NAMES[number]!r}"
            )
        if contract.get("invoked_by") != "orchestrator":
            raise ManifestValidationError(
                f"Stage {number} must declare invoked_by='orchestrator'"
            )
        assertions = contract.get("required_receipt_assertions")
        if (
            not isinstance(assertions, list)
            or not assertions
            or len(assertions) != len(set(assertions))
            or not all(isinstance(value, str) and value for value in assertions)
        ):
            raise ManifestValidationError(
                f"Stage {number} required_receipt_assertions are schema-incompatible"
            )
        expected_next = (
            "pipeline_complete"
            if number == STAGE_NUMBERS[-1]
            else result[number + 1]["document"]["stage"]
        )
        if contract.get("next_stage") != expected_next:
            raise ManifestValidationError(
                f"Stage {number} next_stage must be {expected_next!r}"
            )
        maximum = contract["maximum_attempts"]
        if not isinstance(maximum, int) or not 1 <= maximum <= 2:
            raise ManifestValidationError(
                f"Stage {number} maximum_attempts must be one or two"
            )
        if contract.get("one_shot", False) and maximum != 1:
            raise ManifestValidationError("One-shot stages must allow exactly one attempt")
    return result


def _with_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    base = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    return {**base, field: canonical_json_sha256(base)}


def _event(
    manifest: dict[str, Any],
    *,
    timestamp: str,
    action: str,
    stage_number: int | None,
    details: Mapping[str, Any] | None = None,
) -> None:
    manifest["events"].append(
        {
            "sequence": len(manifest["events"]),
            "timestamp": timestamp,
            "action": action,
            "stage_number": stage_number,
            "details": dict(details or {}),
        }
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> bool:
    manifest["manifest_sha256"] = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return _atomic_write_if_changed(path, manifest)


def create_pipeline_manifest(
    *,
    repository_root: RepositoryPath,
    specimen_id: str,
    config_path: RepositoryPath,
    registration_mode: str,
    manifest_path: RepositoryPath | None = None,
    contracts_directory: RepositoryPath = "analysis/contracts",
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Create an idempotent, self-hashed Stage 0→6 pipeline manifest."""

    _validate_specimen_id(specimen_id)
    if registration_mode not in REGISTRATION_MODES:
        raise OrchestrationError(f"Unsupported registration mode: {registration_mode}")
    root = Path(repository_root).resolve()
    config_resolved, config_relative = _relative_existing_file(
        root, config_path, reject_alias=False
    )
    contracts = _load_contracts(root, contracts_directory)
    created_at = _timestamp(timestamp, clock)
    _, destination = _repository_output_location(
        root,
        manifest_path,
        Path("analysis") / specimen_id / "manifest.json",
        required_directory=Path("analysis") / specimen_id,
        required_name="manifest.json",
    )

    with _manifest_lock(destination):
        if destination.is_file():
            existing = _load_manifest_unlocked(
                destination, root, verify_artifacts=True
            )
            identity = (
                existing["specimen_id"],
                existing["registration_mode"],
                existing["config"]["path"],
                existing["config"]["sha256"],
            )
            requested = (
                specimen_id,
                registration_mode,
                config_relative,
                sha256_file(config_resolved),
            )
            if identity != requested:
                raise ManifestValidationError(
                    "Existing pipeline manifest has different frozen inputs"
                )
            return {"manifest": existing, "path": str(destination), "changed": False}

        stage_records: dict[str, Any] = {}
        for number, item in enumerate(contracts):
            contract = item["document"]
            stage_records[str(number)] = {
                "stage_number": number,
                "name": contract["stage"],
                "owner": contract["owner"],
                "execution_kind": contract["execution_kind"],
                "maximum_attempts": contract["maximum_attempts"],
                "one_shot": bool(contract.get("one_shot", False)),
                "state": "ready" if number == 0 else "locked",
                "attempt_count": 0,
                "current_attempt": None,
                "attempts": [],
                "contract": {
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "version": contract["schema_version"],
                },
                "unlocked_at": created_at if number == 0 else None,
                "completed_at": None,
                "completion_receipt": None,
                "control_halt": None,
            }
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "specimen_id": specimen_id,
            "registration_mode": registration_mode,
            "pipeline_state": "ready",
            "current_stage": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "revision": 0,
            "config": {
                "path": config_relative,
                "sha256": sha256_file(config_resolved),
            },
            "stage_order": list(STAGE_NUMBERS),
            "stages": stage_records,
            "predecessor_receipt_sha256": None,
            "artifact_index": [],
            "sensitive_artifact_hashes": {
                kind: [] for kind in sorted(SENSITIVE_KINDS)
            },
            "sealed_evaluation": {
                "consumed": False,
                "consumed_at": None,
                "stage_attempt": None,
                "run_token": None,
            },
            "events": [],
        }
        _event(
            manifest,
            timestamp=created_at,
            action="pipeline_created",
            stage_number=0,
            details={"registration_mode": registration_mode},
        )
        changed = _write_manifest(destination, manifest)
        return {"manifest": manifest, "path": str(destination), "changed": changed}


def _load_stage_contract(
    manifest: Mapping[str, Any],
    stage_number: int,
    repository_root: Path,
) -> dict[str, Any]:
    stage = manifest["stages"][str(stage_number)]
    path = repository_root / stage["contract"]["path"]
    value = _read_object(path)
    if sha256_file(path) != stage["contract"]["sha256"]:
        raise ManifestValidationError(
            f"Stage {stage_number} contract hash changed after pipeline creation"
        )
    if value.get("schema_version") != stage["contract"]["version"]:
        raise ManifestValidationError(
            f"Stage {stage_number} contract schema changed after pipeline creation"
        )
    return value


def _active_artifact_records(manifest: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for record in manifest.get("artifact_index", []):
        if record.get("state", "active") == "active":
            yield record


def _verify_indexed_artifacts(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    skipped_paths: set[str] | None = None,
) -> None:
    skipped = skipped_paths or set()
    for record in _active_artifact_records(manifest):
        if record["path"] in skipped:
            continue
        _normalize_artifact(repository_root, record)


def _verify_artifact_catalog(
    manifest: Mapping[str, Any], repository_root: Path
) -> None:
    """Re-derive the artifact index and sensitive registry from frozen evidence."""

    derived_index: list[dict[str, Any]] = []
    active_by_path: dict[str, dict[str, Any]] = {}
    sensitive: dict[str, set[str]] = {kind: set() for kind in SENSITIVE_KINDS}
    artifact_fields = {
        "path",
        "sha256",
        "role",
        "phase",
        "consumer",
        "producer",
        "sensitivity",
        "replaces_sha256",
    }

    def remember_sensitive(
        contract: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
    ) -> None:
        for artifact in artifacts:
            kind = _sensitive_kind_for_artifact(contract, artifact)
            if kind is not None:
                sensitive[kind].add(str(artifact["sha256"]))

    def add_output(
        contract: Mapping[str, Any],
        stage_number: int,
        attempt_number: int,
        raw: Mapping[str, Any],
    ) -> None:
        output = {key: copy.deepcopy(value) for key, value in raw.items() if key in artifact_fields}
        if set(raw) - artifact_fields:
            raise ManifestValidationError(
                f"Stage {stage_number} artifact history has undeclared fields"
            )
        kind = _sensitive_kind_for_artifact(contract, output)
        record = {
            **output,
            "stage_number": stage_number,
            "attempt": attempt_number,
            "state": "active",
            "sensitivity": kind,
        }
        existing = active_by_path.get(str(output["path"]))
        identity = (output["path"], output["sha256"], output["role"])
        if existing is not None:
            existing_identity = (
                existing["path"],
                existing["sha256"],
                existing["role"],
            )
            if identity == existing_identity:
                return
            replacement_rules = [
                rule
                for rule in _contract_artifact_policy(contract, "output").get(
                    "allowed", []
                )
                if _rule_matches(rule, output, manifest)
                and rule.get("replaces_role") == existing["role"]
            ]
            if (
                not replacement_rules
                or output.get("replaces_sha256") != existing["sha256"]
            ):
                raise ManifestValidationError(
                    f"Artifact catalog has undeclared replacement at {output['path']}"
                )
            existing["state"] = "superseded"
            existing["superseded_by_sha256"] = output["sha256"]
        elif "replaces_sha256" in output:
            raise ManifestValidationError(
                f"Artifact catalog replacement target is absent at {output['path']}"
            )
        derived_index.append(record)
        active_by_path[str(output["path"])] = record
        if kind is not None:
            sensitive[kind].add(str(output["sha256"]))

    for stage_number in STAGE_NUMBERS:
        stage = manifest["stages"][str(stage_number)]
        contract = _load_stage_contract(manifest, stage_number, repository_root)
        for attempt in stage.get("attempts", []):
            remember_sensitive(contract, attempt.get("input_artifacts", []))
            for output in attempt.get("output_artifacts", []):
                add_output(
                    contract,
                    stage_number,
                    int(attempt["attempt"]),
                    output,
                )
            resolution = attempt.get("manual_review_resolution")
            if isinstance(resolution, Mapping):
                resolution_artifact = {
                    key: copy.deepcopy(value)
                    for key, value in resolution.items()
                    if key in artifact_fields
                }
                add_output(
                    {"output_artifacts": {"sensitive_roles": {}, "allowed": []}},
                    stage_number,
                    int(attempt["attempt"]),
                    resolution_artifact,
                )
            for supplemental_record in attempt.get("supplemental_handoffs", []):
                payload = _verify_hashed_json_record(
                    supplemental_record,
                    repository_root,
                    canonical_field="canonical_handoff_sha256",
                )
                remember_sensitive(contract, payload.get("input_artifacts", []))

    if manifest.get("artifact_index") != derived_index:
        raise ManifestValidationError(
            "Pipeline artifact_index is stale or inconsistent with immutable receipts"
        )
    expected_registry = {
        kind: sorted(hashes) for kind, hashes in sorted(sensitive.items())
    }
    registry = manifest.get("sensitive_artifact_hashes")
    if registry != expected_registry:
        raise ManifestValidationError(
            "Sensitive artifact registry is stale or inconsistent with frozen evidence"
        )


def _validate_manifest_state_topology(manifest: Mapping[str, Any]) -> None:
    _validate_timestamp_text(manifest.get("created_at"))
    _validate_timestamp_text(manifest.get("updated_at"))
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ManifestValidationError("Pipeline stages must be an object")
    states = [stages[str(number)].get("state") for number in STAGE_NUMBERS]
    if all(state == "pass" for state in states):
        if manifest.get("pipeline_state") != "pass" or manifest.get("current_stage") is not None:
            raise ManifestValidationError(
                "Completed pipeline state/current_stage is inconsistent"
            )
    else:
        current = next(
            (number for number, state in enumerate(states) if state != "pass"),
            None,
        )
        if current is None or manifest.get("current_stage") != current:
            raise ManifestValidationError("Pipeline current_stage skips the pass prefix")
        current_state = states[current]
        if current_state not in {"ready", "running", "manual_review", "halt"}:
            raise ManifestValidationError("Current stage is not uniquely actionable")
        if manifest.get("pipeline_state") != current_state:
            raise ManifestValidationError(
                "pipeline_state does not match the current stage state"
            )
        if any(state != "locked" for state in states[current + 1 :]):
            raise ManifestValidationError("A downstream stage was unlocked or completed early")

    for number in STAGE_NUMBERS:
        stage = stages[str(number)]
        for field in ("unlocked_at", "completed_at"):
            if stage.get(field) is not None:
                _validate_timestamp_text(stage[field])
        attempts = stage.get("attempts")
        if not isinstance(attempts, list):
            raise ManifestValidationError(f"Stage {number} attempts must be an array")
        if stage.get("attempt_count") != len(attempts):
            raise ManifestValidationError(
                f"Stage {number} attempt_count does not match attempt history"
            )
        expected_numbers = list(range(1, len(attempts) + 1))
        if [attempt.get("attempt") for attempt in attempts] != expected_numbers:
            raise ManifestValidationError(f"Stage {number} attempt numbering is invalid")
        for attempt in attempts:
            _validate_timestamp_text(attempt.get("started_at"))
            if attempt.get("completed_at") is not None:
                _validate_timestamp_text(attempt["completed_at"])
            resolution = attempt.get("manual_review_resolution")
            if isinstance(resolution, Mapping):
                _validate_timestamp_text(resolution.get("resolved_at"))
        if len(attempts) > stage.get("maximum_attempts", 0):
            raise ManifestValidationError(f"Stage {number} exceeded its attempt limit")
        running = [attempt for attempt in attempts if attempt.get("state") == "running"]
        if running and (len(running) != 1 or running[0] is not attempts[-1]):
            raise ManifestValidationError(f"Stage {number} has invalid running attempts")
        state = stage.get("state")
        current_attempt = stage.get("current_attempt")
        if state == "locked":
            if attempts or current_attempt is not None:
                raise ManifestValidationError(f"Locked Stage {number} has attempt state")
        elif state == "ready":
            if current_attempt is not None:
                raise ManifestValidationError(f"Ready Stage {number} has a current attempt")
            if attempts and attempts[-1].get("effective_terminal_state") != "manual_review":
                raise ManifestValidationError(
                    f"Ready Stage {number} was not explicitly resumed from review"
                )
        elif state == "running":
            if not attempts or current_attempt != attempts[-1].get("attempt"):
                raise ManifestValidationError(f"Running Stage {number} has no current attempt")
            if attempts[-1].get("state") != "running":
                raise ManifestValidationError(f"Running Stage {number} history is inconsistent")
        elif state in TERMINAL_STATES:
            dependency_halt = (
                state == "halt"
                and not attempts
                and stage.get("control_halt", {}).get("code")
                == "missing_or_incompatible_dependency"
            )
            if not dependency_halt:
                if not attempts or attempts[-1].get("effective_terminal_state") != state:
                    raise ManifestValidationError(
                        f"Terminal Stage {number} history is inconsistent"
                    )
                if current_attempt != attempts[-1].get("attempt"):
                    raise ManifestValidationError(
                        f"Terminal Stage {number} current_attempt is inconsistent"
                    )

    events = manifest.get("events")
    revision = manifest.get("revision")
    if (
        not isinstance(events, list)
        or not isinstance(revision, int)
        or revision < 0
        or len(events) != revision + 1
        or [event.get("sequence") for event in events] != list(range(len(events)))
    ):
        raise ManifestValidationError("Pipeline event/revision history is inconsistent")
    for event in events:
        _validate_timestamp_text(event.get("timestamp"))

    stage_five = stages["5"]
    marker = manifest.get("sealed_evaluation", {})
    if stage_five.get("attempt_count", 0) == 0:
        if marker.get("consumed") is not False:
            raise ManifestValidationError("Stage 5 is marked consumed before its first start")
    else:
        first_attempt = stage_five["attempts"][0]
        by_role = {
            artifact["role"]: artifact
            for artifact in first_attempt.get("input_artifacts", [])
        }
        if (
            marker.get("consumed") is not True
            or _validate_timestamp_text(marker.get("consumed_at"))
            != marker.get("consumed_at")
            or marker.get("stage_attempt") != 1
            or marker.get("run_token") != first_attempt.get("run_token")
            or marker.get("config_sha256") != manifest["config"]["sha256"]
            or marker.get("classified_struts_sha256")
            != by_role.get("classified_struts", {}).get("sha256")
            or marker.get("sealed_labels_sha256")
            != by_role.get("sealed_labels", {}).get("sha256")
        ):
            raise ManifestValidationError("Stage 5 one-shot marker is inconsistent")

    last_pass_receipt = None
    for number in STAGE_NUMBERS:
        stage = stages[str(number)]
        if stage.get("state") != "pass":
            break
        record = stage.get("completion_receipt")
        if not isinstance(record, Mapping):
            raise ManifestValidationError(f"Passed Stage {number} has no receipt")
        last_pass_receipt = record.get("canonical_sha256")
    if manifest.get("predecessor_receipt_sha256") != last_pass_receipt:
        raise ManifestValidationError("Predecessor receipt chain head is inconsistent")


def _verify_attempt_control_bindings(
    manifest: Mapping[str, Any],
    stage_number: int,
    attempt: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify one attempt's closed control documents without re-reading science data."""

    stage = manifest["stages"][str(stage_number)]
    contract = _load_stage_contract(manifest, stage_number, repository_root)
    expected_predecessor = (
        None
        if stage_number == 0
        else manifest["stages"][str(stage_number - 1)]["completion_receipt"][
            "canonical_sha256"
        ]
    )
    if attempt.get("predecessor_receipt_sha256") != expected_predecessor:
        raise ReceiptValidationError(
            f"Stage {stage_number} attempt has a stale predecessor receipt"
        )
    inputs = attempt.get("input_artifacts")
    if not isinstance(inputs, list):
        raise ManifestValidationError(
            f"Stage {stage_number} attempt inputs must be an array"
        )
    allowed_artifact_fields = {
        "path",
        "sha256",
        "role",
        "phase",
        "consumer",
        "producer",
        "sensitivity",
        "replaces_sha256",
    }
    for artifact in inputs:
        if (
            not isinstance(artifact, Mapping)
            or not {"path", "sha256", "role", "phase"} <= set(artifact)
            or set(artifact) - allowed_artifact_fields
        ):
            raise ManifestValidationError(
                f"Stage {stage_number} attempt has a malformed input artifact record"
            )
    expected_run_token = canonical_json_sha256(
        {
            "specimen_id": manifest["specimen_id"],
            "stage_number": stage_number,
            "attempt": attempt["attempt"],
            "started_at": attempt["started_at"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": stage["contract"]["sha256"],
            "predecessor_receipt_sha256": attempt["predecessor_receipt_sha256"],
            "input_artifacts": inputs,
        }
    )
    if attempt.get("run_token") != expected_run_token:
        raise ManifestValidationError(f"Stage {stage_number} run token is invalid")
    _validate_artifact_allowlist(
        manifest, contract, inputs, direction="input", require_all=True
    )
    _enforce_sensitive_access(manifest, stage_number, inputs)
    if attempt.get("state") == "running":
        _enforce_artifact_lineage(manifest, inputs)
    else:
        indexed_by_path: dict[str, list[Mapping[str, Any]]] = {}
        for indexed in manifest.get("artifact_index", []):
            indexed_by_path.setdefault(str(indexed.get("path")), []).append(indexed)
        for artifact in inputs:
            indexed = indexed_by_path.get(artifact["path"], [])
            if indexed and not any(
                record.get("sha256") == artifact["sha256"]
                and record.get("role") == artifact["role"]
                for record in indexed
            ):
                raise ArtifactVerificationError(
                    f"Historical input has stale lineage at {artifact['path']}"
                )

    handoff_record = attempt.get("handoff")
    if not isinstance(handoff_record, Mapping) or set(handoff_record) != {
        "path",
        "sha256",
        "canonical_sha256",
    }:
        raise ReceiptValidationError(
            f"Stage {stage_number} handoff record is malformed"
        )
    expected_handoff_path = _default_handoff_path(
        manifest, stage_number, attempt["attempt"]
    ).as_posix()
    if handoff_record.get("path") != expected_handoff_path:
        raise ReceiptValidationError(
            f"Stage {stage_number} handoff path is not canonical"
        )
    supplemental_records = attempt.get("supplemental_handoffs", [])
    scoped_records = attempt.get("scoped_handoffs", [])
    if (
        not isinstance(supplemental_records, list)
        or any(
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "canonical_sha256"}
            for record in supplemental_records
        )
        or not isinstance(scoped_records, list)
        or any(
            not isinstance(record, Mapping)
            or set(record)
            != {"consumer", "path", "sha256", "canonical_sha256"}
            for record in scoped_records
        )
    ):
        raise ReceiptValidationError(
            f"Stage {stage_number} scoped or supplemental handoff record is malformed"
        )
    if supplemental_records:
        expected_supplemental_path = (
            Path("analysis")
            / manifest["specimen_id"]
            / "handoffs"
            / f"stage_2_post_freeze_validation_attempt_{attempt['attempt']}.json"
        ).as_posix()
        if (
            stage_number != 2
            or len(supplemental_records) != 1
            or supplemental_records[0]["path"] != expected_supplemental_path
        ):
            raise ReceiptValidationError(
                f"Stage {stage_number} supplemental handoff path or count is invalid"
            )
    handoff, supplemental, scoped = _verify_attempt_handoffs(
        attempt, repository_root
    )
    shared_inputs, input_scopes = _partition_scoped_inputs(contract, inputs)
    handoff_base = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "specimen_id": manifest["specimen_id"],
        "stage_number": stage_number,
        "stage": stage["name"],
        "owner": stage["owner"],
        "attempt": attempt["attempt"],
        "run_token": attempt["run_token"],
        "created_at": attempt["started_at"],
        "registration_mode": manifest["registration_mode"],
        "config_sha256": manifest["config"]["sha256"],
        "contract_version": stage["contract"]["version"],
        "contract_sha256": stage["contract"]["sha256"],
        "predecessor_receipt_sha256": attempt["predecessor_receipt_sha256"],
        "input_artifacts": shared_inputs,
        "forbidden_operations": contract.get("forbidden_operations", []),
    }
    if handoff != _with_self_hash(handoff_base, "canonical_handoff_sha256"):
        raise ReceiptValidationError(
            f"Stage {stage_number} handoff is stale, open-ended, or misbound"
        )
    if len(scoped_records) != len(input_scopes) or len(scoped) != len(input_scopes):
        raise ReceiptValidationError(
            f"Stage {stage_number} scoped handoff set is incomplete"
        )
    for (specification, scoped_inputs), record, payload in zip(
        input_scopes, scoped_records, scoped, strict=True
    ):
        consumer = specification["consumer"]
        if not isinstance(record, Mapping) or set(record) != {
            "consumer",
            "path",
            "sha256",
            "canonical_sha256",
        }:
            raise ReceiptValidationError(
                f"Stage {stage_number} scoped handoff record is malformed"
            )
        expected_path = _scoped_handoff_path(
            specification, manifest, attempt["attempt"]
        ).as_posix()
        if record.get("consumer") != consumer or record.get("path") != expected_path:
            raise ReceiptValidationError(
                f"Stage {stage_number} scoped handoff path or consumer is invalid"
            )
        scoped_base = {
            key: copy.deepcopy(value)
            for key, value in handoff_base.items()
            if key not in {"owner", "input_artifacts"}
        }
        scoped_base.update(
            {
                "owner": consumer,
                "scope": consumer,
                "parent_handoff": handoff_record,
                "input_artifacts": scoped_inputs,
            }
        )
        if payload != _with_self_hash(scoped_base, "canonical_handoff_sha256"):
            raise ReceiptValidationError(
                f"Stage {stage_number} scoped handoff is stale or open-ended"
            )
    return handoff, supplemental, scoped


def _verify_attempt_evidence(
    manifest: Mapping[str, Any],
    repository_root: Path,
    *,
    verify_artifact_contents: bool = True,
) -> None:
    """Re-hash every immutable attempt record and its artifact bindings."""

    for number in STAGE_NUMBERS:
        stage = manifest["stages"][str(number)]
        for attempt in stage["attempts"]:
            handoff, supplemental, scoped = _verify_attempt_control_bindings(
                manifest, number, attempt, repository_root
            )
            all_handoff_inputs = [*handoff["input_artifacts"]]
            for scoped_payload in scoped:
                all_handoff_inputs.extend(scoped_payload["input_artifacts"])
            normalized_handoff_inputs = (
                _normalize_historical_artifacts(
                    manifest, repository_root, all_handoff_inputs
                )
                if verify_artifact_contents
                else sorted(
                    (dict(value) for value in all_handoff_inputs),
                    key=lambda item: (
                        item["path"],
                        item["role"],
                        str(item.get("consumer", "")),
                    ),
                )
            )
            if normalized_handoff_inputs != attempt["input_artifacts"]:
                raise ReceiptValidationError(
                    f"Stage {number} handoff no longer matches its attempt inputs"
                )
            for supplemental_payload in supplemental:
                expected_keys = {
                    "schema_version",
                    "specimen_id",
                    "stage_number",
                    "attempt",
                    "run_token",
                    "created_at",
                    "config_sha256",
                    "registration_freeze_sha256",
                    "frozen_artifacts",
                    "input_artifacts",
                    "purpose",
                    "canonical_handoff_sha256",
                }
                if (
                    set(supplemental_payload) != expected_keys
                    or _validate_timestamp_text(
                        supplemental_payload.get("created_at")
                    )
                    != supplemental_payload.get("created_at")
                    or not isinstance(
                        supplemental_payload.get("input_artifacts"), list
                    )
                    or len(supplemental_payload["input_artifacts"]) != 1
                    or not isinstance(
                        supplemental_payload["input_artifacts"][0], Mapping
                    )
                ):
                    raise ReceiptValidationError(
                        "Post-freeze handoff is open-ended or schema-incompatible"
                    )
                expected_supplemental = {
                    "schema_version": POST_FREEZE_HANDOFF_SCHEMA_VERSION,
                    "specimen_id": manifest["specimen_id"],
                    "stage_number": 2,
                    "attempt": attempt["attempt"],
                    "run_token": attempt["run_token"],
                    "config_sha256": manifest["config"]["sha256"],
                    "purpose": "optional_post_fit_validation_only",
                }
                stale_supplemental = [
                    key
                    for key, value in expected_supplemental.items()
                    if supplemental_payload.get(key) != value
                ]
                if stale_supplemental:
                    raise ReceiptValidationError(
                        "Post-freeze handoff is stale or misbound: "
                        + ", ".join(stale_supplemental)
                    )
                matching_events = [
                    event
                    for event in manifest.get("events", [])
                    if event.get("action")
                    == "post_freeze_aligned_input_authorized"
                    and event.get("stage_number") == 2
                    and event.get("timestamp")
                    == supplemental_payload["created_at"]
                    and event.get("details")
                    == {
                        "aligned_graph_sha256": supplemental_payload[
                            "input_artifacts"
                        ][0]["sha256"],
                        "freeze_sha256": supplemental_payload[
                            "registration_freeze_sha256"
                        ],
                    }
                ]
                if len(matching_events) != 1:
                    raise ReceiptValidationError(
                        "Post-freeze handoff timestamp is not bound to its event"
                    )
                authorized = _normalize_artifacts(
                    repository_root, supplemental_payload["input_artifacts"]
                )
                if (
                    authorized != supplemental_payload["input_artifacts"]
                    or authorized[0].get("role")
                    != "autonomous_validation_reference"
                    or authorized[0].get("consumer") != "data_prep"
                    or authorized[0].get("phase")
                    != "autonomous_v2_post_freeze_validation"
                ):
                    raise AccessPolicyError(
                        "Post-freeze handoff must contain exactly the authorized aligned reference"
                    )
                stage_two_contract = _load_stage_contract(
                    manifest, 2, repository_root
                )
                _validate_artifact_allowlist(
                    manifest,
                    stage_two_contract,
                    authorized,
                    direction="input",
                    require_all=False,
                )
                _enforce_sensitive_access(manifest, 2, authorized)
            if attempt.get("registration_freeze") is not None:
                freeze = _verify_registration_freeze(
                    manifest, attempt, repository_root
                )
                if any(
                    payload.get("registration_freeze_sha256")
                    != freeze["canonical_freeze_sha256"]
                    or payload.get("frozen_artifacts") != freeze["frozen_artifacts"]
                    for payload in supplemental
                ):
                    raise ReceiptValidationError(
                        "Post-freeze handoff is not bound to the accepted CT-only freeze"
                    )
            elif supplemental:
                raise ReceiptValidationError(
                    "Post-freeze handoff exists without a registration freeze"
                )
            resolution = attempt.get("manual_review_resolution")
            if resolution is not None:
                _normalize_artifact(repository_root, resolution)
            record = attempt.get("receipt")
            if record is None:
                if attempt.get("state") != "running":
                    raise ReceiptValidationError(
                        f"Completed Stage {number} attempt has no receipt"
                    )
                continue
            expected_receipt_path = _receipt_output_path(
                manifest, number, attempt["attempt"]
            ).as_posix()
            if (
                not isinstance(record, Mapping)
                or set(record) != {"path", "sha256", "canonical_sha256"}
                or record.get("path") != expected_receipt_path
            ):
                raise ReceiptValidationError(
                    f"Stage {number} receipt record path is invalid"
                )
            receipt = _verify_hashed_json_record(
                record,
                repository_root,
                canonical_field="canonical_receipt_sha256",
            )
            _validate_stage_receipt_document(
                receipt,
                _load_stage_contract(manifest, number, repository_root),
            )
            expected_receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "contract_version": stage["contract"]["version"],
                "contract_sha256": stage["contract"]["sha256"],
                "specimen_id": manifest["specimen_id"],
                "stage_number": number,
                "stage": stage["name"],
                "owner": stage["owner"],
                "attempt": attempt["attempt"],
                "run_token": attempt["run_token"],
                "config_sha256": manifest["config"]["sha256"],
                "registration_mode": manifest["registration_mode"],
                "predecessor_receipt_sha256": attempt[
                    "predecessor_receipt_sha256"
                ],
                "input_handoff": attempt["handoff"],
                "scoped_handoffs": attempt.get("scoped_handoffs", []),
                "supplemental_handoffs": attempt["supplemental_handoffs"],
                "registration_freeze": attempt["registration_freeze"],
                "terminal_state": attempt["reported_terminal_state"],
                "completed_at": attempt["completed_at"],
            }
            stale_receipt = [
                key for key, value in expected_receipt.items() if receipt.get(key) != value
            ]
            if stale_receipt:
                raise ReceiptValidationError(
                    f"Stage {number} receipt is stale or misbound: "
                    + ", ".join(stale_receipt)
                )
            outputs = (
                _normalize_historical_artifacts(
                    manifest,
                    repository_root,
                    receipt.get("output_artifacts", []),
                )
                if verify_artifact_contents
                else receipt.get("output_artifacts", [])
            )
            if outputs != attempt.get("output_artifacts"):
                raise ReceiptValidationError(
                    f"Stage {number} receipt outputs differ from attempt history"
                )
            if number == 2 and manifest["registration_mode"] == "autonomous_v2" and attempt.get("effective_terminal_state") == "pass":
                _verify_registration_freeze(
                    manifest,
                    attempt,
                    repository_root,
                    completion_outputs=outputs,
                )
            if number == 4 and attempt.get("effective_terminal_state") == "pass":
                _verify_missing_calibration_attestation(
                    outputs,
                    repository_root,
                    manifest=manifest,
                    attempt=attempt,
                )
                _verify_stage4_verifier(
                    outputs,
                    repository_root,
                    manifest=manifest,
                    attempt=attempt,
                )
            if number == 5 and attempt.get("effective_terminal_state") == "pass":
                _verify_stage5_evaluation_result(
                    outputs,
                    repository_root,
                    attempt=attempt,
                )

        completion = stage.get("completion_receipt")
        if completion is None:
            continue
        if not stage["attempts"]:
            expected_dependency_path = (
                Path("analysis")
                / manifest["specimen_id"]
                / "receipts"
                / f"stage_{number}_{stage['name']}_dependency_halt.json"
            ).as_posix()
            if (
                not isinstance(completion, Mapping)
                or set(completion) != {"path", "sha256", "canonical_sha256"}
                or completion.get("path") != expected_dependency_path
            ):
                raise ReceiptValidationError(
                    f"Stage {number} dependency receipt path is invalid"
                )
            receipt = _verify_hashed_json_record(
                completion,
                repository_root,
                canonical_field="canonical_receipt_sha256",
            )
            _validate_stage_receipt_document(
                receipt,
                _load_stage_contract(manifest, number, repository_root),
                dependency_halt=True,
            )
            expected_dependency = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "contract_version": stage["contract"]["version"],
                "contract_sha256": stage["contract"]["sha256"],
                "specimen_id": manifest["specimen_id"],
                "stage_number": number,
                "stage": stage["name"],
                "owner": stage["owner"],
                "attempt": 0,
                "run_token": None,
                "terminal_state": "halt",
                "failure_kind": "dependency",
                "completed_at": stage["completed_at"],
                "config_sha256": manifest["config"]["sha256"],
                "registration_mode": manifest["registration_mode"],
                "predecessor_receipt_sha256": manifest[
                    "predecessor_receipt_sha256"
                ],
                "input_handoff": None,
                "scoped_handoffs": [],
                "supplemental_handoffs": [],
                "registration_freeze": None,
                "output_artifacts": [],
                "assertions": {},
            }
            stale_dependency = [
                key
                for key, value in expected_dependency.items()
                if receipt.get(key) != value
            ]
            matching_events = [
                event
                for event in manifest.get("events", [])
                if event.get("action") == "dependency_halt"
                and event.get("stage_number") == number
                and event.get("timestamp") == receipt["completed_at"]
                and event.get("details") == receipt["error"]
            ]
            if (
                stale_dependency
                or stage.get("state") != "halt"
                or stage.get("control_halt") != receipt.get("error")
                or len(matching_events) != 1
            ):
                raise ReceiptValidationError(
                    f"Stage {number} dependency halt receipt is stale or misbound"
                )
        else:
            completed_receipts = [
                attempt.get("receipt")
                for attempt in stage["attempts"]
                if attempt.get("receipt") is not None
            ]
            expected_completion = (
                stage["attempts"][-1].get("receipt")
                if stage["state"] in TERMINAL_STATES
                else (completed_receipts[-1] if completed_receipts else None)
            )
            if completion != expected_completion:
                raise ReceiptValidationError(
                    f"Stage {number} completion receipt is not the current attempt evidence"
                )


def _verify_control_record_files(
    manifest: Mapping[str, Any], repository_root: Path
) -> None:
    """Re-hash control JSON without requiring superseded artifact contents."""

    for stage in manifest["stages"].values():
        for attempt in stage["attempts"]:
            _verify_hashed_json_record(
                attempt["handoff"],
                repository_root,
                canonical_field="canonical_handoff_sha256",
            )
            for record in [
                *attempt.get("scoped_handoffs", []),
                *attempt.get("supplemental_handoffs", []),
            ]:
                _verify_hashed_json_record(
                    record,
                    repository_root,
                    canonical_field="canonical_handoff_sha256",
                )
            if attempt.get("registration_freeze") is not None:
                _verify_hashed_json_record(
                    attempt["registration_freeze"],
                    repository_root,
                    canonical_field="canonical_freeze_sha256",
                )
            if attempt.get("receipt") is not None:
                _verify_hashed_json_record(
                    attempt["receipt"],
                    repository_root,
                    canonical_field="canonical_receipt_sha256",
                )
        if not stage["attempts"] and stage.get("completion_receipt") is not None:
            _verify_hashed_json_record(
                stage["completion_receipt"],
                repository_root,
                canonical_field="canonical_receipt_sha256",
            )


def _load_manifest_unlocked(
    manifest_path: Path,
    repository_root: Path,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        raise ManifestValidationError("Unsupported pipeline manifest schema")
    if manifest.get("registration_mode") not in REGISTRATION_MODES:
        raise ManifestValidationError("Pipeline registration_mode is invalid")
    specimen_id = manifest.get("specimen_id")
    if not isinstance(specimen_id, str):
        raise ManifestValidationError("Pipeline manifest specimen_id is invalid")
    _validate_specimen_id(specimen_id)
    expected_manifest = (
        repository_root / "analysis" / specimen_id / "manifest.json"
    ).resolve()
    if manifest_path.resolve() != expected_manifest:
        raise ManifestValidationError(
            "Pipeline state must be stored at analysis/<specimen_id>/manifest.json"
        )
    supplied_hash = manifest.get("manifest_sha256")
    expected_hash = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if supplied_hash != expected_hash:
        raise ManifestValidationError("Pipeline manifest canonical hash is invalid")
    if manifest.get("stage_order") != list(STAGE_NUMBERS):
        raise ManifestValidationError("Pipeline stage order is not exactly 0 through 6")
    if set(manifest.get("stages", {})) != {str(value) for value in STAGE_NUMBERS}:
        raise ManifestValidationError("Pipeline stage records are incomplete")
    config = manifest.get("config", {})
    config_resolved, _ = _relative_existing_file(
        repository_root, config.get("path", ""), reject_alias=True
    )
    if sha256_file(config_resolved) != config.get("sha256"):
        raise ManifestValidationError("Frozen config SHA-256 no longer matches")
    for number in STAGE_NUMBERS:
        stage = manifest["stages"][str(number)]
        if stage.get("state") not in CONTROL_STATES:
            raise ManifestValidationError(f"Stage {number} has an invalid state")
        contract = _load_stage_contract(manifest, number, repository_root)
        if contract.get("stage") != stage.get("name"):
            raise ManifestValidationError(f"Stage {number} contract name mismatch")
        if (
            contract.get("owner") != stage.get("owner")
            or contract.get("execution_kind") != stage.get("execution_kind")
            or contract.get("maximum_attempts") != stage.get("maximum_attempts")
            or bool(contract.get("one_shot", False)) != stage.get("one_shot")
        ):
            raise ManifestValidationError(
                f"Stage {number} frozen contract metadata is inconsistent"
            )
    _validate_manifest_state_topology(manifest)
    _verify_artifact_catalog(manifest, repository_root)
    if verify_artifacts:
        _verify_indexed_artifacts(manifest, repository_root)
        _verify_attempt_evidence(manifest, repository_root)
    return manifest


def validate_pipeline_manifest(
    manifest_path: RepositoryPath,
    *,
    repository_root: RepositoryPath,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    with _manifest_lock(path):
        return _load_manifest_unlocked(path, root, verify_artifacts=verify_artifacts)


def _capability_value(entry: Any, version_key: str) -> tuple[bool, str | None]:
    if isinstance(entry, dict) and isinstance(entry.get("available"), bool):
        version = entry.get(version_key)
        return entry["available"], version if isinstance(version, str) else None
    return False, None


def _dependency_failures(
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    if not isinstance(inventory, Mapping):
        return [
            {
                "kind": "inventory",
                "name": "capability_inventory",
                "reason": "missing",
            }
        ]
    dependencies = contract.get("required_dependencies", {})
    agents = inventory.get("agents", {})
    if not isinstance(agents, Mapping):
        agents = {}
    for required in dependencies.get("agents", []):
        name = required["name"]
        available, version = _capability_value(
            agents.get(name), "contract_version"
        )
        expected = required["contract_version"]
        if not available:
            failures.append({"kind": "agent", "name": name, "reason": "missing"})
        elif version != expected:
            failures.append(
                {
                    "kind": "agent",
                    "name": name,
                    "reason": f"schema_incompatible:{version!r}!={expected!r}",
                }
            )

    servers = inventory.get("mcp_servers", {})
    if not isinstance(servers, Mapping):
        servers = {}
    for required in dependencies.get("mcp_tools", []):
        server_name = required.get("server", "segmentation-tools")
        name = required["name"]
        expected = required["response_schema_version"]
        server = servers.get(server_name)
        if not isinstance(server, Mapping) or server.get("healthy") is not True:
            failures.append(
                {
                    "kind": "mcp_server",
                    "name": server_name,
                    "reason": "missing_or_unhealthy",
                }
            )
            continue
        tools = server.get("tools", {})
        entry: Any = tools.get(name) if isinstance(tools, Mapping) else None
        available, version = _capability_value(entry, "response_schema_version")
        if not available:
            failures.append(
                {
                    "kind": "mcp_tool",
                    "name": f"{server_name}.{name}",
                    "reason": "missing",
                }
            )
        elif version != expected:
            failures.append(
                {
                    "kind": "mcp_tool",
                    "name": f"{server_name}.{name}",
                    "reason": f"schema_incompatible:{version!r}!={expected!r}",
                }
            )
    return failures


def _record_dependency_halt(
    manifest_path: Path,
    manifest: dict[str, Any],
    contract: Mapping[str, Any],
    failures: Sequence[Mapping[str, str]],
    *,
    timestamp: str,
    repository_root: Path,
) -> dict[str, Any]:
    stage_number = int(contract["stage_number"])
    stage = manifest["stages"][str(stage_number)]
    receipt_base = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_version": contract["schema_version"],
        "contract_sha256": stage["contract"]["sha256"],
        "specimen_id": manifest["specimen_id"],
        "stage_number": stage_number,
        "stage": stage["name"],
        "owner": stage["owner"],
        "attempt": 0,
        "run_token": None,
        "terminal_state": "halt",
        "failure_kind": "dependency",
        "completed_at": timestamp,
        "config_sha256": manifest["config"]["sha256"],
        "registration_mode": manifest["registration_mode"],
        "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
        "input_handoff": None,
        "scoped_handoffs": [],
        "supplemental_handoffs": [],
        "registration_freeze": None,
        "output_artifacts": [],
        "assertions": {},
        "error": {
            "code": "missing_or_incompatible_dependency",
            "failures": [dict(value) for value in failures],
            "fallback_used": False,
        },
    }
    receipt = _with_self_hash(receipt_base, "canonical_receipt_sha256")
    relative = (
        Path("analysis")
        / manifest["specimen_id"]
        / "receipts"
        / f"stage_{stage_number}_{stage['name']}_dependency_halt.json"
    )
    receipt_path = repository_root / relative
    if receipt_path.is_file():
        existing = _read_object(receipt_path)
        if existing != receipt:
            existing_base = {
                key: value
                for key, value in existing.items()
                if key != "canonical_receipt_sha256"
            }
            comparable_existing = {
                key: value
                for key, value in existing_base.items()
                if key != "completed_at"
            }
            comparable_requested = {
                key: value
                for key, value in receipt_base.items()
                if key != "completed_at"
            }
            if (
                existing.get("canonical_receipt_sha256")
                != canonical_json_sha256(existing_base)
                or comparable_existing != comparable_requested
            ):
                raise ArtifactVerificationError(
                    "Dependency halt receipt path contains incompatible evidence"
                )
            receipt = existing
            timestamp = str(existing["completed_at"])
    else:
        _atomic_write_if_changed(receipt_path, receipt)
    receipt_record = {
        "path": relative.as_posix(),
        "sha256": sha256_file(receipt_path),
        "canonical_sha256": receipt["canonical_receipt_sha256"],
    }
    stage["state"] = "halt"
    stage["completed_at"] = timestamp
    stage["completion_receipt"] = receipt_record
    stage["control_halt"] = receipt["error"]
    manifest["pipeline_state"] = "halt"
    manifest["updated_at"] = timestamp
    manifest["revision"] += 1
    _event(
        manifest,
        timestamp=timestamp,
        action="dependency_halt",
        stage_number=stage_number,
        details=receipt["error"],
    )
    _write_manifest(manifest_path, manifest)
    return {
        "changed": True,
        "state": "halt",
        "receipt": receipt_record,
        "error": receipt["error"],
    }


def _sensitive_kind_for_artifact(
    contract: Mapping[str, Any], artifact: Mapping[str, Any]
) -> str | None:
    explicit = artifact.get("sensitivity")
    if explicit in SENSITIVE_KINDS:
        return str(explicit)
    sensitive_roles = _contract_artifact_policy(contract, "output").get(
        "sensitive_roles", {}
    )
    if isinstance(sensitive_roles, Mapping):
        for role_pattern, kind in sensitive_roles.items():
            if fnmatch.fnmatchcase(str(artifact["role"]), str(role_pattern)):
                if isinstance(kind, str):
                    if kind not in SENSITIVE_KINDS:
                        raise ManifestValidationError(
                            f"Contract declares unsupported sensitivity {kind!r}"
                        )
                    return kind
                # Rich authorization metadata is permitted in contracts. The
                # role still determines the control-plane sensitivity below.
                break
    role = str(artifact["role"])
    if role in {
        "challenge_aligned_json",
        "challenge_aligned_graph",
        "aligned_graph_reference",
        "autonomous_validation_reference",
    }:
        return "aligned_graph"
    if role in {"development_labels", "dev_split"}:
        return "development_labels"
    if role in {"sealed_labels", "sealed_split"}:
        return "sealed_labels"
    if role in {
        "defect_labels",
        "intentional_deletion_labels",
        "prior_defect_labels",
    } or role.startswith("intentional_deletions"):
        return "defect_labels"
    return None


def _record_sensitive_hashes(
    manifest: dict[str, Any],
    contract: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    for artifact in artifacts:
        kind = _sensitive_kind_for_artifact(contract, artifact)
        if kind is None:
            continue
        hashes = manifest["sensitive_artifact_hashes"][kind]
        if artifact["sha256"] not in hashes:
            hashes.append(artifact["sha256"])
            hashes.sort()


def _enforce_sensitive_access(
    manifest: Mapping[str, Any],
    stage_number: int,
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    known = manifest["sensitive_artifact_hashes"]
    for artifact in artifacts:
        digest = artifact["sha256"]
        role = str(artifact["role"])
        consumer = artifact.get("consumer")
        phase = artifact.get("phase", "input")
        kinds = {
            kind
            for kind, hashes in known.items()
            if digest in set(hashes)
        }
        if role in {"development_labels", "dev_split"}:
            kinds.add("development_labels")
        if role in {"sealed_labels", "sealed_split"}:
            kinds.add("sealed_labels")
        if role in {
            "defect_labels",
            "intentional_deletion_labels",
            "prior_defect_labels",
        } or role.startswith("intentional_deletions"):
            kinds.add("defect_labels")
        if role in {
            "challenge_aligned_json",
            "challenge_aligned_graph",
            "aligned_graph_reference",
            "autonomous_validation_reference",
        }:
            kinds.add("aligned_graph")
        if "development_labels" in kinds and not (
            stage_number == 4 and consumer == "missing_strut_agent"
        ):
            raise AccessPolicyError(
                "Development labels are readable only by missing_strut_agent in Stage 4"
            )
        if "sealed_labels" in kinds and not (
            stage_number == 5 and consumer == "eval_agent"
        ):
            raise AccessPolicyError(
                "The sealed split is readable only by eval_agent in Stage 5"
            )
        if "defect_labels" in kinds and stage_number in {0, 2, 3, 4, 6}:
            raise AccessPolicyError(
                f"Stage {stage_number} may not consume defect-label artifacts"
            )
        if "aligned_graph" in kinds and manifest["registration_mode"] == "autonomous_v2":
            if stage_number == 0:
                raise AccessPolicyError(
                    "Autonomous-v2 intake must not receive an aligned JSON"
                )
            if stage_number == 2 and phase not in {
                "post_freeze_validation",
                "autonomous_v2_post_freeze_validation",
            }:
                raise AccessPolicyError(
                    "Autonomous-v2 aligned JSON access requires a frozen CT-only fit"
                )


def _enforce_artifact_lineage(
    manifest: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> None:
    active_by_path = {
        record["path"]: record for record in _active_artifact_records(manifest)
    }
    for artifact in artifacts:
        prior = active_by_path.get(artifact["path"])
        if prior is None:
            continue
        if artifact["sha256"] != prior["sha256"]:
            raise ArtifactVerificationError(
                f"Input artifact has stale lineage at {artifact['path']}"
            )
        if artifact["role"] != prior["role"]:
            raise AccessPolicyError(
                f"Input role {artifact['role']!r} does not match frozen role "
                f"{prior['role']!r} for {artifact['path']}"
            )


def _normalize_artifacts(
    repository_root: Path,
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [_normalize_artifact(repository_root, value) for value in values]
    identities = [(item["path"], item["role"], item.get("consumer")) for item in normalized]
    if len(identities) != len(set(identities)):
        raise ArtifactVerificationError("Duplicate artifact records are forbidden")
    paths = [item["path"] for item in normalized]
    if len(paths) != len(set(paths)):
        raise ArtifactVerificationError(
            "One handoff or receipt may not assign multiple roles to the same path"
        )
    return sorted(
        normalized,
        key=lambda item: (
            item["path"], item["role"], str(item.get("consumer", ""))
        ),
    )


def _normalize_historical_artifact(
    manifest: Mapping[str, Any],
    repository_root: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    superseded = next(
        (
            record
            for record in manifest.get("artifact_index", [])
            if record.get("state") == "superseded"
            and record.get("path") == value.get("path")
            and record.get("sha256") == value.get("sha256")
            and record.get("role") == value.get("role")
        ),
        None,
    )
    if superseded is None:
        return _normalize_artifact(repository_root, value)
    active = next(
        (
            record
            for record in _active_artifact_records(manifest)
            if record.get("path") == superseded["path"]
            and record.get("sha256") == superseded.get("superseded_by_sha256")
        ),
        None,
    )
    if active is None:
        raise ArtifactVerificationError(
            f"Superseded artifact has no active replacement: {superseded['path']}"
        )
    _, relative = _relative_existing_file(
        repository_root, value["path"], reject_alias=True
    )
    if relative != value["path"]:
        raise ArtifactVerificationError("Historical artifact path is non-canonical")
    result = {
        "path": relative,
        "sha256": value["sha256"],
        "role": value["role"],
        "phase": str(value.get("phase", "input")),
    }
    for key in ("consumer", "producer", "sensitivity", "replaces_sha256"):
        if key in value:
            result[key] = value[key]
    return result


def _normalize_historical_artifacts(
    manifest: Mapping[str, Any],
    repository_root: Path,
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [
        _normalize_historical_artifact(manifest, repository_root, value)
        for value in values
    ]
    paths = [item["path"] for item in normalized]
    if len(paths) != len(set(paths)):
        raise ArtifactVerificationError("Historical artifact paths are duplicated")
    return sorted(
        normalized,
        key=lambda item: (
            item["path"], item["role"], str(item.get("consumer", ""))
        ),
    )


def _default_handoff_path(
    manifest: Mapping[str, Any], stage_number: int, attempt: int
) -> Path:
    stage_name = manifest["stages"][str(stage_number)]["name"]
    return (
        Path("analysis")
        / manifest["specimen_id"]
        / "handoffs"
        / f"stage_{stage_number}_{stage_name}_attempt_{attempt}.json"
    )


def _partition_scoped_inputs(
    contract: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    """Remove contract-declared secrets from the shared stage handoff."""

    specifications = contract.get("scoped_handoffs", [])
    if not isinstance(specifications, list):
        raise ManifestValidationError("scoped_handoffs must be an array")
    claimed: set[int] = set()
    scopes: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for raw_specification in specifications:
        if not isinstance(raw_specification, dict):
            raise ManifestValidationError("Each scoped_handoff must be an object")
        specification = dict(raw_specification)
        consumer = specification.get("consumer")
        path = specification.get("path")
        required_roles = specification.get("required_roles", [])
        forbidden_roles = specification.get("forbidden_roles", [])
        if (
            not isinstance(consumer, str)
            or not consumer
            or not isinstance(path, str)
            or not path
            or not isinstance(required_roles, list)
            or not all(isinstance(role, str) and role for role in required_roles)
            or not isinstance(forbidden_roles, list)
        ):
            raise ManifestValidationError("scoped_handoff schema is incompatible")
        selected: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            if artifact["role"] in forbidden_roles:
                raise AccessPolicyError(
                    f"Scoped handoff for {consumer} forbids {artifact['role']}"
                )
            if artifact["role"] in required_roles:
                if artifact.get("consumer") != consumer:
                    raise AccessPolicyError(
                        f"{artifact['role']} must be scoped only to {consumer}"
                    )
                selected.append(dict(artifact))
                claimed.add(index)
        missing = sorted(set(required_roles) - {item["role"] for item in selected})
        if missing:
            raise AccessPolicyError(
                f"Scoped handoff for {consumer} is missing: " + ", ".join(missing)
            )
        scopes.append((specification, selected))
    shared = [dict(artifact) for index, artifact in enumerate(artifacts) if index not in claimed]
    return shared, scopes


def _scoped_handoff_path(
    specification: Mapping[str, Any], manifest: Mapping[str, Any], attempt: int
) -> Path:
    rendered = (
        str(specification["path"])
        .replace("<specimen_id>", str(manifest["specimen_id"]))
        .replace("<attempt>", str(attempt))
    )
    if re.search(r"<[^>]+>", rendered):
        raise ManifestValidationError(
            f"Unresolved scoped handoff path placeholder: {rendered}"
        )
    return Path(rendered)


def start_stage(
    manifest_path: RepositoryPath,
    stage_number: int,
    *,
    input_artifacts: Sequence[Mapping[str, Any]],
    capability_inventory: Mapping[str, Any] | None,
    repository_root: RepositoryPath,
    handoff_path: RepositoryPath | None = None,
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Atomically reserve and start only the currently unlocked stage."""

    if stage_number not in STAGE_NUMBERS:
        raise IllegalTransitionError(f"Unknown stage number: {stage_number}")
    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    started_at = _timestamp(timestamp, clock)
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=True)
        stage = manifest["stages"][str(stage_number)]
        contract = _load_stage_contract(manifest, stage_number, root)
        is_running = stage["state"] == "running"
        if not is_running and manifest["pipeline_state"] in {
            "manual_review",
            "halt",
            "pass",
        }:
            raise IllegalTransitionError(
                f"Pipeline state {manifest['pipeline_state']} blocks stage starts"
            )
        if not is_running:
            if manifest["current_stage"] != stage_number or stage["state"] != "ready":
                raise IllegalTransitionError(
                    f"Stage {stage_number} is not the uniquely unlocked ready stage"
                )
            for prior in range(stage_number):
                if manifest["stages"][str(prior)]["state"] != "pass":
                    raise IllegalTransitionError(
                        f"Stage {stage_number} cannot skip non-pass Stage {prior}"
                    )
            failures = _dependency_failures(contract, capability_inventory)
            if failures:
                return _record_dependency_halt(
                    path,
                    manifest,
                    contract,
                    failures,
                    timestamp=started_at,
                    repository_root=root,
                )
        attempt_number = stage["attempt_count"] + 1
        if not is_running and attempt_number > stage["maximum_attempts"]:
            raise IllegalTransitionError(
                f"Stage {stage_number} exhausted its attempt limit"
            )
        if (
            not is_running
            and stage["one_shot"]
            and manifest["sealed_evaluation"]["consumed"]
        ):
            raise IllegalTransitionError("Stage 5 sealed evaluation is already consumed")

        normalized_inputs = _normalize_artifacts(root, input_artifacts)
        _enforce_sensitive_access(manifest, stage_number, normalized_inputs)
        _validate_artifact_allowlist(
            manifest,
            contract,
            normalized_inputs,
            direction="input",
            require_all=True,
        )
        _enforce_artifact_lineage(manifest, normalized_inputs)
        _record_sensitive_hashes(manifest, contract, normalized_inputs)
        if (
            manifest["registration_mode"] == "autonomous_v2"
            and stage_number == 2
            and any(
                _sensitive_kind_for_artifact(contract, artifact) == "aligned_graph"
                for artifact in normalized_inputs
            )
        ):
            raise AccessPolicyError(
                "Autonomous-v2 Stage 2 start cannot consume aligned JSON before "
                "a frozen CT-only fit; use the post-freeze authorization handoff"
            )
        if is_running:
            if stage["one_shot"]:
                raise IllegalTransitionError(
                    "Stage 5 is already reserved; monitor the existing attempt instead of restarting it"
                )
            attempt = stage["attempts"][-1]
            if attempt.get("input_artifacts") != normalized_inputs:
                raise IllegalTransitionError(
                    "A running stage cannot be restarted with different inputs"
                )
            return {
                "changed": False,
                "state": "running",
                "run_token": attempt["run_token"],
                "handoff": attempt["handoff"],
                "scoped_handoffs": attempt.get("scoped_handoffs", []),
            }

        run_token = canonical_json_sha256(
            {
                "specimen_id": manifest["specimen_id"],
                "stage_number": stage_number,
                "attempt": attempt_number,
                "started_at": started_at,
                "config_sha256": manifest["config"]["sha256"],
                "contract_sha256": stage["contract"]["sha256"],
                "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
                "input_artifacts": normalized_inputs,
            }
        )
        shared_inputs, input_scopes = _partition_scoped_inputs(
            contract, normalized_inputs
        )
        handoff_base = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "specimen_id": manifest["specimen_id"],
            "stage_number": stage_number,
            "stage": stage["name"],
            "owner": stage["owner"],
            "attempt": attempt_number,
            "run_token": run_token,
            "created_at": started_at,
            "registration_mode": manifest["registration_mode"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_version": stage["contract"]["version"],
            "contract_sha256": stage["contract"]["sha256"],
            "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
            "input_artifacts": shared_inputs,
            "forbidden_operations": contract.get("forbidden_operations", []),
        }
        handoff = _with_self_hash(handoff_base, "canonical_handoff_sha256")
        relative, handoff_destination = _repository_output_location(
            root,
            handoff_path,
            _default_handoff_path(manifest, stage_number, attempt_number),
            required_directory=Path("analysis")
            / manifest["specimen_id"]
            / "handoffs",
            required_name=(
                f"stage_{stage_number}_{stage['name']}_attempt_{attempt_number}.json"
            ),
        )
        if handoff_destination.is_file():
            existing = _read_object(handoff_destination)
            if existing != handoff:
                orphan_started_at = existing.get("created_at")
                orphan_run_token = existing.get("run_token")
                expected_orphan_token = canonical_json_sha256(
                    {
                        "specimen_id": manifest["specimen_id"],
                        "stage_number": stage_number,
                        "attempt": attempt_number,
                        "started_at": orphan_started_at,
                        "config_sha256": manifest["config"]["sha256"],
                        "contract_sha256": stage["contract"]["sha256"],
                        "predecessor_receipt_sha256": manifest[
                            "predecessor_receipt_sha256"
                        ],
                        "input_artifacts": normalized_inputs,
                    }
                )
                orphan_base = {
                    **handoff_base,
                    "created_at": orphan_started_at,
                    "run_token": orphan_run_token,
                }
                expected_orphan = _with_self_hash(
                    orphan_base, "canonical_handoff_sha256"
                )
                if (
                    not isinstance(orphan_started_at, str)
                    or orphan_run_token != expected_orphan_token
                    or existing != expected_orphan
                ):
                    raise ArtifactVerificationError(
                        "Immutable handoff path contains an incompatible orphan: "
                        f"{relative}"
                    )
                started_at = orphan_started_at
                run_token = orphan_run_token
                handoff_base = orphan_base
                handoff = existing
        else:
            _atomic_write_if_changed(handoff_destination, handoff)
        handoff_record = {
            "path": relative.as_posix(),
            "sha256": sha256_file(handoff_destination),
            "canonical_sha256": handoff["canonical_handoff_sha256"],
        }
        scoped_handoff_records: list[dict[str, Any]] = []
        for specification, scoped_inputs in input_scopes:
            scoped_base = {
                key: copy.deepcopy(value)
                for key, value in handoff_base.items()
                if key not in {"owner", "input_artifacts"}
            }
            scoped_base.update(
                {
                    "owner": specification["consumer"],
                    "scope": specification["consumer"],
                    "parent_handoff": handoff_record,
                    "input_artifacts": scoped_inputs,
                }
            )
            scoped_handoff = _with_self_hash(
                scoped_base, "canonical_handoff_sha256"
            )
            scoped_default = _scoped_handoff_path(
                specification, manifest, attempt_number
            )
            scoped_relative, scoped_destination = _repository_output_location(
                root,
                None,
                scoped_default,
                required_directory=Path("analysis")
                / manifest["specimen_id"]
                / "handoffs",
                required_name=scoped_default.name,
            )
            if scoped_destination.is_file():
                if _read_object(scoped_destination) != scoped_handoff:
                    raise ArtifactVerificationError(
                        "Immutable scoped handoff already exists with different content: "
                        f"{scoped_relative.as_posix()}"
                    )
            else:
                _atomic_write_if_changed(scoped_destination, scoped_handoff)
            scoped_handoff_records.append(
                {
                    "consumer": specification["consumer"],
                    "path": scoped_relative.as_posix(),
                    "sha256": sha256_file(scoped_destination),
                    "canonical_sha256": scoped_handoff[
                        "canonical_handoff_sha256"
                    ],
                }
            )
        attempt = {
            "attempt": attempt_number,
            "run_token": run_token,
            "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
            "state": "running",
            "started_at": started_at,
            "completed_at": None,
            "handoff": handoff_record,
            "scoped_handoffs": scoped_handoff_records,
            "supplemental_handoffs": [],
            "registration_freeze": None,
            "receipt": None,
            "reported_terminal_state": None,
            "effective_terminal_state": None,
            "failure_kind": None,
            "input_artifacts": normalized_inputs,
            "output_artifacts": [],
        }
        stage["attempts"].append(attempt)
        stage["attempt_count"] = attempt_number
        stage["current_attempt"] = attempt_number
        stage["state"] = "running"
        manifest["pipeline_state"] = "running"
        if stage["one_shot"]:
            by_role = {artifact["role"]: artifact for artifact in normalized_inputs}
            manifest["sealed_evaluation"] = {
                "consumed": True,
                "consumed_at": started_at,
                "stage_attempt": attempt_number,
                "run_token": run_token,
                "config_sha256": manifest["config"]["sha256"],
                "classified_struts_sha256": by_role["classified_struts"][
                    "sha256"
                ],
                "sealed_labels_sha256": by_role["sealed_labels"]["sha256"],
            }
        manifest["updated_at"] = started_at
        manifest["revision"] += 1
        _event(
            manifest,
            timestamp=started_at,
            action="stage_started",
            stage_number=stage_number,
            details={"attempt": attempt_number, "run_token": run_token},
        )
        changed = _write_manifest(path, manifest)
        return {
            "changed": changed,
            "state": "running",
            "run_token": run_token,
            "handoff": handoff_record,
            "scoped_handoffs": scoped_handoff_records,
        }


def _verify_hashed_json_record(
    record: Mapping[str, Any],
    repository_root: Path,
    *,
    canonical_field: str,
) -> dict[str, Any]:
    path, relative = _relative_existing_file(
        repository_root, record["path"], reject_alias=True
    )
    if sha256_file(path) != record.get("sha256"):
        raise ReceiptValidationError(f"Raw receipt/handoff hash mismatch: {relative}")
    payload = _read_object(path)
    expected = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != canonical_field}
    )
    if payload.get(canonical_field) != expected:
        raise ReceiptValidationError(f"Canonical hash mismatch: {relative}")
    if record.get("canonical_sha256") != expected:
        raise ReceiptValidationError(f"Recorded canonical hash is stale: {relative}")
    return payload


def _receipt_output_path(
    manifest: Mapping[str, Any], stage_number: int, attempt: int
) -> Path:
    stage = manifest["stages"][str(stage_number)]
    return (
        Path("analysis")
        / manifest["specimen_id"]
        / "receipts"
        / f"stage_{stage_number}_{stage['name']}_attempt_{attempt}.json"
    )


def _validate_manual_review_outputs(
    manifest: Mapping[str, Any],
    stage_number: int,
    attempt_number: int,
    outputs: Sequence[Mapping[str, Any]],
) -> None:
    prefix = (
        f"analysis/{manifest['specimen_id']}/reviews/"
        f"stage_{stage_number}_attempt_{attempt_number}/"
    )
    invalid = [
        artifact["path"]
        for artifact in outputs
        if artifact["role"] != "manual_review_evidence"
        or not artifact["path"].startswith(prefix)
    ]
    if invalid:
        raise ReceiptValidationError(
            "manual_review outputs must be immutable attempt-scoped evidence under "
            + prefix
        )


def _validate_stage_receipt_document(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    dependency_halt: bool = False,
) -> None:
    """Reject open-ended receipts and label-bearing assertion/error payloads."""

    if set(receipt) != RECEIPT_FIELDS:
        raise ReceiptValidationError(
            "Stage receipt has undeclared or missing root fields"
        )
    terminal = receipt.get("terminal_state")
    if terminal not in TERMINAL_STATES:
        raise ReceiptValidationError("Stage receipt terminal_state is invalid")
    if (
        _validate_timestamp_text(receipt.get("completed_at"))
        != receipt.get("completed_at")
        or not isinstance(receipt.get("output_artifacts"), list)
        or not isinstance(receipt.get("scoped_handoffs"), list)
        or not isinstance(receipt.get("supplemental_handoffs"), list)
    ):
        raise ReceiptValidationError("Stage receipt collection fields are invalid")
    allowed_artifact_fields = {
        "path",
        "sha256",
        "role",
        "phase",
        "consumer",
        "producer",
        "sensitivity",
        "replaces_sha256",
    }
    if any(
        not isinstance(artifact, Mapping)
        or not {"path", "sha256", "role", "phase"} <= set(artifact)
        or set(artifact) - allowed_artifact_fields
        for artifact in receipt["output_artifacts"]
    ):
        raise ReceiptValidationError(
            "Stage receipt output artifact records are malformed"
        )
    assertions = receipt.get("assertions")
    if not isinstance(assertions, Mapping) or any(
        type(value) is not bool for value in assertions.values()
    ):
        raise ReceiptValidationError("Stage receipt assertions must be boolean")
    required = set(contract.get("required_receipt_assertions", []))
    allowed = set(required)
    if contract.get("stage_number") == 5:
        allowed.add("optimization_performed")
    if dependency_halt:
        if assertions != {}:
            raise ReceiptValidationError(
                "Dependency halt receipt may not carry assertions"
            )
    elif terminal == "pass":
        if set(assertions) != allowed:
            raise ReceiptValidationError(
                "Pass receipt assertion set is incomplete or open-ended"
            )
        if any(assertions.get(name) is not True for name in required):
            raise ReceiptValidationError("Pass receipt contains a failed assertion")
        if contract.get("stage_number") == 5 and (
            assertions.get("optimization_performed") is not False
        ):
            raise ReceiptValidationError(
                "Stage 5 pass receipt must attest that optimization was not performed"
            )
    elif set(assertions) - allowed:
        raise ReceiptValidationError(
            "Non-pass receipt contains an undeclared assertion"
        )

    failure_kind = receipt.get("failure_kind")
    error = receipt.get("error")
    if terminal == "pass":
        if failure_kind is not None or error is not None:
            raise ReceiptValidationError(
                "Pass receipt may not carry failure or error content"
            )
        return
    if dependency_halt:
        if failure_kind != "dependency":
            raise ReceiptValidationError(
                "Dependency halt receipt failure_kind is invalid"
            )
        if (
            not isinstance(error, Mapping)
            or set(error) != {"code", "failures", "fallback_used"}
            or error.get("code") != "missing_or_incompatible_dependency"
            or error.get("fallback_used") is not False
            or not isinstance(error.get("failures"), list)
            or any(
                not isinstance(failure, Mapping)
                or set(failure) != {"kind", "name", "reason"}
                or not all(
                    isinstance(failure.get(key), str) and failure[key]
                    for key in ("kind", "name", "reason")
                )
                for failure in error.get("failures", [])
            )
        ):
            raise ReceiptValidationError(
                "Dependency halt receipt error schema is invalid"
            )
        return
    if failure_kind is not None and failure_kind not in RECEIPT_FAILURE_KINDS:
        raise ReceiptValidationError("Stage receipt failure_kind is invalid")
    if error is not None and (
        not isinstance(error, Mapping)
        or set(error) != {"code"}
        or error.get("code") not in RECEIPT_ERROR_CODES
    ):
        raise ReceiptValidationError(
            "Stage receipt error must use a bounded aggregate-only code"
        )
    if error is not None and failure_kind is None:
        raise ReceiptValidationError(
            "Stage receipt error requires a declared failure_kind"
        )
    if failure_kind == "deterministic_gate" and terminal != "halt":
        raise ReceiptValidationError(
            "A deterministic gate failure must terminate as halt"
        )


def build_stage_receipt(
    manifest_path: RepositoryPath,
    stage_number: int,
    *,
    terminal_state: Literal["pass", "manual_review", "halt"],
    output_artifacts: Sequence[Mapping[str, Any]],
    assertions: Mapping[str, bool],
    repository_root: RepositoryPath,
    output_path: RepositoryPath | None = None,
    failure_kind: str | None = None,
    error: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Build an immutable attempt-bound receipt for a running stage."""

    if stage_number not in STAGE_NUMBERS:
        raise IllegalTransitionError(f"Unknown stage number: {stage_number}")
    if terminal_state not in TERMINAL_STATES:
        raise ReceiptValidationError(f"Unsupported terminal state: {terminal_state}")
    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    completed_at = _timestamp(timestamp, clock)
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=False)
        stage = manifest["stages"][str(stage_number)]
        if stage["state"] != "running" or not stage["attempts"]:
            raise IllegalTransitionError(f"Stage {stage_number} is not running")
        attempt = stage["attempts"][-1]
        contract = _load_stage_contract(manifest, stage_number, root)
        _verify_attempt_evidence(
            manifest, root, verify_artifact_contents=False
        )
        normalized_outputs = _normalize_artifacts(root, output_artifacts)
        _validate_artifact_allowlist(
            manifest,
            contract,
            normalized_outputs,
            direction="output",
            require_all=terminal_state == "pass",
        )
        if terminal_state == "manual_review":
            _validate_manual_review_outputs(
                manifest,
                stage_number,
                attempt["attempt"],
                normalized_outputs,
            )
        if failure_kind == "deterministic_gate" and terminal_state != "halt":
            raise ReceiptValidationError(
                "A deterministic gate failure must terminate as halt"
            )
        required_assertions = contract.get("required_receipt_assertions", [])
        if terminal_state == "pass":
            failed = [name for name in required_assertions if assertions.get(name) is not True]
            if failed:
                raise ReceiptValidationError(
                    "Required receipt assertions failed: " + ", ".join(failed)
                )
        receipt_base = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "contract_version": stage["contract"]["version"],
            "contract_sha256": stage["contract"]["sha256"],
            "specimen_id": manifest["specimen_id"],
            "stage_number": stage_number,
            "stage": stage["name"],
            "owner": stage["owner"],
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "terminal_state": terminal_state,
            "failure_kind": failure_kind,
            "completed_at": completed_at,
            "config_sha256": manifest["config"]["sha256"],
            "registration_mode": manifest["registration_mode"],
            "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
            "input_handoff": attempt["handoff"],
            "scoped_handoffs": attempt.get("scoped_handoffs", []),
            "supplemental_handoffs": attempt["supplemental_handoffs"],
            "registration_freeze": attempt["registration_freeze"],
            "output_artifacts": normalized_outputs,
            "assertions": dict(assertions),
            "error": dict(error) if error is not None else None,
        }
        receipt = _with_self_hash(receipt_base, "canonical_receipt_sha256")
        _validate_stage_receipt_document(receipt, contract)
        relative, destination = _repository_output_location(
            root,
            output_path,
            _receipt_output_path(manifest, stage_number, attempt["attempt"]),
            required_directory=Path("analysis")
            / manifest["specimen_id"]
            / "receipts",
            required_name=(
                f"stage_{stage_number}_{stage['name']}_attempt_{attempt['attempt']}.json"
            ),
        )
        if destination.is_file():
            existing = _read_object(destination)
            if existing != receipt:
                raise ReceiptValidationError(
                    f"Immutable receipt already exists with different content: {relative}"
                )
            changed = False
        else:
            changed = _atomic_write_if_changed(destination, receipt)
        return {
            "receipt": receipt,
            "path": str(destination),
            "changed": changed,
        }


def _replacement_map(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    replacements: dict[str, dict[str, Any]] = {}
    allowed_rules = _contract_artifact_policy(contract, "output").get("allowed", [])
    active_by_role = {
        record["role"]: record for record in _active_artifact_records(manifest)
    }
    for output in outputs:
        matching = [
            rule
            for rule in allowed_rules
            if _rule_matches(rule, output, manifest) and rule.get("replaces_role")
        ]
        if not matching:
            if "replaces_sha256" in output:
                raise ReceiptValidationError(
                    f"Undeclared artifact replacement for {output['path']}"
                )
            continue
        role = matching[0]["replaces_role"]
        prior = active_by_role.get(role)
        if prior is None or prior["path"] != output["path"]:
            raise ReceiptValidationError(
                f"Replacement target {role!r} is unavailable at {output['path']}"
            )
        if output.get("replaces_sha256") != prior["sha256"]:
            raise ReceiptValidationError(
                f"Replacement for {output['path']} does not bind the prior hash"
            )
        replacements[output["path"]] = prior
    return replacements


def _verify_attempt_handoffs(
    attempt: Mapping[str, Any], repository_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    handoff = _verify_hashed_json_record(
        attempt["handoff"],
        repository_root,
        canonical_field="canonical_handoff_sha256",
    )
    supplemental = [
        _verify_hashed_json_record(
            record,
            repository_root,
            canonical_field="canonical_handoff_sha256",
        )
        for record in attempt.get("supplemental_handoffs", [])
    ]
    scoped = [
        _verify_hashed_json_record(
            record,
            repository_root,
            canonical_field="canonical_handoff_sha256",
        )
        for record in attempt.get("scoped_handoffs", [])
    ]
    return handoff, supplemental, scoped


def _verify_registration_freeze(
    manifest: Mapping[str, Any],
    attempt: Mapping[str, Any],
    repository_root: Path,
    *,
    completion_outputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    record = attempt.get("registration_freeze")
    if record is None:
        raise ReceiptValidationError(
            "Autonomous-v2 Stage 2 requires a CT-only registration freeze receipt"
        )
    expected_path = (
        Path("analysis")
        / manifest["specimen_id"]
        / "receipts"
        / f"stage_2_registration_freeze_attempt_{attempt['attempt']}.json"
    ).as_posix()
    if (
        not isinstance(record, Mapping)
        or set(record) != {"path", "sha256", "canonical_sha256"}
        or record.get("path") != expected_path
    ):
        raise ReceiptValidationError("Registration freeze record is malformed")
    payload = _verify_hashed_json_record(
        record,
        repository_root,
        canonical_field="canonical_freeze_sha256",
    )
    expected_keys = {
        "schema_version",
        "specimen_id",
        "stage_number",
        "attempt",
        "run_token",
        "frozen_at",
        "config_sha256",
        "contract_sha256",
        "predecessor_receipt_sha256",
        "input_handoff_sha256",
        "aligned_graph_accessed",
        "frozen_artifacts",
        "canonical_freeze_sha256",
    }
    if (
        set(payload) != expected_keys
        or _validate_timestamp_text(payload.get("frozen_at"))
        != payload.get("frozen_at")
    ):
        raise ReceiptValidationError(
            "Registration freeze is open-ended or schema-incompatible"
        )
    expected = {
        "schema_version": REGISTRATION_FREEZE_SCHEMA_VERSION,
        "specimen_id": manifest["specimen_id"],
        "stage_number": 2,
        "attempt": attempt["attempt"],
        "run_token": attempt["run_token"],
        "config_sha256": manifest["config"]["sha256"],
        "contract_sha256": manifest["stages"]["2"]["contract"]["sha256"],
        "predecessor_receipt_sha256": attempt["predecessor_receipt_sha256"],
        "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
        "aligned_graph_accessed": False,
    }
    stale = [key for key, value in expected.items() if payload.get(key) != value]
    if stale:
        raise ReceiptValidationError(
            "Registration freeze is stale or misbound: " + ", ".join(stale)
        )
    matching_events = [
        event
        for event in manifest.get("events", [])
        if event.get("action") == "autonomous_registration_frozen"
        and event.get("stage_number") == 2
        and event.get("timestamp") == payload["frozen_at"]
        and event.get("details")
        == {"freeze_sha256": payload["canonical_freeze_sha256"]}
    ]
    if len(matching_events) != 1:
        raise ReceiptValidationError(
            "Registration freeze timestamp is not bound to its event"
        )
    artifacts = _normalize_artifacts(
        repository_root, payload.get("frozen_artifacts", [])
    )
    if sorted(item["role"] for item in artifacts) != [
        "registered_graph",
        "registration_report",
    ]:
        raise ReceiptValidationError(
            "Registration freeze does not contain exactly the CT-only fit outputs"
        )
    contract = _load_stage_contract(manifest, 2, repository_root)
    _validate_artifact_allowlist(
        manifest,
        contract,
        artifacts,
        direction="output",
        require_all=False,
    )
    if completion_outputs is not None:
        by_role = {item["role"]: item for item in completion_outputs}
        if any(by_role.get(item["role"]) != item for item in artifacts):
            raise ReceiptValidationError(
                "Stage 2 completion outputs do not match the frozen CT-only fit"
            )
    return payload


def _verify_missing_calibration_attestation(
    outputs: Sequence[Mapping[str, Any]],
    repository_root: Path,
    *,
    manifest: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    by_role = {record["role"]: record for record in outputs}
    record = by_role.get("missing_calibration_attestation")
    findings = by_role.get("findings_missing")
    metrics = next(
        (
            artifact
            for artifact in attempt["input_artifacts"]
            if artifact["role"] == "per_strut_metrics"
        ),
        None,
    )
    scope = next(
        (
            value
            for value in attempt.get("scoped_handoffs", [])
            if value.get("consumer") == "missing_strut_agent"
        ),
        None,
    )
    if any(value is None for value in (record, findings, metrics, scope)):
        raise ReceiptValidationError(
            "Missing-strut calibration lacks findings, metrics, or scoped handoff"
        )
    attestation = _read_object(repository_root / record["path"])
    expected_keys = {
        "schema_version",
        "owner",
        "gate",
        "specimen_id",
        "stage_number",
        "attempt",
        "run_token",
        "scoped_handoff_sha256",
        "per_strut_metrics_sha256",
        "findings_missing_sha256",
        "development_split_accessed",
        "raw_development_labels_included",
        "calibration_summary",
    }
    if set(attestation) != expected_keys:
        raise ReceiptValidationError(
            "Missing calibration attestation has undeclared or missing fields"
        )
    expected = {
        "schema_version": "missing-calibration-attestation/1.0.0",
        "owner": "missing_strut_agent",
        "gate": "pass",
        "specimen_id": manifest["specimen_id"],
        "stage_number": 4,
        "attempt": attempt["attempt"],
        "run_token": attempt["run_token"],
        "scoped_handoff_sha256": scope["canonical_sha256"],
        "per_strut_metrics_sha256": metrics["sha256"],
        "findings_missing_sha256": findings["sha256"],
        "development_split_accessed": True,
        "raw_development_labels_included": False,
    }
    stale = [key for key, value in expected.items() if attestation.get(key) != value]
    summary = attestation.get("calibration_summary")
    if (
        stale
        or not isinstance(summary, Mapping)
        or set(summary)
        != {"method", "development_sample_count", "selected_missing_boundary"}
        or summary.get("method") != "development_split_calibration"
        or type(summary.get("development_sample_count")) is not int
        or summary["development_sample_count"] <= 0
        or type(summary.get("selected_missing_boundary")) not in {int, float}
        or not math.isfinite(summary["selected_missing_boundary"])
    ):
        raise ReceiptValidationError(
            "Missing calibration attestation is stale, leaking, or schema-incompatible"
        )


def _verify_stage4_verifier(
    outputs: Sequence[Mapping[str, Any]],
    repository_root: Path,
    *,
    manifest: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> None:
    by_role = {record["role"]: record for record in outputs}
    verifier_record = by_role.get("classifier_verifier_report")
    if verifier_record is None:
        raise ReceiptValidationError("Stage 4 requires classifier_verifier_report")
    verifier = _read_object(repository_root / verifier_record["path"])
    if verifier.get("schema_version") != "classifier-verifier-report/1.0.0":
        raise ReceiptValidationError("Classifier verifier schema is incompatible")
    expected_root_keys = {
        "schema_version",
        "owner",
        "gate",
        "specimen_id",
        "stage_number",
        "attempt",
        "run_token",
        "config_sha256",
        "contract_sha256",
        "predecessor_receipt_sha256",
        "input_handoff_sha256",
        "participated_in_classification",
        "label_access",
        "bindings",
        "self_verification",
    }
    if set(verifier) != expected_root_keys:
        raise ReceiptValidationError(
            "Classifier verifier report has undeclared or missing fields"
        )
    if verifier.get("owner") != "classifier_verifier" or verifier.get("gate") != "pass":
        raise ReceiptValidationError("Independent classifier verifier did not pass")
    expected_context = {
        "specimen_id": manifest["specimen_id"],
        "stage_number": 4,
        "attempt": attempt["attempt"],
        "run_token": attempt["run_token"],
        "config_sha256": manifest["config"]["sha256"],
        "contract_sha256": manifest["stages"]["4"]["contract"]["sha256"],
        "predecessor_receipt_sha256": attempt["predecessor_receipt_sha256"],
        "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
    }
    stale_context = [
        key for key, value in expected_context.items() if verifier.get(key) != value
    ]
    if stale_context:
        raise ReceiptValidationError(
            "Classifier verifier is stale or misbound: " + ", ".join(stale_context)
        )
    if verifier.get("participated_in_classification") is not False:
        raise ReceiptValidationError("Classifier verifier is not independent")
    access = verifier.get("label_access", {})
    if set(access) != {"development_split_read", "sealed_split_read"}:
        raise ReceiptValidationError("Classifier verifier label_access is invalid")
    if access.get("development_split_read") is not False:
        raise ReceiptValidationError("Classifier verifier must not read the dev split")
    if access.get("sealed_split_read") is not False:
        raise ReceiptValidationError("Classifier verifier must not read the sealed split")
    required_bindings = {
        "classified_struts_sha256": ("classified_struts",),
        "thresholds_sha256": ("classification_thresholds",),
        "decision_log_sha256": ("decision_log", "classification_decision_log"),
    }
    bindings = verifier.get("bindings", {})
    if set(bindings) != {
        *required_bindings,
        "evidence_set_sha256",
        "per_strut_metrics_sha256",
        "specialist_findings_sha256",
    }:
        raise ReceiptValidationError("Classifier verifier bindings are incomplete")
    for field, roles in required_bindings.items():
        artifact = next((by_role.get(role) for role in roles if by_role.get(role)), None)
        if artifact is None or bindings.get(field) != artifact["sha256"]:
            raise ReceiptValidationError(
                f"Classifier verifier has a stale or missing {field} binding"
            )
    evidence = sorted(
        (
            {"path": record["path"], "sha256": record["sha256"]}
            for record in outputs
            if record["role"]
            in {"evidence_index", "evidence_packet_manifest", "evidence_packets"}
        ),
        key=lambda record: (record["path"], record["sha256"]),
    )
    if (
        not evidence
        or bindings.get("evidence_set_sha256")
        != canonical_json_sha256(evidence)
    ):
        raise ReceiptValidationError(
            "Classifier verifier has a stale or incomplete evidence-set binding"
        )
    metrics = next(
        (
            artifact
            for artifact in attempt["input_artifacts"]
            if artifact["role"] == "per_strut_metrics"
        ),
        None,
    )
    if metrics is None or bindings.get("per_strut_metrics_sha256") != metrics["sha256"]:
        raise ReceiptValidationError(
            "Classifier verifier has a stale or missing per_strut_metrics_sha256 binding"
        )
    specialist_roles = (
        "findings_missing",
        "missing_calibration_attestation",
        "findings_thin",
        "findings_bent",
        "findings_broken",
    )
    expected_specialists = {
        role: by_role[role]["sha256"] for role in specialist_roles
    }
    if bindings.get("specialist_findings_sha256") != expected_specialists:
        raise ReceiptValidationError(
            "Classifier verifier specialist findings bindings are stale or incomplete"
        )
    checks = verifier.get("self_verification", {})
    required_checks = (
        "every_strut_labeled_once",
        "fixed_precedence_respected",
        "bent_kept_separate",
        "every_adjudication_logged",
        "evidence_support_checked",
        "cutoffs_audited",
        "decision_log_matches_execution",
        "development_split_not_accessed",
        "sealed_split_not_accessed",
    )
    failed = [name for name in required_checks if checks.get(name) is not True]
    if failed or set(checks) != set(required_checks):
        raise ReceiptValidationError(
            "Classifier verifier failed checks: " + ", ".join(failed)
        )


def _verify_stage5_evaluation_result(
    outputs: Sequence[Mapping[str, Any]],
    repository_root: Path,
    *,
    attempt: Mapping[str, Any],
) -> None:
    by_role = {record["role"]: record for record in outputs}
    result_record = by_role.get("sealed_evaluation_result")
    if result_record is None:
        raise ReceiptValidationError("Stage 5 requires one sealed_evaluation_result")
    result = _read_object(repository_root / result_record["path"])
    allowed_keys = {
        "schema_version",
        "gate",
        "overall_pass",
        "protocol",
        "sealed_strut_count",
        "strict_recall",
        "lenient_recall",
        "confusion_matrix",
        "omitted_metrics",
        "artifacts",
        "hashes",
        "provenance",
        "warnings",
    }
    if set(result) != allowed_keys:
        raise ReceiptValidationError(
            "Sealed evaluation result has undeclared or missing fields that could leak labels"
        )
    if (
        result.get("schema_version") != "part2-detection-metrics/1.0.0"
        or result.get("gate") != "pass"
        or result.get("overall_pass") is not True
        or result.get("protocol") != "one_shot_reporting_not_pass_fail"
    ):
        raise ReceiptValidationError("Sealed evaluation result schema/protocol is invalid")
    sealed_count = result.get("sealed_strut_count")
    if type(sealed_count) is not int or sealed_count <= 0:
        raise ReceiptValidationError("sealed_strut_count must be a positive integer")
    definitions = {
        "strict_recall": "predicted missing",
        "lenient_recall": "predicted missing or broken",
    }
    for name, definition in definitions.items():
        value = result.get(name)
        if not isinstance(value, Mapping) or set(value) != {
            "definition",
            "detected",
            "total",
            "value",
            "wilson_95_ci",
        }:
            raise ReceiptValidationError(f"{name} is incomplete or schema-incompatible")
        interval = value.get("wilson_95_ci")
        detected = value.get("detected")
        total = value.get("total")
        point = value.get("value")
        if (
            value.get("definition") != definition
            or type(detected) is not int
            or detected < 0
            or type(total) is not int
            or total != sealed_count
            or detected > total
            or type(point) not in {int, float}
            or not math.isfinite(point)
            or not 0 <= point <= 1
            or not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                type(bound) in {int, float}
                and math.isfinite(bound)
                and 0 <= bound <= 1
                for bound in interval
            )
            or interval[0] > interval[1]
        ):
            raise ReceiptValidationError(f"{name} lacks a Wilson 95% interval")
    confusion = result.get("confusion_matrix")
    classes = ["missing", "broken", "thin", "present"]
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != {"class_order", "rows_actual_columns_predicted"}
        or confusion.get("class_order") != classes
        or set(confusion.get("rows_actual_columns_predicted", {})) != set(classes)
        or any(
            not isinstance(row, Mapping) or set(row) != set(classes)
            for row in confusion.get("rows_actual_columns_predicted", {}).values()
        )
        or any(
            type(count) is not int or count < 0
            for row in confusion.get("rows_actual_columns_predicted", {}).values()
            for count in row.values()
        )
    ):
        raise ReceiptValidationError("Stage 5 confusion matrix is incomplete")
    hashes = result.get("hashes", {})
    if (
        not isinstance(hashes, Mapping)
        or set(hashes) != {"classifications_sha256", "sealed_labels_sha256"}
    ):
        raise ReceiptValidationError("Stage 5 result hash bindings are incomplete")
    classified = next(
        (
            artifact
            for artifact in attempt["input_artifacts"]
            if artifact["role"] == "classified_struts"
        ),
        None,
    )
    sealed = next(
        (
            artifact
            for artifact in attempt["input_artifacts"]
            if artifact["role"] == "sealed_labels"
        ),
        None,
    )
    if (
        classified is None
        or sealed is None
        or hashes.get("classifications_sha256") != classified["sha256"]
        or hashes.get("sealed_labels_sha256") != sealed["sha256"]
    ):
        raise ReceiptValidationError(
            "Stage 5 scoring result is not bound to frozen classifications and sealed split"
        )
    omitted = result.get("omitted_metrics", {})
    if omitted != {
        "precision": (
            "undefined because detections outside sealed intentional deletions "
            "may be unintentional defects"
        ),
        "f1": "not computed because precision is undefined",
    }:
        raise ReceiptValidationError("Stage 5 must explicitly omit precision and F1")
    provenance = result.get("provenance", {})
    if (
        set(provenance) != {"eval_side", "sealed_labels_read"}
        or provenance.get("eval_side") is not True
        or provenance.get("sealed_labels_read") is not True
    ):
        raise ReceiptValidationError("Stage 5 evaluation provenance is invalid")
    if result.get("artifacts") != {} or result.get("warnings") != []:
        raise ReceiptValidationError(
            "Stage 5 aggregate result contains undeclared artifact or warning payloads"
        )


def _register_output_artifacts(
    manifest: dict[str, Any],
    contract: Mapping[str, Any],
    stage_number: int,
    attempt_number: int,
    outputs: Sequence[Mapping[str, Any]],
    replacements: Mapping[str, Mapping[str, Any]],
) -> None:
    for path, prior in replacements.items():
        for record in manifest["artifact_index"]:
            if record is prior or (
                record.get("state", "active") == "active"
                and record["path"] == path
                and record["sha256"] == prior["sha256"]
            ):
                record["state"] = "superseded"
                record["superseded_by_sha256"] = next(
                    item["sha256"] for item in outputs if item["path"] == path
                )
    active = {(item["path"], item["sha256"], item["role"]) for item in _active_artifact_records(manifest)}
    paths = {item["path"]: item for item in _active_artifact_records(manifest)}
    for output in outputs:
        identity = (output["path"], output["sha256"], output["role"])
        existing = paths.get(output["path"])
        if existing is not None and identity not in active:
            raise ArtifactVerificationError(
                f"Immutable output path collision at {output['path']}"
            )
        if identity not in active:
            kind = _sensitive_kind_for_artifact(contract, output)
            record = {
                **dict(output),
                "stage_number": stage_number,
                "attempt": attempt_number,
                "state": "active",
                "sensitivity": kind,
            }
            manifest["artifact_index"].append(record)
            if kind is not None:
                hashes = manifest["sensitive_artifact_hashes"][kind]
                if output["sha256"] not in hashes:
                    hashes.append(output["sha256"])
                    hashes.sort()


def complete_stage(
    manifest_path: RepositoryPath,
    receipt_path: RepositoryPath,
    *,
    repository_root: RepositoryPath,
) -> dict[str, Any]:
    """Verify a current receipt and apply exactly one legal terminal transition."""

    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    receipt_resolved, receipt_relative = _relative_existing_file(
        root, receipt_path, reject_alias=False
    )
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=False)
        _verify_control_record_files(manifest, root)
        _verify_attempt_evidence(
            manifest, root, verify_artifact_contents=False
        )
        receipt = _read_object(receipt_resolved)
        canonical = canonical_json_sha256(
            {key: value for key, value in receipt.items() if key != "canonical_receipt_sha256"}
        )
        if receipt.get("canonical_receipt_sha256") != canonical:
            raise ReceiptValidationError("Completion receipt canonical hash is invalid")
        stage_number = receipt.get("stage_number")
        if stage_number not in STAGE_NUMBERS:
            raise ReceiptValidationError("Receipt stage_number is invalid")
        stage = manifest["stages"][str(stage_number)]
        receipt_attempt = receipt.get("attempt")
        if type(receipt_attempt) is not int or receipt_attempt < 1:
            raise ReceiptValidationError("Receipt attempt is invalid")
        expected_receipt_relative = _receipt_output_path(
            manifest, stage_number, receipt_attempt
        ).as_posix()
        if receipt_relative != expected_receipt_relative:
            raise ReceiptValidationError(
                "Completion receipt path is not the canonical attempt path"
            )
        raw_receipt_hash = sha256_file(receipt_resolved)

        for attempt in stage["attempts"]:
            record = attempt.get("receipt")
            if record and record.get("sha256") == raw_receipt_hash:
                _verify_indexed_artifacts(manifest, root)
                _verify_attempt_evidence(manifest, root)
                _verify_hashed_json_record(
                    record, root, canonical_field="canonical_receipt_sha256"
                )
                before = path.read_bytes()
                return {
                    "changed": False,
                    "state": stage["state"],
                    "receipt": record,
                    "manifest_bytes_unchanged": path.read_bytes() == before,
                }
        if stage["state"] != "running" or not stage["attempts"]:
            raise IllegalTransitionError(f"Stage {stage_number} is not running")
        attempt = stage["attempts"][-1]
        contract = _load_stage_contract(manifest, stage_number, root)
        _validate_stage_receipt_document(receipt, contract)
        expected = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "contract_version": stage["contract"]["version"],
            "contract_sha256": stage["contract"]["sha256"],
            "specimen_id": manifest["specimen_id"],
            "stage_number": stage_number,
            "stage": stage["name"],
            "owner": stage["owner"],
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "config_sha256": manifest["config"]["sha256"],
            "registration_mode": manifest["registration_mode"],
            "predecessor_receipt_sha256": manifest["predecessor_receipt_sha256"],
            "input_handoff": attempt["handoff"],
            "scoped_handoffs": attempt.get("scoped_handoffs", []),
            "supplemental_handoffs": attempt["supplemental_handoffs"],
            "registration_freeze": attempt["registration_freeze"],
        }
        stale = [key for key, value in expected.items() if receipt.get(key) != value]
        if stale:
            raise ReceiptValidationError(
                "Completion receipt is stale or misbound: " + ", ".join(stale)
            )
        terminal = receipt.get("terminal_state")
        if terminal not in TERMINAL_STATES:
            raise ReceiptValidationError("Receipt terminal_state is invalid")
        if receipt.get("failure_kind") == "deterministic_gate" and terminal != "halt":
            raise ReceiptValidationError(
                "Deterministic gate failures are non-retryable halts"
            )
        handoff, supplemental, scoped = _verify_attempt_handoffs(attempt, root)
        scoped_inputs = [
            artifact
            for scoped_handoff in scoped
            for artifact in scoped_handoff.get("input_artifacts", [])
        ]
        combined_inputs = sorted(
            [*handoff.get("input_artifacts", []), *scoped_inputs],
            key=lambda item: (
                item["path"], item["role"], str(item.get("consumer", ""))
            ),
        )
        if combined_inputs != attempt["input_artifacts"]:
            raise ReceiptValidationError("Attempt handoff inputs changed")
        outputs = _normalize_artifacts(root, receipt.get("output_artifacts", []))
        _validate_artifact_allowlist(
            manifest,
            contract,
            outputs,
            direction="output",
            require_all=terminal == "pass",
        )
        if terminal == "manual_review":
            _validate_manual_review_outputs(
                manifest,
                stage_number,
                attempt["attempt"],
                outputs,
            )
        required_assertions = contract.get("required_receipt_assertions", [])
        if terminal == "pass":
            failed = [
                name
                for name in required_assertions
                if receipt.get("assertions", {}).get(name) is not True
            ]
            if failed:
                raise ReceiptValidationError(
                    "Completion assertions failed: " + ", ".join(failed)
                )
        replacements = _replacement_map(manifest, contract, outputs)
        skipped = set(replacements)
        _verify_indexed_artifacts(manifest, root, skipped_paths=skipped)
        for input_artifact in handoff["input_artifacts"]:
            if input_artifact["path"] in replacements:
                if replacements[input_artifact["path"]]["sha256"] != input_artifact["sha256"]:
                    raise ReceiptValidationError("Replacement lineage is stale")
            else:
                _normalize_artifact(root, input_artifact)
        for supplemental_handoff in supplemental:
            for artifact in supplemental_handoff.get("input_artifacts", []):
                _normalize_artifact(root, artifact)
        if stage_number == 4 and terminal == "pass":
            _verify_missing_calibration_attestation(
                outputs,
                root,
                manifest=manifest,
                attempt=attempt,
            )
            _verify_stage4_verifier(
                outputs,
                root,
                manifest=manifest,
                attempt=attempt,
            )
        if stage_number == 2 and manifest["registration_mode"] == "autonomous_v2":
            if terminal == "pass":
                _verify_registration_freeze(
                    manifest,
                    attempt,
                    root,
                    completion_outputs=outputs,
                )
        if stage_number == 5 and terminal == "pass":
            _verify_stage5_evaluation_result(
                outputs,
                root,
                attempt=attempt,
            )
            if receipt.get("assertions", {}).get("optimization_performed") is not False:
                raise ReceiptValidationError(
                    "Stage 5 is one-shot reporting and may not optimize upstream decisions"
                )

        receipt_record = {
            "path": receipt_relative,
            "sha256": raw_receipt_hash,
            "canonical_sha256": canonical,
        }
        effective = terminal
        control_halt: dict[str, Any] | None = None
        if terminal == "manual_review" and attempt["attempt"] >= stage["maximum_attempts"]:
            effective = "halt"
            control_halt = {
                "code": "attempts_exhausted",
                "reported_terminal_state": "manual_review",
                "maximum_attempts": stage["maximum_attempts"],
            }
        attempt["state"] = effective
        attempt["completed_at"] = receipt["completed_at"]
        attempt["receipt"] = receipt_record
        attempt["reported_terminal_state"] = terminal
        attempt["effective_terminal_state"] = effective
        attempt["failure_kind"] = receipt.get("failure_kind")
        attempt["output_artifacts"] = outputs
        stage["state"] = effective
        stage["completed_at"] = receipt["completed_at"]
        stage["completion_receipt"] = receipt_record
        stage["control_halt"] = control_halt
        _register_output_artifacts(
            manifest,
            contract,
            stage_number,
            attempt["attempt"],
            outputs,
            replacements,
        )
        if effective == "pass":
            manifest["predecessor_receipt_sha256"] = canonical
            if stage_number == STAGE_NUMBERS[-1]:
                manifest["pipeline_state"] = "pass"
                manifest["current_stage"] = None
            else:
                next_stage = manifest["stages"][str(stage_number + 1)]
                if next_stage["state"] != "locked":
                    raise IllegalTransitionError("Next stage is not locked")
                next_stage["state"] = "ready"
                next_stage["unlocked_at"] = receipt["completed_at"]
                manifest["pipeline_state"] = "ready"
                manifest["current_stage"] = stage_number + 1
        elif effective == "manual_review":
            manifest["pipeline_state"] = "manual_review"
            manifest["current_stage"] = stage_number
        else:
            manifest["pipeline_state"] = "halt"
            manifest["current_stage"] = stage_number
        manifest["updated_at"] = receipt["completed_at"]
        manifest["revision"] += 1
        _event(
            manifest,
            timestamp=receipt["completed_at"],
            action="stage_completed",
            stage_number=stage_number,
            details={
                "attempt": attempt["attempt"],
                "reported_state": terminal,
                "effective_state": effective,
                "receipt_sha256": canonical,
            },
        )
        changed = _write_manifest(path, manifest)
        return {
            "changed": changed,
            "state": effective,
            "reported_state": terminal,
            "receipt": receipt_record,
            "next_stage": manifest["current_stage"],
        }


def resume_manual_review(
    manifest_path: RepositoryPath,
    stage_number: int,
    *,
    resolution_artifact: Mapping[str, Any],
    reason: str,
    repository_root: RepositoryPath,
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Explicitly reopen a retryable judgment stage without erasing evidence."""

    if stage_number not in STAGE_NUMBERS:
        raise IllegalTransitionError(f"Unknown stage number: {stage_number}")
    if not isinstance(reason, str) or not reason.strip():
        raise IllegalTransitionError("Manual review resumption requires a reason")
    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    resumed_at = _timestamp(timestamp, clock)
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=True)
        stage = manifest["stages"][str(stage_number)]
        resolution = _normalize_artifact(root, resolution_artifact)
        if resolution["role"] != "manual_review_resolution":
            raise AccessPolicyError(
                "Resolution artifact role must be manual_review_resolution"
            )
        expected_prefix = f"analysis/{manifest['specimen_id']}/reviews/"
        if not resolution["path"].startswith(expected_prefix):
            raise AccessPolicyError(
                f"Resolution artifact must be stored under {expected_prefix}"
            )
        if stage["attempts"]:
            recorded = next(
                (
                    attempt.get("manual_review_resolution")
                    for attempt in reversed(stage["attempts"])
                    if attempt.get("manual_review_resolution") is not None
                ),
                None,
            )
            if (
                isinstance(recorded, dict)
                and recorded.get("path") == resolution["path"]
                and recorded.get("sha256") == resolution["sha256"]
                and recorded.get("reason") == reason
                and stage["state"] != "manual_review"
            ):
                return {
                    "changed": False,
                    "state": stage["state"],
                    "stage_number": stage_number,
                }
        if manifest["pipeline_state"] != "manual_review" or stage["state"] != "manual_review":
            raise IllegalTransitionError(f"Stage {stage_number} is not in manual_review")
        if manifest["current_stage"] != stage_number:
            raise IllegalTransitionError("Manual review stage is not current")
        if stage["one_shot"]:
            raise IllegalTransitionError("A one-shot sealed evaluation cannot be resumed")
        if stage["attempt_count"] >= stage["maximum_attempts"]:
            raise IllegalTransitionError("Stage attempt limit is exhausted")
        latest = stage["attempts"][-1]
        if latest.get("failure_kind") == "deterministic_gate":
            raise IllegalTransitionError("Deterministic gate failures cannot be retried")
        _register_output_artifacts(
            manifest,
            {"output_artifacts": {"sensitive_roles": {}}},
            stage_number,
            latest["attempt"],
            [resolution],
            {},
        )
        latest["manual_review_resolution"] = {
            **resolution,
            "reason": reason,
            "resolved_at": resumed_at,
        }
        stage["state"] = "ready"
        stage["completed_at"] = None
        stage["current_attempt"] = None
        manifest["pipeline_state"] = "ready"
        manifest["updated_at"] = resumed_at
        manifest["revision"] += 1
        _event(
            manifest,
            timestamp=resumed_at,
            action="manual_review_resumed",
            stage_number=stage_number,
            details={
                "prior_attempt": latest["attempt"],
                "resolution_sha256": resolution["sha256"],
                "reason": reason,
            },
        )
        changed = _write_manifest(path, manifest)
        return {"changed": changed, "state": "ready", "stage_number": stage_number}


def record_autonomous_registration_freeze(
    manifest_path: RepositoryPath,
    *,
    frozen_artifacts: Sequence[Mapping[str, Any]],
    repository_root: RepositoryPath,
    output_path: RepositoryPath | None = None,
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Seal Stage 2 CT-only registration before any aligned-JSON validation."""

    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    frozen_at = _timestamp(timestamp, clock)
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=True)
        if manifest["registration_mode"] != "autonomous_v2":
            raise IllegalTransitionError("Registration freeze is autonomous-v2 only")
        stage = manifest["stages"]["2"]
        if stage["state"] != "running" or not stage["attempts"]:
            raise IllegalTransitionError("Stage 2 must be running before registration freeze")
        attempt = stage["attempts"][-1]
        contract = _load_stage_contract(manifest, 2, root)
        artifacts = _normalize_artifacts(root, frozen_artifacts)
        _validate_artifact_allowlist(
            manifest,
            contract,
            artifacts,
            direction="output",
            require_all=False,
        )
        roles = [artifact["role"] for artifact in artifacts]
        if sorted(roles) != ["registered_graph", "registration_report"]:
            raise ReceiptValidationError(
                "Registration freeze accepts exactly registered_graph and registration_report"
            )
        if any(
            _sensitive_kind_for_artifact(contract, artifact) is not None
            for artifact in artifacts
        ):
            raise AccessPolicyError("Registration freeze must contain CT-only outputs")
        if attempt["registration_freeze"] is not None:
            payload = _verify_hashed_json_record(
                attempt["registration_freeze"],
                root,
                canonical_field="canonical_freeze_sha256",
            )
            if payload.get("frozen_artifacts") != artifacts:
                raise ReceiptValidationError(
                    "Registration freeze replay requested different artifacts"
                )
            return {
                "changed": False,
                "freeze": attempt["registration_freeze"],
                "frozen_artifacts": payload["frozen_artifacts"],
            }
        if any(
            artifact.get("role")
            in {
                "challenge_aligned_json",
                "challenge_aligned_graph",
                "aligned_graph_reference",
                "autonomous_validation_reference",
            }
            for artifact in attempt["input_artifacts"]
        ):
            raise AccessPolicyError("CT-only registration handoff contains aligned JSON")
        freeze_base = {
            "schema_version": REGISTRATION_FREEZE_SCHEMA_VERSION,
            "specimen_id": manifest["specimen_id"],
            "stage_number": 2,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "frozen_at": frozen_at,
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": stage["contract"]["sha256"],
            "predecessor_receipt_sha256": attempt[
                "predecessor_receipt_sha256"
            ],
            "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
            "aligned_graph_accessed": False,
            "frozen_artifacts": artifacts,
        }
        freeze = _with_self_hash(freeze_base, "canonical_freeze_sha256")
        freeze_name = f"stage_2_registration_freeze_attempt_{attempt['attempt']}.json"
        relative, destination = _repository_output_location(
            root,
            output_path,
            Path("analysis")
            / manifest["specimen_id"]
            / "receipts"
            / freeze_name,
            required_directory=Path("analysis")
            / manifest["specimen_id"]
            / "receipts",
            required_name=freeze_name,
        )
        if destination.is_file():
            existing = _read_object(destination)
            if existing != freeze:
                existing_base = {
                    key: value
                    for key, value in existing.items()
                    if key != "canonical_freeze_sha256"
                }
                comparable_existing = {
                    key: value for key, value in existing_base.items() if key != "frozen_at"
                }
                comparable_requested = {
                    key: value for key, value in freeze_base.items() if key != "frozen_at"
                }
                if (
                    existing.get("canonical_freeze_sha256")
                    != canonical_json_sha256(existing_base)
                    or comparable_existing != comparable_requested
                ):
                    raise ArtifactVerificationError(
                        "Immutable registration freeze already exists with different content"
                    )
                freeze = existing
                frozen_at = str(existing["frozen_at"])
        else:
            _atomic_write_if_changed(destination, freeze)
        record = {
            "path": relative.as_posix(),
            "sha256": sha256_file(destination),
            "canonical_sha256": freeze["canonical_freeze_sha256"],
        }
        attempt["registration_freeze"] = record
        manifest["updated_at"] = frozen_at
        manifest["revision"] += 1
        _event(
            manifest,
            timestamp=frozen_at,
            action="autonomous_registration_frozen",
            stage_number=2,
            details={"freeze_sha256": freeze["canonical_freeze_sha256"]},
        )
        _write_manifest(path, manifest)
        return {"changed": True, "freeze": record, "frozen_artifacts": artifacts}


def authorize_post_freeze_aligned_input(
    manifest_path: RepositoryPath,
    *,
    aligned_artifact: Mapping[str, Any],
    repository_root: RepositoryPath,
    output_path: RepositoryPath | None = None,
    timestamp: str | None = None,
    clock: Clock | None = None,
) -> dict[str, Any]:
    """Create a bounded post-freeze validator handoff for autonomous-v2."""

    root = Path(repository_root).resolve()
    path = _pipeline_manifest_location(root, manifest_path)
    authorized_at = _timestamp(timestamp, clock)
    with _manifest_lock(path):
        manifest = _load_manifest_unlocked(path, root, verify_artifacts=True)
        if manifest["registration_mode"] != "autonomous_v2":
            raise IllegalTransitionError("Post-freeze authorization is autonomous-v2 only")
        stage = manifest["stages"]["2"]
        if stage["state"] != "running" or not stage["attempts"]:
            raise IllegalTransitionError("Stage 2 is not running")
        attempt = stage["attempts"][-1]
        freeze_record = attempt.get("registration_freeze")
        if freeze_record is None:
            raise AccessPolicyError(
                "Aligned JSON remains inaccessible until CT-only registration is frozen"
            )
        freeze = _verify_hashed_json_record(
            freeze_record,
            root,
            canonical_field="canonical_freeze_sha256",
        )
        artifact = _normalize_artifact(root, aligned_artifact)
        artifact["consumer"] = artifact.get("consumer", "data_prep")
        artifact["phase"] = "autonomous_v2_post_freeze_validation"
        if artifact["consumer"] not in {"data_prep", "registration_validator"} or artifact["role"] not in {
            "challenge_aligned_json",
            "challenge_aligned_graph",
            "aligned_graph_reference",
            "autonomous_validation_reference",
        }:
            raise AccessPolicyError(
                "Post-freeze input must be the aligned graph for registration_validator"
            )
        contract = _load_stage_contract(manifest, 2, root)
        _validate_artifact_allowlist(
            manifest,
            contract,
            [artifact],
            direction="input",
            require_all=False,
        )
        _enforce_sensitive_access(manifest, 2, [artifact])
        _record_sensitive_hashes(manifest, contract, [artifact])
        existing_records = attempt.get("supplemental_handoffs", [])
        if existing_records:
            if len(existing_records) != 1:
                raise ManifestValidationError(
                    "Autonomous Stage 2 has multiple post-freeze handoffs"
                )
            existing_record = existing_records[0]
            existing = _verify_hashed_json_record(
                existing_record,
                root,
                canonical_field="canonical_handoff_sha256",
            )
            if (
                existing.get("registration_freeze_sha256")
                == freeze["canonical_freeze_sha256"]
                and existing.get("frozen_artifacts") == freeze["frozen_artifacts"]
                and existing.get("input_artifacts") == [artifact]
                and existing.get("run_token") == attempt["run_token"]
            ):
                return {"changed": False, "handoff": existing_record}
            raise ArtifactVerificationError(
                "Post-freeze handoff already authorizes a different request"
            )
        handoff_base = {
            "schema_version": POST_FREEZE_HANDOFF_SCHEMA_VERSION,
            "specimen_id": manifest["specimen_id"],
            "stage_number": 2,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "created_at": authorized_at,
            "config_sha256": manifest["config"]["sha256"],
            "registration_freeze_sha256": freeze["canonical_freeze_sha256"],
            "frozen_artifacts": freeze["frozen_artifacts"],
            "input_artifacts": [artifact],
            "purpose": "optional_post_fit_validation_only",
        }
        handoff = _with_self_hash(handoff_base, "canonical_handoff_sha256")
        handoff_name = (
            f"stage_2_post_freeze_validation_attempt_{attempt['attempt']}.json"
        )
        relative, destination = _repository_output_location(
            root,
            output_path,
            Path("analysis")
            / manifest["specimen_id"]
            / "handoffs"
            / handoff_name,
            required_directory=Path("analysis")
            / manifest["specimen_id"]
            / "handoffs",
            required_name=handoff_name,
        )
        if destination.is_file():
            existing = _read_object(destination)
            if existing != handoff:
                existing_base = {
                    key: value
                    for key, value in existing.items()
                    if key != "canonical_handoff_sha256"
                }
                comparable_existing = {
                    key: value for key, value in existing_base.items() if key != "created_at"
                }
                comparable_requested = {
                    key: value for key, value in handoff_base.items() if key != "created_at"
                }
                if (
                    existing.get("canonical_handoff_sha256")
                    != canonical_json_sha256(existing_base)
                    or comparable_existing != comparable_requested
                ):
                    raise ArtifactVerificationError(
                        "Post-freeze handoff already exists with different content"
                    )
                handoff = existing
                authorized_at = str(existing["created_at"])
        else:
            _atomic_write_if_changed(destination, handoff)
        record = {
            "path": relative.as_posix(),
            "sha256": sha256_file(destination),
            "canonical_sha256": handoff["canonical_handoff_sha256"],
        }
        attempt["supplemental_handoffs"].append(record)
        manifest["updated_at"] = authorized_at
        manifest["revision"] += 1
        _event(
            manifest,
            timestamp=authorized_at,
            action="post_freeze_aligned_input_authorized",
            stage_number=2,
            details={
                "aligned_graph_sha256": artifact["sha256"],
                "freeze_sha256": freeze["canonical_freeze_sha256"],
            },
        )
        _write_manifest(path, manifest)
        return {"changed": True, "handoff": record}


def pipeline_status(
    manifest_path: RepositoryPath,
    *,
    repository_root: RepositoryPath,
) -> dict[str, Any]:
    manifest = validate_pipeline_manifest(
        manifest_path, repository_root=repository_root, verify_artifacts=True
    )
    return {
        "schema_version": manifest["schema_version"],
        "specimen_id": manifest["specimen_id"],
        "pipeline_state": manifest["pipeline_state"],
        "current_stage": manifest["current_stage"],
        "registration_mode": manifest["registration_mode"],
        "config_sha256": manifest["config"]["sha256"],
        "sealed_evaluation_consumed": manifest["sealed_evaluation"]["consumed"],
        "stages": {
            number: {
                "name": stage["name"],
                "state": stage["state"],
                "attempt_count": stage["attempt_count"],
                "maximum_attempts": stage["maximum_attempts"],
            }
            for number, stage in manifest["stages"].items()
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }
