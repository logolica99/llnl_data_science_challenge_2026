"""Shared path policy and closed response handling for production MCP tools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from llnl_nde.core import error_response as _error_response
from llnl_nde.core import success_response as _success_response

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class MCPErrorEnvelope(BaseModel):
    """Closed error member for the shared Part 2 MCP response envelope."""

    model_config = ConfigDict(extra="forbid")

    code: str
    type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MCPResponseEnvelope(BaseModel):
    """Closed top-level response shape used by production Part 2 tools."""

    model_config = ConfigDict(extra="forbid")

    response_schema_version: Literal["part2-mcp-response/1.0.0"]
    tool: str
    status: Literal["ok", "error"]
    gate: Literal["pass", "halt", "manual_review"]
    summary: str
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    hashes: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: MCPErrorEnvelope | None = None

    def __getitem__(self, key: str) -> Any:
        """Retain compact direct-Python access without opening outputSchema."""

        if key in type(self).model_fields:
            return getattr(self, key)
        return self.result[key]


def _repository_path(
    filepath: str,
    *,
    must_exist: bool,
    expected_suffixes: set[str] | None = None,
) -> tuple[Path, str]:
    """Resolve one new Part 2 tool path without allowing repository escape."""

    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Input file does not exist: {relative.as_posix()}")
    if expected_suffixes and resolved.suffix.lower() not in expected_suffixes:
        choices = ", ".join(sorted(expected_suffixes))
        raise ValueError(f"Expected one of [{choices}], found {relative.as_posix()}")
    return resolved, relative.as_posix()


def _repository_output_directory(filepath: str) -> tuple[Path, str]:
    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(
            f"Output directory is an existing file: {relative.as_posix()}"
        )
    return resolved, relative.as_posix()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _structured_failure(tool: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        code = "input_not_found"
    elif isinstance(exc, FileExistsError):
        code = "artifact_exists"
    elif isinstance(exc, (ValueError, TypeError, IndexError)):
        code = "invalid_input"
    else:
        code = "tool_execution_failed"
    return _error_response(
        tool=tool,
        code=code,
        error_type=type(exc).__name__,
        message=str(exc),
    )


def _run_structured_tool(
    tool: str,
    operation: Callable[[], dict[str, Any]],
) -> MCPResponseEnvelope:
    try:
        return MCPResponseEnvelope.model_validate(operation())
    except Exception as exc:
        return MCPResponseEnvelope.model_validate(_structured_failure(tool, exc))


def _relative_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Convert core artifact paths to repository-relative MCP paths."""

    result: dict[str, Any] = {}
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            result[name] = metadata
            continue
        item = dict(metadata)
        if "path" in item:
            artifact_path = Path(item["path"])
            item["path"] = (
                artifact_path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
                if artifact_path.is_absolute()
                else artifact_path.as_posix()
            )
        result[name] = item
    return result


def _core_response(
    tool: str,
    summary: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Expose a deterministic core result without adding computation."""

    result = (
        dict(payload["result"])
        if isinstance(payload.get("result"), dict)
        else {
            key: value
            for key, value in payload.items()
            if key not in {"artifacts", "hashes", "warnings"}
        }
    )
    return _success_response(
        tool=tool,
        gate=payload["gate"],
        summary=summary,
        result=result,
        artifacts=_relative_artifacts(payload.get("artifacts", {})),
        hashes=dict(payload.get("hashes", {})),
        warnings=list(payload.get("warnings", [])),
    )
