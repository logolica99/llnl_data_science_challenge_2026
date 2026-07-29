"""Versioned structured response envelopes shared by Part 2 MCP tools."""

from __future__ import annotations

from typing import Any, Literal


RESPONSE_SCHEMA_VERSION = "part2-mcp-response/1.0.0"
GATES = {"pass", "halt", "manual_review"}
Gate = Literal["pass", "halt", "manual_review"]


def _gate(value: str) -> Gate:
    if value not in GATES:
        raise ValueError(f"Unsupported gate {value!r}; expected one of {sorted(GATES)}")
    return value  # type: ignore[return-value]


def success_response(
    *,
    tool: str,
    gate: Gate,
    summary: str,
    result: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    hashes: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a successful computation envelope, including non-pass gates."""

    return {
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "tool": tool,
        "status": "ok",
        "gate": _gate(gate),
        "summary": summary,
        "result": result,
        "artifacts": artifacts or {},
        "hashes": hashes or {},
        "warnings": warnings or [],
        "error": None,
    }


def error_response(
    *,
    tool: str,
    code: str,
    message: str,
    error_type: str,
    details: dict[str, Any] | None = None,
    gate: Gate = "halt",
) -> dict[str, Any]:
    """Build a deterministic structured error instead of returning prose."""

    return {
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "tool": tool,
        "status": "error",
        "gate": _gate(gate),
        "summary": message,
        "result": {},
        "artifacts": {},
        "hashes": {},
        "warnings": [],
        "error": {
            "code": code,
            "type": error_type,
            "message": message,
            "details": details or {},
        },
    }
