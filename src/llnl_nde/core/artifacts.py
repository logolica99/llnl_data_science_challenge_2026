"""Deterministic artifact helpers shared by the Part 2 production core."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of one file without reading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value canonically for stable hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {resolved}")
    return value


def write_json_atomic(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write stable pretty JSON atomically and return path/hash metadata."""

    destination = Path(path).expanduser().resolve()
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "changed": False,
            }
        raise FileExistsError(
            "Artifact already exists with different bytes; choose a new path: "
            f"{destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "changed": True,
    }


def write_text_atomic(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write UTF-8 text atomically with exact-replay idempotency."""

    destination = Path(path).expanduser().resolve()
    payload = text.encode("utf-8")
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "changed": False,
            }
        raise FileExistsError(
            f"Artifact already exists with different bytes: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "changed": True,
    }


def require_new_path(path: str | Path, overwrite: bool) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Artifact already exists; enable overwrite explicitly: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
