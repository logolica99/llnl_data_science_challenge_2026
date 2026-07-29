"""Explicit persisted Stage 0 MCP-response fixtures for deterministic tests.

This module reads only fixture headers to emulate the header facts in an actual
MCP call receipt. It never reads voxel content for scientific computation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import tifffile


UNKNOWN = "unknown"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.byteorder == "|":
        return "not_applicable"
    if dtype.byteorder == "=":
        return sys.byteorder
    return "little" if dtype.byteorder == "<" else "big"


def _actual_header_facts(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        dtype = np.dtype(array.dtype)
        shape = [int(value) for value in array.shape]
        axes = UNKNOWN
        byte_order = _byte_order(dtype)
        volume_format = "npy"
    elif suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            dtype = np.dtype(series.dtype)
            shape = [int(value) for value in series.shape]
            axes = str(series.axes) if series.axes else UNKNOWN
            byte_order = (
                "not_applicable"
                if dtype.itemsize == 1
                else "little" if tif.byteorder == "<" else "big"
            )
        volume_format = "tiff"
    else:
        raise ValueError(f"Unsupported Stage 0 fixture volume: {path}")
    voxel_count = int(np.prod(shape, dtype=np.int64))
    return {
        "file_bytes": path.stat().st_size,
        "format": volume_format,
        "shape": shape,
        "ndim": len(shape),
        "dtype": dtype.name,
        "dtype_string": dtype.str,
        "byte_order": byte_order,
        "axes": axes,
        "voxel_count": voxel_count,
        "array_bytes": voxel_count * dtype.itemsize,
    }


def _unknown_spacing() -> dict[str, Any]:
    return {
        axis: {
            "value": UNKNOWN,
            "unit": UNKNOWN,
            "provenance": {
                "source": UNKNOWN,
                "field": UNKNOWN,
                "raw_value": UNKNOWN,
            },
        }
        for axis in ("z", "y", "x")
    }


def write_ct_metadata_response_fixture(
    *,
    repository_root: Path,
    specimen_id: str,
    ct_path: Path,
    shape: Sequence[int],
    dtype: str,
    dtype_string: str,
    byte_order: str,
    volume_format: str,
    retention: str,
    axes: str = UNKNOWN,
    array_axes: list[str] | str = UNKNOWN,
    voxel_spacing: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Write a closed authoritative response artifact with explicit metadata."""

    root = repository_root.resolve()
    resolved_ct = ct_path.resolve()
    ct_relative = resolved_ct.relative_to(root).as_posix()
    output_relative = Path(
        "analysis", specimen_id, "config", "ct_metadata_response.json"
    )
    call_receipt_relative = Path(
        "analysis",
        specimen_id,
        "config",
        "ct_metadata_mcp_call_receipt.json",
    )
    output = root / output_relative
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(resolved_ct.read_bytes()).hexdigest()
    normalized_shape = [int(value) for value in shape]
    normalized_dtype = np.dtype(dtype)
    count = int(np.prod(normalized_shape, dtype=np.int64))
    spacing = voxel_spacing or _unknown_spacing()
    result = {
        "status": "ok",
        "authoritative": True,
        "inspection_mode": "header_only",
        "method": "volume_metadata",
        "method_version": "1.0.0",
        "output_schema_version": "volume-metadata/1.0.0",
        "path": ct_relative,
        "sha256": digest,
        "file_bytes": resolved_ct.stat().st_size,
        "format": volume_format,
        "shape": normalized_shape,
        "ndim": len(normalized_shape),
        "dtype": normalized_dtype.name,
        "dtype_string": dtype_string,
        "byte_order": byte_order,
        "axes": axes,
        "voxel_count": count,
        "array_bytes": count * normalized_dtype.itemsize,
        "voxel_spacing": spacing,
        "statistics": {
            "status": "not_computed",
            "minimum": UNKNOWN,
            "maximum": UNKNOWN,
            "mean": UNKNOWN,
            "finite_count": UNKNOWN,
            "nonfinite_count": UNKNOWN,
        },
        "manifest_fragment": {
            "ct_volume": {
                "path": ct_relative,
                "sha256": digest,
                "role": "ct_volume",
                "retention": retention,
            },
            "ct_metadata": {
                "format": volume_format,
                "shape": normalized_shape,
                "dtype": normalized_dtype.name,
                "byte_order": byte_order,
                "array_axes": array_axes,
                "voxel_spacing": spacing,
            },
        },
    }
    summary = "Persisted authoritative header-only CT metadata response"
    evidence = {
        "schema_version": "volume-metadata-mcp-evidence/1.1.0",
        "response_schema_version": "part2-mcp-response/1.0.0",
        "tool": "inspect_volume_metadata",
        "status": "ok",
        "gate": "pass",
        "summary": summary,
        "request": {
            "input_filepath": ct_relative,
            "output_filepath": output_relative.as_posix(),
            "call_receipt_filepath": call_receipt_relative.as_posix(),
            "header_only": True,
            "include_sha256": True,
            "retention": retention,
        },
        "result": result,
        "warnings": [],
        "error": None,
    }
    output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    request = evidence["request"]
    header_facts = _actual_header_facts(resolved_ct)
    call_receipt_base = {
        "schema_version": "volume-metadata-mcp-call-receipt/1.0.0",
        "response_schema_version": "part2-mcp-response/1.0.0",
        "tool": "inspect_volume_metadata",
        "status": "ok",
        "gate": "pass",
        "summary": summary,
        "request": request,
        "header_facts": header_facts,
        "artifacts": {
            "metadata_response": {
                "path": output_relative.as_posix(),
                "sha256": evidence_sha256,
                "role": "ct_metadata_mcp_response",
                "retention": "committed",
            }
        },
        "hashes": {
            "input_sha256": digest,
            "request_sha256": _canonical_sha256(request),
            "result_sha256": _canonical_sha256(result),
            "header_facts_sha256": _canonical_sha256(header_facts),
            "metadata_response_sha256": evidence_sha256,
        },
        "warnings": [],
        "error": None,
    }
    call_receipt = {
        **call_receipt_base,
        "canonical_call_receipt_sha256": _canonical_sha256(call_receipt_base),
    }
    call_receipt_path = root / call_receipt_relative
    call_receipt_path.write_text(
        json.dumps(call_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output, evidence_sha256
