"""Deterministic intake core for new lattice specimens.

This module validates and associates scientist-supplied inputs.  CT inspection
is an MCP-boundary operation: intake consumes only a persisted, hash-bound
``inspect_volume_metadata`` response.  It intentionally does not run volume
inspection, segmentation, registration, local recentering, or defect labeling.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Iterable

import numpy as np
import trimesh

from .contracts import (
    DEFAULT_SCHEMA,
    ManifestValidationError,
    canonical_json_sha256,
    sha256_file,
    validate_manifest,
)
METHOD_NAME = "specimen_ingest"
METHOD_VERSION = "1.3.0"
REQUEST_SCHEMA_VERSION = "ingest-request/1.3.0"
RECEIPT_SCHEMA_VERSION = "ingest-receipt/1.3.0"
VOLUME_METADATA_EVIDENCE_SCHEMA_VERSION = (
    "volume-metadata-mcp-evidence/1.1.0"
)
VOLUME_METADATA_CALL_RECEIPT_SCHEMA_VERSION = (
    "volume-metadata-mcp-call-receipt/1.0.0"
)
MCP_RESPONSE_SCHEMA_VERSION = "part2-mcp-response/1.0.0"
UNKNOWN = "unknown"
SPECIMEN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REGISTRATION_MODES = {"autonomous_v2"}
ANALYSIS_SCOPES = {"roi_screening", "direct_metrology"}


class SpecimenIngestError(ValueError):
    """Raised when intake cannot produce a trustworthy provisional manifest."""


def _strict_object(
    value: Any,
    expected_keys: set[str],
    *,
    field: str,
    optional_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecimenIngestError(f"{field} must be an object")
    optional = optional_keys or set()
    missing = sorted(expected_keys - set(value))
    extra = sorted(set(value) - expected_keys - optional)
    if missing or extra:
        raise SpecimenIngestError(
            f"{field} has missing keys {missing} and unexpected keys {extra}"
        )
    return value


def _sha256_value(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SpecimenIngestError(
            f"{field} must be a lowercase SHA-256 hex digest"
        )
    return value


def _validate_spacing(value: Any, *, field: str) -> dict[str, Any]:
    spacing = _strict_object(value, {"z", "y", "x"}, field=field)
    for axis in ("z", "y", "x"):
        record = _strict_object(
            spacing[axis],
            {"value", "unit", "provenance"},
            field=f"{field}.{axis}",
        )
        scalar = record["value"]
        if scalar != UNKNOWN and (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(float(scalar))
            or float(scalar) <= 0.0
        ):
            raise SpecimenIngestError(
                f"{field}.{axis}.value must be positive finite or 'unknown'"
            )
        if not isinstance(record["unit"], str) or not record["unit"]:
            raise SpecimenIngestError(f"{field}.{axis}.unit must be non-empty")
        provenance = _strict_object(
            record["provenance"],
            {"source", "field", "raw_value"},
            optional_keys={"resolution_unit"},
            field=f"{field}.{axis}.provenance",
        )
        if any(
            not isinstance(provenance[name], (str, int, float))
            for name in ("source", "field", "raw_value")
        ):
            raise SpecimenIngestError(
                f"{field}.{axis}.provenance contains an unsupported value"
            )
    return spacing


def validate_ct_metadata_mcp_evidence(
    *,
    response_path: Path,
    expected_response_sha256: str,
    call_receipt_path: Path,
    expected_call_receipt_sha256: str,
    resolved_ct: Path,
    ct_relative_path: str,
    repository_root: Path,
    specimen_id: str,
    retention: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Validate the persisted evidence and full MCP call receipt as one chain.

    The call receipt is the repository's explicit MCP trust boundary.  Its
    hashes provide deterministic integrity and lineage, not cryptographic
    authentication of the local server process.
    """

    root = repository_root.resolve()
    candidate = response_path.expanduser()
    resolved_response = (
        (root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    expected_relative = Path(
        "analysis", specimen_id, "config", "ct_metadata_response.json"
    )
    expected_path = (root / expected_relative).resolve()
    try:
        expected_path.relative_to(root)
    except ValueError as exc:
        raise SpecimenIngestError(
            "CT metadata MCP response path escapes the repository"
        ) from exc
    if resolved_response != expected_path:
        raise SpecimenIngestError(
            "CT metadata MCP response must use the fixed specimen config path "
            f"{expected_relative.as_posix()}"
        )
    if not resolved_response.is_file():
        raise SpecimenIngestError(
            f"CT metadata MCP response does not exist: {expected_relative.as_posix()}"
        )
    expected_hash = _sha256_value(
        expected_response_sha256,
        field="ct_metadata_response_sha256",
    )
    actual_response_hash = sha256_file(resolved_response)
    if actual_response_hash != expected_hash:
        raise SpecimenIngestError(
            "CT metadata MCP response SHA-256 does not match the MCP artifact binding"
        )
    try:
        evidence = json.loads(resolved_response.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecimenIngestError(
            "CT metadata MCP response is not readable JSON"
        ) from exc
    evidence = _strict_object(
        evidence,
        {
            "schema_version",
            "response_schema_version",
            "tool",
            "status",
            "gate",
            "summary",
            "request",
            "result",
            "warnings",
            "error",
        },
        field="ct_metadata_response",
    )
    if evidence["schema_version"] != VOLUME_METADATA_EVIDENCE_SCHEMA_VERSION:
        raise SpecimenIngestError("CT metadata MCP evidence schema is incompatible")
    if evidence["response_schema_version"] != MCP_RESPONSE_SCHEMA_VERSION:
        raise SpecimenIngestError("CT metadata MCP response schema is incompatible")
    if (
        evidence["tool"] != "inspect_volume_metadata"
        or evidence["status"] != "ok"
        or evidence["gate"] != "pass"
        or evidence["error"] is not None
        or evidence["warnings"] != []
        or evidence["summary"]
        != "Persisted authoritative header-only CT metadata response"
    ):
        raise SpecimenIngestError(
            "CT metadata MCP response is not an authoritative successful result"
        )
    expected_call_relative = Path(
        "analysis", specimen_id, "config", "ct_metadata_mcp_call_receipt.json"
    )
    expected_call_path = (root / expected_call_relative).resolve()
    try:
        expected_call_path.relative_to(root)
    except ValueError as exc:
        raise SpecimenIngestError(
            "CT metadata MCP call-receipt path escapes the repository"
        ) from exc
    call_candidate = call_receipt_path.expanduser()
    resolved_call_receipt = (
        (root / call_candidate).resolve()
        if not call_candidate.is_absolute()
        else call_candidate.resolve()
    )
    if resolved_call_receipt != expected_call_path:
        raise SpecimenIngestError(
            "CT metadata MCP call receipt must use the fixed specimen config path "
            f"{expected_call_relative.as_posix()}"
        )
    if not resolved_call_receipt.is_file():
        raise SpecimenIngestError(
            "CT metadata MCP call receipt does not exist: "
            f"{expected_call_relative.as_posix()}"
        )
    expected_call_hash = _sha256_value(
        expected_call_receipt_sha256,
        field="ct_metadata_call_receipt_sha256",
    )
    actual_call_hash = sha256_file(resolved_call_receipt)
    if actual_call_hash != expected_call_hash:
        raise SpecimenIngestError(
            "CT metadata MCP call-receipt SHA-256 does not match its artifact binding"
        )

    request = _strict_object(
        evidence["request"],
        {
            "input_filepath",
            "output_filepath",
            "call_receipt_filepath",
            "header_only",
            "include_sha256",
            "retention",
        },
        field="ct_metadata_response.request",
    )
    if request != {
        "input_filepath": ct_relative_path,
        "output_filepath": expected_relative.as_posix(),
        "call_receipt_filepath": expected_call_relative.as_posix(),
        "header_only": True,
        "include_sha256": True,
        "retention": retention,
    }:
        raise SpecimenIngestError(
            "CT metadata MCP request binding differs from the intake request"
        )
    result = _strict_object(
        evidence["result"],
        {
            "status",
            "authoritative",
            "inspection_mode",
            "method",
            "method_version",
            "output_schema_version",
            "path",
            "sha256",
            "file_bytes",
            "format",
            "shape",
            "ndim",
            "dtype",
            "dtype_string",
            "byte_order",
            "axes",
            "voxel_count",
            "array_bytes",
            "voxel_spacing",
            "statistics",
            "manifest_fragment",
        },
        field="ct_metadata_response.result",
    )
    if (
        result["status"] != "ok"
        or result["authoritative"] is not True
        or result["inspection_mode"] != "header_only"
        or result["method"] != "volume_metadata"
        or result["method_version"] != "1.0.0"
        or result["output_schema_version"] != "volume-metadata/1.0.0"
        or result["path"] != ct_relative_path
    ):
        raise SpecimenIngestError(
            "CT metadata result identity, mode, or source path is invalid"
        )
    actual_ct_hash = sha256_file(resolved_ct)
    if _sha256_value(result["sha256"], field="result.sha256") != actual_ct_hash:
        raise SpecimenIngestError(
            "CT metadata result SHA-256 does not match the exact CT file"
        )
    shape = result["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or result["ndim"] != 3
    ):
        raise SpecimenIngestError(
            "CT input must be 3D with 3 positive dimensions in MCP metadata"
        )
    if type(result["file_bytes"]) is not int or result["file_bytes"] != resolved_ct.stat().st_size:
        raise SpecimenIngestError("CT metadata file_bytes differs from the CT file")
    try:
        dtype = np.dtype(result["dtype"])
        dtype_string = np.dtype(result["dtype_string"])
    except (TypeError, ValueError) as exc:
        raise SpecimenIngestError("CT metadata dtype is invalid") from exc
    if (
        dtype.fields
        or dtype.subdtype
        or dtype.kind not in "buif"
        or dtype.name != dtype_string.name
    ):
        raise SpecimenIngestError("CT metadata dtype is unsupported or inconsistent")
    voxel_count = int(np.prod(shape, dtype=np.int64))
    if (
        result["voxel_count"] != voxel_count
        or result["array_bytes"] != voxel_count * dtype.itemsize
    ):
        raise SpecimenIngestError("CT metadata voxel or byte counts are inconsistent")
    suffix_format = {
        ".npy": "npy",
        ".tif": "tiff",
        ".tiff": "tiff",
    }.get(resolved_ct.suffix.lower())
    if result["format"] != suffix_format:
        raise SpecimenIngestError("CT metadata format differs from the CT suffix")
    if result["byte_order"] not in {"little", "big", "not_applicable"}:
        raise SpecimenIngestError("CT metadata byte order is invalid")
    _validate_spacing(result["voxel_spacing"], field="result.voxel_spacing")
    statistics = _strict_object(
        result["statistics"],
        {
            "status",
            "minimum",
            "maximum",
            "mean",
            "finite_count",
            "nonfinite_count",
        },
        field="result.statistics",
    )
    if statistics != {
        "status": "not_computed",
        "minimum": UNKNOWN,
        "maximum": UNKNOWN,
        "mean": UNKNOWN,
        "finite_count": UNKNOWN,
        "nonfinite_count": UNKNOWN,
    }:
        raise SpecimenIngestError(
            "CT metadata evidence must be header-only without voxel statistics"
        )
    fragment = _strict_object(
        result["manifest_fragment"],
        {"ct_volume", "ct_metadata"},
        field="result.manifest_fragment",
    )
    expected_volume = {
        "path": ct_relative_path,
        "sha256": actual_ct_hash,
        "role": "ct_volume",
        "retention": retention,
    }
    if _strict_object(
        fragment["ct_volume"],
        {"path", "sha256", "role", "retention"},
        field="result.manifest_fragment.ct_volume",
    ) != expected_volume:
        raise SpecimenIngestError("CT manifest fragment volume binding is invalid")
    metadata = _strict_object(
        fragment["ct_metadata"],
        {"format", "shape", "dtype", "byte_order", "array_axes", "voxel_spacing"},
        field="result.manifest_fragment.ct_metadata",
    )
    expected_metadata = {
        "format": result["format"],
        "shape": shape,
        "dtype": result["dtype"],
        "byte_order": result["byte_order"],
        "array_axes": metadata["array_axes"],
        "voxel_spacing": result["voxel_spacing"],
    }
    if metadata != expected_metadata:
        raise SpecimenIngestError("CT manifest fragment metadata is inconsistent")
    axes = metadata["array_axes"]
    if axes != UNKNOWN and (
        not isinstance(axes, list)
        or len(axes) != 3
        or any(axis not in {"x", "y", "z"} for axis in axes)
        or len(set(axes)) != 3
    ):
        raise SpecimenIngestError("CT manifest fragment array_axes is invalid")

    try:
        call_receipt = json.loads(
            resolved_call_receipt.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecimenIngestError(
            "CT metadata MCP call receipt is not readable JSON"
        ) from exc
    call_receipt = _strict_object(
        call_receipt,
        {
            "schema_version",
            "response_schema_version",
            "tool",
            "status",
            "gate",
            "summary",
            "request",
            "header_facts",
            "artifacts",
            "hashes",
            "warnings",
            "error",
            "canonical_call_receipt_sha256",
        },
        field="ct_metadata_mcp_call_receipt",
    )
    call_receipt_base = {
        key: value
        for key, value in call_receipt.items()
        if key != "canonical_call_receipt_sha256"
    }
    if (
        call_receipt["schema_version"]
        != VOLUME_METADATA_CALL_RECEIPT_SCHEMA_VERSION
        or call_receipt["response_schema_version"]
        != MCP_RESPONSE_SCHEMA_VERSION
        or call_receipt["tool"] != "inspect_volume_metadata"
        or call_receipt["status"] != evidence["status"]
        or call_receipt["gate"] != evidence["gate"]
        or call_receipt["summary"] != evidence["summary"]
        or call_receipt["request"] != request
        or call_receipt["warnings"] != evidence["warnings"]
        or call_receipt["error"] is not None
        or call_receipt["canonical_call_receipt_sha256"]
        != canonical_json_sha256(call_receipt_base)
    ):
        raise SpecimenIngestError(
            "CT metadata MCP call receipt is stale, incompatible, or has an invalid self-hash"
        )
    header_fields = {
        "file_bytes",
        "format",
        "shape",
        "ndim",
        "dtype",
        "dtype_string",
        "byte_order",
        "axes",
        "voxel_count",
        "array_bytes",
    }
    header_facts = _strict_object(
        call_receipt["header_facts"],
        header_fields,
        field="ct_metadata_mcp_call_receipt.header_facts",
    )
    expected_header_facts = {field: result[field] for field in header_fields}
    if header_facts != expected_header_facts:
        raise SpecimenIngestError(
            "CT metadata MCP call-receipt header facts differ from persisted evidence"
        )
    call_artifacts = _strict_object(
        call_receipt["artifacts"],
        {"metadata_response"},
        field="ct_metadata_mcp_call_receipt.artifacts",
    )
    metadata_binding = _strict_object(
        call_artifacts["metadata_response"],
        {"path", "sha256", "role", "retention"},
        field="ct_metadata_mcp_call_receipt.artifacts.metadata_response",
    )
    if metadata_binding != {
        "path": expected_relative.as_posix(),
        "sha256": actual_response_hash,
        "role": "ct_metadata_mcp_response",
        "retention": "committed",
    }:
        raise SpecimenIngestError(
            "CT metadata MCP call receipt does not bind the exact evidence artifact"
        )
    call_hashes = _strict_object(
        call_receipt["hashes"],
        {
            "input_sha256",
            "request_sha256",
            "result_sha256",
            "header_facts_sha256",
            "metadata_response_sha256",
        },
        field="ct_metadata_mcp_call_receipt.hashes",
    )
    expected_call_hashes = {
        "input_sha256": actual_ct_hash,
        "request_sha256": canonical_json_sha256(request),
        "result_sha256": canonical_json_sha256(result),
        "header_facts_sha256": canonical_json_sha256(header_facts),
        "metadata_response_sha256": actual_response_hash,
    }
    if call_hashes != expected_call_hashes:
        raise SpecimenIngestError(
            "CT metadata MCP call receipt hash bindings are stale or inconsistent"
        )
    return result, {
        "path": expected_relative.as_posix(),
        "sha256": actual_response_hash,
    }, {
        "path": expected_call_relative.as_posix(),
        "sha256": actual_call_hash,
        "canonical_sha256": call_receipt["canonical_call_receipt_sha256"],
    }


def _resolve_input(
    path: Path, *, repository_root: Path, allowed_roots: Iterable[Path]
) -> tuple[Path, str]:
    root = repository_root.expanduser().resolve()
    candidate = path.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SpecimenIngestError(
            f"Input path escapes repository root {root}: {resolved}"
        ) from exc
    allowed = [
        (
            (root / item.expanduser()).resolve()
            if not item.expanduser().is_absolute()
            else item.expanduser().resolve()
        )
        for item in allowed_roots
    ]
    if not any(resolved == item or item in resolved.parents for item in allowed):
        raise SpecimenIngestError(
            f"Input path is outside configured data roots: {relative.as_posix()}"
        )
    if not resolved.is_file():
        raise SpecimenIngestError(f"Input file does not exist: {relative.as_posix()}")
    return resolved, relative.as_posix()


def _load_graph(path: Path) -> dict[str, Any]:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecimenIngestError(f"Unreadable lattice graph {path}: {exc}") from exc
    if not isinstance(graph, dict):
        raise SpecimenIngestError(f"Lattice graph must be a JSON object: {path}")
    return graph


def _unique_integer_ids(items: Any, *, section: str, path: Path) -> set[int]:
    if not isinstance(items, list) or not items:
        raise SpecimenIngestError(f"{path}: {section} must be a non-empty array")
    identifiers: list[int] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise SpecimenIngestError(
                f"{path}: {section}[{index}].id must be an integer"
            )
        identifiers.append(item["id"])
    if len(identifiers) != len(set(identifiers)):
        raise SpecimenIngestError(f"{path}: {section} contains duplicate IDs")
    return set(identifiers)


def inspect_lattice_graph(
    path: Path,
    *,
    repository_root: Path,
    allowed_roots: Iterable[Path],
) -> dict[str, Any]:
    """Validate graph identifiers/references and return its canonical topology."""
    resolved, relative_path = _resolve_input(
        path, repository_root=repository_root, allowed_roots=allowed_roots
    )
    if resolved.suffix.lower() != ".json":
        raise SpecimenIngestError(f"Expected a .json lattice graph: {relative_path}")
    graph = _load_graph(resolved)
    required = {"junctions", "struts", "unit_cells"}
    missing = sorted(required - set(graph))
    if missing:
        raise SpecimenIngestError(
            f"{relative_path}: missing graph keys: {', '.join(missing)}"
        )
    junction_ids = _unique_integer_ids(
        graph["junctions"], section="junctions", path=resolved
    )
    strut_ids = _unique_integer_ids(graph["struts"], section="struts", path=resolved)
    unit_cell_ids = _unique_integer_ids(
        graph["unit_cells"], section="unit_cells", path=resolved
    )

    for index, junction in enumerate(graph["junctions"]):
        position = junction.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in position
            )
        ):
            raise SpecimenIngestError(
                f"{relative_path}: junctions[{index}].position must be 3 finite numbers"
            )

    topology_struts: list[list[int]] = []
    for index, strut in enumerate(graph["struts"]):
        endpoints = (strut.get("junction0"), strut.get("junction1"))
        if not all(isinstance(value, int) for value in endpoints):
            raise SpecimenIngestError(
                f"{relative_path}: struts[{index}] endpoints must be integer IDs"
            )
        if endpoints[0] == endpoints[1]:
            raise SpecimenIngestError(
                f"{relative_path}: struts[{index}] is a self-loop"
            )
        unknown = sorted(set(endpoints) - junction_ids)
        if unknown:
            raise SpecimenIngestError(
                f"{relative_path}: struts[{index}] references unknown junctions {unknown}"
            )
        topology_struts.append(
            [strut["id"], min(endpoints), max(endpoints)]
        )

    topology_cells: list[list[Any]] = []
    for index, unit_cell in enumerate(graph["unit_cells"]):
        members = unit_cell.get("struts")
        if not isinstance(members, list) or not all(
            isinstance(value, int) for value in members
        ):
            raise SpecimenIngestError(
                f"{relative_path}: unit_cells[{index}].struts must contain integer IDs"
            )
        if len(members) != len(set(members)):
            raise SpecimenIngestError(
                f"{relative_path}: unit_cells[{index}].struts contains duplicate IDs"
            )
        unknown = sorted(set(members) - strut_ids)
        if unknown:
            raise SpecimenIngestError(
                f"{relative_path}: unit_cells[{index}] references unknown struts {unknown}"
            )
        topology_cells.append([unit_cell["id"], sorted(members)])

    topology = {
        "junction_ids": sorted(junction_ids),
        "struts": sorted(topology_struts),
        "unit_cells": sorted(topology_cells),
    }
    return {
        "method": "canonical_lattice_topology",
        "method_version": "1.0.0",
        "path": relative_path,
        "sha256": sha256_file(resolved),
        "junction_count": len(junction_ids),
        "strut_count": len(strut_ids),
        "unit_cell_count": len(unit_cell_ids),
        "topology_sha256": canonical_json_sha256(topology),
        "id_reference_integrity": True,
        "extra_top_level_keys": sorted(set(graph) - required),
    }


def inspect_cad_stl(
    path: Path,
    *,
    repository_root: Path,
    allowed_roots: Iterable[Path],
    units: str = UNKNOWN,
    units_provenance: str = UNKNOWN,
) -> dict[str, Any]:
    """Verify a supplied STL and record bounds without processing other inputs."""
    resolved, relative_path = _resolve_input(
        path, repository_root=repository_root, allowed_roots=allowed_roots
    )
    if resolved.suffix.lower() != ".stl":
        raise SpecimenIngestError(f"Expected an .stl CAD file: {relative_path}")
    file_size = resolved.stat().st_size
    face_count: int
    vertex_count: int
    bounds: np.ndarray[Any, Any]
    method: str
    with resolved.open("rb") as stream:
        header = stream.read(84)
    binary_face_count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else 0
    if binary_face_count > 0 and file_size == 84 + binary_face_count * 50:
        record_dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        records = np.memmap(
            resolved,
            dtype=record_dtype,
            mode="r",
            offset=84,
            shape=(binary_face_count,),
        )
        minimum = np.full(3, np.inf, dtype=np.float64)
        maximum = np.full(3, -np.inf, dtype=np.float64)
        for start in range(0, binary_face_count, 250_000):
            vertices = np.asarray(
                records[start : start + 250_000]["vertices"], dtype=np.float64
            )
            if not np.all(np.isfinite(vertices)):
                raise SpecimenIngestError(
                    f"STL contains non-finite vertices: {relative_path}"
                )
            minimum = np.minimum(minimum, np.min(vertices, axis=(0, 1)))
            maximum = np.maximum(maximum, np.max(vertices, axis=(0, 1)))
        bounds = np.stack((minimum, maximum))
        face_count = binary_face_count
        vertex_count = binary_face_count * 3
        method = "binary_stl_stream"
    else:
        try:
            mesh = trimesh.load_mesh(resolved, file_type="stl", process=False)
        except Exception as exc:
            raise SpecimenIngestError(
                f"Unreadable STL {relative_path}: {exc}"
            ) from exc
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise SpecimenIngestError(
                f"STL must contain one non-empty triangle mesh: {relative_path}"
            )
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        face_count = int(len(mesh.faces))
        vertex_count = int(len(mesh.vertices))
        method = "trimesh_ascii_stl"
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise SpecimenIngestError(f"STL has invalid bounds: {relative_path}")
    return {
        "method": method,
        "method_version": "1.0.0",
        "path": relative_path,
        "sha256": sha256_file(resolved),
        "format": "stl",
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds": {
            "minimum": [float(value) for value in bounds[0]],
            "maximum": [float(value) for value in bounds[1]],
        },
        "units": units,
        "units_provenance": units_provenance,
        "readable": True,
    }


def _axes(
    value: str,
    expected: str,
    *,
    resolve_unknown_to_expected: bool = False,
) -> list[str] | str:
    normalized = value.lower()
    if normalized == UNKNOWN:
        return list(expected) if resolve_unknown_to_expected else UNKNOWN
    if normalized != expected:
        raise SpecimenIngestError(
            f"Expected axes {expected!r} or 'unknown', found {value!r}"
        )
    return list(normalized)


def _default_analysis_parameters(
    *,
    requested_analysis_scope: str,
    registration_mode: str,
    ct_dtype: str,
    graph_axes: list[str] | str,
    array_axes: list[str] | str,
    aligned_graph_units: str,
) -> dict[str, Any]:
    return {
        "requested_analysis_scope": requested_analysis_scope,
        "registration": {
            "mode": registration_mode,
            "local_recenter_required": True,
        },
        "coordinates": {
            "graph_axes": graph_axes,
            "array_axes": array_axes,
            "numpy_index_expression": (
                "volume[round(z), round(y), round(x)]"
                if array_axes == ["z", "y", "x"]
                else UNKNOWN
            ),
            "aligned_graph_units": aligned_graph_units,
        },
        "segmentation": {
            "method": "exact_histogram_otsu",
            "method_version": "2.0.0",
            "comparison": "value >= threshold",
            "histogram_bins": 65536,
            "histogram_encoding": (
                "native_uint16"
                if ct_dtype == "uint16"
                else "full_volume_affine_uint16"
            ),
            "edge_slices_excluded": 0,
            "chunk_depth": 8,
            "coarse_bins": 1024,
            "peak_smoothing_sigma_bins": 2.0,
            "peak_prominence_fraction": 0.003,
            "minimum_significant_peaks": 2,
            "minimum_foreground_fraction": 0.01,
            "maximum_foreground_fraction": 0.35,
            "minimum_otsu_separability": 0.45,
            "minimum_class_mean_separation_sigma": 0.75,
        },
        "localization_policy": {
            "schema_version": "stage2-localization-policy/1.1.0",
            "patch_radius_voxels": 10,
            "search_radius_voxels": 8.0,
            "maximum_shift_voxels": 8.0,
            "smoothing_sigma_voxels": 1.25,
            "mean_shift_bandwidth_voxels": 4.0,
            "mean_shift_max_iterations": 12,
            "mean_shift_tolerance_voxels": 0.05,
            "seed_perturbation_voxels": 2.0,
            "seed_cluster_radius_voxels": 2.0,
            "minimum_seed_consensus_fraction": 0.7,
            "minimum_candidate_support": 0.05,
            "minimum_relative_support_improvement": 0.01,
            "incident_sample_distances_voxels": [3.0, 5.0, 7.0],
            "core_support_weight": 0.7,
            "incident_support_weight": 0.3,
            "minimum_primary_or_stable_coarse_fraction": 0.95,
            "maximum_fallback_fraction": 0.05,
            "maximum_ambiguous_fraction": 0.05,
            "maximum_rejected_fraction": 0.0,
            "maximum_boundary_limited_fraction": 0.0,
        },
        "qa_policy": {
            "schema_version": "stage2-qa-policy/1.1.0",
            "junction_patch_radius_voxels": 2,
            "corridor_axial_samples": 9,
            "corridor_radius_voxels": 6.0,
            "corridor_angular_samples": 8,
            "roi_padding_fraction": 0.2,
            "spatial_bins_per_axis": 5,
            "radial_foreground_probability": 0.5,
            "minimum_mean_junction_foreground_fraction": 0.85,
            "minimum_median_corridor_foreground_fraction": 0.08,
            "maximum_spatial_bin_median_range": 0.25,
            "minimum_roi_in_bounds_fraction": 0.99,
            "maximum_uncertainty_to_radius_ratio": 1.0,
        },
        "budgets": {
            "local_recenter_radius_voxels": 8.0,
            "roi_padding_fraction": 0.2,
            "metrology_uncertainty_voxels": 2.0,
            "maximum_agent_retries": 2,
        },
        "artifact_schema_versions": {
            "specimen_manifest": "2.1.0",
            "node_localization": "1.2.0",
            "registration_qa": "1.2.0",
            "per_strut_metrics": "1.0.0",
            "classified_struts": "1.0.0",
            "nde_report": "1.0.0",
        },
    }


def _artifact(path: str, digest: str, role: str, retention: str) -> dict[str, str]:
    return {
        "path": path,
        "sha256": digest,
        "role": role,
        "retention": retention,
    }


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


def ingest_specimen(
    *,
    repository_root: Path,
    specimen_id: str,
    design_id: str,
    requested_analysis_scope: str,
    cad_path: Path | None = None,
    design_graph_path: Path,
    ct_path: Path,
    ct_metadata_response_path: Path,
    ct_metadata_response_sha256: str,
    ct_metadata_call_receipt_path: Path,
    ct_metadata_call_receipt_sha256: str,
    registration_mode: str,
    association_confirmed: bool,
    allowed_data_roots: Iterable[Path] | None = None,
    aligned_graph_path: Path | None = None,
    design_transform_declaration_path: Path | None = None,
    cad_units: str = UNKNOWN,
    cad_units_provenance: str = UNKNOWN,
    graph_axes: str = "xyz",
    array_axes: str = UNKNOWN,
    aligned_graph_units: str = UNKNOWN,
    retention: str = "committed",
    schema_path: Path = DEFAULT_SCHEMA,
    normalized_graph_path: Path | None = None,
    normalized_graph_sha256: str | None = None,
) -> dict[str, Any]:
    """Inspect explicit inputs and write idempotent intake artifacts."""
    root = repository_root.expanduser().resolve()
    if not SPECIMEN_ID_PATTERN.fullmatch(specimen_id):
        raise SpecimenIngestError(f"Invalid specimen_id: {specimen_id!r}")
    if not SPECIMEN_ID_PATTERN.fullmatch(design_id):
        raise SpecimenIngestError(f"Invalid design_id: {design_id!r}")
    if requested_analysis_scope not in ANALYSIS_SCOPES:
        raise SpecimenIngestError(
            f"Unsupported requested_analysis_scope: {requested_analysis_scope!r}"
        )
    if registration_mode not in REGISTRATION_MODES:
        raise SpecimenIngestError(
            f"Unsupported registration mode: {registration_mode!r}"
        )
    if not association_confirmed:
        raise SpecimenIngestError(
            "Scientist must explicitly confirm the nominal-graph/CT association"
        )
    if retention not in {"committed", "external", "regenerable"}:
        raise SpecimenIngestError(f"Unsupported retention policy: {retention!r}")
    data_roots = list(allowed_data_roots or [root / "data"])

    design = inspect_lattice_graph(
        design_graph_path, repository_root=root, allowed_roots=data_roots
    )
    cad = (
        inspect_cad_stl(
            cad_path,
            repository_root=root,
            allowed_roots=data_roots,
            units=cad_units,
            units_provenance=cad_units_provenance,
        )
        if cad_path is not None
        else None
    )
    normalized_graph: dict[str, Any] | None = None
    if normalized_graph_path is not None:
        resolved_normalized, normalized_relative = _resolve_input(
            normalized_graph_path,
            repository_root=root,
            allowed_roots=[root / "analysis"],
        )
        if resolved_normalized.suffix.lower() != ".npz":
            raise SpecimenIngestError("Normalized nominal graph must be an .npz artifact")
        actual_normalized_sha256 = sha256_file(resolved_normalized)
        if normalized_graph_sha256 != actual_normalized_sha256:
            raise SpecimenIngestError("Normalized nominal graph SHA-256 mismatch")
        normalized_graph = _artifact(
            normalized_relative,
            actual_normalized_sha256,
            "normalized_nominal_graph",
            retention,
        )
    elif normalized_graph_sha256 is not None:
        raise SpecimenIngestError(
            "normalized_graph_sha256 requires normalized_graph_path"
        )
    resolved_ct, ct_relative = _resolve_input(
        ct_path, repository_root=root, allowed_roots=data_roots
    )
    ct, ct_metadata_response, ct_metadata_call_receipt = (
        validate_ct_metadata_mcp_evidence(
            response_path=ct_metadata_response_path,
            expected_response_sha256=ct_metadata_response_sha256,
            call_receipt_path=ct_metadata_call_receipt_path,
            expected_call_receipt_sha256=ct_metadata_call_receipt_sha256,
            resolved_ct=resolved_ct,
            ct_relative_path=ct_relative,
            repository_root=root,
            specimen_id=specimen_id,
            retention=retention,
        )
    )

    aligned: dict[str, Any] | None = None
    if aligned_graph_path is not None:
        aligned = inspect_lattice_graph(
            aligned_graph_path, repository_root=root, allowed_roots=data_roots
        )
        if aligned["topology_sha256"] != design["topology_sha256"]:
            raise SpecimenIngestError(
                "Nominal and aligned graphs have different canonical topology"
            )
    if registration_mode == "challenge_aligned_json" and aligned is None:
        raise SpecimenIngestError(
            "challenge_aligned_json requires a scientist-supplied aligned graph"
        )
    if registration_mode == "autonomous_v2" and aligned is not None:
        raise SpecimenIngestError(
            "autonomous_v2 provisional intake must not accept a precomputed aligned graph"
        )

    transform_declaration: dict[str, Any] | None = None
    transform_declaration_verification: dict[str, Any] | None = None
    if design_transform_declaration_path is not None:
        raise SpecimenIngestError(
            "Production intake accepts only a nominal graph JSON and specimen CT; "
            "graph-to-CAD transform declarations are not supported"
        )

    # Nominal lattice JSON positions use the repository's canonical XYZ
    # component order. Preserve explicit validation, but do not force a manual
    # review when a caller leaves this canonical convention unspecified.
    declared_graph_axes = _axes(
        graph_axes,
        "xyz",
        resolve_unknown_to_expected=True,
    )
    declared_array_axes = _axes(array_axes, "zyx")
    observed_array_axes = ct["manifest_fragment"]["ct_metadata"]["array_axes"]
    if declared_array_axes == UNKNOWN and observed_array_axes == ["z", "y", "x"]:
        declared_array_axes = list(observed_array_axes)
    if (
        observed_array_axes != UNKNOWN
        and declared_array_axes != UNKNOWN
        and observed_array_axes != declared_array_axes
    ):
        raise SpecimenIngestError(
            "Scientist-declared CT array axes differ from the MCP metadata response"
        )
    if aligned_graph_units not in {"voxel", "simulation_voxel", UNKNOWN}:
        raise SpecimenIngestError(
            f"Unsupported aligned graph units: {aligned_graph_units!r}"
        )
    unresolved_fields: list[str] = []
    declared_fields = [
        ("analysis_parameters.coordinates.graph_axes", declared_graph_axes),
        ("analysis_parameters.coordinates.array_axes", declared_array_axes),
    ]
    if aligned is not None:
        declared_fields.append(
            ("analysis_parameters.coordinates.aligned_graph_units", aligned_graph_units)
        )
    for field, value in declared_fields:
        if value == UNKNOWN:
            unresolved_fields.append(field)

    lifecycle_state = (
        "provisional" if unresolved_fields else "ready_for_data_prep"
    )
    analysis_parameters = _default_analysis_parameters(
        requested_analysis_scope=requested_analysis_scope,
        registration_mode=registration_mode,
        ct_dtype=ct["dtype"],
        graph_axes=declared_graph_axes,
        array_axes=declared_array_axes,
        aligned_graph_units=aligned_graph_units,
    )
    config_hash = canonical_json_sha256(analysis_parameters)

    inputs: dict[str, Any] = {
        "ct": _artifact(ct["path"], ct["sha256"], "ct_volume", retention),
        "ct_metadata": {
            "format": ct["format"],
            "shape": ct["shape"],
            "dtype": ct["dtype"],
            "byte_order": ct["byte_order"],
            "array_axes": declared_array_axes,
            "voxel_spacing": ct["voxel_spacing"],
        },
        "design_graph": _artifact(
            design["path"], design["sha256"], "design_graph", retention
        ),
    }
    if cad is not None:
        inputs["cad"] = _artifact(cad["path"], cad["sha256"], "cad", retention)
    if normalized_graph is not None:
        inputs["normalized_nominal_graph"] = normalized_graph
    graph_input_hashes = [design["sha256"]]
    if aligned is not None:
        inputs["aligned_graph"] = _artifact(
            aligned["path"], aligned["sha256"], "aligned_graph", retention
        )
        graph_input_hashes = sorted(
            set(graph_input_hashes) | {aligned["sha256"]}
        )
    if transform_declaration is not None:
        inputs["design_transform_declaration"] = transform_declaration

    graph_values = {
        key: design[key]
        for key in (
            "junction_count",
            "strut_count",
            "unit_cell_count",
            "topology_sha256",
        )
    }
    graph_summary: dict[str, Any] = {
        "method": design["method"],
        "method_version": design["method_version"],
        "provenance": {
            "source": "scientist-supplied graph schema and reference inspection",
            "input_sha256": graph_input_hashes,
            "config_sha256": config_hash,
        },
        "values": graph_values,
    }
    if aligned is not None:
        graph_summary["aligned_values"] = {
            key: aligned[key]
            for key in (
                "junction_count",
                "strut_count",
                "unit_cell_count",
                "topology_sha256",
            )
        }

    manifest = {
        "schema_version": "2.1.0",
        "specimen_id": specimen_id,
        "design_id": design_id,
        "lifecycle_state": lifecycle_state,
        "unresolved_fields": sorted(unresolved_fields),
        "inputs": inputs,
        "intake": {
            "association": {
                "source": "scientist_explicit",
                "confirmed": True,
                "ct_to_specimen": True,
            },
            "registration_mode_selection": {
                "mode": registration_mode,
                "source": "scientist_explicit",
            },
            "graph_inspection": design,
            "volume_metadata": {
                "method": ct["method"],
                "method_version": ct["method_version"],
                "output_schema_version": ct["output_schema_version"],
            },
        },
        "analysis_parameters": analysis_parameters,
        "analysis_parameters_sha256": config_hash,
        "derived": {"graph_summary": graph_summary},
    }
    if cad is not None:
        manifest["intake"]["cad_inspection"] = cad
        manifest["intake"]["association"]["design_graph_to_cad"] = True

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "specimen_id": specimen_id,
        "design_id": design_id,
        "requested_analysis_scope": requested_analysis_scope,
        "paths": {
            "cad": cad["path"] if cad else None,
            "design_graph": design["path"],
            "ct": ct["path"],
            "ct_metadata_response": ct_metadata_response["path"],
            "ct_metadata_mcp_call_receipt": ct_metadata_call_receipt["path"],
            "aligned_graph": aligned["path"] if aligned else None,
            "design_transform_declaration": (
                transform_declaration["path"] if transform_declaration else None
            ),
        },
        "registration_mode": registration_mode,
        "association_confirmed": True,
        "mcp_response_binding": {
            "tool": "inspect_volume_metadata",
            "response_schema_version": MCP_RESPONSE_SCHEMA_VERSION,
            "artifact_path": ct_metadata_response["path"],
            "artifact_sha256": ct_metadata_response["sha256"],
            "call_receipt_path": ct_metadata_call_receipt["path"],
            "call_receipt_sha256": ct_metadata_call_receipt["sha256"],
            "canonical_call_receipt_sha256": ct_metadata_call_receipt[
                "canonical_sha256"
            ],
            "ct_sha256": ct["sha256"],
            "authoritative": True,
            "header_only": True,
        },
        "declared": {
            "cad_units": cad_units,
            "cad_units_provenance": cad_units_provenance,
            "graph_axes": declared_graph_axes,
            "array_axes": declared_array_axes,
            "aligned_graph_units": aligned_graph_units,
            "retention": retention,
            "design_transform_declaration_verification": (
                transform_declaration_verification
            ),
        },
    }
    request_hash = canonical_json_sha256(request)
    manifest_hash = canonical_json_sha256(manifest)
    warnings = [
        f"{field} remains unknown" for field in sorted(unresolved_fields)
    ]
    if all(
        ct["voxel_spacing"][axis]["value"] == UNKNOWN for axis in ("z", "y", "x")
    ):
        warnings.append("CT voxel spacing is unavailable from file metadata")

    config_directory = root / "analysis" / specimen_id / "config"
    manifest_path = config_directory / "specimen_manifest.json"
    request_path = config_directory / "ingest_request.json"
    receipt_path = config_directory / "ingest_receipt.json"

    # Validate from a temporary manifest before replacing a prior valid intake.
    config_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=config_directory, suffix=".json", delete=False
    ) as stream:
        temporary_manifest = Path(stream.name)
        stream.write(_json_bytes(manifest))
    try:
        validate_manifest(
            temporary_manifest,
            schema_path=schema_path,
            repository_root=root,
            verify_files=False,
        )
    except (ManifestValidationError, OSError) as exc:
        raise SpecimenIngestError(f"Generated manifest failed validation: {exc}") from exc
    finally:
        temporary_manifest.unlink(missing_ok=True)

    receipt_base = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "specimen_id": specimen_id,
        "design_id": design_id,
        "requested_analysis_scope": requested_analysis_scope,
        "lifecycle_state": lifecycle_state,
        "input_sha256": {
            "design_graph": design["sha256"],
            "ct": ct["sha256"],
            "ct_metadata_response": ct_metadata_response["sha256"],
            "ct_metadata_mcp_call_receipt": ct_metadata_call_receipt["sha256"],
            **({"aligned_graph": aligned["sha256"]} if aligned else {}),
            **({"cad": cad["sha256"]} if cad else {}),
            **(
                {"normalized_nominal_graph": normalized_graph["sha256"]}
                if normalized_graph
                else {}
            ),
            **(
                {"design_transform_declaration": transform_declaration["sha256"]}
                if transform_declaration
                else {}
            ),
        },
        "request_sha256": request_hash,
        "manifest_sha256": manifest_hash,
        "design_transform_declaration_verification": (
            transform_declaration_verification
        ),
        "warnings": sorted(warnings),
        "unresolved_fields": sorted(unresolved_fields),
        "self_verification": {
            "association_explicit": True,
            "all_paths_repository_relative": True,
            "all_inputs_hashed": True,
            "ct_metadata_mcp_integrity_chain_valid": True,
            "ct_metadata_response_schema_closed": True,
            "ct_metadata_response_hash_bound": True,
            "ct_metadata_response_header_only": True,
            "ct_metadata_response_path_and_ct_hash_match": True,
            "ct_metadata_mcp_call_receipt_closed": True,
            "ct_metadata_mcp_call_receipt_hash_bound": True,
            "ct_metadata_header_facts_bound_to_call_receipt": True,
            "cad_readable": cad is None or cad["readable"] is True,
            "cad_readable_or_not_supplied": cad is None or cad["readable"] is True,
            "graph_id_reference_integrity": True,
            "normalized_graph_hash_bound": normalized_graph is not None or cad is not None,
            "manifest_schema_valid": True,
            "design_transform_declaration_valid": (
                transform_declaration_verification is not None
                if transform_declaration is not None
                else True
            ),
            "design_transform_declaration_hash_bound": (
                transform_declaration_verification is not None
                if transform_declaration is not None
                else True
            ),
            "segmentation_not_run": True,
            "registration_not_run": True,
            "defect_labels_not_derived": True,
        },
    }
    receipt = {
        **receipt_base,
        "canonical_receipt_sha256": canonical_json_sha256(receipt_base),
    }
    changed = {
        "ingest_request": _atomic_write_if_changed(request_path, request),
        "specimen_manifest": _atomic_write_if_changed(manifest_path, manifest),
        "ingest_receipt": _atomic_write_if_changed(receipt_path, receipt),
    }
    return {
        "specimen_id": specimen_id,
        "design_id": design_id,
        "requested_analysis_scope": requested_analysis_scope,
        "lifecycle_state": lifecycle_state,
        "paths": {
            "ingest_request": str(request_path),
            "specimen_manifest": str(manifest_path),
            "ingest_receipt": str(receipt_path),
            "ct_metadata_response": str(
                root / ct_metadata_response["path"]
            ),
            "ct_metadata_mcp_call_receipt": str(
                root / ct_metadata_call_receipt["path"]
            ),
        },
        "canonical_hashes": {
            "request": request_hash,
            "manifest": manifest_hash,
            "receipt": receipt["canonical_receipt_sha256"],
        },
        "changed": changed,
        "warnings": sorted(warnings),
        "unresolved_fields": sorted(unresolved_fields),
    }


def validate_ingest_artifact_bundle(
    *,
    repository_root: Path,
    manifest_path: Path,
    request_path: Path,
    receipt_path: Path,
    ct_metadata_response_path: Path,
    ct_metadata_call_receipt_path: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    expected_specimen_id: str | None = None,
    expected_design_id: str | None = None,
    expected_analysis_scope: str | None = None,
    expected_registration_mode: str | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Revalidate the complete Stage 0 intake chain without producing outputs."""

    root = repository_root.expanduser().resolve()

    def fixed_config_path(value: Path, name: str, specimen_id: str) -> Path:
        candidate = value.expanduser()
        resolved = (
            (root / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        expected = (root / "analysis" / specimen_id / "config" / name).resolve()
        try:
            resolved.relative_to(root)
            expected.relative_to(root)
        except ValueError as exc:
            raise SpecimenIngestError(
                f"Stage 0 config artifact escapes the repository: {name}"
            ) from exc
        if resolved != expected or not resolved.is_file():
            raise SpecimenIngestError(
                f"Stage 0 config artifact must exist at "
                f"analysis/{specimen_id}/config/{name}"
            )
        return resolved

    manifest_candidate = manifest_path.expanduser()
    resolved_manifest = (
        (root / manifest_candidate).resolve()
        if not manifest_candidate.is_absolute()
        else manifest_candidate.resolve()
    )
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecimenIngestError("Stage 0 specimen manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise SpecimenIngestError("Stage 0 specimen manifest must be an object")
    specimen_id = manifest.get("specimen_id")
    if not isinstance(specimen_id, str) or not SPECIMEN_ID_PATTERN.fullmatch(
        specimen_id
    ):
        raise SpecimenIngestError("Stage 0 specimen manifest has an invalid specimen_id")
    expected_manifest = fixed_config_path(
        resolved_manifest, "specimen_manifest.json", specimen_id
    )
    try:
        validate_manifest(
            expected_manifest,
            schema_path=schema_path,
            repository_root=root,
            verify_files=True,
        )
    except (ManifestValidationError, OSError) as exc:
        raise SpecimenIngestError(
            f"Stage 0 specimen manifest failed semantic validation: {exc}"
        ) from exc

    resolved_request = fixed_config_path(
        request_path, "ingest_request.json", specimen_id
    )
    resolved_receipt = fixed_config_path(
        receipt_path, "ingest_receipt.json", specimen_id
    )
    resolved_response = fixed_config_path(
        ct_metadata_response_path, "ct_metadata_response.json", specimen_id
    )
    resolved_call_receipt = fixed_config_path(
        ct_metadata_call_receipt_path,
        "ct_metadata_mcp_call_receipt.json",
        specimen_id,
    )
    try:
        request = json.loads(resolved_request.read_text(encoding="utf-8"))
        receipt = json.loads(resolved_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecimenIngestError("Stage 0 request or receipt is unreadable") from exc
    request = _strict_object(
        request,
        {
            "schema_version",
            "method",
            "method_version",
            "specimen_id",
            "design_id",
            "requested_analysis_scope",
            "paths",
            "registration_mode",
            "association_confirmed",
            "mcp_response_binding",
            "declared",
        },
        field="ingest_request",
    )
    receipt = _strict_object(
        receipt,
        {
            "schema_version",
            "method",
            "method_version",
            "specimen_id",
            "design_id",
            "requested_analysis_scope",
            "lifecycle_state",
            "input_sha256",
            "request_sha256",
            "manifest_sha256",
            "design_transform_declaration_verification",
            "warnings",
            "unresolved_fields",
            "self_verification",
            "canonical_receipt_sha256",
        },
        field="ingest_receipt",
    )
    design_id = manifest.get("design_id")
    scope = manifest.get("analysis_parameters", {}).get(
        "requested_analysis_scope"
    )
    registration_mode = manifest.get("analysis_parameters", {}).get(
        "registration", {}
    ).get("mode")
    expected_identity = {
        "specimen_id": specimen_id,
        "design_id": design_id,
        "requested_analysis_scope": scope,
    }
    for document_name, document in (("request", request), ("receipt", receipt)):
        stale = [
            key
            for key, value in expected_identity.items()
            if document.get(key) != value
        ]
        if stale:
            raise SpecimenIngestError(
                f"Stage 0 {document_name} identity is stale: " + ", ".join(stale)
            )
    externally_expected = {
        "specimen_id": expected_specimen_id,
        "design_id": expected_design_id,
        "requested_analysis_scope": expected_analysis_scope,
        "registration_mode": expected_registration_mode,
    }
    actual_identity = {**expected_identity, "registration_mode": registration_mode}
    stale_external = [
        key
        for key, value in externally_expected.items()
        if value is not None and actual_identity.get(key) != value
    ]
    if stale_external:
        raise SpecimenIngestError(
            "Stage 0 bundle differs from the frozen pipeline identity: "
            + ", ".join(stale_external)
        )
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION or receipt.get(
        "schema_version"
    ) != RECEIPT_SCHEMA_VERSION:
        raise SpecimenIngestError("Stage 0 request or receipt schema is incompatible")
    if any(
        document.get("method") != METHOD_NAME
        or document.get("method_version") != METHOD_VERSION
        for document in (request, receipt)
    ):
        raise SpecimenIngestError("Stage 0 request or receipt method is incompatible")
    if (
        request.get("registration_mode") != registration_mode
        or request.get("association_confirmed") is not True
        or receipt.get("lifecycle_state") != manifest.get("lifecycle_state")
        or receipt.get("unresolved_fields") != manifest.get("unresolved_fields")
    ):
        raise SpecimenIngestError("Stage 0 request, receipt, and manifest disagree")
    if require_ready and (
        manifest.get("lifecycle_state") != "ready_for_data_prep"
        or manifest.get("unresolved_fields") != []
    ):
        raise SpecimenIngestError("Stage 0 pass requires a ready intake manifest")

    receipt_base = {
        key: value
        for key, value in receipt.items()
        if key != "canonical_receipt_sha256"
    }
    if receipt.get("canonical_receipt_sha256") != canonical_json_sha256(
        receipt_base
    ):
        raise SpecimenIngestError(
            "Stage 0 ingest receipt canonical hash is invalid"
        )
    if receipt.get("request_sha256") != canonical_json_sha256(request):
        raise SpecimenIngestError("Stage 0 ingest request hash is invalid")
    if receipt.get("manifest_sha256") != canonical_json_sha256(manifest):
        raise SpecimenIngestError(
            "Stage 0 receipt manifest_sha256 is invalid"
        )

    request_paths = _strict_object(
        request["paths"],
        {
            "cad",
            "design_graph",
            "ct",
            "ct_metadata_response",
            "ct_metadata_mcp_call_receipt",
            "aligned_graph",
            "design_transform_declaration",
        },
        field="ingest_request.paths",
    )
    manifest_inputs = manifest["inputs"]
    expected_paths = {
        "cad": manifest_inputs.get("cad", {}).get("path"),
        "design_graph": manifest_inputs["design_graph"]["path"],
        "ct": manifest_inputs["ct"]["path"],
        "ct_metadata_response": (
            Path("analysis")
            / specimen_id
            / "config"
            / "ct_metadata_response.json"
        ).as_posix(),
        "ct_metadata_mcp_call_receipt": (
            Path("analysis")
            / specimen_id
            / "config"
            / "ct_metadata_mcp_call_receipt.json"
        ).as_posix(),
        "aligned_graph": manifest_inputs.get("aligned_graph", {}).get("path"),
        "design_transform_declaration": manifest_inputs.get(
            "design_transform_declaration", {}
        ).get("path"),
    }
    if request_paths != expected_paths:
        raise SpecimenIngestError("Stage 0 ingest request paths are stale")

    receipt_hashes = _strict_object(
        receipt["input_sha256"],
        {
            name
            for name in manifest_inputs
            if name != "ct_metadata"
        }
        | {"ct_metadata_response", "ct_metadata_mcp_call_receipt"},
        field="ingest_receipt.input_sha256",
    )
    expected_response_sha256 = receipt_hashes["ct_metadata_response"]
    expected_call_sha256 = receipt_hashes["ct_metadata_mcp_call_receipt"]
    for name, artifact in manifest_inputs.items():
        if name != "ct_metadata" and receipt_hashes.get(name) != artifact["sha256"]:
            raise SpecimenIngestError(
                f"Stage 0 receipt input hash differs from manifest input {name}"
            )
    ct_artifact = manifest_inputs["ct"]
    ct_path = (root / ct_artifact["path"]).resolve()
    ct_result, response_binding, call_binding = (
        validate_ct_metadata_mcp_evidence(
            response_path=resolved_response,
            expected_response_sha256=expected_response_sha256,
            call_receipt_path=resolved_call_receipt,
            expected_call_receipt_sha256=expected_call_sha256,
            resolved_ct=ct_path,
            ct_relative_path=ct_artifact["path"],
            repository_root=root,
            specimen_id=specimen_id,
            retention=ct_artifact["retention"],
        )
    )
    manifest_metadata = manifest_inputs["ct_metadata"]
    for field in ("format", "shape", "dtype", "byte_order", "voxel_spacing"):
        if manifest_metadata.get(field) != ct_result.get(field):
            raise SpecimenIngestError(
                f"Stage 0 manifest CT metadata differs from MCP evidence: {field}"
            )
    mcp_binding = _strict_object(
        request["mcp_response_binding"],
        {
            "tool",
            "response_schema_version",
            "artifact_path",
            "artifact_sha256",
            "call_receipt_path",
            "call_receipt_sha256",
            "canonical_call_receipt_sha256",
            "ct_sha256",
            "authoritative",
            "header_only",
        },
        field="ingest_request.mcp_response_binding",
    )
    if mcp_binding != {
        "tool": "inspect_volume_metadata",
        "response_schema_version": MCP_RESPONSE_SCHEMA_VERSION,
        "artifact_path": response_binding["path"],
        "artifact_sha256": response_binding["sha256"],
        "call_receipt_path": call_binding["path"],
        "call_receipt_sha256": call_binding["sha256"],
        "canonical_call_receipt_sha256": call_binding["canonical_sha256"],
        "ct_sha256": ct_artifact["sha256"],
        "authoritative": True,
        "header_only": True,
    }:
        raise SpecimenIngestError("Stage 0 request MCP binding is stale")
    declared = _strict_object(
        request["declared"],
        {
            "cad_units",
            "cad_units_provenance",
            "graph_axes",
            "array_axes",
            "aligned_graph_units",
            "retention",
            "design_transform_declaration_verification",
        },
        field="ingest_request.declared",
    )
    if (
        declared["design_transform_declaration_verification"]
        != receipt["design_transform_declaration_verification"]
    ):
        raise SpecimenIngestError(
            "Stage 0 transform-declaration verification is inconsistent"
        )

    coordinates = manifest["analysis_parameters"]["coordinates"]
    cad_inspection = manifest.get("intake", {}).get("cad_inspection")
    expected_declared = {
        "cad_units": (
            cad_inspection.get("units") if cad_inspection else declared["cad_units"]
        ),
        "cad_units_provenance": (
            cad_inspection.get("units_provenance")
            if cad_inspection
            else declared["cad_units_provenance"]
        ),
        "graph_axes": coordinates["graph_axes"],
        "array_axes": coordinates["array_axes"],
        "aligned_graph_units": coordinates["aligned_graph_units"],
        "retention": manifest_inputs["ct"]["retention"],
        "design_transform_declaration_verification": receipt[
            "design_transform_declaration_verification"
        ],
    }
    if declared != expected_declared:
        stale_declarations = sorted(
            name
            for name, value in expected_declared.items()
            if declared.get(name) != value
        )
        raise SpecimenIngestError(
            "Stage 0 declared intake values differ from the manifest: "
            + ", ".join(stale_declarations)
        )
    if manifest_inputs["ct_metadata"]["array_axes"] != declared["array_axes"]:
        raise SpecimenIngestError(
            "Stage 0 declared array axes differ from manifest CT metadata"
        )
    artifact_retentions = {
        artifact["retention"]
        for name, artifact in manifest_inputs.items()
        if name != "ct_metadata"
    }
    if artifact_retentions != {declared["retention"]}:
        raise SpecimenIngestError(
            "Stage 0 declared retention differs from a manifest input"
        )

    try:
        current_graph = inspect_lattice_graph(
            root / manifest_inputs["design_graph"]["path"],
            repository_root=root,
            allowed_roots=[root],
        )
    except SpecimenIngestError as exc:
        raise SpecimenIngestError(
            f"Stage 0 source artifact failed semantic reinspection: {exc}"
        ) from exc
    if "cad" in manifest_inputs:
        current_cad = inspect_cad_stl(
            root / manifest_inputs["cad"]["path"],
            repository_root=root,
            allowed_roots=[root],
            units=declared["cad_units"],
            units_provenance=declared["cad_units_provenance"],
        )
        if current_cad != manifest["intake"].get("cad_inspection"):
            raise SpecimenIngestError(
                "Stage 0 CAD inspection record differs from the current STL"
            )
    if current_graph != manifest["intake"]["graph_inspection"]:
        raise SpecimenIngestError(
            "Stage 0 graph inspection record differs from the current graph"
        )
    aligned_artifact = manifest_inputs.get("aligned_graph")
    if aligned_artifact is not None:
        try:
            current_aligned = inspect_lattice_graph(
                root / aligned_artifact["path"],
                repository_root=root,
                allowed_roots=[root],
            )
        except SpecimenIngestError as exc:
            raise SpecimenIngestError(
                f"Stage 0 aligned graph failed semantic reinspection: {exc}"
            ) from exc
        current_aligned_summary = {
            name: current_aligned[name]
            for name in (
                "junction_count",
                "strut_count",
                "unit_cell_count",
                "topology_sha256",
            )
        }
        if (
            manifest["derived"]["graph_summary"].get("aligned_values")
            != current_aligned_summary
        ):
            raise SpecimenIngestError(
                "Stage 0 aligned graph summary differs from the current graph"
            )
    verification = _strict_object(
        receipt["self_verification"],
        {
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
            "design_transform_declaration_valid",
            "design_transform_declaration_hash_bound",
            "segmentation_not_run",
            "registration_not_run",
            "defect_labels_not_derived",
        },
        field="ingest_receipt.self_verification",
    )
    failed = sorted(name for name, value in verification.items() if value is not True)
    if failed:
        raise SpecimenIngestError(
            "Stage 0 ingest self-verification failed: " + ", ".join(failed)
        )
    return {
        "manifest": manifest,
        "request": request,
        "receipt": receipt,
        "ct_metadata": ct_result,
        "ct_metadata_response": response_binding,
        "ct_metadata_mcp_call_receipt": call_binding,
    }
