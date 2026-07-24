"""Deterministic intake core for new lattice specimens.

This module inspects and associates scientist-supplied inputs. It intentionally
does not run segmentation, registration, local recentering, or defect labeling.
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

from specimen_manifest import (
    DEFAULT_SCHEMA,
    ManifestValidationError,
    canonical_json_sha256,
    sha256_file,
    validate_manifest,
)
from volume_metadata import UNKNOWN, inspect_volume


METHOD_NAME = "specimen_ingest"
METHOD_VERSION = "1.0.0"
REQUEST_SCHEMA_VERSION = "ingest-request/1.0.0"
RECEIPT_SCHEMA_VERSION = "ingest-receipt/1.0.0"
SPECIMEN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REGISTRATION_MODES = {"challenge_aligned_json", "autonomous_v2"}


class SpecimenIngestError(ValueError):
    """Raised when intake cannot produce a trustworthy provisional manifest."""


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


def _axes(value: str, expected: str) -> list[str] | str:
    normalized = value.lower()
    if normalized == UNKNOWN:
        return UNKNOWN
    if normalized != expected:
        raise SpecimenIngestError(
            f"Expected axes {expected!r} or 'unknown', found {value!r}"
        )
    return list(normalized)


def _default_analysis_parameters(
    *,
    registration_mode: str,
    ct_dtype: str,
    graph_axes: list[str] | str,
    array_axes: list[str] | str,
    aligned_graph_units: str,
) -> dict[str, Any]:
    return {
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
        "budgets": {
            "local_recenter_radius_voxels": 8.0,
            "roi_padding_fraction": 0.2,
            "metrology_uncertainty_voxels": 2.0,
            "maximum_agent_retries": 2,
        },
        "artifact_schema_versions": {
            "specimen_manifest": "2.0.0",
            "registration_qa": "1.0.0",
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
    cad_path: Path,
    design_graph_path: Path,
    ct_path: Path,
    registration_mode: str,
    association_confirmed: bool,
    allowed_data_roots: Iterable[Path] | None = None,
    aligned_graph_path: Path | None = None,
    cad_units: str = UNKNOWN,
    cad_units_provenance: str = UNKNOWN,
    graph_axes: str = "xyz",
    array_axes: str = UNKNOWN,
    aligned_graph_units: str = UNKNOWN,
    retention: str = "committed",
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Inspect explicit inputs and write idempotent intake artifacts."""
    root = repository_root.expanduser().resolve()
    if not SPECIMEN_ID_PATTERN.fullmatch(specimen_id):
        raise SpecimenIngestError(f"Invalid specimen_id: {specimen_id!r}")
    if registration_mode not in REGISTRATION_MODES:
        raise SpecimenIngestError(
            f"Unsupported registration mode: {registration_mode!r}"
        )
    if not association_confirmed:
        raise SpecimenIngestError(
            "Scientist must explicitly confirm the CAD/graph/CT association"
        )
    if retention not in {"committed", "external", "regenerable"}:
        raise SpecimenIngestError(f"Unsupported retention policy: {retention!r}")
    data_roots = list(allowed_data_roots or [root / "data"])

    design = inspect_lattice_graph(
        design_graph_path, repository_root=root, allowed_roots=data_roots
    )
    cad = inspect_cad_stl(
        cad_path,
        repository_root=root,
        allowed_roots=data_roots,
        units=cad_units,
        units_provenance=cad_units_provenance,
    )
    resolved_ct, _ = _resolve_input(
        ct_path, repository_root=root, allowed_roots=data_roots
    )
    ct = inspect_volume(resolved_ct, repository_root=root, header_only=True)
    if ct["ndim"] != 3:
        raise SpecimenIngestError(
            f"CT input must be 3D, found shape {ct['shape']} at {ct['path']}"
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

    declared_graph_axes = _axes(graph_axes, "xyz")
    declared_array_axes = _axes(array_axes, "zyx")
    if aligned_graph_units not in {"voxel", "simulation_voxel", UNKNOWN}:
        raise SpecimenIngestError(
            f"Unsupported aligned graph units: {aligned_graph_units!r}"
        )
    unresolved_fields: list[str] = []
    for field, value in (
        ("intake.cad_inspection.units", cad_units),
        ("analysis_parameters.coordinates.graph_axes", declared_graph_axes),
        ("analysis_parameters.coordinates.array_axes", declared_array_axes),
        (
            "analysis_parameters.coordinates.aligned_graph_units",
            aligned_graph_units,
        ),
    ):
        if value == UNKNOWN:
            unresolved_fields.append(field)

    lifecycle_state = (
        "provisional" if unresolved_fields else "ready_for_data_prep"
    )
    analysis_parameters = _default_analysis_parameters(
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
        "cad": _artifact(cad["path"], cad["sha256"], "cad", retention),
    }
    graph_input_hashes = [design["sha256"]]
    if aligned is not None:
        inputs["aligned_graph"] = _artifact(
            aligned["path"], aligned["sha256"], "aligned_graph", retention
        )
        graph_input_hashes = sorted(
            set(graph_input_hashes) | {aligned["sha256"]}
        )

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
        "schema_version": "2.0.0",
        "specimen_id": specimen_id,
        "lifecycle_state": lifecycle_state,
        "unresolved_fields": sorted(unresolved_fields),
        "inputs": inputs,
        "intake": {
            "association": {
                "source": "scientist_explicit",
                "confirmed": True,
                "design_graph_to_cad": True,
                "ct_to_specimen": True,
            },
            "registration_mode_selection": {
                "mode": registration_mode,
                "source": "scientist_explicit",
            },
            "cad_inspection": cad,
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

    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "specimen_id": specimen_id,
        "paths": {
            "cad": cad["path"],
            "design_graph": design["path"],
            "ct": ct["path"],
            "aligned_graph": aligned["path"] if aligned else None,
        },
        "registration_mode": registration_mode,
        "association_confirmed": True,
        "declared": {
            "cad_units": cad_units,
            "cad_units_provenance": cad_units_provenance,
            "graph_axes": declared_graph_axes,
            "array_axes": declared_array_axes,
            "aligned_graph_units": aligned_graph_units,
            "retention": retention,
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
        "lifecycle_state": lifecycle_state,
        "input_sha256": {
            "cad": cad["sha256"],
            "design_graph": design["sha256"],
            "ct": ct["sha256"],
            **({"aligned_graph": aligned["sha256"]} if aligned else {}),
        },
        "request_sha256": request_hash,
        "manifest_sha256": manifest_hash,
        "warnings": sorted(warnings),
        "unresolved_fields": sorted(unresolved_fields),
        "self_verification": {
            "association_explicit": True,
            "all_paths_repository_relative": True,
            "all_inputs_hashed": True,
            "cad_readable": True,
            "graph_id_reference_integrity": True,
            "manifest_schema_valid": True,
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
        "lifecycle_state": lifecycle_state,
        "paths": {
            "ingest_request": str(request_path),
            "specimen_manifest": str(manifest_path),
            "ingest_receipt": str(receipt_path),
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
