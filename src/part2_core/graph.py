"""Deterministic normalization of LLNL lattice JSON graphs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np


GRAPH_SCHEMA_VERSION = "normalized-lattice-graph/1.0.0"


class GraphNormalizationError(ValueError):
    """Raised when graph identities or references are unsafe to normalize."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphNormalizationError(f"Unreadable lattice graph {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GraphNormalizationError(f"Lattice graph must be a JSON object: {path}")
    return value


def _items_with_ids(graph: dict[str, Any], key: str, path: Path) -> list[dict[str, Any]]:
    items = graph.get(key)
    if not isinstance(items, list) or not items:
        raise GraphNormalizationError(f"{path}: {key} must be a non-empty array")
    normalized: list[dict[str, Any]] = []
    identifiers: set[int] = set()
    for row, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise GraphNormalizationError(f"{path}: {key}[{row}].id must be an integer")
        identifier = item["id"]
        if identifier in identifiers:
            raise GraphNormalizationError(f"{path}: {key} contains duplicate ID {identifier}")
        identifiers.add(identifier)
        normalized.append(item)
    return sorted(normalized, key=lambda item: item["id"])


def _finite_triplet(
    item: dict[str, Any],
    field: str,
    *,
    required: bool,
    context: str,
) -> tuple[float, float, float] | None:
    value = item.get(field)
    if value is None and not required:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(
            isinstance(component, (int, float)) and math.isfinite(component)
            for component in value
        )
    ):
        raise GraphNormalizationError(f"{context}.{field} must contain 3 finite numbers")
    return tuple(float(component) for component in value)


def _optional_number(
    item: dict[str, Any],
    field: str,
    *,
    context: str,
) -> float:
    value = item.get(field)
    if value is None:
        return math.nan
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GraphNormalizationError(f"{context}.{field} must be a finite number")
    return float(value)


def _contiguous(ids: np.ndarray[Any, Any]) -> bool:
    if not ids.size:
        return False
    return bool(np.array_equal(ids, np.arange(ids[0], ids[0] + ids.size)))


def _grid_shape(indices: np.ndarray[Any, Any]) -> tuple[list[int] | None, list[str]]:
    warnings: list[str] = []
    if np.isnan(indices).any():
        return None, ["unit-cell indices are missing; lattice dimensions need review"]
    rounded = np.rint(indices)
    if not np.array_equal(indices, rounded):
        return None, ["unit-cell indices are non-integral; lattice dimensions need review"]
    integer = rounded.astype(np.int64)
    minimum = integer.min(axis=0)
    maximum = integer.max(axis=0)
    shape = (maximum - minimum + 1).tolist()
    expected = int(np.prod(shape, dtype=np.int64))
    if expected != integer.shape[0]:
        warnings.append(
            f"unit-cell indices occupy {integer.shape[0]} of {expected} bounding-grid positions"
        )
    if len(set(shape)) != 1:
        warnings.append(f"unit-cell grid is not cubic: {shape}")
    return [int(value) for value in shape], warnings


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray[Any, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise GraphNormalizationError(
            f"Normalized graph already exists; enable overwrite explicitly: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_lattice_graph(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate identities/references and write a normalized NPZ artifact.

    Rows are sorted by explicit IDs.  Endpoint and membership joins are stored
    both as source IDs and as explicit ``*_id_rows`` maps; callers never infer
    identity from the original JSON list offsets.
    """

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source.suffix.lower() != ".json" or not source.is_file():
        raise GraphNormalizationError(f"Expected an existing .json graph: {source}")
    if destination.suffix.lower() != ".npz":
        raise GraphNormalizationError(f"Normalized graph must use .npz: {destination}")
    graph = _load_json(source)
    required = {"junctions", "struts", "unit_cells"}
    missing = sorted(required - set(graph))
    if missing:
        raise GraphNormalizationError(
            f"{source}: missing graph keys: {', '.join(missing)}"
        )

    nodes = _items_with_ids(graph, "junctions", source)
    edges = _items_with_ids(graph, "struts", source)
    cells = _items_with_ids(graph, "unit_cells", source)

    node_ids = np.asarray([item["id"] for item in nodes], dtype=np.int64)
    edge_ids = np.asarray([item["id"] for item in edges], dtype=np.int64)
    cell_ids = np.asarray([item["id"] for item in cells], dtype=np.int64)
    node_id_to_row = {int(identifier): row for row, identifier in enumerate(node_ids)}
    edge_id_to_row = {int(identifier): row for row, identifier in enumerate(edge_ids)}

    node_positions = np.asarray(
        [
            _finite_triplet(
                item,
                "position",
                required=True,
                context=f"junction ID {item['id']}",
            )
            for item in nodes
        ],
        dtype=np.float64,
    )
    node_indices = np.asarray(
        [
            _finite_triplet(
                item,
                "indices",
                required=False,
                context=f"junction ID {item['id']}",
            )
            or (math.nan, math.nan, math.nan)
            for item in nodes
        ],
        dtype=np.float64,
    )

    edge_node_ids: list[tuple[int, int]] = []
    edge_node_rows: list[tuple[int, int]] = []
    edge_pairs: set[tuple[int, int]] = set()
    edge_unit_cell_index: list[int] = []
    edge_thickness: list[float] = []
    for item in edges:
        context = f"strut ID {item['id']}"
        endpoints = (item.get("junction0"), item.get("junction1"))
        if not all(isinstance(value, int) for value in endpoints):
            raise GraphNormalizationError(f"{context} endpoints must be integer node IDs")
        first, second = int(endpoints[0]), int(endpoints[1])
        if first == second:
            raise GraphNormalizationError(f"{context} is a self-loop")
        unknown = [value for value in (first, second) if value not in node_id_to_row]
        if unknown:
            raise GraphNormalizationError(f"{context} references unknown nodes {unknown}")
        pair = tuple(sorted((first, second)))
        if pair in edge_pairs:
            raise GraphNormalizationError(
                f"{context} duplicates physical node pair {pair}"
            )
        edge_pairs.add(pair)
        edge_node_ids.append((first, second))
        edge_node_rows.append((node_id_to_row[first], node_id_to_row[second]))
        cell_index = item.get("unit_cell_edge_idx", -1)
        if not isinstance(cell_index, int):
            raise GraphNormalizationError(f"{context}.unit_cell_edge_idx must be an integer")
        edge_unit_cell_index.append(cell_index)
        edge_thickness.append(_optional_number(item, "thickness", context=context))

    cell_indices: list[tuple[float, float, float]] = []
    cell_edge_ids_flat: list[int] = []
    cell_edge_rows_flat: list[int] = []
    cell_edge_offsets = [0]
    for item in cells:
        context = f"unit_cell ID {item['id']}"
        members = item.get("struts")
        if not isinstance(members, list) or not all(
            isinstance(value, int) for value in members
        ):
            raise GraphNormalizationError(f"{context}.struts must contain integer edge IDs")
        if len(members) != len(set(members)):
            raise GraphNormalizationError(f"{context}.struts contains duplicate edge IDs")
        unknown = sorted(set(members) - set(edge_id_to_row))
        if unknown:
            raise GraphNormalizationError(f"{context} references unknown edges {unknown}")
        ordered_members = sorted(int(value) for value in members)
        cell_edge_ids_flat.extend(ordered_members)
        cell_edge_rows_flat.extend(edge_id_to_row[value] for value in ordered_members)
        cell_edge_offsets.append(len(cell_edge_ids_flat))
        cell_indices.append(
            _finite_triplet(item, "indices", required=False, context=context)
            or (math.nan, math.nan, math.nan)
        )

    arrays: dict[str, np.ndarray[Any, Any]] = {
        "node_ids": node_ids,
        "node_positions_xyz": node_positions,
        "node_indices_xyz": node_indices,
        "node_id_keys": node_ids.copy(),
        "node_id_rows": np.arange(node_ids.size, dtype=np.int64),
        "edge_ids": edge_ids,
        "edge_node_ids": np.asarray(edge_node_ids, dtype=np.int64),
        "edge_node_rows": np.asarray(edge_node_rows, dtype=np.int64),
        "edge_unit_cell_index": np.asarray(edge_unit_cell_index, dtype=np.int64),
        "edge_thickness": np.asarray(edge_thickness, dtype=np.float64),
        "edge_id_keys": edge_ids.copy(),
        "edge_id_rows": np.arange(edge_ids.size, dtype=np.int64),
        "cell_ids": cell_ids,
        "cell_indices_xyz": np.asarray(cell_indices, dtype=np.float64),
        "cell_edge_ids": np.asarray(cell_edge_ids_flat, dtype=np.int64),
        "cell_edge_rows": np.asarray(cell_edge_rows_flat, dtype=np.int64),
        "cell_edge_offsets": np.asarray(cell_edge_offsets, dtype=np.int64),
        "cell_id_keys": cell_ids.copy(),
        "cell_id_rows": np.arange(cell_ids.size, dtype=np.int64),
    }
    _atomic_savez(destination, arrays, overwrite)

    grid_shape, warnings = _grid_shape(arrays["cell_indices_xyz"])
    ids_contiguous = {
        "nodes": _contiguous(node_ids),
        "edges": _contiguous(edge_ids),
        "cells": _contiguous(cell_ids),
    }
    if not all(ids_contiguous.values()):
        warnings.append(
            "one or more ID domains are non-contiguous; explicit ID maps are required"
        )
    coordinate_min = node_positions.min(axis=0)
    coordinate_max = node_positions.max(axis=0)
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "source_path": str(source),
        "output_path": str(destination),
        "counts": {
            "nodes": int(node_ids.size),
            "edges": int(edge_ids.size),
            "cells": int(cell_ids.size),
        },
        "cell_grid_shape_xyz": grid_shape,
        "coordinate_bounds_xyz": {
            "minimum": coordinate_min.tolist(),
            "maximum": coordinate_max.tolist(),
            "span": (coordinate_max - coordinate_min).tolist(),
        },
        "ids_contiguous": ids_contiguous,
        "id_reference_integrity": True,
        "explicit_id_maps": {
            "nodes": ["node_id_keys", "node_id_rows"],
            "edges": ["edge_id_keys", "edge_id_rows"],
            "cells": ["cell_id_keys", "cell_id_rows"],
        },
        "array_names": sorted(arrays),
        "warnings": warnings,
        "source_sha256": _sha256_file(source),
        "artifact_sha256": _sha256_file(destination),
    }
