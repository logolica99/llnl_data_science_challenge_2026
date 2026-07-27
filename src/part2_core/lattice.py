"""ID-safe in-memory views of nominal and registered lattice JSON graphs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import read_json_object, sha256_file


@dataclass(frozen=True)
class LatticeGraph:
    path: Path
    document: dict[str, Any]
    node_ids: np.ndarray
    node_positions_xyz: np.ndarray
    edge_ids: np.ndarray
    edge_node_ids: np.ndarray
    edge_node_rows: np.ndarray
    cell_ids: np.ndarray
    source_sha256: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            "nodes": int(self.node_ids.size),
            "edges": int(self.edge_ids.size),
            "cells": int(self.cell_ids.size),
        }

    def document_with_positions(self, positions_xyz: np.ndarray) -> dict[str, Any]:
        positions = np.asarray(positions_xyz, dtype=np.float64)
        if positions.shape != self.node_positions_xyz.shape:
            raise ValueError(
                f"Expected node positions with shape {self.node_positions_xyz.shape}, "
                f"got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("Registered node positions must all be finite")
        result = copy.deepcopy(self.document)
        by_id = {
            int(identifier): positions[row].tolist()
            for row, identifier in enumerate(self.node_ids)
        }
        for node in result["junctions"]:
            node["position"] = by_id[int(node["id"])]
        return result


def _id_items(
    document: dict[str, Any],
    key: str,
    path: Path,
) -> list[dict[str, Any]]:
    items = document.get(key)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: {key} must be a non-empty array")
    seen: set[int] = set()
    result: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("id"), int):
            raise ValueError(f"{path}: {key}[{row}].id must be an integer")
        identifier = int(item["id"])
        if identifier in seen:
            raise ValueError(f"{path}: duplicate {key} ID {identifier}")
        seen.add(identifier)
        result.append(item)
    return sorted(result, key=lambda item: int(item["id"]))


def load_lattice_json(path: str | Path) -> LatticeGraph:
    """Load a graph while resolving all edge endpoints by explicit node IDs."""

    resolved = Path(path).expanduser().resolve()
    document = read_json_object(resolved)
    nodes = _id_items(document, "junctions", resolved)
    edges = _id_items(document, "struts", resolved)
    cells = _id_items(document, "unit_cells", resolved)
    node_ids = np.asarray([int(node["id"]) for node in nodes], dtype=np.int64)
    positions = np.asarray([node.get("position") for node in nodes], dtype=np.float64)
    if positions.shape != (len(nodes), 3) or not np.isfinite(positions).all():
        raise ValueError(
            f"{resolved}: every junction position must be 3 finite numbers"
        )
    node_rows = {int(identifier): row for row, identifier in enumerate(node_ids)}
    edge_ids = np.asarray([int(edge["id"]) for edge in edges], dtype=np.int64)
    edge_node_ids = np.empty((len(edges), 2), dtype=np.int64)
    edge_node_rows = np.empty((len(edges), 2), dtype=np.int64)
    physical_edges: set[tuple[int, int]] = set()
    for row, edge in enumerate(edges):
        endpoints = edge.get("junction0"), edge.get("junction1")
        if not all(isinstance(value, int) for value in endpoints):
            raise ValueError(f"{resolved}: strut ID {edge['id']} has invalid endpoints")
        first, second = int(endpoints[0]), int(endpoints[1])
        if first == second or first not in node_rows or second not in node_rows:
            raise ValueError(
                f"{resolved}: strut ID {edge['id']} has unsafe endpoint IDs "
                f"{[first, second]}"
            )
        pair = tuple(sorted((first, second)))
        if pair in physical_edges:
            raise ValueError(f"{resolved}: duplicate physical strut {pair}")
        physical_edges.add(pair)
        edge_node_ids[row] = first, second
        edge_node_rows[row] = node_rows[first], node_rows[second]
    cell_ids = np.asarray([int(cell["id"]) for cell in cells], dtype=np.int64)
    known_edges = set(int(identifier) for identifier in edge_ids)
    for cell in cells:
        members = cell.get("struts")
        if (
            not isinstance(members, list)
            or not all(isinstance(value, int) for value in members)
            or not set(members).issubset(known_edges)
        ):
            raise ValueError(
                f"{resolved}: unit_cell ID {cell['id']} has invalid strut references"
            )
    return LatticeGraph(
        path=resolved,
        document=document,
        node_ids=node_ids,
        node_positions_xyz=positions,
        edge_ids=edge_ids,
        edge_node_ids=edge_node_ids,
        edge_node_rows=edge_node_rows,
        cell_ids=cell_ids,
        source_sha256=sha256_file(resolved),
    )


def compare_topology(
    nominal: LatticeGraph,
    candidate: LatticeGraph,
) -> dict[str, Any]:
    """Compare graph identity/topology without relying on JSON list offsets."""

    node_ids_match = bool(np.array_equal(nominal.node_ids, candidate.node_ids))
    edge_ids_match = bool(np.array_equal(nominal.edge_ids, candidate.edge_ids))
    cell_ids_match = bool(np.array_equal(nominal.cell_ids, candidate.cell_ids))
    edge_endpoints_match = bool(
        edge_ids_match
        and np.array_equal(nominal.edge_node_ids, candidate.edge_node_ids)
    )
    counts_match = nominal.counts == candidate.counts
    gates = {
        "counts_match": counts_match,
        "node_ids_match": node_ids_match,
        "edge_ids_match": edge_ids_match,
        "cell_ids_match": cell_ids_match,
        "edge_endpoints_match": edge_endpoints_match,
    }
    return {"gates": gates, "overall_pass": bool(all(gates.values()))}


def graph_bounds(graph: LatticeGraph) -> dict[str, list[float]]:
    minimum = graph.node_positions_xyz.min(axis=0)
    maximum = graph.node_positions_xyz.max(axis=0)
    return {
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "span": (maximum - minimum).tolist(),
    }


def positions_in_volume(
    positions_xyz: np.ndarray,
    shape_zyx: tuple[int, int, int],
    margin: float = 0.0,
) -> np.ndarray:
    shape_xyz = np.asarray(shape_zyx[::-1], dtype=np.float64)
    positions = np.asarray(positions_xyz, dtype=np.float64)
    return np.all(
        (positions >= margin) & (positions < shape_xyz - margin),
        axis=1,
    )
