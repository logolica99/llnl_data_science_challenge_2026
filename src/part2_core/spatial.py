"""Artifact-backed spatial summaries and graph rendering for Stage 4."""

from __future__ import annotations

from collections import defaultdict, deque
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import LatticeGraph, load_lattice_json
from .struts import read_metrics_csv


SPATIAL_STATISTICS_SCHEMA_VERSION = "part2-spatial-statistics/1.0.0"
LATTICE_RENDER_SCHEMA_VERSION = "part2-lattice-render/1.0.0"
CLASS_ORDER = ("missing", "broken", "thin", "present")
CLASS_COLORS = {
    "missing": "#7048e8",
    "broken": "#e03131",
    "thin": "#f08c00",
    "present": "#adb5bd",
}


def _classification_map(path: str | Path) -> tuple[dict[int, dict[str, Any]], str]:
    payload = read_json_object(path)
    rows = payload.get("classifications")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Classification artifact must contain a non-empty array")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("strut_id"), int):
            raise ValueError("Every classification must have an integer strut_id")
        identifier = int(row["strut_id"])
        label = row.get("class")
        if identifier in result or label not in CLASS_ORDER:
            raise ValueError(f"Invalid or duplicate classification for strut {identifier}")
        result[identifier] = row
    return result, sha256_file(path)


def _validate_coverage(graph: LatticeGraph, rows: dict[int, Any], label: str) -> None:
    graph_ids = {int(value) for value in graph.edge_ids}
    row_ids = set(rows)
    if graph_ids != row_ids:
        missing = sorted(graph_ids - row_ids)
        extra = sorted(row_ids - graph_ids)
        raise ValueError(
            f"{label} strut-ID coverage does not match the localized graph; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )


def _metric_map(graph: LatticeGraph, path: str | Path) -> tuple[dict[int, dict[str, Any]], str]:
    result: dict[int, dict[str, Any]] = {}
    endpoint_by_id = {
        int(edge_id): (int(endpoints[0]), int(endpoints[1]))
        for edge_id, endpoints in zip(graph.edge_ids, graph.edge_node_ids, strict=True)
    }
    for row in read_metrics_csv(path):
        identifier = int(row["strut_id"])
        if identifier in result:
            raise ValueError(f"Duplicate metrics row for strut {identifier}")
        actual = (int(row["junction0_id"]), int(row["junction1_id"]))
        if identifier not in endpoint_by_id or set(actual) != set(endpoint_by_id[identifier]):
            raise ValueError(f"Metrics endpoints contradict graph strut {identifier}")
        result[identifier] = row
    _validate_coverage(graph, result, "Metrics")
    return result, sha256_file(path)


def _edge_segments(graph: LatticeGraph) -> dict[int, np.ndarray]:
    return {
        int(edge_id): np.asarray(
            [
                graph.node_positions_xyz[int(rows[0])],
                graph.node_positions_xyz[int(rows[1])],
            ],
            dtype=np.float64,
        )
        for edge_id, rows in zip(graph.edge_ids, graph.edge_node_rows, strict=True)
    }


def _defect_clusters(
    graph: LatticeGraph,
    classifications: dict[int, dict[str, Any]],
) -> list[list[int]]:
    defective = {
        identifier
        for identifier, row in classifications.items()
        if row["class"] != "present"
    }
    incident: dict[int, list[int]] = defaultdict(list)
    for edge_id, endpoint_ids in zip(graph.edge_ids, graph.edge_node_ids, strict=True):
        identifier = int(edge_id)
        if identifier not in defective:
            continue
        incident[int(endpoint_ids[0])].append(identifier)
        incident[int(endpoint_ids[1])].append(identifier)
    adjacency: dict[int, set[int]] = {identifier: set() for identifier in defective}
    for members in incident.values():
        ordered = sorted(members)
        for identifier in ordered:
            adjacency[identifier].update(value for value in ordered if value != identifier)
    clusters: list[list[int]] = []
    unseen = set(defective)
    while unseen:
        seed = min(unseen)
        queue: deque[int] = deque([seed])
        unseen.remove(seed)
        cluster: list[int] = []
        while queue:
            identifier = queue.popleft()
            cluster.append(identifier)
            for neighbor in sorted(adjacency[identifier]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        clusters.append(sorted(cluster))
    return sorted(clusters, key=lambda values: (-len(values), values[0]))


def _axis_thirds(midpoints: dict[int, np.ndarray], bounds: np.ndarray) -> dict[str, list[int]]:
    minimum, maximum = bounds
    span = maximum - minimum
    result: dict[str, list[int]] = {}
    for axis, name in enumerate(("x", "y", "z")):
        counts = [0, 0, 0]
        for point in midpoints.values():
            normalized = 0.5 if span[axis] == 0 else (point[axis] - minimum[axis]) / span[axis]
            counts[min(2, max(0, int(math.floor(float(normalized) * 3.0))))] += 1
        result[name] = counts
    return result


def _write_figure_atomic(
    path: str | Path,
    figure: plt.Figure,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".png":
        raise ValueError(f"Figure output must use .png: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".png",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            dpi=160,
            bbox_inches="tight",
            metadata={"Software": "part2-core"},
        )
        changed = True
        if destination.exists():
            if destination.is_file() and destination.read_bytes() == temporary.read_bytes():
                changed = False
            elif not overwrite:
                raise FileExistsError(
                    f"Artifact already exists with different bytes: {destination}"
                )
            else:
                os.replace(temporary, destination)
        else:
            os.replace(temporary, destination)
        return {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "changed": changed,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _equal_3d_axes(axes: Any, positions: np.ndarray) -> dict[str, list[float]]:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.5)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    axes.set_box_aspect((1.0, 1.0, 1.0))
    return {
        "minimum_xyz": minimum.tolist(),
        "maximum_xyz": maximum.tolist(),
        "span_xyz": (maximum - minimum).tolist(),
    }


def render_lattice_3d(
    localized_graph_path: str | Path,
    classifications_path: str | Path,
    output_path: str | Path,
    *,
    elevation: float = 24.0,
    azimuth: float = 38.0,
    line_width: float = 0.8,
    node_size: float = 1.5,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render every classified graph edge with deterministic class colors."""

    for name, value in {
        "elevation": elevation,
        "azimuth": azimuth,
        "line_width": line_width,
        "node_size": node_size,
    }.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if line_width <= 0 or node_size < 0:
        raise ValueError("line_width must be positive and node_size non-negative")
    graph = load_lattice_json(localized_graph_path)
    classifications, classifications_sha256 = _classification_map(classifications_path)
    _validate_coverage(graph, classifications, "Classification")
    segments = _edge_segments(graph)
    counts = {label: 0 for label in CLASS_ORDER}
    figure = plt.figure(figsize=(10, 8))
    axes = figure.add_subplot(111, projection="3d")
    try:
        for label in reversed(CLASS_ORDER):
            identifiers = sorted(
                identifier
                for identifier, row in classifications.items()
                if row["class"] == label
            )
            counts[label] = len(identifiers)
            if identifiers:
                axes.add_collection3d(
                    Line3DCollection(
                        [segments[identifier] for identifier in identifiers],
                        colors=CLASS_COLORS[label],
                        linewidths=line_width * (1.8 if label != "present" else 1.0),
                        alpha=0.95 if label != "present" else 0.35,
                    )
                )
        if node_size:
            positions = graph.node_positions_xyz
            axes.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                s=node_size,
                c="#495057",
                alpha=0.25,
                depthshade=False,
            )
        bounds = _equal_3d_axes(axes, graph.node_positions_xyz)
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_xlabel("X (CT voxels)")
        axes.set_ylabel("Y (CT voxels)")
        axes.set_zlabel("Z (CT voxels)")
        axes.set_title("Classified nominal lattice")
        axes.legend(
            handles=[
                Line2D([0], [0], color=CLASS_COLORS[label], lw=3, label=label)
                for label in CLASS_ORDER
            ],
            loc="upper right",
        )
        figure.tight_layout()
        artifact = _write_figure_atomic(output_path, figure, overwrite=overwrite)
    finally:
        plt.close(figure)
    config_sha256 = sha256_json(
        {
            "elevation": float(elevation),
            "azimuth": float(azimuth),
            "line_width": float(line_width),
            "node_size": float(node_size),
            "class_colors": CLASS_COLORS,
        }
    )
    return {
        "schema_version": LATTICE_RENDER_SCHEMA_VERSION,
        "gate": "pass",
        "counts": {**counts, "total": int(graph.edge_ids.size)},
        "bent_count": sum(bool(row.get("bent", False)) for row in classifications.values()),
        "bounds": bounds,
        "view": {"elevation": float(elevation), "azimuth": float(azimuth)},
        "artifacts": {
            "lattice_3d_render": {
                **artifact,
                "role": "lattice_3d_render",
                "retention": "committed",
            }
        },
        "hashes": {
            "localized_graph_sha256": graph.source_sha256,
            "classifications_sha256": classifications_sha256,
            "render_sha256": artifact["sha256"],
            "config_sha256": config_sha256,
        },
        "provenance": {
            "artifact_backed": True,
            "sealed_labels_read": False,
            "intentional_attribution_claimed": False,
        },
        "warnings": [],
    }


def compute_spatial_stats(
    localized_graph_path: str | Path,
    classifications_path: str | Path,
    metrics_path: str | Path,
    output_statistics_path: str | Path,
    output_figure_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write ID-complete spatial and per-class summaries from frozen artifacts."""

    graph = load_lattice_json(localized_graph_path)
    classifications, classifications_sha256 = _classification_map(classifications_path)
    _validate_coverage(graph, classifications, "Classification")
    metrics, metrics_sha256 = _metric_map(graph, metrics_path)
    segments = _edge_segments(graph)
    midpoints = {identifier: segment.mean(axis=0) for identifier, segment in segments.items()}
    bounds_array = np.asarray(
        [graph.node_positions_xyz.min(axis=0), graph.node_positions_xyz.max(axis=0)]
    )
    class_counts = {
        label: sum(row["class"] == label for row in classifications.values())
        for label in CLASS_ORDER
    }
    total = len(classifications)
    clusters = _defect_clusters(graph, classifications)
    metric_fields = {
        "corridor_foreground_fraction": "median_corridor_foreground_fraction",
        "maximum_axial_gap_fraction": "median_maximum_axial_gap_fraction",
        "edt_radius_median_voxels": "median_edt_radius_voxels",
        "centerline_curvature_rms_voxels": "median_curvature_rms_voxels",
    }
    metric_summary: dict[str, dict[str, float | None]] = {}
    for label in CLASS_ORDER:
        identifiers = [
            identifier
            for identifier, row in classifications.items()
            if row["class"] == label
        ]
        summary: dict[str, float | None] = {}
        for source, destination in metric_fields.items():
            values = [
                float(metrics[identifier][source])
                for identifier in identifiers
                if metrics[identifier][source] is not None
                and math.isfinite(float(metrics[identifier][source]))
            ]
            summary[destination] = float(np.median(values)) if values else None
        metric_summary[label] = summary

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        axes[0].bar(
            CLASS_ORDER,
            [class_counts[label] for label in CLASS_ORDER],
            color=[CLASS_COLORS[label] for label in CLASS_ORDER],
        )
        axes[0].set_title("Strut classifications")
        axes[0].set_ylabel("Count")
        for label in CLASS_ORDER:
            identifiers = [
                identifier
                for identifier, row in classifications.items()
                if row["class"] == label
            ]
            if identifiers:
                points = np.asarray([midpoints[identifier] for identifier in identifiers])
                axes[1].scatter(
                    points[:, 0],
                    points[:, 1],
                    s=10 if label != "present" else 3,
                    alpha=0.85 if label != "present" else 0.2,
                    color=CLASS_COLORS[label],
                    label=label,
                )
        axes[1].set_title("Strut midpoint distribution")
        axes[1].set_xlabel("X (CT voxels)")
        axes[1].set_ylabel("Y (CT voxels)")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].legend()
        figure.tight_layout()
        figure_artifact = _write_figure_atomic(
            output_figure_path, figure, overwrite=overwrite
        )
    finally:
        plt.close(figure)

    persistent_figure_artifact = {
        key: value for key, value in figure_artifact.items() if key != "changed"
    }
    report = {
        "schema_version": SPATIAL_STATISTICS_SCHEMA_VERSION,
        "gate": "pass",
        "overall_pass": True,
        "counts": {
            "nodes": int(graph.node_ids.size),
            "struts": total,
            "cells": int(graph.cell_ids.size),
            "bent": sum(bool(row.get("bent", False)) for row in classifications.values()),
            "primary_defects": total - class_counts["present"],
        },
        "class_counts": class_counts,
        "class_fractions": {
            label: class_counts[label] / total for label in CLASS_ORDER
        },
        "defect_clusters": {
            "count": len(clusters),
            "largest_strut_count": len(clusters[0]) if clusters else 0,
            "cluster_strut_ids": clusters,
        },
        "axis_thirds_all_struts": _axis_thirds(midpoints, bounds_array),
        "metric_medians_by_class": metric_summary,
        "bounds": {
            "minimum_xyz": bounds_array[0].tolist(),
            "maximum_xyz": bounds_array[1].tolist(),
            "span_xyz": (bounds_array[1] - bounds_array[0]).tolist(),
        },
        "artifacts": {
            "spatial_statistics_figure": {
                **persistent_figure_artifact,
                "role": "spatial_statistics_figure",
                "retention": "committed",
            }
        },
        "hashes": {
            "localized_graph_sha256": graph.source_sha256,
            "classifications_sha256": classifications_sha256,
            "metrics_sha256": metrics_sha256,
            "spatial_statistics_figure_sha256": figure_artifact["sha256"],
        },
        "provenance": {
            "artifact_backed": True,
            "classification_recomputed": False,
            "sealed_labels_read": False,
            "intentional_attribution_claimed": False,
        },
        "warnings": [],
    }
    statistics_artifact = write_json_atomic(
        output_statistics_path, report, overwrite=overwrite
    )
    report["artifacts"]["spatial_statistics"] = {
        **statistics_artifact,
        "role": "spatial_statistics",
        "retention": "committed",
    }
    report["artifacts"]["spatial_statistics_figure"]["changed"] = figure_artifact[
        "changed"
    ]
    report["hashes"]["spatial_statistics_sha256"] = statistics_artifact["sha256"]
    return report
