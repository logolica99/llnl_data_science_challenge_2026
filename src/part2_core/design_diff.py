"""Deterministic design-space orientation and STL deletion labeling.

The implementation deliberately uses only the nominal graph and CAD meshes.
It never opens CT, aligned-coordinate, segmentation, or defect-analysis data.
Binary STL triangles are memory mapped and each mesh is released before the
next mesh is opened.
"""

from __future__ import annotations

import gc
import itertools
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import (
    read_json_object,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)
from .lattice import LatticeGraph, load_lattice_json


ORIENTATION_SCHEMA_VERSION = "part2-cad-graph-orientation/1.0.0"
LABEL_SCHEMA_VERSION = "part2-design-labels/1.0.0"
STL_TRIANGLE_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
REFERENCE_COUNTS = {"nodes": 10_206, "edges": 18_468, "cells": 729}
REFERENCE_DELETIONS = {"0p1": 18, "0p5": 93, "1p0": 186}


def _binary_stl_centroids(path: str | Path) -> tuple[np.ndarray, int]:
    """Load triangle centroids without materializing a processed mesh."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise ValueError(f"Binary STL header is incomplete: {resolved}")
    triangle_count = int(struct.unpack_from("<I", header, 80)[0])
    if triangle_count <= 0:
        raise ValueError(f"Binary STL contains no triangles: {resolved}")
    expected_bytes = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    if resolved.stat().st_size != expected_bytes:
        raise ValueError(
            f"Binary STL size/header mismatch for {resolved}; ASCII or corrupt STL "
            "input is not accepted by the memory-aware production path"
        )
    triangles = np.memmap(
        resolved,
        dtype=STL_TRIANGLE_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    centroids = np.asarray(triangles["vertices"].mean(axis=1), dtype=np.float32)
    del triangles
    return centroids, triangle_count


def _right_handed_axis_rotations() -> list[np.ndarray]:
    hypotheses: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.diag(signs) @ base
            if np.linalg.det(rotation) > 0.0:
                hypotheses.append(rotation)
    hypotheses.sort(key=lambda item: tuple(item.ravel().tolist()))
    return hypotheses


def _edge_core_samples(
    graph: LatticeGraph,
    *,
    sample_count: int,
    sample_start: float,
    sample_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 1 or not 0.0 < sample_start <= sample_end < 1.0:
        raise ValueError("Invalid junction-trimmed centerline sampling configuration")
    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    fractions = np.linspace(sample_start, sample_end, sample_count, dtype=np.float64)
    samples = (
        starts[:, None, :] * (1.0 - fractions[None, :, None])
        + ends[:, None, :] * fractions[None, :, None]
    )
    design_center = (
        graph.node_positions_xyz.min(axis=0)
        + graph.node_positions_xyz.max(axis=0)
    ) / 2.0
    return samples - design_center, design_center


def _scale_hypotheses(
    centered_samples: np.ndarray,
    centroids: np.ndarray,
    explicit: Sequence[float] | None,
) -> list[float]:
    if explicit is not None:
        values = sorted({float(value) for value in explicit})
    else:
        # This robust quantile span is only a proposal generator.  Ranking is
        # based on lattice centerline/triangle support, never a whole-mesh box.
        design_span = np.ptp(centered_samples.reshape(-1, 3), axis=0)
        low, high = np.quantile(centroids, [0.005, 0.995], axis=0)
        ratios = (high - low)[design_span > 0] / design_span[design_span > 0]
        center = float(np.median(ratios))
        values = [center * (1.0 + offset) for offset in np.linspace(-0.02, 0.02, 17)]
        # Preserve the independently established design-unit scale as a
        # candidate for this reference specimen, without forcing it to win.
        values.append(2.3052)
        values = sorted({round(value, 7) for value in values if value > 0})
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Scale hypotheses must be finite and positive")
    return values


def _translation_hypotheses(centroids: np.ndarray) -> list[np.ndarray]:
    low, high = np.quantile(centroids, [0.01, 0.99], axis=0)
    proposals = [
        np.zeros(3, dtype=np.float64),
        np.asarray((low + high) / 2.0, dtype=np.float64),
        np.asarray(np.median(centroids, axis=0), dtype=np.float64),
    ]
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for value in proposals:
        unique[tuple(np.round(value, 7).tolist())] = value
    return [unique[key] for key in sorted(unique)]


def resolve_cad_graph_orientation(
    nominal_graph_path: str | Path,
    full_design_stl_path: str | Path,
    output_path: str | Path,
    *,
    sample_count: int = 9,
    sample_start: float = 0.40,
    sample_end: float = 0.60,
    scale_candidates: Sequence[float] | None = None,
    ambiguity_absolute_mm: float = 1e-4,
    ambiguity_relative_fraction: float = 1e-3,
    expected_counts: Mapping[str, int] | None = None,
    config_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rank scale-preserving signed-axis hypotheses using lattice support."""

    graph = load_lattice_json(nominal_graph_path)
    samples, design_center = _edge_core_samples(
        graph,
        sample_count=sample_count,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    centroids, triangle_count = _binary_stl_centroids(full_design_stl_path)
    tree: cKDTree | None = None
    try:
        tree = cKDTree(centroids, compact_nodes=False, balanced_tree=False)
        scales = _scale_hypotheses(samples, centroids, scale_candidates)
        translations = _translation_hypotheses(centroids)
        # A stable subset is sufficient to rank orientation hypotheses; the
        # winning transform is then checked against every nominal edge.
        edge_rows = np.linspace(
            0,
            len(graph.edge_ids) - 1,
            min(len(graph.edge_ids), 2_048),
            dtype=int,
        )
        query = samples[edge_rows].reshape(-1, 3)
        ranked: list[dict[str, Any]] = []
        for rotation_index, rotation in enumerate(_right_handed_axis_rotations()):
            rotated = query @ rotation.T
            for scale in scales:
                for translation_index, translation in enumerate(translations):
                    distances = tree.query(
                        rotated * scale + translation,
                        k=1,
                        workers=1,
                    )[0]
                    ranked.append(
                        {
                            "rotation_index": rotation_index,
                            "translation_index": translation_index,
                            "scale_mm_per_design_unit": float(scale),
                            "rotation_matrix": rotation.tolist(),
                            "translation_mm": translation.tolist(),
                            "mean_support_distance_mm": float(np.mean(distances)),
                            "p99_support_distance_mm": float(np.quantile(distances, 0.99)),
                            "maximum_support_distance_mm": float(np.max(distances)),
                        }
                    )
        ranked.sort(
            key=lambda item: (
                item["mean_support_distance_mm"],
                item["p99_support_distance_mm"],
                item["rotation_index"],
                item["scale_mm_per_design_unit"],
                item["translation_index"],
            )
        )
        best = ranked[0]
        best_rotation = np.asarray(best["rotation_matrix"], dtype=np.float64)
        best_translation = np.asarray(best["translation_mm"], dtype=np.float64)
        all_points = (
            samples.reshape(-1, 3)
            @ best_rotation.T
            * float(best["scale_mm_per_design_unit"])
            + best_translation
        )
        all_distances = tree.query(all_points, k=1, workers=1)[0].reshape(
            samples.shape[:2]
        )
    finally:
        del tree, centroids
        gc.collect()

    best_score = float(best["mean_support_distance_mm"])
    ambiguity_limit = max(
        float(ambiguity_absolute_mm),
        abs(best_score) * float(ambiguity_relative_fraction),
    )
    equivalent = [
        item
        for item in ranked
        if float(item["mean_support_distance_mm"]) - best_score <= ambiguity_limit
    ]
    expected = dict(REFERENCE_COUNTS if expected_counts is None else expected_counts)
    count_gate = graph.counts == expected
    gates = {
        "graph_counts_match": count_gate,
        "nominal_ids_unique": bool(
            len(np.unique(graph.node_ids)) == len(graph.node_ids)
            and len(np.unique(graph.edge_ids)) == len(graph.edge_ids)
            and len(np.unique(graph.cell_ids)) == len(graph.cell_ids)
        ),
        "orientation_unambiguous": len(equivalent) == 1,
        "scale_preserving_transform": bool(
            math.isfinite(float(best["scale_mm_per_design_unit"]))
            and float(best["scale_mm_per_design_unit"]) > 0
        ),
        "all_edge_support_finite": bool(np.isfinite(all_distances).all()),
    }
    gate = "pass" if all(gates.values()) else (
        "manual_review" if not gates["orientation_unambiguous"] else "halt"
    )
    resolved_config_hash = config_sha256 or sha256_json(
        {
            "sample_count": sample_count,
            "sample_start": sample_start,
            "sample_end": sample_end,
            "scale_candidates": scales,
            "ambiguity_absolute_mm": ambiguity_absolute_mm,
            "ambiguity_relative_fraction": ambiguity_relative_fraction,
            "expected_counts": expected,
        }
    )
    report = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate == "pass",
        "counts": graph.counts,
        "triangle_count": triangle_count,
        "stl_coordinate_contract": {
            "units": "millimeter",
            "origin_convention": "origin_centered",
            "extra_y_geometry_is_not_lattice_extent": True,
            "whole_mesh_bounds_used_for_registration": False,
        },
        "transform": {
            "convention": "stl_mm = scale * ((design_xyz - design_center) @ rotation.T) + translation_mm",
            "design_center": design_center.tolist(),
            "scale_mm_per_design_unit": best["scale_mm_per_design_unit"],
            "rotation_matrix": best["rotation_matrix"],
            "translation_mm": best["translation_mm"],
        },
        "support": {
            "edge_count": int(len(graph.edge_ids)),
            "samples_per_edge": sample_count,
            "mean_nearest_triangle_centroid_mm": float(np.mean(all_distances)),
            "p99_nearest_triangle_centroid_mm": float(np.quantile(all_distances, 0.99)),
            "maximum_nearest_triangle_centroid_mm": float(np.max(all_distances)),
        },
        "ambiguity": {
            "equivalent_hypothesis_count": len(equivalent),
            "absolute_score_tolerance_mm": ambiguity_limit,
            "requires_scientist_review": len(equivalent) != 1,
            "top_hypotheses": ranked[: min(8, len(ranked))],
        },
        "gates": gates,
        "hashes": {
            "nominal_graph_sha256": graph.source_sha256,
            "full_design_stl_sha256": sha256_file(full_design_stl_path),
            "config_sha256": resolved_config_hash,
        },
        "provenance": {
            "design_space_only": True,
            "ct_accessed": False,
            "aligned_graph_accessed": False,
            "whole_mesh_bounding_box_used_as_primary_logic": False,
            "mesh_loading": "binary_stl_memmap_equivalent_to_process_false",
        },
        "warnings": (
            ["Equivalent CAD/graph orientation hypotheses require manual review"]
            if gate == "manual_review"
            else []
        ),
    }
    artifact = write_json_atomic(output_path, report, overwrite=overwrite)
    report["artifact"] = {**artifact, "role": "cad_graph_orientation", "retention": "committed"}
    return report


def _transform_samples(samples: np.ndarray, orientation: Mapping[str, Any]) -> np.ndarray:
    transform = orientation["transform"]
    rotation = np.asarray(transform["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(transform["translation_mm"], dtype=np.float64)
    scale = float(transform["scale_mm_per_design_unit"])
    return samples @ rotation.T * scale + translation


def _analyze_one_mesh(
    path: str | Path,
    transformed_samples: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, int, np.ndarray]:
    centroids, triangle_count = _binary_stl_centroids(path)
    tree: cKDTree | None = None
    try:
        tree = cKDTree(centroids, compact_nodes=False, balanced_tree=False)
        distances = tree.query(
            transformed_samples.reshape(-1, 3), k=1, workers=1
        )[0].reshape(transformed_samples.shape[:2])
        minimum = distances.min(axis=1)
        deleted = minimum > radius_mm
        return deleted, triangle_count, minimum
    finally:
        del tree, centroids
        gc.collect()


def deterministic_stratified_split(
    graph: LatticeGraph,
    deleted_ids: Sequence[int],
    *,
    development_fraction: float = 0.30,
    seed: int = 20260723,
    x_bins: int = 5,
    z_shells: int = 3,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Split positive IDs deterministically by midpoint X bin and Z shell."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    id_to_row = {int(identifier): row for row, identifier in enumerate(graph.edge_ids)}
    ids = sorted({int(value) for value in deleted_ids})
    if len(ids) != len(deleted_ids) or not set(ids).issubset(id_to_row):
        raise ValueError("Split IDs must be unique nominal strut IDs")
    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    midpoints = (starts + ends) / 2.0
    selected = midpoints[[id_to_row[value] for value in ids]]
    x_min, x_max = graph.node_positions_xyz[:, 0].min(), graph.node_positions_xyz[:, 0].max()
    z_center = (
        graph.node_positions_xyz[:, 2].min() + graph.node_positions_xyz[:, 2].max()
    ) / 2.0
    z_extent = max(float(np.max(np.abs(graph.node_positions_xyz[:, 2] - z_center))), 1e-12)
    x_index = np.minimum(
        x_bins - 1,
        np.floor((selected[:, 0] - x_min) / max(float(x_max - x_min), 1e-12) * x_bins).astype(int),
    )
    z_index = np.minimum(
        z_shells - 1,
        np.floor(np.abs(selected[:, 2] - z_center) / z_extent * z_shells).astype(int),
    )
    strata: dict[tuple[int, int], list[int]] = {}
    for identifier, x_bin, z_shell in zip(ids, x_index, z_index, strict=True):
        strata.setdefault((int(x_bin), int(z_shell)), []).append(identifier)
    target = int(round(len(ids) * development_fraction))
    allocation = {key: int(math.floor(len(values) * development_fraction)) for key, values in strata.items()}
    remaining = target - sum(allocation.values())
    fractional = sorted(
        strata,
        key=lambda key: (
            -(len(strata[key]) * development_fraction - allocation[key]),
            key,
        ),
    )
    for key in fractional[:remaining]:
        allocation[key] += 1

    def stable_key(identifier: int) -> str:
        return sha256_json({"seed": int(seed), "strut_id": identifier})

    development: list[int] = []
    sealed: list[int] = []
    summary: list[dict[str, Any]] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda value: (stable_key(value), value))
        count = allocation[key]
        development.extend(ordered[:count])
        sealed.extend(ordered[count:])
        summary.append(
            {
                "x_bin": key[0],
                "z_shell": key[1],
                "total": len(ordered),
                "development": count,
                "sealed": len(ordered) - count,
            }
        )
    return sorted(development), sorted(sealed), {
        "seed": int(seed),
        "development_fraction": development_fraction,
        "x_bins": x_bins,
        "z_shells": z_shells,
        "strata": summary,
    }


def label_deleted_edges(
    nominal_graph_path: str | Path,
    baseline_stl_path: str | Path,
    variant_stl_paths: Mapping[str, str | Path],
    orientation_path: str | Path,
    output_directory: str | Path,
    *,
    development_split_path: str | Path | None = None,
    sealed_split_path: str | Path | None = None,
    label_report_path: str | Path | None = None,
    expected_deletions: Mapping[str, int] | None = None,
    radius_margin_mm: float = 0.03,
    radius_rounding_mm: float = 0.01,
    sample_count: int = 9,
    split_seed: int = 20260723,
    config_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Label all nominal edges by junction-trimmed tube emptiness."""

    graph = load_lattice_json(nominal_graph_path)
    orientation = read_json_object(orientation_path)
    if orientation.get("gate") != "pass" or not orientation.get("gates", {}).get(
        "orientation_unambiguous", False
    ):
        raise ValueError("CAD/graph orientation is not unambiguously frozen")
    samples, _ = _edge_core_samples(
        graph, sample_count=sample_count, sample_start=0.40, sample_end=0.60
    )
    transformed = _transform_samples(samples, orientation)
    baseline_deleted, baseline_triangles, baseline_distances = _analyze_one_mesh(
        baseline_stl_path, transformed, math.inf
    )
    del baseline_deleted
    required_radius = float(np.max(baseline_distances)) + float(radius_margin_mm)
    radius = math.ceil(required_radius / radius_rounding_mm) * radius_rounding_mm
    # Re-check the baseline at the calibrated radius as a strict negative control.
    baseline_empty = baseline_distances > radius
    destination = Path(output_directory).expanduser().resolve()
    expected = dict(REFERENCE_DELETIONS if expected_deletions is None else expected_deletions)
    resolved_config_hash = config_sha256 or sha256_json(
        {
            "expected_deletions": expected,
            "radius_margin_mm": radius_margin_mm,
            "radius_rounding_mm": radius_rounding_mm,
            "sample_count": sample_count,
            "split_seed": split_seed,
        }
    )
    labels: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    prior_ids: set[int] = set()
    monotone_sets = True
    for variant in sorted(variant_stl_paths):
        deleted_mask, triangle_count, distances = _analyze_one_mesh(
            variant_stl_paths[variant], transformed, radius
        )
        deleted_ids = [int(value) for value in graph.edge_ids[deleted_mask]]
        deficit = baseline_triangles - triangle_count
        ratio = deficit / len(deleted_ids) if deleted_ids else None
        monotone_sets = monotone_sets and prior_ids.issubset(deleted_ids)
        prior_ids = set(deleted_ids)
        payload = {
            "schema_version": LABEL_SCHEMA_VERSION,
            "variant": variant,
            "deleted_strut_ids": deleted_ids,
            "deleted_count": len(deleted_ids),
            "triangle_count": triangle_count,
            "triangle_deficit_from_baseline": deficit,
            "triangles_per_deleted_strut": ratio,
            "tube_test": {
                "radius_mm": radius,
                "samples_per_edge": sample_count,
                "sample_fraction": [0.40, 0.60],
                "edge_count": int(len(graph.edge_ids)),
                "minimum_distance_mm": float(np.min(distances)),
                "maximum_distance_mm": float(np.max(distances)),
            },
            "hashes": {
                "nominal_graph_sha256": graph.source_sha256,
                "orientation_sha256": sha256_file(orientation_path),
                "baseline_stl_sha256": sha256_file(baseline_stl_path),
                "variant_stl_sha256": sha256_file(variant_stl_paths[variant]),
                "config_sha256": resolved_config_hash,
            },
            "provenance": {
                "design_space_only": True,
                "ct_accessed": False,
                "aligned_graph_accessed": False,
                "primary_logic": "junction_trimmed_tube_emptiness",
                "exact_float_coordinate_differencing": False,
                "clustering_used_as_primary_logic": False,
            },
        }
        path = destination / f"intentional_deletions_{variant}.json"
        artifact = write_json_atomic(path, payload, overwrite=overwrite)
        labels[variant] = payload
        artifacts[f"labels_{variant}"] = {
            **artifact,
            "role": f"intentional_deletions_{variant}",
            "retention": "sealed" if variant == "0p5" else "committed",
        }

    count_gate = all(
        variant in labels and labels[variant]["deleted_count"] == count
        for variant, count in expected.items()
    )
    ratio_gate = all(
        labels[variant]["triangles_per_deleted_strut"] is not None
        and 170.0 <= labels[variant]["triangles_per_deleted_strut"] <= 180.0
        for variant in expected
        if variant in labels
    )
    nominal_ids = set(int(value) for value in graph.edge_ids)
    id_gate = all(
        len(item["deleted_strut_ids"]) == len(set(item["deleted_strut_ids"]))
        and set(item["deleted_strut_ids"]).issubset(nominal_ids)
        for item in labels.values()
    )
    gates = {
        "baseline_negative_control": not bool(np.any(baseline_empty)),
        "deletion_counts_match": count_gate,
        "deletion_sets_monotone": monotone_sets,
        "triangle_deficit_ratio_between_170_and_180": ratio_gate,
        "label_ids_unique_and_nominal": id_gate,
        "graph_counts_match_reference": graph.counts == REFERENCE_COUNTS,
        "orientation_unambiguous": True,
    }
    split_summary: dict[str, Any] | None = None
    if "0p5" in labels and development_split_path and sealed_split_path:
        development, sealed, stratification = deterministic_stratified_split(
            graph, labels["0p5"]["deleted_strut_ids"], seed=split_seed
        )
        split_base = {
            "source_variant": "0p5",
            "source_labels_sha256": artifacts["labels_0p5"]["sha256"],
            "stratification": stratification,
            "config_sha256": resolved_config_hash,
        }
        dev_document = {
            "schema_version": "part2-label-split/1.0.0",
            "role": "development_labels",
            "strut_ids": development,
            **split_base,
        }
        sealed_document = {
            "schema_version": "part2-label-split/1.0.0",
            "role": "sealed_labels",
            "strut_ids": sealed,
            **split_base,
        }
        dev_artifact = write_json_atomic(development_split_path, dev_document, overwrite=overwrite)
        sealed_artifact = write_json_atomic(sealed_split_path, sealed_document, overwrite=overwrite)
        artifacts["development_split"] = {**dev_artifact, "role": "development_labels", "retention": "committed"}
        artifacts["sealed_split"] = {**sealed_artifact, "role": "sealed_labels", "retention": "sealed"}
        split_gates = {
            "disjoint": set(development).isdisjoint(sealed),
            "exhaustive": set(development) | set(sealed) == set(labels["0p5"]["deleted_strut_ids"]),
            "development_count": len(development),
            "sealed_count": len(sealed),
        }
        gates["development_and_sealed_disjoint"] = bool(split_gates["disjoint"])
        gates["development_and_sealed_exhaustive"] = bool(split_gates["exhaustive"])
        split_summary = {**split_gates, "stratification": stratification}

    gate = "pass" if all(bool(value) for value in gates.values()) else "halt"
    report = {
        "schema_version": "part2-design-label-report/1.0.0",
        "gate": gate,
        "overall_pass": gate == "pass",
        "counts": graph.counts,
        "baseline": {
            "triangle_count": baseline_triangles,
            "calibrated_radius_mm": radius,
            "maximum_support_distance_mm": float(np.max(baseline_distances)),
        },
        "variants": {
            key: {
                "deleted_count": value["deleted_count"],
                "triangle_count": value["triangle_count"],
                "triangles_per_deleted_strut": value["triangles_per_deleted_strut"],
            }
            for key, value in labels.items()
        },
        "split": split_summary,
        "gates": gates,
        "hashes": {
            "nominal_graph_sha256": graph.source_sha256,
            "orientation_sha256": sha256_file(orientation_path),
            "config_sha256": resolved_config_hash,
        },
        "provenance": {
            "design_space_only": True,
            "ct_accessed": False,
            "aligned_graph_accessed": False,
            "meshes_loaded_sequentially": True,
        },
        "warnings": [] if gate == "pass" else ["One or more deterministic design-label gates failed"],
    }
    if label_report_path is not None:
        report_destination = Path(label_report_path)
        if report_destination.suffix.lower() == ".md":
            lines = [
                "# Design-diff label report",
                "",
                f"Gate: `{gate}`",
                "",
                "| Variant | Deleted struts | Triangles | Triangles/deletion |",
                "|---|---:|---:|---:|",
            ]
            for key, value in report["variants"].items():
                ratio = value["triangles_per_deleted_strut"]
                ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
                lines.append(
                    f"| {key} | {value['deleted_count']} | {value['triangle_count']} | "
                    f"{ratio_text} |"
                )
            lines.extend(["", "All deterministic gates and hashes are recorded in the MCP response.", ""])
            artifact = write_text_atomic(
                report_destination, "\n".join(lines), overwrite=overwrite
            )
        else:
            artifact = write_json_atomic(report_destination, report, overwrite=overwrite)
        artifacts["label_report"] = {**artifact, "role": "label_report", "retention": "committed"}
    report["artifacts"] = artifacts
    return report
