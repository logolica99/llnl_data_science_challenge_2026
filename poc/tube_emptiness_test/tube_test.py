#!/usr/bin/env python3
"""Detect deleted octet-truss struts with a junction-trimmed tube test.

The script reads the design JSON and binary STL files directly, calibrates the
JSON-to-STL scale and query radius on 0.stl, analyzes all four designs, and
writes the proof-of-concept artifacts under results/.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree


STL_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)
DESIGN_CENTER = 9.0
DESIGN_PATH = Path("data/missing_struts/octet_truss_9x9x9.json")
STL_DIR = Path("data/missing_struts/stls")
MESH_LABELS = ("0", "0.1", "0.5", "1")
SCALE_CANDIDATES = tuple(np.round(np.arange(2.290, 2.321, 0.001), 3)) + (2.3052,)
RADIUS_SAFETY_MARGIN_MM = 0.03
RADIUS_ROUNDING_MM = 0.01
SAMPLE_START = 0.40
SAMPLE_END = 0.60
DEFAULT_SAMPLE_COUNT = 9


@dataclass(frozen=True)
class Design:
    strut_ids: np.ndarray
    endpoint_positions: np.ndarray


@dataclass(frozen=True)
class MeshResult:
    label: str
    triangle_count: int
    deleted_ids: np.ndarray
    nearest_distance_min_mm: np.ndarray
    elapsed_seconds: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_dir.parents[1],
        help="Repository root (automatically inferred by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
        help="Destination for JSON and PNG artifacts.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        help="Skip scale search and use this mm/design-unit value.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Skip radius calibration and use this tube radius in mm.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Centerline samples per strut (default: {DEFAULT_SAMPLE_COUNT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="cKDTree query workers; -1 uses all available cores.",
    )
    return parser.parse_args()


def load_design(path: Path) -> Design:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    positions_by_id = {
        int(junction["id"]): np.asarray(junction["position"], dtype=np.float32)
        for junction in raw["junctions"]
    }
    strut_ids = np.asarray([int(strut["id"]) for strut in raw["struts"]], dtype=np.int64)
    endpoint_positions = np.asarray(
        [
            [
                positions_by_id[int(strut["junction0"])],
                positions_by_id[int(strut["junction1"])],
            ]
            for strut in raw["struts"]
        ],
        dtype=np.float32,
    )
    if len(np.unique(strut_ids)) != len(strut_ids):
        raise ValueError("Design contains duplicate strut IDs")
    return Design(strut_ids=strut_ids, endpoint_positions=endpoint_positions)


def read_triangle_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(84)
    if len(header) != 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    triangle_count = struct.unpack_from("<I", header, 80)[0]
    expected_size = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path} size does not match its binary STL header "
            f"({actual_size:,} bytes, expected {expected_size:,})"
        )
    return triangle_count


def load_triangle_centroids(path: Path) -> tuple[np.ndarray, int]:
    triangle_count = read_triangle_count(path)
    triangles = np.memmap(
        path,
        dtype=STL_TRIANGLE_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    # The result is an in-memory float32 array; the STL itself remains memory-mapped.
    centroids = np.asarray(triangles["vertices"].mean(axis=1), dtype=np.float32)
    del triangles
    return centroids, triangle_count


def raw_centerline_samples(design: Design, sample_count: int) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("--samples must be at least 1")
    fractions = np.linspace(SAMPLE_START, SAMPLE_END, sample_count, dtype=np.float32)
    start = design.endpoint_positions[:, 0, None, :]
    end = design.endpoint_positions[:, 1, None, :]
    points = start * (1.0 - fractions[None, :, None]) + end * fractions[None, :, None]
    return points - DESIGN_CENTER


def nearest_distance_min(
    tree: cKDTree,
    raw_samples: np.ndarray,
    scale: float,
    workers: int,
) -> np.ndarray:
    distances = tree.query(
        (raw_samples * scale).reshape(-1, 3),
        k=1,
        workers=workers,
    )[0]
    return distances.reshape(raw_samples.shape[:2]).min(axis=1)


def calibrate_scale(
    tree: cKDTree,
    raw_samples: np.ndarray,
    workers: int,
) -> tuple[float, list[dict[str, float]]]:
    trials: list[dict[str, float]] = []
    for scale in sorted(set(float(value) for value in SCALE_CANDIDATES)):
        distances = nearest_distance_min(tree, raw_samples, scale, workers)
        trials.append(
            {
                "scale_mm_per_design_unit": scale,
                "mean_nearest_distance_mm": float(distances.mean()),
                "p99_nearest_distance_mm": float(np.quantile(distances, 0.99)),
                "max_nearest_distance_mm": float(distances.max()),
            }
        )
    best = min(trials, key=lambda item: item["mean_nearest_distance_mm"])
    return float(best["scale_mm_per_design_unit"]), trials


def calibrate_radius(baseline_distances: np.ndarray) -> float:
    required = float(baseline_distances.max()) + RADIUS_SAFETY_MARGIN_MM
    return math.ceil(required / RADIUS_ROUNDING_MM) * RADIUS_ROUNDING_MM


def build_tree(points: np.ndarray) -> cKDTree:
    # Disabling balancing/compaction makes construction fast for this uniform cloud.
    return cKDTree(points, compact_nodes=False, balanced_tree=False)


def analyze_mesh(
    label: str,
    path: Path,
    design: Design,
    raw_samples: np.ndarray,
    scale: float,
    radius: float,
    workers: int,
) -> MeshResult:
    started = time.perf_counter()
    centroids, triangle_count = load_triangle_centroids(path)
    tree = build_tree(centroids)
    distances = nearest_distance_min(tree, raw_samples, scale, workers)
    # A strut is empty only when none of its sampled core points has a surface
    # centroid inside the radius. Shared junction regions are deliberately trimmed.
    deleted_mask = distances > radius
    result = MeshResult(
        label=label,
        triangle_count=triangle_count,
        deleted_ids=design.strut_ids[deleted_mask],
        nearest_distance_min_mm=distances,
        elapsed_seconds=time.perf_counter() - started,
    )
    del tree, centroids
    gc.collect()
    return result


def write_deleted_result(
    path: Path,
    result: MeshResult,
    baseline_triangles: int,
    scale: float,
    radius: float,
) -> None:
    deleted_count = int(len(result.deleted_ids))
    deficit = baseline_triangles - result.triangle_count
    payload = {
        "stl": f"{result.label}.stl",
        "deleted_count": deleted_count,
        "deleted_strut_ids": [int(value) for value in result.deleted_ids],
        "triangle_count": result.triangle_count,
        "triangle_deficit_from_0_stl": deficit,
        "triangles_per_deleted_strut": (
            deficit / deleted_count if deleted_count else None
        ),
        "scale_mm_per_design_unit": scale,
        "radius_mm": radius,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def result_record(result: MeshResult, baseline_triangles: int) -> dict[str, Any]:
    deleted_count = int(len(result.deleted_ids))
    deficit = baseline_triangles - result.triangle_count
    return {
        "design": result.label,
        "stl": f"{result.label}.stl",
        "triangle_count": result.triangle_count,
        "triangle_deficit_from_0_stl": deficit,
        "deleted_count": deleted_count,
        "deleted_percent_of_design": deleted_count / 18_468 * 100.0,
        "triangles_per_deleted_strut": deficit / deleted_count if deleted_count else None,
        "analysis_seconds": result.elapsed_seconds,
    }


def evaluate_gates(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {record["design"]: record for record in records}
    counts = [by_label[label]["deleted_count"] for label in MESH_LABELS]
    ratios = {
        label: by_label[label]["triangles_per_deleted_strut"]
        for label in MESH_LABELS[1:]
    }
    negative_control = counts[0] == 0
    half_percent = 88 <= counts[2] <= 96
    monotonic = counts[0] < counts[1] < counts[2] < counts[3]
    deficit = all(value is not None and 170 <= value <= 180 for value in ratios.values())
    return {
        "negative_control_0_stl": {
            "expected": "0 deleted struts",
            "actual": counts[0],
            "pass": negative_control,
        },
        "deleted_count_0.5_stl": {
            "expected": "88-96 deleted struts",
            "actual": counts[2],
            "pass": half_percent,
        },
        "monotonic_counts": {
            "expected": "0.stl < 0.1.stl < 0.5.stl < 1.stl",
            "actual": dict(zip(MESH_LABELS, counts)),
            "pass": monotonic,
        },
        "triangle_deficit_ratio": {
            "expected": "170-180 triangles per deleted strut for every variant",
            "actual": ratios,
            "pass": deficit,
        },
        "overall_pass": negative_control and half_percent and monotonic and deficit,
    }


def make_deleted_centerline_plot(
    path: Path,
    design: Design,
    deleted_ids: np.ndarray,
    scale: float,
) -> None:
    id_to_index = {int(value): index for index, value in enumerate(design.strut_ids)}
    indices = np.asarray([id_to_index[int(value)] for value in deleted_ids], dtype=int)
    segments = (design.endpoint_positions[indices] - DESIGN_CENTER) * scale
    midpoints = segments.mean(axis=1)

    fig = plt.figure(figsize=(9.2, 8.0), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    line_collection = Line3DCollection(
        segments,
        colors="#e34a33",
        linewidths=1.7,
        alpha=0.90,
    )
    axis.add_collection3d(line_collection)
    axis.scatter(
        midpoints[:, 0],
        midpoints[:, 1],
        midpoints[:, 2],
        s=18,
        c="#2b8cbe",
        edgecolors="white",
        linewidths=0.25,
        depthshade=False,
        label="deleted-strut midpoint",
    )
    bound = DESIGN_CENTER * scale
    axis.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound))
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("STL X (mm)")
    axis.set_ylabel("STL Y (mm)")
    axis.set_zlabel("STL Z (mm)")
    axis.set_title(f"Tube-emptiness detections in 0.5.stl ({len(deleted_ids)} struts)")
    axis.view_init(elev=24, azim=-55)
    axis.legend(loc="upper left", frameon=False)
    axis.grid(alpha=0.22)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    path: Path,
    design_path: Path,
    stl_dir: Path,
    records: list[dict[str, Any]],
    gates: dict[str, Any],
    scale: float,
    radius: float,
    sample_count: int,
    baseline_distances: np.ndarray,
    scale_trials: list[dict[str, float]],
    scale_overridden: bool,
    radius_overridden: bool,
) -> None:
    payload = {
        "method": "cKDTree tube-emptiness queries against binary-STL triangle centroids",
        "design_json": str(design_path),
        "stl_directory": str(stl_dir),
        "design_strut_count": 18_468,
        "transform": {
            "formula": "stl_mm = (json_coordinate - 9.0) * scale",
            "center_design_units": [DESIGN_CENTER] * 3,
            "scale_mm_per_design_unit": scale,
        },
        "tube_test": {
            "radius_mm": radius,
            "point_cloud": "triangle centroids",
            "sample_count": sample_count,
            "sample_fraction_start": SAMPLE_START,
            "sample_fraction_end": SAMPLE_END,
            "deletion_rule": "all sampled core points have no centroid within radius",
        },
        "calibration": {
            "baseline": "0.stl",
            "scale_overridden": scale_overridden,
            "radius_overridden": radius_overridden,
            "scale_objective": "minimize mean per-strut nearest-centroid distance on 0.stl",
            "scale_trials": scale_trials,
            "baseline_max_nearest_distance_mm": float(baseline_distances.max()),
            "radius_safety_margin_mm": RADIUS_SAFETY_MARGIN_MM,
            "radius_rounding_mm": RADIUS_ROUNDING_MM,
        },
        "results": records,
        "gates": gates,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    design_path = repo_root / DESIGN_PATH
    stl_dir = repo_root / STL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    design = load_design(design_path)
    raw_samples = raw_centerline_samples(design, args.samples)

    baseline_path = stl_dir / "0.stl"
    baseline_centroids, baseline_triangle_count = load_triangle_centroids(baseline_path)
    baseline_tree = build_tree(baseline_centroids)

    if args.scale is None:
        scale, scale_trials = calibrate_scale(baseline_tree, raw_samples, args.workers)
    else:
        scale = args.scale
        scale_trials = []
    baseline_distances = nearest_distance_min(
        baseline_tree, raw_samples, scale, args.workers
    )
    radius = args.radius if args.radius is not None else calibrate_radius(baseline_distances)
    baseline_result = MeshResult(
        label="0",
        triangle_count=baseline_triangle_count,
        deleted_ids=design.strut_ids[baseline_distances > radius],
        nearest_distance_min_mm=baseline_distances,
        elapsed_seconds=0.0,
    )
    del baseline_tree, baseline_centroids
    gc.collect()

    print(f"Calibrated scale={scale:.4f} mm/design-unit, radius={radius:.3f} mm")
    print(
        f"0.stl: {len(baseline_result.deleted_ids)} deleted / "
        f"{baseline_result.triangle_count:,} triangles"
    )
    mesh_results = [baseline_result]
    for label in MESH_LABELS[1:]:
        result = analyze_mesh(
            label,
            stl_dir / f"{label}.stl",
            design,
            raw_samples,
            scale,
            radius,
            args.workers,
        )
        mesh_results.append(result)
        print(
            f"{label}.stl: {len(result.deleted_ids)} deleted / "
            f"{result.triangle_count:,} triangles ({result.elapsed_seconds:.2f}s)"
        )

    for result in mesh_results:
        write_deleted_result(
            output_dir / f"deleted_struts_{result.label}.json",
            result,
            baseline_triangle_count,
            scale,
            radius,
        )

    records = [result_record(result, baseline_triangle_count) for result in mesh_results]
    gates = evaluate_gates(records)
    write_summary(
        output_dir / "summary.json",
        design_path,
        stl_dir,
        records,
        gates,
        scale,
        radius,
        args.samples,
        baseline_distances,
        scale_trials,
        args.scale is not None,
        args.radius is not None,
    )
    half_percent = next(result for result in mesh_results if result.label == "0.5")
    make_deleted_centerline_plot(
        output_dir / "deleted_centerlines_0.5.png",
        design,
        half_percent.deleted_ids,
        scale,
    )

    print(f"Overall validation: {'PASS' if gates['overall_pass'] else 'FAIL'}")
    print(f"Wrote artifacts to {output_dir}")
    return 0 if gates["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
