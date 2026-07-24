#!/usr/bin/env python3
"""Compare a completed CT-only fit with a held-out registered graph.

This is deliberately a separate executable.  It refuses to run unless the
CT-only completion marker and artifact hashes verify first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from registration_core import (
    SimilarityTransform,
    load_design,
    load_json,
    rotation_difference_deg,
    sha256_file,
    solve_similarity,
    write_json,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fit-dir",
        type=Path,
        default=script_dir / "results/current",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=repo_root / "data/missing_struts/octet_truss_9x9x9.json",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=repo_root
        / "data/missing_struts/registered_jsons/"
        "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config.default.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: <fit-dir>/held_out_validation.json",
    )
    return parser.parse_args()


def verify_completion(fit_dir: Path) -> dict[str, Any]:
    marker_path = fit_dir / "FIT_COMPLETE.json"
    if not marker_path.is_file():
        raise RuntimeError(
            "Refusing ground-truth access: FIT_COMPLETE.json is absent"
        )
    marker = load_json(marker_path)
    if marker.get("ground_truth_used_for_fit") is not False:
        raise RuntimeError("Fit completion marker does not prove ground-truth isolation")
    if marker.get("fit_completed_before_ground_truth_access") is not True:
        raise RuntimeError("Fit completion ordering is not proven")
    for key in ("fit_manifest", "fitted_transform", "our_registered"):
        record = marker[key]
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Fit artifact failed integrity check: {key}")
    manifest = load_json(Path(marker["fit_manifest"]["path"]))
    if manifest.get("ground_truth_used_for_fit") is not False:
        raise RuntimeError("Fit manifest reports ground-truth use")
    return marker


def positions_by_id(document: dict[str, Any], node_ids: np.ndarray) -> np.ndarray:
    lookup = {
        int(node["id"]): np.asarray(node["position"], dtype=np.float64)
        for node in document["junctions"]
    }
    return np.stack([lookup[int(node_id)] for node_id in node_ids])


def save_histogram(path: Path, errors: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    upper = max(10.0, float(np.quantile(errors, 0.995)))
    axis.hist(
        errors,
        bins=np.linspace(0.0, upper, 55),
        color="#2b8cbe",
        edgecolor="white",
    )
    axis.axvline(
        np.median(errors),
        color="#e34a33",
        linewidth=2,
        label=f"median {np.median(errors):.2f}",
    )
    axis.axvline(
        np.quantile(errors, 0.95),
        color="#fdbb84",
        linewidth=2,
        label=f"p95 {np.quantile(errors, 0.95):.2f}",
    )
    axis.set_xlabel("Per-node registration error (voxels)")
    axis.set_ylabel("Design-node records")
    axis.set_title("CT-only registration vs held-out node positions")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    fit_dir = args.fit_dir.resolve()
    design_path = args.design.resolve()
    ground_truth_path = args.ground_truth.resolve()
    config = load_json(args.config.resolve())
    output_path = (
        args.output.resolve()
        if args.output
        else fit_dir / "held_out_validation.json"
    )

    completion = verify_completion(fit_dir)
    # HARD BOUNDARY: this is the first ground-truth read in this process.
    ground_truth_read_started = time.time()
    ground_truth = load_json(ground_truth_path)
    ground_truth_sha256 = sha256_file(ground_truth_path)

    design = load_design(design_path)
    ours = load_json(fit_dir / "our_registered.json")
    our_positions = positions_by_id(ours, design.node_ids)
    ground_truth_positions = positions_by_id(ground_truth, design.node_ids)
    errors_xyz = our_positions - ground_truth_positions
    errors = np.linalg.norm(errors_xyz, axis=1)
    expected_transform = solve_similarity(
        design.node_positions, ground_truth_positions
    )
    fit_payload = load_json(fit_dir / "fitted_transform.json")
    fitted_transform = SimilarityTransform(
        scale=float(fit_payload["scale"]),
        rotation=np.asarray(fit_payload["rotation_matrix"], dtype=np.float64),
        translation=np.asarray(fit_payload["translation"], dtype=np.float64),
    )
    translation_delta = fitted_transform.translation - expected_transform.translation
    relative_scale_error = abs(
        fitted_transform.scale / expected_transform.scale - 1.0
    )
    rotation_magnitude_error = abs(
        fitted_transform.rotation_deg - expected_transform.rotation_deg
    )
    rotation_matrix_error = rotation_difference_deg(
        fitted_transform.rotation, expected_transform.rotation
    )

    spatial: dict[str, Any] = {}
    for axis, name in enumerate("xyz"):
        order = np.argsort(ground_truth_positions[:, axis])
        bins = np.array_split(order, 5)
        medians = [float(np.median(errors[index])) for index in bins]
        spatial[name] = {
            "quintile_median_node_error_voxels": medians,
            "range": float(max(medians) - min(medians)),
        }

    image_validation = load_json(fit_dir / "image_validation.json")
    downstream = load_json(fit_dir / "downstream_tolerance.json")
    params = config["held_out_validation"]
    gates = {
        "median_node_error": bool(
            np.median(errors)
            <= float(params["maximum_median_node_error_voxels"])
        ),
        "p95_node_error": bool(
            np.quantile(errors, 0.95)
            <= float(params["maximum_p95_node_error_voxels"])
        ),
        "relative_scale_error": bool(
            relative_scale_error
            <= float(params["maximum_relative_scale_error"])
        ),
        "rotation_magnitude_error": bool(
            rotation_magnitude_error
            <= float(params["maximum_rotation_magnitude_error_degrees"])
        ),
        "translation_error": bool(
            np.linalg.norm(translation_delta)
            <= float(params["maximum_translation_error_norm_voxels"])
        ),
        "independent_node_foreground": bool(
            image_validation["junctions"][
                "mean_5x5x5_foreground_fraction"
            ]
            >= float(params["minimum_node_foreground_fraction"])
        ),
    }
    measured_radius = float(
        image_validation["corridors"]["measured_strut_radius_voxels"]
    )
    corridor_margin = float(
        downstream["recommended_corridor_margin_voxels"]
    )
    downstream_validation = {
        "held_out_p95_error_voxels": float(np.quantile(errors, 0.95)),
        "measured_strut_radius_voxels": measured_radius,
        "recommended_corridor_margin_voxels": corridor_margin,
        "p95_within_radius_plus_margin": bool(
            np.quantile(errors, 0.95) <= measured_radius + corridor_margin
        ),
        "median_within_measured_radius": bool(
            np.median(errors) <= measured_radius
        ),
    }
    downstream_validation["overall_pass"] = bool(
        downstream_validation["p95_within_radius_plus_margin"]
        and downstream_validation["median_within_measured_radius"]
    )
    gates["downstream_tolerance_against_held_out"] = downstream_validation[
        "overall_pass"
    ]
    overall_pass = bool(all(gates.values()))
    payload = {
        "schema_version": 1,
        "validation_only_ground_truth": str(ground_truth_path),
        "ground_truth_sha256": ground_truth_sha256,
        "ground_truth_read_phase": (
            "after FIT_COMPLETE.json and all CT-only artifact hashes verified"
        ),
        "ground_truth_read_unix_time": ground_truth_read_started,
        "ground_truth_used_for_fit": False,
        "fit_completion_marker": completion,
        "node_count": int(len(errors)),
        "node_error_voxels": {
            "median": float(np.median(errors)),
            "mean": float(np.mean(errors)),
            "p95": float(np.quantile(errors, 0.95)),
            "p99": float(np.quantile(errors, 0.99)),
            "maximum": float(np.max(errors)),
        },
        "signed_axis_error_voxels": {
            "mean_xyz": errors_xyz.mean(axis=0).tolist(),
            "median_xyz": np.median(errors_xyz, axis=0).tolist(),
            "minimum_xyz": errors_xyz.min(axis=0).tolist(),
            "maximum_xyz": errors_xyz.max(axis=0).tolist(),
        },
        "fitted_transform": fitted_transform.to_dict(),
        "held_out_transform": expected_transform.to_dict(),
        "transform_error": {
            "relative_scale": relative_scale_error,
            "rotation_magnitude_degrees": rotation_magnitude_error,
            "rotation_matrix_delta_degrees": rotation_matrix_error,
            "translation_delta_xyz": translation_delta.tolist(),
            "translation_delta_norm_voxels": float(
                np.linalg.norm(translation_delta)
            ),
        },
        "spatial_error": spatial,
        "downstream_tolerance": downstream_validation,
        "gates": gates,
        "overall_pass": overall_pass,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output_path, payload)
    save_histogram(fit_dir / "held_out_error_histogram.png", errors)
    print(
        f"Held-out comparison: median={np.median(errors):.3f}, "
        f"p95={np.quantile(errors, 0.95):.3f}, "
        f"overall={'PASS' if overall_pass else 'FAIL'}"
    )
    print(f"Wrote {output_path}")
    return 0 if overall_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
