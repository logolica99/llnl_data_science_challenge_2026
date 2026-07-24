#!/usr/bin/env python3
"""Recover the ideal lattice-to-CT registration from CT intensities alone.

The held-out registered JSON is opened only after the CT-only fit and registered
JSON have been written. It is used exclusively by the final validation phase.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree


THRESHOLD = 40_129
DOWNSAMPLE = 2
DETECTION_Z_MIN = 50
DETECTION_Z_MAX = 710
EDT_PEAK_THRESHOLD = 2.0  # downsampled voxels
MIN_COMPONENT_VOXELS = 2
MAX_COMPONENT_VOXELS = 999
ICP_KEEP_FRACTION = 0.70
ICP_MAX_ITERATIONS = 60
LOCAL_REFINEMENT_PASSES = 2
LOCAL_PATCH_RADIUS = 14
LOCAL_SEARCH_RADIUS = 8.0
LOCAL_MIN_EDT = 2.5
LOCAL_MAX_EDT = 10.0  # excludes thick CT end-cap regions
LOCAL_MAX_SHIFT = 7.5
LOCAL_KEEP_FRACTION = 0.80

DESIGN_RELATIVE_PATH = Path("data/missing_struts/octet_truss_9x9x9.json")
CT_RELATIVE_PATH = Path(
    "data/missing_struts/tif_stacks/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
GROUND_TRUTH_RELATIVE_PATH = Path(
    "data/missing_struts/registered_jsons/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
)


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation

    @property
    def rotation_deg(self) -> float:
        cosine = np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))


@dataclass
class DesignData:
    document: dict[str, Any]
    node_ids: np.ndarray
    node_positions: np.ndarray
    unique_positions: np.ndarray


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_dir.parents[1],
        help="Repository root (inferred automatically by default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
        help="Artifact destination (default: results beside this script).",
    )
    return parser.parse_args()


def load_design(path: Path) -> DesignData:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    junctions = document.get("junctions")
    if not isinstance(junctions, list) or not junctions:
        raise ValueError(f"{path} has no non-empty junctions list")
    node_ids = np.asarray([int(node["id"]) for node in junctions], dtype=np.int64)
    node_positions = np.asarray([node["position"] for node in junctions], dtype=np.float64)
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Design junction IDs are not unique")
    unique_positions = np.unique(node_positions, axis=0)
    return DesignData(document, node_ids, node_positions, unique_positions)


def foreground_count(volume: np.ndarray, chunk_depth: int = 16) -> int:
    count = 0
    for start in range(0, volume.shape[0], chunk_depth):
        count += int(np.count_nonzero(volume[start : start + chunk_depth] >= THRESHOLD))
    return count


def mask_bounds_xyz(mask_zyx: np.ndarray, factor: int, z_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    occupied_z = np.flatnonzero(mask_zyx.any(axis=(1, 2)))
    occupied_y = np.flatnonzero(mask_zyx.any(axis=(0, 2)))
    occupied_x = np.flatnonzero(mask_zyx.any(axis=(0, 1)))
    if not len(occupied_x) or not len(occupied_y) or not len(occupied_z):
        raise ValueError("Thresholded CT mask is empty")
    low = np.array(
        [occupied_x[0] * factor, occupied_y[0] * factor, occupied_z[0] * factor + z_offset],
        dtype=np.float64,
    )
    high = np.array(
        [occupied_x[-1] * factor, occupied_y[-1] * factor, occupied_z[-1] * factor + z_offset],
        dtype=np.float64,
    )
    return low, high


def detect_ct_nodes(ct_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    volume = tifffile.memmap(ct_path)
    if volume.ndim != 3 or volume.dtype.kind != "u":
        raise ValueError(f"Expected a 3D unsigned-integer TIFF, got {volume.shape} {volume.dtype}")
    source_shape = list(map(int, volume.shape))

    count = foreground_count(volume)
    sampled_mask = np.asarray(volume[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE] >= THRESHOLD)
    full_low, full_high = mask_bounds_xyz(sampled_mask, DOWNSAMPLE)

    z_start_index = math.ceil(DETECTION_Z_MIN / DOWNSAMPLE)
    z_stop_index = DETECTION_Z_MAX // DOWNSAMPLE + 1
    detection_mask = sampled_mask[z_start_index:z_stop_index]
    z_offset = z_start_index * DOWNSAMPLE
    detection_low, detection_high = mask_bounds_xyz(detection_mask, DOWNSAMPLE, z_offset)
    del sampled_mask, volume
    gc.collect()

    edt = ndimage.distance_transform_edt(detection_mask)
    high_radius = edt >= EDT_PEAK_THRESHOLD
    labels, component_count = ndimage.label(high_radius)
    component_sizes = np.bincount(labels.ravel())[1:]
    kept_labels = (
        np.flatnonzero(
            (component_sizes >= MIN_COMPONENT_VOXELS)
            & (component_sizes <= MAX_COMPONENT_VOXELS)
        )
        + 1
    )
    centers_zyx = np.asarray(
        ndimage.center_of_mass(high_radius, labels, kept_labels), dtype=np.float64
    )
    detected_xyz = centers_zyx[:, ::-1] * DOWNSAMPLE
    detected_xyz[:, 2] += z_offset

    metadata = {
        "ct_shape_zyx": list(map(int, detection_mask.shape)),
        "source_ct_shape_zyx": source_shape,
        "threshold": THRESHOLD,
        "foreground_voxel_count": count,
        "downsample_factor": DOWNSAMPLE,
        "detection_z_range_full_resolution": [DETECTION_Z_MIN, DETECTION_Z_MAX],
        "edt_peak_threshold_downsampled_voxels": EDT_PEAK_THRESHOLD,
        "all_component_count": int(component_count),
        "detected_node_count": int(len(detected_xyz)),
        "component_size_range_kept": [MIN_COMPONENT_VOXELS, MAX_COMPONENT_VOXELS],
        "full_sampled_foreground_bounds_xyz": [full_low.tolist(), full_high.tolist()],
        "detection_foreground_bounds_xyz": [
            detection_low.tolist(),
            detection_high.tolist(),
        ],
        "detected_bounds_xyz": [detected_xyz.min(axis=0).tolist(), detected_xyz.max(axis=0).tolist()],
        "elapsed_seconds": time.perf_counter() - started,
    }
    del detection_mask, edt, high_radius, labels, component_sizes, centers_zyx
    gc.collect()
    return detected_xyz, metadata


def solve_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must be matching N x 3 arrays")
    if len(source) < 4:
        raise ValueError("At least four point pairs are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ np.diag(signs) @ right_transpose
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def coarse_initialization(
    source: np.ndarray,
    detection_metadata: dict[str, Any],
) -> SimilarityTransform:
    low, high = np.asarray(
        detection_metadata["detection_foreground_bounds_xyz"], dtype=np.float64
    )
    design_span = np.ptp(source, axis=0)
    scale = float(np.median((high[:2] - low[:2]) / design_span[:2]))
    target_center = np.array(
        [
            (low[0] + high[0]) / 2.0,
            (low[1] + high[1]) / 2.0,
            (detection_metadata["source_ct_shape_zyx"][0] - 1) / 2.0,
        ]
    )
    rotation = np.eye(3)
    translation = target_center - scale * (rotation @ source.mean(axis=0))
    return SimilarityTransform(scale, rotation, translation)


def trimmed_icp(
    source: np.ndarray,
    detected: np.ndarray,
    initial: SimilarityTransform,
) -> tuple[SimilarityTransform, list[dict[str, Any]]]:
    tree = cKDTree(detected)
    transform = initial
    previous_mean = math.inf
    history: list[dict[str, Any]] = []

    for iteration in range(ICP_MAX_ITERATIONS):
        transformed = transform.apply(source)
        distances, indices = tree.query(transformed, k=1)
        cutoff = float(np.quantile(distances, ICP_KEEP_FRACTION))
        keep = distances <= cutoff
        mean_residual = float(distances[keep].mean())
        history.append(
            {
                "iteration": iteration,
                "kept_pairs": int(np.count_nonzero(keep)),
                "trimmed_mean_residual_voxels": mean_residual,
                "all_pair_median_residual_voxels": float(np.median(distances)),
                "cutoff_voxels": cutoff,
            }
        )
        updated = solve_similarity(source[keep], detected[indices[keep]])
        transform = updated
        if abs(previous_mean - mean_residual) < 1e-5:
            break
        previous_mean = mean_residual
    return transform, history


def localize_full_resolution_nodes(
    volume: np.ndarray,
    predictions_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume_shape_xyz = np.asarray(volume.shape[::-1], dtype=int)
    localized: list[np.ndarray] = []
    peak_radii: list[float] = []
    shifts: list[float] = []

    for prediction in predictions_xyz:
        center = np.rint(prediction).astype(int)
        low = np.maximum(center - LOCAL_PATCH_RADIUS, 0)
        high = np.minimum(center + LOCAL_PATCH_RADIUS + 1, volume_shape_xyz)
        patch = np.asarray(
            volume[low[2] : high[2], low[1] : high[1], low[0] : high[0]] >= THRESHOLD
        )
        edt = ndimage.distance_transform_edt(patch)
        z_index, y_index, x_index = np.indices(patch.shape)
        global_x = x_index + low[0]
        global_y = y_index + low[1]
        global_z = z_index + low[2]
        near_prediction = (
            (global_x - prediction[0]) ** 2
            + (global_y - prediction[1]) ** 2
            + (global_z - prediction[2]) ** 2
            <= LOCAL_SEARCH_RADIUS**2
        )
        restricted = np.where(near_prediction, edt, -1.0)
        peak_index = np.unravel_index(int(np.argmax(restricted)), patch.shape)
        peak_radius = float(restricted[peak_index])
        peak_z, peak_y, peak_x = peak_index
        near_peak = (
            (x_index - peak_x) ** 2
            + (y_index - peak_y) ** 2
            + (z_index - peak_z) ** 2
            <= 3.5**2
        )
        weights = (
            np.clip(edt - (peak_radius - 1.0), 0.0, None) ** 2
            * near_prediction
            * near_peak
        )
        if weights.sum() == 0:
            location = np.full(3, np.nan)
        else:
            location = np.array(
                [
                    np.sum(weights * global_x),
                    np.sum(weights * global_y),
                    np.sum(weights * global_z),
                ]
            ) / weights.sum()
        localized.append(location)
        peak_radii.append(peak_radius)
        shifts.append(float(np.linalg.norm(location - prediction)))
    return np.asarray(localized), np.asarray(peak_radii), np.asarray(shifts)


def refine_transform_from_full_resolution_ct(
    ct_path: Path,
    source: np.ndarray,
    initial: SimilarityTransform,
) -> tuple[SimilarityTransform, list[dict[str, Any]]]:
    volume = tifffile.memmap(ct_path)
    transform = initial
    history: list[dict[str, Any]] = []
    for pass_index in range(LOCAL_REFINEMENT_PASSES):
        predictions = transform.apply(source)
        localized, radii, shifts = localize_full_resolution_nodes(volume, predictions)
        valid = (
            np.isfinite(localized[:, 0])
            & (radii >= LOCAL_MIN_EDT)
            & (radii <= LOCAL_MAX_EDT)
            & (shifts < LOCAL_MAX_SHIFT)
        )
        shift_cutoff = float(np.quantile(shifts[valid], LOCAL_KEEP_FRACTION))
        fit = valid & (shifts <= shift_cutoff)
        transform = solve_similarity(source[fit], localized[fit])
        history.append(
            {
                "pass": pass_index,
                "valid_localizations": int(np.count_nonzero(valid)),
                "kept_localizations": int(np.count_nonzero(fit)),
                "shift_cutoff_voxels": shift_cutoff,
                "median_shift_voxels": float(np.median(shifts[valid])),
                "p95_shift_voxels": float(np.quantile(shifts[valid], 0.95)),
                "scale": transform.scale,
                "rotation_deg": transform.rotation_deg,
                "translation": transform.translation.tolist(),
            }
        )
    del volume
    gc.collect()
    return transform, history


def transform_payload(
    transform: SimilarityTransform,
    detection_metadata: dict[str, Any],
    coarse: SimilarityTransform,
    icp_history: list[dict[str, Any]],
    refinement_history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "scale": transform.scale,
        "rotation_matrix": transform.rotation.tolist(),
        "translation": transform.translation.tolist(),
        "rotation_deg": transform.rotation_deg,
        "convention": "ct_xyz = scale * (design_xyz @ rotation_matrix.T) + translation",
        "fit_inputs": "ideal design JSON and CT intensities only",
        "ground_truth_used_for_fit": False,
        "coarse_initialization": {
            "scale": coarse.scale,
            "rotation_matrix": coarse.rotation.tolist(),
            "translation": coarse.translation.tolist(),
        },
        "detection": detection_metadata,
        "trimmed_icp": {
            "keep_fraction": ICP_KEEP_FRACTION,
            "iterations": len(icp_history),
            "history": icp_history,
        },
        "full_resolution_refinement": {
            "passes": LOCAL_REFINEMENT_PASSES,
            "history": refinement_history,
        },
    }


def write_registered_json(
    path: Path,
    design: DesignData,
    transform: SimilarityTransform,
) -> None:
    registered = copy.deepcopy(design.document)
    transformed = transform.apply(design.node_positions)
    for node, position in zip(registered["junctions"], transformed):
        node["position"] = position.tolist()
    path.write_text(json.dumps(registered, indent=2) + "\n", encoding="utf-8")


def fit_ground_truth_transform(
    design: DesignData,
    ground_truth_document: dict[str, Any],
) -> tuple[SimilarityTransform, np.ndarray]:
    positions_by_id = {
        int(node["id"]): np.asarray(node["position"], dtype=np.float64)
        for node in ground_truth_document["junctions"]
    }
    target = np.stack([positions_by_id[int(node_id)] for node_id in design.node_ids])
    transform = solve_similarity(design.node_positions, target)
    return transform, target


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first @ second.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def ct_neighborhood_stats(ct_path: Path, positions_xyz: np.ndarray) -> dict[str, float]:
    volume = tifffile.memmap(ct_path)
    fractions = []
    any_hits = []
    center_hits = []
    for position in positions_xyz:
        x, y, z = np.rint(position).astype(int)
        z0, z1 = max(0, z - 2), min(volume.shape[0], z + 3)
        y0, y1 = max(0, y - 2), min(volume.shape[1], y + 3)
        x0, x1 = max(0, x - 2), min(volume.shape[2], x + 3)
        patch = np.asarray(volume[z0:z1, y0:y1, x0:x1] >= THRESHOLD)
        fractions.append(float(patch.mean()) if patch.size else 0.0)
        any_hits.append(bool(patch.any()))
        center_hits.append(
            bool(0 <= z < volume.shape[0] and 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2] and volume[z, y, x] >= THRESHOLD)
        )
    del volume
    return {
        "foreground_voxel_fraction_5x5x5": float(np.mean(fractions)),
        "nodes_with_any_foreground_5x5x5": float(np.mean(any_hits)),
        "center_voxel_foreground_rate": float(np.mean(center_hits)),
    }


def save_error_histogram(path: Path, errors: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    upper = max(10.0, float(np.quantile(errors, 0.995)))
    axis.hist(errors, bins=np.linspace(0.0, upper, 55), color="#2b8cbe", edgecolor="white")
    axis.axvline(np.median(errors), color="#e34a33", linewidth=2, label=f"median {np.median(errors):.2f}")
    axis.axvline(np.quantile(errors, 0.95), color="#fdbb84", linewidth=2, label=f"p95 {np.quantile(errors, 0.95):.2f}")
    axis.set_xlabel("Per-node registration error (voxels)")
    axis.set_ylabel("Design-node records")
    axis.set_title("CT-only registration error against held-out node positions")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def validate_after_fit(
    ground_truth_path: Path,
    ct_path: Path,
    design: DesignData,
    fitted: SimilarityTransform,
    output_dir: Path,
) -> dict[str, Any]:
    # HARD-RULE BOUNDARY: this is the first and only ground-truth read.
    with ground_truth_path.open(encoding="utf-8") as handle:
        ground_truth_document = json.load(handle)
    ground_truth_transform, ground_truth_positions = fit_ground_truth_transform(
        design, ground_truth_document
    )
    our_positions = fitted.apply(design.node_positions)
    errors = np.linalg.norm(our_positions - ground_truth_positions, axis=1)
    neighborhoods = ct_neighborhood_stats(ct_path, our_positions)
    translation_delta = fitted.translation - ground_truth_transform.translation
    relative_scale_error = abs(fitted.scale / ground_truth_transform.scale - 1.0)
    rotation_magnitude_error = abs(fitted.rotation_deg - ground_truth_transform.rotation_deg)
    rotation_matrix_error = rotation_difference_deg(fitted.rotation, ground_truth_transform.rotation)

    gates = {
        "median_node_error_under_5_voxels": bool(np.median(errors) < 5.0),
        "scale_within_1_percent": bool(relative_scale_error < 0.01),
        "rotation_magnitude_within_0.2_deg": bool(rotation_magnitude_error < 0.2),
        "translation_within_few_voxels": bool(
            np.linalg.norm(translation_delta) < 6.0 and np.max(np.abs(translation_delta)) < 5.0
        ),
        "foreground_fraction_5x5x5_at_least_85_percent": bool(
            neighborhoods["foreground_voxel_fraction_5x5x5"] >= 0.85
        ),
    }
    payload = {
        "validation_only_ground_truth": str(ground_truth_path),
        "ground_truth_read_phase": "after detected_nodes.npy, fitted_transform.json, and our_registered.json were written",
        "ground_truth_used_for_fit": False,
        "node_count": int(len(errors)),
        "node_error_voxels": {
            "median": float(np.median(errors)),
            "mean": float(np.mean(errors)),
            "p95": float(np.quantile(errors, 0.95)),
            "p99": float(np.quantile(errors, 0.99)),
            "maximum": float(np.max(errors)),
        },
        "fitted_transform": {
            "scale": fitted.scale,
            "rotation_deg": fitted.rotation_deg,
            "translation": fitted.translation.tolist(),
            "rotation_matrix": fitted.rotation.tolist(),
        },
        "held_out_transform": {
            "scale": ground_truth_transform.scale,
            "rotation_deg": ground_truth_transform.rotation_deg,
            "translation": ground_truth_transform.translation.tolist(),
            "rotation_matrix": ground_truth_transform.rotation.tolist(),
        },
        "transform_error": {
            "relative_scale": relative_scale_error,
            "rotation_magnitude_deg": rotation_magnitude_error,
            "rotation_matrix_delta_deg": rotation_matrix_error,
            "translation_delta_xyz": translation_delta.tolist(),
            "translation_delta_norm": float(np.linalg.norm(translation_delta)),
        },
        "ct_node_sampling": neighborhoods,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
    }
    save_error_histogram(output_dir / "registration_error_hist.png", errors)
    return payload


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = repo_root / DESIGN_RELATIVE_PATH
    ct_path = repo_root / CT_RELATIVE_PATH
    ground_truth_path = repo_root / GROUND_TRUTH_RELATIVE_PATH

    started = time.perf_counter()
    design = load_design(design_path)
    print(f"Loaded {len(design.node_ids):,} node records ({len(design.unique_positions):,} unique positions)")
    detected, detection_metadata = detect_ct_nodes(ct_path)
    np.save(output_dir / "detected_nodes.npy", detected)
    print(f"Detected {len(detected):,} CT node candidates")

    coarse = coarse_initialization(design.unique_positions, detection_metadata)
    icp_transform, icp_history = trimmed_icp(
        design.unique_positions, detected, coarse
    )
    fitted, refinement_history = refine_transform_from_full_resolution_ct(
        ct_path, design.unique_positions, icp_transform
    )
    fitted_payload = transform_payload(
        fitted,
        detection_metadata,
        coarse,
        icp_history,
        refinement_history,
    )
    fitted_payload["elapsed_before_validation_seconds"] = time.perf_counter() - started
    (output_dir / "fitted_transform.json").write_text(
        json.dumps(fitted_payload, indent=2) + "\n", encoding="utf-8"
    )
    write_registered_json(output_dir / "our_registered.json", design, fitted)
    print(
        f"CT-only fit frozen: scale={fitted.scale:.6f}, "
        f"rotation={fitted.rotation_deg:.4f} deg, translation={fitted.translation}"
    )

    validation = validate_after_fit(
        ground_truth_path, ct_path, design, fitted, output_dir
    )
    validation["total_elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    error = validation["node_error_voxels"]
    print(
        f"Held-out validation: median={error['median']:.3f}, "
        f"mean={error['mean']:.3f}, p95={error['p95']:.3f} voxels"
    )
    print(f"Overall validation: {'PASS' if validation['overall_pass'] else 'FAIL'}")
    print(f"Wrote artifacts to {output_dir}")
    return 0 if validation["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
