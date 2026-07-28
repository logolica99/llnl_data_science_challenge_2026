"""Core algorithms and trust gates for autonomous CT lattice registration.

This module never opens a supplied/ground-truth registration.  Ground-truth
comparison is isolated in ``validate_against_ground_truth.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from numbers import Real
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "rotation_matrix": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "rotation_deg": self.rotation_deg,
            "convention": "ct_xyz = scale * (design_xyz @ rotation_matrix.T) + translation",
        }


@dataclass(frozen=True)
class DesignData:
    document: dict[str, Any]
    node_ids: np.ndarray
    node_positions: np.ndarray
    unique_positions: np.ndarray
    strut_ids: np.ndarray
    strut_junction_ids: np.ndarray


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_ct_volume(path: Path) -> np.ndarray:
    """Open a 3-D TIFF or NumPy CT volume without copying it into memory."""
    if path.suffix.lower() == ".npy":
        volume = np.load(path, mmap_mode="r")
    else:
        volume = tifffile.memmap(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D CT volume, got {volume.shape} from {path}")
    return volume


def load_design(path: Path) -> DesignData:
    document = load_json(path)
    junctions = document.get("junctions")
    struts = document.get("struts")
    if not isinstance(junctions, list) or not junctions:
        raise ValueError(f"{path} has no non-empty junctions list")
    if not isinstance(struts, list) or not struts:
        raise ValueError(f"{path} has no non-empty struts list")
    node_ids = np.asarray([int(node["id"]) for node in junctions], dtype=np.int64)
    node_positions = np.asarray(
        [node["position"] for node in junctions], dtype=np.float64
    )
    strut_ids = np.asarray([int(strut["id"]) for strut in struts], dtype=np.int64)
    strut_junction_ids = np.asarray(
        [
            [int(strut["junction0"]), int(strut["junction1"])]
            for strut in struts
        ],
        dtype=np.int64,
    )
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("Design junction IDs are not unique")
    if len(np.unique(strut_ids)) != len(strut_ids):
        raise ValueError("Design strut IDs are not unique")
    if set(node_ids.tolist()) != set(range(len(node_ids))):
        raise ValueError("V2 currently requires sequential junction IDs")
    return DesignData(
        document=document,
        node_ids=node_ids,
        node_positions=node_positions,
        unique_positions=np.unique(node_positions, axis=0),
        strut_ids=strut_ids,
        strut_junction_ids=strut_junction_ids,
    )


def exact_uint16_histogram(
    volume: np.ndarray,
    chunk_depth: int,
    edge_slices_excluded: int = 0,
) -> tuple[np.ndarray, int]:
    if volume.ndim != 3 or volume.dtype.kind != "u":
        raise ValueError(f"Expected 3D unsigned CT volume, got {volume.shape} {volume.dtype}")
    start = int(edge_slices_excluded)
    stop = volume.shape[0] - int(edge_slices_excluded)
    if not 0 <= start < stop <= volume.shape[0]:
        raise ValueError("edge_slices_excluded removes the complete CT volume")
    histogram = np.zeros(65536, dtype=np.int64)
    voxel_count = 0
    for z0 in range(start, stop, chunk_depth):
        chunk = np.asarray(volume[z0 : min(z0 + chunk_depth, stop)])
        histogram += np.bincount(chunk.ravel(), minlength=65536)
        voxel_count += int(chunk.size)
    return histogram, voxel_count


def deterministic_65536_histogram(
    volume: np.ndarray,
    chunk_depth: int,
    edge_slices_excluded: int = 0,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Count every voxel in a deterministic 65,536-bin intensity histogram.

    Native uint16 data uses its exact integer levels. Floating-point CT data is
    first mapped affinely from its full-volume finite minimum and maximum to
    uint16 bins. The full source volume is scanned (not sampled), and the
    mapping is recorded so the Otsu bin can be converted back to native units.
    """
    if volume.dtype == np.uint16:
        histogram, voxel_count = exact_uint16_histogram(
            volume,
            chunk_depth=chunk_depth,
            edge_slices_excluded=edge_slices_excluded,
        )
        return histogram, voxel_count, {
            "encoding": "native_uint16_exact",
            "native_dtype": str(volume.dtype),
            "native_min": 0.0,
            "native_max": 65535.0,
            "bin_count": 65536,
        }
    if volume.dtype.kind not in "fiu":
        raise ValueError(
            f"Expected numeric CT volume, got {volume.shape} {volume.dtype}"
        )
    start = int(edge_slices_excluded)
    stop = volume.shape[0] - int(edge_slices_excluded)
    if not 0 <= start < stop <= volume.shape[0]:
        raise ValueError("edge_slices_excluded removes the complete CT volume")

    native_min = math.inf
    native_max = -math.inf
    voxel_count = 0
    for z0 in range(start, stop, chunk_depth):
        chunk = np.asarray(volume[z0 : min(z0 + chunk_depth, stop)])
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            native_min = min(native_min, float(finite.min()))
            native_max = max(native_max, float(finite.max()))
            voxel_count += int(finite.size)
    if not math.isfinite(native_min) or native_max <= native_min:
        raise ValueError("CT volume has no usable finite intensity range")

    histogram = np.zeros(65536, dtype=np.int64)
    scale = 65535.0 / (native_max - native_min)
    for z0 in range(start, stop, chunk_depth):
        chunk = np.asarray(
            volume[z0 : min(z0 + chunk_depth, stop)], dtype=np.float64
        )
        finite = np.isfinite(chunk)
        quantized = np.rint((chunk[finite] - native_min) * scale)
        quantized = np.clip(quantized, 0, 65535).astype(np.uint16)
        histogram += np.bincount(quantized, minlength=65536)
    return histogram, voxel_count, {
        "encoding": "full_volume_affine_uint16",
        "native_dtype": str(volume.dtype),
        "native_min": native_min,
        "native_max": native_max,
        "native_units_per_bin": (native_max - native_min) / 65535.0,
        "bin_count": 65536,
    }


def otsu_from_histogram(histogram: np.ndarray) -> tuple[int, float]:
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.shape != (65536,) or histogram.sum() <= 0:
        raise ValueError("Otsu requires a non-empty 65536-bin uint16 histogram")
    levels = np.arange(histogram.size, dtype=np.float64)
    total = histogram.sum()
    cumulative_weight = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * levels)
    background_weight = cumulative_weight[:-1]
    foreground_weight = total - background_weight
    valid = (background_weight > 0) & (foreground_weight > 0)
    background_mean = np.zeros_like(background_weight)
    foreground_mean = np.zeros_like(background_weight)
    background_mean[valid] = cumulative_sum[:-1][valid] / background_weight[valid]
    foreground_mean[valid] = (
        cumulative_sum[-1] - cumulative_sum[:-1][valid]
    ) / foreground_weight[valid]
    between = np.zeros_like(background_weight)
    between[valid] = (
        background_weight[valid]
        * foreground_weight[valid]
        * (background_mean[valid] - foreground_mean[valid]) ** 2
    )
    threshold = int(np.argmax(between))
    mean = float(np.sum(histogram * levels) / total)
    total_variance = float(np.sum(histogram * (levels - mean) ** 2))
    separability = float(between[threshold] / (total * total_variance))
    return threshold, separability


def histogram_diagnostics(
    histogram: np.ndarray,
    threshold: int,
    separability: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    params = config["histogram"]
    histogram = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(histogram.size, dtype=np.float64)
    total = float(histogram.sum())
    background = histogram[:threshold]
    foreground = histogram[threshold:]
    background_weight = float(background.sum())
    foreground_weight = float(foreground.sum())
    foreground_fraction = foreground_weight / total

    def weighted_stats(counts: np.ndarray, values: np.ndarray) -> tuple[float, float]:
        weight = float(counts.sum())
        if weight <= 0:
            return math.nan, math.nan
        mean = float(np.sum(counts * values) / weight)
        variance = float(np.sum(counts * (values - mean) ** 2) / weight)
        return mean, variance

    bg_mean, bg_var = weighted_stats(background, levels[:threshold])
    fg_mean, fg_var = weighted_stats(foreground, levels[threshold:])
    pooled_sigma = math.sqrt(max((bg_var + fg_var) / 2.0, 1e-12))
    class_separation_sigma = abs(fg_mean - bg_mean) / pooled_sigma

    coarse_bins = int(params["coarse_bins"])
    if 65536 % coarse_bins:
        raise ValueError("histogram.coarse_bins must evenly divide 65536")
    coarse = histogram.reshape(coarse_bins, -1).sum(axis=1)
    smoothed = ndimage.gaussian_filter1d(
        coarse.astype(np.float64),
        float(params["peak_smoothing_sigma_bins"]),
    )
    prominence = max(
        1.0,
        float(smoothed.max()) * float(params["peak_prominence_fraction"]),
    )
    peaks, properties = signal.find_peaks(
        smoothed,
        prominence=prominence,
        distance=max(2, coarse_bins // 128),
    )
    significant_peak_count = int(len(peaks))
    gates = {
        "foreground_fraction_plausible": bool(
            float(params["minimum_foreground_fraction"])
            <= foreground_fraction
            <= float(params["maximum_foreground_fraction"])
        ),
        "otsu_separability_sufficient": bool(
            separability >= float(params["minimum_otsu_separability"])
        ),
        "class_mean_separation_sufficient": bool(
            class_separation_sigma
            >= float(params["minimum_class_mean_separation_sigma"])
        ),
        "histogram_not_unimodal": bool(
            significant_peak_count >= int(params["minimum_significant_peaks"])
        ),
    }
    return {
        "threshold": threshold,
        "voxel_count": int(total),
        "foreground_voxel_count": int(foreground_weight),
        "foreground_fraction": foreground_fraction,
        "otsu_separability": separability,
        "background_mean": bg_mean,
        "foreground_mean": fg_mean,
        "class_mean_separation_sigma": class_separation_sigma,
        "significant_peak_count": significant_peak_count,
        "significant_peak_centers_uint16": (
            (peaks + 0.5) * (65536 / coarse_bins)
        ).tolist(),
        "significant_peak_prominences": properties.get(
            "prominences", np.array([], dtype=float)
        ).tolist(),
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
    }


def compute_per_scan_threshold(
    ct_path: Path,
    config: dict[str, Any],
) -> tuple[float | int, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    volume = load_ct_volume(ct_path)
    params = config["histogram"]
    histogram, voxel_count, encoding = deterministic_65536_histogram(
        volume,
        chunk_depth=int(params["chunk_depth"]),
        edge_slices_excluded=int(params["edge_slices_excluded"]),
    )
    del volume
    threshold_bin, separability = otsu_from_histogram(histogram)
    diagnostics = histogram_diagnostics(
        histogram, threshold_bin, separability, config
    )
    if encoding["encoding"] == "native_uint16_exact":
        threshold: float | int = threshold_bin
    else:
        threshold = float(
            encoding["native_min"]
            + threshold_bin * encoding["native_units_per_bin"]
        )
    diagnostics["threshold_histogram_bin"] = threshold_bin
    diagnostics["threshold"] = threshold
    diagnostics["histogram_encoding"] = encoding
    diagnostics["source_voxel_count"] = voxel_count
    diagnostics["elapsed_seconds"] = time.perf_counter() - started
    if not diagnostics["overall_pass"]:
        failed = [name for name, passed in diagnostics["gates"].items() if not passed]
        raise ValueError(
            "Per-scan histogram rejected before registration: " + ", ".join(failed)
        )
    return threshold, histogram, diagnostics


def mask_bounds_xyz(
    mask_zyx: np.ndarray,
    factor: int,
    z_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
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


def detect_ct_nodes(
    ct_path: Path,
    threshold: Real,
    config: dict[str, Any],
    *,
    edt_peak_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    params = config["detection"]
    factor = int(params["downsample_factor"])
    edt_threshold = float(
        params["edt_peak_threshold_downsampled_voxels"]
        if edt_peak_threshold is None
        else edt_peak_threshold
    )
    volume = load_ct_volume(ct_path)
    source_shape = np.asarray(volume.shape, dtype=int)
    sampled_mask = np.asarray(volume[::factor, ::factor, ::factor] >= threshold)
    full_low, full_high = mask_bounds_xyz(sampled_mask, factor)
    margin = int(
        round(source_shape[0] * float(params["central_z_margin_fraction"]))
    )
    z_start_index = math.ceil(margin / factor)
    z_stop_index = (source_shape[0] - margin - 1) // factor + 1
    detection_mask = sampled_mask[z_start_index:z_stop_index]
    z_offset = z_start_index * factor
    detection_low, detection_high = mask_bounds_xyz(
        detection_mask, factor, z_offset
    )
    del sampled_mask, volume

    edt = ndimage.distance_transform_edt(detection_mask)
    high_radius = edt >= edt_threshold
    labels, component_count = ndimage.label(high_radius)
    component_sizes = np.bincount(labels.ravel())[1:]
    kept_labels = (
        np.flatnonzero(
            (component_sizes >= int(params["minimum_component_voxels"]))
            & (component_sizes <= int(params["maximum_component_voxels"]))
        )
        + 1
    )
    centers_zyx = np.asarray(
        ndimage.center_of_mass(high_radius, labels, kept_labels),
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(centers_zyx):
        raise ValueError(
            "Node detection produced no components inside the configured size range"
        )
    detected_xyz = centers_zyx[:, ::-1] * factor
    detected_xyz[:, 2] += z_offset
    metadata = {
        "threshold": threshold,
        "source_ct_shape_zyx": source_shape.tolist(),
        "downsample_factor": factor,
        "central_z_margin_full_resolution_voxels": margin,
        "detection_z_range_full_resolution": [
            z_offset,
            int((z_stop_index - 1) * factor),
        ],
        "edt_peak_threshold_downsampled_voxels": edt_threshold,
        "all_component_count": int(component_count),
        "detected_node_count": int(len(detected_xyz)),
        "component_size_range_kept": [
            int(params["minimum_component_voxels"]),
            int(params["maximum_component_voxels"]),
        ],
        "full_sampled_foreground_bounds_xyz": [
            full_low.tolist(),
            full_high.tolist(),
        ],
        "detection_foreground_bounds_xyz": [
            detection_low.tolist(),
            detection_high.tolist(),
        ],
        "detected_bounds_xyz": [
            detected_xyz.min(axis=0).tolist(),
            detected_xyz.max(axis=0).tolist(),
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    return detected_xyz, metadata


def solve_similarity(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
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
    if source_variance <= 0:
        raise ValueError("Degenerate source points have zero variance")
    scale = float(np.sum(singular_values * signs) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = first @ second.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def coarse_initialization(
    source: np.ndarray,
    bounds_xyz: Iterable[Iterable[float]],
) -> SimilarityTransform:
    low, high = np.asarray(bounds_xyz, dtype=np.float64)
    design_span = np.ptp(source, axis=0)
    valid = design_span > 0
    scale = float(np.median((high[valid] - low[valid]) / design_span[valid]))
    rotation = np.eye(3)
    target_center = (low + high) / 2.0
    translation = target_center - scale * (rotation @ source.mean(axis=0))
    return SimilarityTransform(scale, rotation, translation)


def trimmed_icp(
    source: np.ndarray,
    detected: np.ndarray,
    initial: SimilarityTransform,
    *,
    keep_fraction: float,
    max_iterations: int,
    convergence_epsilon: float,
) -> tuple[SimilarityTransform, list[dict[str, Any]], float]:
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    tree = cKDTree(detected)
    transform = initial
    previous_mean = math.inf
    history: list[dict[str, Any]] = []
    for iteration in range(max_iterations):
        transformed = transform.apply(source)
        distances, indices = tree.query(transformed, k=1)
        cutoff = float(np.quantile(distances, keep_fraction))
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
        if abs(previous_mean - mean_residual) < convergence_epsilon:
            break
        previous_mean = mean_residual
    transformed = transform.apply(source)
    distances, _ = tree.query(transformed, k=1)
    objective = float(np.mean(np.sort(distances)[: max(4, int(len(distances) * keep_fraction))]))
    return transform, history, objective


def make_multistarts(
    source: np.ndarray,
    base: SimilarityTransform,
    config: dict[str, Any],
) -> list[SimilarityTransform]:
    params = config["fitting"]
    target_center = base.apply(source).mean(axis=0)
    source_center = source.mean(axis=0)
    starts: list[SimilarityTransform] = []
    for scale_multiplier in params["scale_multipliers"]:
        for angles in params["rotation_perturbations_degrees_xyz"]:
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            scale = base.scale * float(scale_multiplier)
            translation = target_center - scale * (rotation @ source_center)
            starts.append(SimilarityTransform(scale, rotation, translation))
    return starts


def multistart_fit(
    source: np.ndarray,
    detected: np.ndarray,
    base: SimilarityTransform,
    config: dict[str, Any],
    *,
    keep_fraction: float | None = None,
) -> tuple[SimilarityTransform, dict[str, Any]]:
    params = config["fitting"]
    keep = float(
        params["icp_keep_fraction"] if keep_fraction is None else keep_fraction
    )
    results: list[dict[str, Any]] = []
    transforms: list[SimilarityTransform] = []
    for index, initial in enumerate(make_multistarts(source, base, config)):
        fitted, history, objective = trimmed_icp(
            source,
            detected,
            initial,
            keep_fraction=keep,
            max_iterations=int(params["icp_max_iterations"]),
            convergence_epsilon=float(params["icp_convergence_epsilon"]),
        )
        transforms.append(fitted)
        results.append(
            {
                "start_index": index,
                "initial": initial.to_dict(),
                "fitted": fitted.to_dict(),
                "objective": objective,
                "iterations": len(history),
                "final_iteration": history[-1],
            }
        )
    objectives = np.asarray([entry["objective"] for entry in results])
    best_index = int(np.argmin(objectives))
    best = transforms[best_index]
    objective_limit = float(
        objectives[best_index]
        * (1.0 + float(params["near_optimal_objective_fraction"]))
    )
    near_indices = np.flatnonzero(objectives <= objective_limit)
    near_shifts = []
    converged_count = 0
    for index in near_indices:
        shift = np.linalg.norm(
            transforms[int(index)].apply(source) - best.apply(source),
            axis=1,
        )
        near_shifts.extend(shift.tolist())
        if float(np.quantile(shift, 0.95)) <= float(
            params["maximum_multistart_p95_spread_voxels"]
        ):
            converged_count += 1
    p95_spread = float(np.quantile(near_shifts, 0.95)) if near_shifts else math.inf
    converged_fraction = (
        converged_count / len(near_indices) if len(near_indices) else 0.0
    )
    gates = {
        "enough_near_optimal_starts": bool(
            len(near_indices) >= int(params["minimum_near_optimal_starts"])
        ),
        "near_optimal_starts_agree": bool(
            p95_spread
            <= float(params["maximum_multistart_p95_spread_voxels"])
        ),
        "converged_start_fraction_sufficient": bool(
            converged_fraction >= float(params["minimum_converged_start_fraction"])
        ),
    }
    return best, {
        "keep_fraction": keep,
        "best_start_index": best_index,
        "best_objective": float(objectives[best_index]),
        "near_optimal_start_indices": near_indices.tolist(),
        "near_optimal_p95_prediction_spread_voxels": p95_spread,
        "near_optimal_converged_fraction": converged_fraction,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
        "starts": results,
    }


def localize_full_resolution_nodes(
    volume: np.ndarray,
    predictions_xyz: np.ndarray,
    threshold: Real,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = config["refinement"]
    patch_radius = int(params["patch_radius_voxels"])
    search_radius = float(params["search_radius_voxels"])
    volume_shape_xyz = np.asarray(volume.shape[::-1], dtype=int)
    localized: list[np.ndarray] = []
    peak_radii: list[float] = []
    shifts: list[float] = []
    for prediction in predictions_xyz:
        center = np.rint(prediction).astype(int)
        low = np.maximum(center - patch_radius, 0)
        high = np.minimum(center + patch_radius + 1, volume_shape_xyz)
        patch = np.asarray(
            volume[
                low[2] : high[2],
                low[1] : high[1],
                low[0] : high[0],
            ]
            >= threshold
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
            <= search_radius**2
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
    return (
        np.asarray(localized),
        np.asarray(peak_radii),
        np.asarray(shifts),
    )


def refine_transform(
    ct_path: Path,
    source: np.ndarray,
    initial: SimilarityTransform,
    threshold: Real,
    config: dict[str, Any],
) -> tuple[SimilarityTransform, list[dict[str, Any]]]:
    params = config["refinement"]
    volume = load_ct_volume(ct_path)
    transform = initial
    history: list[dict[str, Any]] = []
    for pass_index in range(int(params["passes"])):
        predictions = transform.apply(source)
        localized, radii, shifts = localize_full_resolution_nodes(
            volume, predictions, threshold, config
        )
        valid = (
            np.isfinite(localized[:, 0])
            & (radii >= float(params["minimum_local_edt_voxels"]))
            & (radii <= float(params["maximum_local_edt_voxels"]))
            & (shifts < float(params["maximum_local_shift_voxels"]))
        )
        if np.count_nonzero(valid) < 4:
            raise ValueError("Full-resolution refinement has fewer than four valid nodes")
        cutoff = float(np.quantile(shifts[valid], float(params["keep_fraction"])))
        fit = valid & (shifts <= cutoff)
        transform = solve_similarity(source[fit], localized[fit])
        history.append(
            {
                "pass": pass_index,
                "valid_localizations": int(np.count_nonzero(valid)),
                "kept_localizations": int(np.count_nonzero(fit)),
                "shift_cutoff_voxels": cutoff,
                "median_shift_voxels": float(np.median(shifts[valid])),
                "p95_shift_voxels": float(np.quantile(shifts[valid], 0.95)),
                "transform": transform.to_dict(),
            }
        )
    del volume
    return transform, history


def split_candidates(
    candidates: np.ndarray,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("candidate holdout fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(candidates))
    holdout_count = max(1, int(round(len(candidates) * holdout_fraction)))
    holdout_indices = np.sort(permutation[:holdout_count])
    fit_indices = np.sort(permutation[holdout_count:])
    return (
        candidates[fit_indices],
        candidates[holdout_indices],
        fit_indices,
        holdout_indices,
    )


def write_registered_json(
    path: Path,
    design: DesignData,
    transform: SimilarityTransform,
) -> None:
    registered = copy.deepcopy(design.document)
    transformed = transform.apply(design.node_positions)
    for node, position in zip(registered["junctions"], transformed):
        node["position"] = position.tolist()
    write_json(path, registered)


def longest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def perpendicular_bases(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    references = np.tile(np.array([0.0, 0.0, 1.0]), (len(directions), 1))
    near_parallel = np.abs(directions[:, 2]) > 0.9
    references[near_parallel] = np.array([0.0, 1.0, 0.0])
    first = np.cross(directions, references)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(directions, first)
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    return first, second


def corridor_image_validation(
    ct_path: Path,
    design: DesignData,
    transform: SimilarityTransform,
    threshold: Real,
    holdout_candidates: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    params = config["image_validation"]
    registered_nodes = transform.apply(design.node_positions)
    volume = load_ct_volume(ct_path)
    shape_xyz = np.asarray(volume.shape[::-1], dtype=int)
    patch_radius = int(params["junction_patch_radius_voxels"])
    junction_fractions = []
    junction_any = []
    for x_f, y_f, z_f in registered_nodes:
        x, y, z = np.rint([x_f, y_f, z_f]).astype(int)
        z0, z1 = max(0, z - patch_radius), min(volume.shape[0], z + patch_radius + 1)
        y0, y1 = max(0, y - patch_radius), min(volume.shape[1], y + patch_radius + 1)
        x0, x1 = max(0, x - patch_radius), min(volume.shape[2], x + patch_radius + 1)
        patch = np.asarray(volume[z0:z1, y0:y1, x0:x1] >= threshold)
        junction_fractions.append(float(patch.mean()) if patch.size else 0.0)
        junction_any.append(bool(patch.any()))

    predicted_unique = transform.apply(design.unique_positions)
    predicted_tree = cKDTree(predicted_unique)
    holdout_distances, _ = predicted_tree.query(holdout_candidates, k=1)

    edge_nodes = design.strut_junction_ids
    start = registered_nodes[edge_nodes[:, 0]]
    end = registered_nodes[edge_nodes[:, 1]]
    midpoints = (start + end) / 2.0
    directions = end - start
    first_basis, second_basis = perpendicular_bases(directions)
    axial_count = int(params["corridor_axial_samples"])
    max_radius = int(params["corridor_max_radius_voxels"])
    angular_count = int(params["corridor_angular_samples"])
    axial_t = np.linspace(0.15, 0.85, axial_count)
    radii = np.arange(max_radius + 1, dtype=float)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_count, endpoint=False)
    radial_cos = np.concatenate(([0.0], np.repeat(radii[1:], angular_count) * np.tile(np.cos(angles), max_radius)))
    radial_sin = np.concatenate(([0.0], np.repeat(radii[1:], angular_count) * np.tile(np.sin(angles), max_radius)))
    radial_ids = np.concatenate(([0], np.repeat(np.arange(1, max_radius + 1), angular_count)))

    edge_occupancy = np.empty(len(edge_nodes), dtype=np.float64)
    edge_max_gap = np.empty(len(edge_nodes), dtype=np.int64)
    radial_foreground = np.zeros(max_radius + 1, dtype=np.float64)
    radial_samples = np.zeros(max_radius + 1, dtype=np.int64)
    chunk_size = int(params["corridor_chunk_edges"])
    for chunk_start in range(0, len(edge_nodes), chunk_size):
        chunk_stop = min(chunk_start + chunk_size, len(edge_nodes))
        local_start = start[chunk_start:chunk_stop]
        local_direction = directions[chunk_start:chunk_stop]
        local_first = first_basis[chunk_start:chunk_stop]
        local_second = second_basis[chunk_start:chunk_stop]
        base = (
            local_start[:, None, :]
            + axial_t[None, :, None] * local_direction[:, None, :]
        )
        offsets = (
            local_first[:, None, :] * radial_cos[None, :, None]
            + local_second[:, None, :] * radial_sin[None, :, None]
        )
        coordinates = base[:, :, None, :] + offsets[:, None, :, :]
        indices = np.rint(coordinates).astype(int)
        valid = np.all((indices >= 0) & (indices < shape_xyz), axis=3)
        x = np.clip(indices[..., 0], 0, shape_xyz[0] - 1)
        y = np.clip(indices[..., 1], 0, shape_xyz[1] - 1)
        z = np.clip(indices[..., 2], 0, shape_xyz[2] - 1)
        foreground = (volume[z, y, x] >= threshold) & valid
        edge_occupancy[chunk_start:chunk_stop] = foreground.mean(axis=(1, 2))
        axial_support = foreground.any(axis=2)
        edge_max_gap[chunk_start:chunk_stop] = np.asarray(
            [longest_false_run(row) for row in axial_support],
            dtype=np.int64,
        )
        for radius in range(max_radius + 1):
            selected = foreground[:, :, radial_ids == radius]
            radial_foreground[radius] += float(selected.sum())
            radial_samples[radius] += int(selected.size)
    del volume

    high_threshold = float(
        np.quantile(
            edge_occupancy,
            float(params["high_confidence_edge_quantile"]),
        )
    )
    high_edges = edge_occupancy >= high_threshold
    # Recompute a conservative radial profile from the global sample. The high-edge
    # threshold is reported for downstream auditing; global probability avoids a
    # second full CT pass.
    radial_probability = radial_foreground / np.maximum(radial_samples, 1)
    eligible = np.flatnonzero(
        radial_probability >= float(params["radial_foreground_probability"])
    )
    measured_radius = float(eligible.max()) if len(eligible) else 0.0

    spatial: dict[str, Any] = {}
    spatial_ranges = []
    bin_count = int(params["spatial_bins_per_axis"])
    for axis, name in enumerate("xyz"):
        order = np.argsort(midpoints[:, axis])
        bins = np.array_split(order, bin_count)
        medians = [float(np.median(edge_occupancy[index])) for index in bins]
        value_range = float(max(medians) - min(medians))
        spatial_ranges.append(value_range)
        spatial[name] = {
            "bin_median_corridor_foreground_fraction": medians,
            "median_range": value_range,
        }

    gates = {
        "junction_foreground_sufficient": bool(
            np.mean(junction_fractions)
            >= float(params["minimum_mean_junction_foreground_fraction"])
        ),
        "corridor_foreground_sufficient": bool(
            np.median(edge_occupancy)
            >= float(params["minimum_median_corridor_foreground_fraction"])
        ),
        "spatial_bias_within_limit": bool(
            max(spatial_ranges)
            <= float(params["maximum_spatial_bin_median_range"])
        ),
        "holdout_candidates_supported": bool(
            np.median(holdout_distances)
            <= float(params["maximum_holdout_candidate_median_distance_voxels"])
        ),
        "measured_radius_positive": bool(measured_radius > 0.0),
    }
    payload = {
        "junctions": {
            "record_count": len(registered_nodes),
            "mean_5x5x5_foreground_fraction": float(np.mean(junction_fractions)),
            "median_5x5x5_foreground_fraction": float(np.median(junction_fractions)),
            "nodes_with_any_foreground_fraction": float(np.mean(junction_any)),
        },
        "candidate_holdout": {
            "candidate_count": int(len(holdout_candidates)),
            "median_distance_to_predicted_unique_node_voxels": float(
                np.median(holdout_distances)
            ),
            "p95_distance_to_predicted_unique_node_voxels": float(
                np.quantile(holdout_distances, 0.95)
            ),
        },
        "corridors": {
            "edge_count": int(len(edge_occupancy)),
            "axial_samples_per_edge": axial_count,
            "maximum_sampled_radius_voxels": max_radius,
            "median_foreground_fraction": float(np.median(edge_occupancy)),
            "p10_foreground_fraction": float(np.quantile(edge_occupancy, 0.10)),
            "p90_foreground_fraction": float(np.quantile(edge_occupancy, 0.90)),
            "edges_with_complete_axial_gap_fraction": float(
                np.mean(edge_max_gap == axial_count)
            ),
            "high_confidence_edge_threshold": high_threshold,
            "high_confidence_edge_count": int(np.count_nonzero(high_edges)),
            "radial_foreground_probability": radial_probability.tolist(),
            "measured_strut_radius_voxels": measured_radius,
        },
        "spatial_bias": spatial,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
    }
    return payload, edge_occupancy


def run_robustness_suite(
    ct_path: Path,
    design: DesignData,
    threshold: Real,
    baseline_candidates: np.ndarray,
    detection_metadata: dict[str, Any],
    baseline_transform: SimilarityTransform,
    config: dict[str, Any],
) -> dict[str, Any]:
    params = config["robustness"]
    fitting = config["fitting"]
    source = design.unique_positions
    cases: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[float, float], tuple[np.ndarray, dict[str, Any]]] = {
        (
            float(threshold),
            float(config["detection"]["edt_peak_threshold_downsampled_voxels"]),
        ): (baseline_candidates, detection_metadata)
    }

    variants: list[tuple[str, float, float, float]] = []
    base_edt = float(config["detection"]["edt_peak_threshold_downsampled_voxels"])
    base_keep = float(fitting["icp_keep_fraction"])
    for offset in params["threshold_offsets"]:
        offset_value = float(offset)
        variants.append(
            (
                f"threshold_{offset_value:+g}",
                float(threshold) + offset_value,
                base_edt,
                base_keep,
            )
        )
    for edt in params["edt_peak_thresholds"]:
        variants.append(
            (f"edt_{float(edt):.2f}", float(threshold), float(edt), base_keep)
        )
    for keep in params["icp_keep_fractions"]:
        variants.append(
            (f"trim_{float(keep):.2f}", float(threshold), base_edt, float(keep))
        )

    seen = set()
    for name, case_threshold, edt_threshold, keep_fraction in variants:
        key = (name, case_threshold, edt_threshold, keep_fraction)
        if key in seen:
            continue
        seen.add(key)
        candidate_key = (case_threshold, edt_threshold)
        if candidate_key not in candidate_cache:
            detected_candidates, variant_metadata = detect_ct_nodes(
                ct_path,
                case_threshold,
                config,
                edt_peak_threshold=edt_threshold,
            )
            variant_fit_candidates, _, _, _ = split_candidates(
                detected_candidates,
                float(config["detection"]["candidate_holdout_fraction"]),
                int(config["random_seed"]),
            )
            candidate_cache[candidate_key] = (
                variant_fit_candidates,
                variant_metadata,
            )
        candidates, metadata = candidate_cache[candidate_key]
        base = coarse_initialization(
            source, metadata["detection_foreground_bounds_xyz"]
        )
        pre_refinement, _, objective = trimmed_icp(
            source,
            candidates,
            base,
            keep_fraction=keep_fraction,
            max_iterations=int(fitting["icp_max_iterations"]),
            convergence_epsilon=float(fitting["icp_convergence_epsilon"]),
        )
        fitted, refinement_history = refine_transform(
            ct_path,
            source,
            pre_refinement,
            case_threshold,
            config,
        )
        shift = np.linalg.norm(
            fitted.apply(source) - baseline_transform.apply(source),
            axis=1,
        )
        cases.append(
            {
                "name": name,
                "threshold": case_threshold,
                "edt_peak_threshold": edt_threshold,
                "icp_keep_fraction": keep_fraction,
                "candidate_count": int(len(candidates)),
                "objective": objective,
                "pre_refinement_transform": pre_refinement.to_dict(),
                "transform": fitted.to_dict(),
                "full_resolution_refinement": refinement_history,
                "prediction_shift_voxels": {
                    "median": float(np.median(shift)),
                    "p95": float(np.quantile(shift, 0.95)),
                    "maximum": float(np.max(shift)),
                },
            }
        )

    # Bootstrap the paired, trimmed baseline correspondences. Independently
    # resampling source and target clouds destroys those correspondences and
    # measures artificial missing-data bias instead of sampling uncertainty.
    baseline_tree = cKDTree(baseline_candidates)
    baseline_predictions = baseline_transform.apply(source)
    baseline_distances, baseline_indices = baseline_tree.query(
        baseline_predictions, k=1
    )
    baseline_cutoff = float(np.quantile(baseline_distances, base_keep))
    baseline_inliers = baseline_distances <= baseline_cutoff
    paired_source = source[baseline_inliers]
    paired_target = baseline_candidates[
        baseline_indices[baseline_inliers]
    ]
    if len(paired_source) < 4:
        raise ValueError(
            "Too few baseline inlier correspondences for bootstrap"
        )

    rng = np.random.default_rng(int(config["random_seed"]) + 101)
    fraction = float(params["bootstrap_fraction"])
    bootstrap_size = max(4, int(len(paired_source) * fraction))
    for repeat in range(int(params["bootstrap_repeats"])):
        pair_index = rng.choice(
            len(paired_source), bootstrap_size, replace=True
        )
        pre_refinement = solve_similarity(
            paired_source[pair_index],
            paired_target[pair_index],
        )
        pair_residuals = np.linalg.norm(
            pre_refinement.apply(paired_source[pair_index])
            - paired_target[pair_index],
            axis=1,
        )
        objective = float(np.mean(pair_residuals))
        fitted, refinement_history = refine_transform(
            ct_path,
            source,
            pre_refinement,
            threshold,
            config,
        )
        shift = np.linalg.norm(
            fitted.apply(source) - baseline_transform.apply(source),
            axis=1,
        )
        cases.append(
            {
                "name": f"bootstrap_{repeat}",
                "threshold": threshold,
                "edt_peak_threshold": base_edt,
                "icp_keep_fraction": base_keep,
                "objective": objective,
                "baseline_correspondence_count": int(len(paired_source)),
                "bootstrap_pair_count": int(bootstrap_size),
                "unique_bootstrap_pair_count": int(
                    len(np.unique(pair_index))
                ),
                "pre_refinement_transform": pre_refinement.to_dict(),
                "transform": fitted.to_dict(),
                "full_resolution_refinement": refinement_history,
                "prediction_shift_voxels": {
                    "median": float(np.median(shift)),
                    "p95": float(np.quantile(shift, 0.95)),
                    "maximum": float(np.max(shift)),
                },
            }
        )

    worst_p95 = max(
        entry["prediction_shift_voxels"]["p95"] for entry in cases
    )
    gate = bool(
        worst_p95 <= float(params["maximum_p95_prediction_shift_voxels"])
    )
    return {
        "case_count": len(cases),
        "baseline_transform": baseline_transform.to_dict(),
        "maximum_case_p95_prediction_shift_voxels": worst_p95,
        "maximum_allowed_p95_prediction_shift_voxels": float(
            params["maximum_p95_prediction_shift_voxels"]
        ),
        "gates": {"robustness_within_corridor_budget": gate},
        "overall_pass": gate,
        "cases": cases,
    }


def transform_errors(
    fitted: SimilarityTransform,
    expected: SimilarityTransform,
) -> dict[str, float]:
    return {
        "relative_scale": abs(fitted.scale / expected.scale - 1.0),
        "rotation_degrees": rotation_difference_deg(
            fitted.rotation, expected.rotation
        ),
        "translation_norm_voxels": float(
            np.linalg.norm(fitted.translation - expected.translation)
        ),
    }


def run_synthetic_suite(
    source: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    params = config["synthetic"]
    rng = np.random.default_rng(int(config["random_seed"]) + 202)
    cases = []
    for case_index in range(int(params["case_count"])):
        angles = rng.uniform(-2.0, 2.0, size=3)
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        expected = SimilarityTransform(
            scale=float(rng.uniform(30.0, 50.0)),
            rotation=rotation,
            translation=rng.uniform(20.0, 80.0, size=3),
        )
        target = expected.apply(source)
        target += rng.normal(
            0.0, float(params["noise_sigma_voxels"]), size=target.shape
        )
        keep = rng.random(len(target)) >= float(params["missing_fraction"])
        target = target[keep]
        outlier_count = int(round(len(target) * float(params["outlier_fraction"])))
        low = np.quantile(target, 0.05, axis=0)
        high = np.quantile(target, 0.95, axis=0)
        outliers = rng.uniform(low, high, size=(outlier_count, 3))
        detected = np.concatenate([target, outliers], axis=0)
        robust_low = np.quantile(detected, 0.01, axis=0)
        robust_high = np.quantile(detected, 0.99, axis=0)
        base = coarse_initialization(source, [robust_low, robust_high])
        fitted, diagnostics = multistart_fit(source, detected, base, config)
        errors = transform_errors(fitted, expected)
        gates = {
            "scale": errors["relative_scale"]
            <= float(params["maximum_relative_scale_error"]),
            "rotation": errors["rotation_degrees"]
            <= float(params["maximum_rotation_error_degrees"]),
            "translation": errors["translation_norm_voxels"]
            <= float(params["maximum_translation_error_voxels"]),
            "multistart": diagnostics["overall_pass"],
        }
        cases.append(
            {
                "case": case_index,
                "expected": expected.to_dict(),
                "fitted": fitted.to_dict(),
                "errors": errors,
                "gates": gates,
                "pass": bool(all(gates.values())),
            }
        )
    pass_fraction = float(np.mean([entry["pass"] for entry in cases]))
    gate = bool(pass_fraction >= float(params["minimum_pass_fraction"]))
    return {
        "case_count": len(cases),
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": float(params["minimum_pass_fraction"]),
        "gates": {"synthetic_recovery_pass_fraction": gate},
        "overall_pass": gate,
        "cases": cases,
    }


def downstream_tolerance_gate(
    multistart: dict[str, Any],
    robustness: dict[str, Any],
    image_validation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    params = config["downstream_gate"]
    measured_radius = float(
        image_validation["corridors"]["measured_strut_radius_voxels"]
    )
    holdout_median = float(
        image_validation["candidate_holdout"][
            "median_distance_to_predicted_unique_node_voxels"
        ]
    )
    uncertainty = max(
        float(multistart["near_optimal_p95_prediction_spread_voxels"]),
        float(robustness["maximum_case_p95_prediction_shift_voxels"]),
        holdout_median,
    )
    margin = max(
        float(params["minimum_corridor_margin_voxels"]),
        measured_radius * float(params["corridor_margin_radius_fraction"]),
    )
    ratio = uncertainty / measured_radius if measured_radius > 0 else math.inf
    gates = {
        "uncertainty_within_corridor_margin": bool(uncertainty <= margin),
        "uncertainty_within_measured_radius_ratio": bool(
            ratio
            <= float(params["maximum_uncertainty_to_measured_radius_ratio"])
        ),
    }
    return {
        "estimated_registration_uncertainty_voxels": uncertainty,
        "measured_strut_radius_voxels": measured_radius,
        "recommended_corridor_margin_voxels": margin,
        "uncertainty_to_measured_radius_ratio": ratio,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
        "classification_allowed": bool(all(gates.values())),
    }
