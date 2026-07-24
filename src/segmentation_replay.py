"""Deterministic per-scan exact-histogram Otsu replay for Part 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy import ndimage, signal
import tifffile

from specimen_manifest import (
    REPOSITORY_ROOT,
    load_json,
    sha256_file,
    validate_manifest,
)


def load_ct_volume(path: Path, format_name: str) -> np.ndarray:
    """Memory-map a supported three-dimensional CT input."""
    if format_name == "tiff":
        volume = tifffile.memmap(path, mode="r")
    elif format_name == "npy":
        volume = np.load(path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError(f"Unsupported CT format: {format_name}")
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D CT volume, found shape {volume.shape}")
    return volume


def deterministic_histogram(
    volume: np.ndarray,
    *,
    chunk_depth: int,
    edge_slices_excluded: int,
    encoding: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Count all finite voxels in a deterministic 65,536-bin histogram."""
    start = edge_slices_excluded
    stop = volume.shape[0] - edge_slices_excluded
    if not 0 <= start < stop <= volume.shape[0]:
        raise ValueError("edge_slices_excluded removes the complete CT volume")

    if encoding == "native_uint16":
        if volume.dtype.kind != "u" or volume.dtype.itemsize != 2:
            raise ValueError(
                f"native_uint16 requires a uint16 volume, found {volume.dtype}"
            )
        histogram = np.zeros(65_536, dtype=np.int64)
        for z0 in range(start, stop, chunk_depth):
            chunk = np.asarray(volume[z0 : min(z0 + chunk_depth, stop)])
            histogram += np.bincount(chunk.ravel(), minlength=65_536)
        return histogram, {
            "encoding": encoding,
            "native_dtype": str(volume.dtype),
            "native_min": 0.0,
            "native_max": 65_535.0,
            "native_units_per_bin": 1.0,
        }

    if encoding != "full_volume_affine_uint16":
        raise ValueError(f"Unsupported histogram encoding: {encoding}")
    if volume.dtype.kind not in "fiu":
        raise ValueError(f"Expected a numeric CT volume, found {volume.dtype}")

    native_min = math.inf
    native_max = -math.inf
    finite_count = 0
    for z0 in range(start, stop, chunk_depth):
        chunk = np.asarray(volume[z0 : min(z0 + chunk_depth, stop)])
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            native_min = min(native_min, float(finite.min()))
            native_max = max(native_max, float(finite.max()))
            finite_count += int(finite.size)
    if not math.isfinite(native_min) or native_max <= native_min:
        raise ValueError("CT volume has no usable finite intensity range")

    histogram = np.zeros(65_536, dtype=np.int64)
    scale = 65_535.0 / (native_max - native_min)
    for z0 in range(start, stop, chunk_depth):
        chunk = np.asarray(
            volume[z0 : min(z0 + chunk_depth, stop)], dtype=np.float64
        )
        finite = np.isfinite(chunk)
        quantized = np.rint((chunk[finite] - native_min) * scale)
        quantized = np.clip(quantized, 0, 65_535).astype(np.uint16)
        histogram += np.bincount(quantized, minlength=65_536)
    if int(histogram.sum()) != finite_count:
        raise RuntimeError("Histogram count does not match the finite source voxels")
    return histogram, {
        "encoding": encoding,
        "native_dtype": str(volume.dtype),
        "native_min": native_min,
        "native_max": native_max,
        "native_units_per_bin": (native_max - native_min) / 65_535.0,
    }


def histogram_sha256(histogram: np.ndarray) -> str:
    """Hash histogram counts using a platform-independent uint64 encoding."""
    counts = np.asarray(histogram, dtype=">u8")
    return hashlib.sha256(counts.tobytes()).hexdigest()


def otsu_from_histogram(histogram: np.ndarray) -> tuple[int, float]:
    """Return the Otsu threshold bin and between-class separability."""
    counts = np.asarray(histogram, dtype=np.float64)
    if counts.shape != (65_536,) or counts.sum() <= 0:
        raise ValueError("Otsu requires a non-empty 65,536-bin histogram")
    levels = np.arange(counts.size, dtype=np.float64)
    total = counts.sum()
    cumulative_weight = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * levels)
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
    threshold_bin = int(np.argmax(between))
    mean = float(np.sum(counts * levels) / total)
    total_variance = float(np.sum(counts * (levels - mean) ** 2))
    separability = float(between[threshold_bin] / (total * total_variance))
    return threshold_bin, separability


def histogram_diagnostics(
    histogram: np.ndarray,
    threshold_bin: int,
    separability: float,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    """Compute the frozen v2 histogram diagnostics and rejection gates."""
    counts = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(counts.size, dtype=np.float64)
    total = float(counts.sum())
    background = counts[:threshold_bin]
    foreground = counts[threshold_bin:]
    foreground_fraction = float(foreground.sum() / total)

    def weighted_stats(
        values: np.ndarray, weights: np.ndarray
    ) -> tuple[float, float]:
        weight = float(weights.sum())
        mean = float(np.sum(weights * values) / weight)
        variance = float(np.sum(weights * (values - mean) ** 2) / weight)
        return mean, variance

    background_mean, background_variance = weighted_stats(
        levels[:threshold_bin], background
    )
    foreground_mean, foreground_variance = weighted_stats(
        levels[threshold_bin:], foreground
    )
    pooled_sigma = math.sqrt(
        max((background_variance + foreground_variance) / 2.0, 1e-12)
    )
    class_separation = abs(foreground_mean - background_mean) / pooled_sigma

    coarse_bins = int(recipe["coarse_bins"])
    if 65_536 % coarse_bins:
        raise ValueError("coarse_bins must evenly divide 65,536")
    coarse = counts.reshape(coarse_bins, -1).sum(axis=1)
    smoothed = ndimage.gaussian_filter1d(
        coarse, float(recipe["peak_smoothing_sigma_bins"])
    )
    prominence = max(
        1.0, float(smoothed.max()) * float(recipe["peak_prominence_fraction"])
    )
    peaks, _ = signal.find_peaks(
        smoothed,
        prominence=prominence,
        distance=max(2, coarse_bins // 128),
    )
    modes = ((peaks + 0.5) * (65_536 / coarse_bins)).tolist()
    gates = {
        "foreground_fraction_plausible": (
            recipe["minimum_foreground_fraction"]
            <= foreground_fraction
            <= recipe["maximum_foreground_fraction"]
        ),
        "otsu_separability_sufficient": (
            separability >= recipe["minimum_otsu_separability"]
        ),
        "class_mean_separation_sufficient": (
            class_separation >= recipe["minimum_class_mean_separation_sigma"]
        ),
        "histogram_not_unimodal": (
            len(peaks) >= recipe["minimum_significant_peaks"]
        ),
    }
    return {
        "voxel_count": int(total),
        "foreground_voxel_count": int(foreground.sum()),
        "foreground_fraction": foreground_fraction,
        "otsu_separability": separability,
        "background_mean": background_mean,
        "foreground_mean": foreground_mean,
        "class_mean_separation_sigma": class_separation,
        "significant_modes": modes,
        "histogram_sha256": histogram_sha256(histogram),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def replay_manifest(
    manifest_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay a manifest's segmentation recipe and enforce its frozen result."""
    validate_manifest(manifest_path)
    manifest = load_json(manifest_path)
    artifact = manifest["inputs"]["ct"]
    ct_path = repository_root / artifact["path"]
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT input is unavailable: {ct_path}")
    actual_input_hash = sha256_file(ct_path)
    if actual_input_hash != artifact["sha256"]:
        raise ValueError(
            f"CT SHA-256 mismatch: expected {artifact['sha256']}, "
            f"found {actual_input_hash}"
        )

    started = time.perf_counter()
    metadata = manifest["inputs"]["ct_metadata"]
    recipe = manifest["analysis_parameters"]["segmentation"]
    volume = load_ct_volume(ct_path, metadata["format"])
    if list(volume.shape) != metadata["shape"]:
        raise ValueError(
            f"CT shape mismatch: expected {metadata['shape']}, found {volume.shape}"
        )
    histogram, encoding = deterministic_histogram(
        volume,
        chunk_depth=recipe["chunk_depth"],
        edge_slices_excluded=recipe["edge_slices_excluded"],
        encoding=recipe["histogram_encoding"],
    )
    threshold_bin, separability = otsu_from_histogram(histogram)
    result = histogram_diagnostics(histogram, threshold_bin, separability, recipe)
    if encoding["encoding"] == "native_uint16":
        threshold: int | float = threshold_bin
    else:
        threshold = (
            encoding["native_min"]
            + threshold_bin * encoding["native_units_per_bin"]
        )
    result["threshold"] = threshold
    result["threshold_histogram_bin"] = threshold_bin
    result["histogram_encoding"] = encoding
    result["elapsed_seconds"] = time.perf_counter() - started

    expected = manifest["derived"]["segmentation_result"]["values"]
    mismatches: list[str] = []
    exact_fields = (
        "threshold",
        "voxel_count",
        "foreground_voxel_count",
        "significant_modes",
        "histogram_sha256",
        "overall_pass",
    )
    for field in exact_fields:
        if result[field] != expected[field]:
            mismatches.append(
                f"{field}: expected {expected[field]!r}, found {result[field]!r}"
            )
    float_fields = (
        "foreground_fraction",
        "otsu_separability",
        "background_mean",
        "foreground_mean",
        "class_mean_separation_sigma",
    )
    for field in float_fields:
        if not math.isclose(
            result[field], expected[field], rel_tol=1e-12, abs_tol=1e-12
        ):
            mismatches.append(
                f"{field}: expected {expected[field]!r}, found {result[field]!r}"
            )
    if mismatches:
        raise ValueError("Segmentation replay mismatch:\n- " + "\n- ".join(mismatches))
    if not result["overall_pass"]:
        failed = [name for name, passed in result["gates"].items() if not passed]
        raise ValueError("Histogram rejection gates failed: " + ", ".join(failed))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "exact_histogram_uint16.npy", histogram)
        with (output_dir / "histogram_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="root used to resolve manifest input paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optionally persist the regenerable histogram and report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = replay_manifest(
            args.manifest,
            repository_root=args.repository_root.resolve(),
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(
        "PASS "
        f"threshold={result['threshold']} "
        f"foreground={result['foreground_voxel_count']}/"
        f"{result['voxel_count']} "
        f"histogram_sha256={result['histogram_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
