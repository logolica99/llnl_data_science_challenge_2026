"""Exact-histogram Otsu replay promoted from the registration POC."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
from scipy import ndimage, signal

from .volume import VolumeView, iter_array_chunks, load_volume


OTSU_METHOD_VERSION = "2.0.0"
OTSU_RESULT_SCHEMA_VERSION = "exact-otsu-replay/1.0.0"
DEFAULT_OTSU_RECIPE: dict[str, int | float | str] = {
    "histogram_encoding": "auto",
    "edge_slices_excluded": 0,
    "chunk_voxels": 8 * 1024 * 1024,
    "coarse_bins": 1024,
    "peak_smoothing_sigma_bins": 2.0,
    "peak_prominence_fraction": 0.003,
    "minimum_significant_peaks": 2,
    "minimum_foreground_fraction": 0.01,
    "maximum_foreground_fraction": 0.35,
    "minimum_otsu_separability": 0.45,
    "minimum_class_mean_separation_sigma": 0.75,
}


class OtsuReplayError(ValueError):
    """Raised when an exact replay cannot be computed deterministically."""


def _recipe(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    recipe = dict(DEFAULT_OTSU_RECIPE)
    if overrides:
        unknown = sorted(set(overrides) - set(recipe))
        if unknown:
            raise OtsuReplayError(f"Unknown Otsu recipe fields: {', '.join(unknown)}")
        recipe.update(overrides)
    if int(recipe["chunk_voxels"]) <= 0:
        raise OtsuReplayError("chunk_voxels must be positive")
    return recipe


def _selected_volume(volume: np.ndarray[Any, Any], edge_slices_excluded: int) -> np.ndarray[Any, Any]:
    start = int(edge_slices_excluded)
    stop = int(volume.shape[0]) - start
    if not 0 <= start < stop <= int(volume.shape[0]):
        raise OtsuReplayError("edge_slices_excluded removes the complete CT volume")
    return volume[start:stop]


def deterministic_histogram(
    volume: np.ndarray[Any, Any],
    *,
    chunk_voxels: int = 8 * 1024 * 1024,
    edge_slices_excluded: int = 0,
    encoding: str = "auto",
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], dict[str, Any]]:
    """Count finite voxels in a deterministic 65,536-bin histogram.

    Native uint16 values use one exact bin per intensity.  Other real numeric
    types use a deterministic full-volume affine mapping to uint16, computed in
    bounded chunks without converting the full volume to float.
    """

    selected = _selected_volume(volume, edge_slices_excluded)
    dtype = np.dtype(selected.dtype)
    if dtype.fields or dtype.subdtype or dtype.kind not in "buif":
        raise OtsuReplayError(f"Expected a real numeric CT volume, found {dtype}")
    resolved_encoding = (
        "native_uint16"
        if encoding == "auto" and dtype.kind == "u" and dtype.itemsize == 2
        else "full_volume_affine_uint16"
        if encoding == "auto"
        else encoding
    )

    if resolved_encoding == "native_uint16":
        if dtype.kind != "u" or dtype.itemsize != 2:
            raise OtsuReplayError(
                f"native_uint16 requires a uint16 volume, found {dtype}"
            )
        histogram = np.zeros(65_536, dtype=np.int64)
        for chunk in iter_array_chunks(selected, int(chunk_voxels)):
            values = np.asarray(chunk, dtype=np.int64)
            histogram += np.bincount(values, minlength=65_536)
        return histogram, {
            "encoding": resolved_encoding,
            "native_dtype": str(dtype),
            "native_min": 0.0,
            "native_max": 65_535.0,
            "native_units_per_bin": 1.0,
        }

    if resolved_encoding != "full_volume_affine_uint16":
        raise OtsuReplayError(f"Unsupported histogram encoding: {resolved_encoding}")

    native_min = math.inf
    native_max = -math.inf
    finite_count = 0
    for chunk in iter_array_chunks(selected, int(chunk_voxels)):
        values = np.asarray(chunk)
        finite = values[np.isfinite(values)]
        if finite.size:
            native_min = min(native_min, float(np.min(finite)))
            native_max = max(native_max, float(np.max(finite)))
            finite_count += int(finite.size)
    if not math.isfinite(native_min) or not math.isfinite(native_max):
        raise OtsuReplayError("CT volume has no finite intensities")
    if native_max <= native_min:
        raise OtsuReplayError("CT volume has no usable finite intensity range")

    histogram = np.zeros(65_536, dtype=np.int64)
    scale = 65_535.0 / (native_max - native_min)
    for chunk in iter_array_chunks(selected, int(chunk_voxels)):
        values = np.asarray(chunk)
        finite_mask = np.isfinite(values)
        finite_values = np.asarray(values[finite_mask], dtype=np.float64)
        quantized = np.rint((finite_values - native_min) * scale)
        quantized = np.clip(quantized, 0, 65_535).astype(np.uint16)
        histogram += np.bincount(quantized.astype(np.int64), minlength=65_536)
    if int(histogram.sum()) != finite_count:
        raise OtsuReplayError("Histogram count does not match finite source voxels")
    return histogram, {
        "encoding": resolved_encoding,
        "native_dtype": str(dtype),
        "native_min": native_min,
        "native_max": native_max,
        "native_units_per_bin": (native_max - native_min) / 65_535.0,
    }


def histogram_sha256(histogram: np.ndarray[Any, Any]) -> str:
    """Hash counts using a platform-independent unsigned 64-bit encoding."""

    counts = np.asarray(histogram, dtype=">u8")
    return hashlib.sha256(counts.tobytes()).hexdigest()


def otsu_from_histogram(histogram: np.ndarray[Any, Any]) -> tuple[int, float]:
    """Return the frozen v2 Otsu threshold bin and separability."""

    counts = np.asarray(histogram, dtype=np.float64)
    if counts.shape != (65_536,) or counts.sum() <= 0:
        raise OtsuReplayError("Otsu requires a non-empty 65,536-bin histogram")
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
    histogram: np.ndarray[Any, Any],
    threshold_bin: int,
    separability: float,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute v2 diagnostics and deterministic histogram rejection gates."""

    counts = np.asarray(histogram, dtype=np.float64)
    levels = np.arange(counts.size, dtype=np.float64)
    total = float(counts.sum())
    background = counts[:threshold_bin]
    foreground = counts[threshold_bin:]
    if background.sum() <= 0 or foreground.sum() <= 0:
        raise OtsuReplayError("Otsu threshold did not produce two populated classes")
    foreground_fraction = float(foreground.sum() / total)

    def weighted_stats(
        values: np.ndarray[Any, Any],
        weights: np.ndarray[Any, Any],
    ) -> tuple[float, float]:
        weight = float(weights.sum())
        mean = float(np.sum(weights * values) / weight)
        variance = float(np.sum(weights * (values - mean) ** 2) / weight)
        return mean, variance

    background_mean, background_variance = weighted_stats(
        levels[:threshold_bin],
        background,
    )
    foreground_mean, foreground_variance = weighted_stats(
        levels[threshold_bin:],
        foreground,
    )
    pooled_sigma = math.sqrt(
        max((background_variance + foreground_variance) / 2.0, 1e-12)
    )
    class_separation = abs(foreground_mean - background_mean) / pooled_sigma

    coarse_bins = int(recipe["coarse_bins"])
    if coarse_bins <= 0 or 65_536 % coarse_bins:
        raise OtsuReplayError("coarse_bins must be a positive divisor of 65,536")
    coarse = counts.reshape(coarse_bins, -1).sum(axis=1)
    smoothed = ndimage.gaussian_filter1d(
        coarse,
        float(recipe["peak_smoothing_sigma_bins"]),
    )
    prominence = max(
        1.0,
        float(smoothed.max()) * float(recipe["peak_prominence_fraction"]),
    )
    peaks, _ = signal.find_peaks(
        smoothed,
        prominence=prominence,
        distance=max(2, coarse_bins // 128),
    )
    modes = ((peaks + 0.5) * (65_536 / coarse_bins)).tolist()
    gates = {
        "foreground_fraction_plausible": (
            float(recipe["minimum_foreground_fraction"])
            <= foreground_fraction
            <= float(recipe["maximum_foreground_fraction"])
        ),
        "otsu_separability_sufficient": (
            separability >= float(recipe["minimum_otsu_separability"])
        ),
        "class_mean_separation_sufficient": (
            class_separation
            >= float(recipe["minimum_class_mean_separation_sigma"])
        ),
        "histogram_not_unimodal": (
            len(peaks) >= int(recipe["minimum_significant_peaks"])
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


def replay_exact_otsu(
    volume: VolumeView | str | Path,
    *,
    recipe: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray[Any, np.dtype[np.int64]]]:
    """Replay exact Otsu for one scan and return compact data plus histogram."""

    view = load_volume(volume) if isinstance(volume, (str, Path)) else volume
    resolved_recipe = _recipe(recipe)
    histogram, encoding = deterministic_histogram(
        view.array,
        chunk_voxels=int(resolved_recipe["chunk_voxels"]),
        edge_slices_excluded=int(resolved_recipe["edge_slices_excluded"]),
        encoding=str(resolved_recipe["histogram_encoding"]),
    )
    threshold_bin, separability = otsu_from_histogram(histogram)
    result = histogram_diagnostics(
        histogram,
        threshold_bin,
        separability,
        resolved_recipe,
    )
    threshold: int | float
    if encoding["encoding"] == "native_uint16":
        threshold = threshold_bin
    else:
        threshold = (
            encoding["native_min"]
            + threshold_bin * encoding["native_units_per_bin"]
        )
    result.update(
        {
            "schema_version": OTSU_RESULT_SCHEMA_VERSION,
            "method": "exact_histogram_otsu",
            "method_version": OTSU_METHOD_VERSION,
            "threshold": threshold,
            "threshold_histogram_bin": threshold_bin,
            "threshold_comparison": "value >= threshold",
            "histogram_encoding": encoding,
            "recipe": resolved_recipe,
        }
    )
    return result, histogram


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise OtsuReplayError(
            f"Otsu artifact already exists; enable overwrite explicitly: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(path: Path, value: np.ndarray[Any, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise OtsuReplayError(
            f"Otsu artifact already exists; enable overwrite explicitly: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npy",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.save(temporary, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_otsu_artifacts(
    output_directory: str | Path,
    result: dict[str, Any],
    histogram: np.ndarray[Any, Any],
    *,
    overwrite: bool = False,
) -> dict[str, dict[str, Any]]:
    """Persist deterministic histogram/report artifacts and return their hashes."""

    destination = Path(output_directory).expanduser().resolve()
    histogram_path = destination / "exact_histogram_uint16.npy"
    report_path = destination / "histogram_report.json"
    _atomic_npy(histogram_path, np.asarray(histogram, dtype=np.int64), overwrite)
    try:
        _atomic_json(report_path, result, overwrite)
    except Exception:
        if not overwrite:
            histogram_path.unlink(missing_ok=True)
        raise
    return {
        "histogram": {
            "path": str(histogram_path),
            "sha256": _sha256_file(histogram_path),
            "role": "exact_histogram",
            "retention": "regenerable",
        },
        "report": {
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "role": "otsu_replay_report",
            "retention": "regenerable",
        },
    }
