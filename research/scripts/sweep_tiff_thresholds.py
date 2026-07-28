#!/usr/bin/env python3
"""Run a bounded Otsu-centered TIFF threshold sweep and optionally write a mask."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
import tifffile


DEFAULT_OFFSETS = (-5.0, -2.0, 0.0, 2.0, 5.0)


def inspect_tiff(path: Path, slice_index: int) -> tuple[tuple[int, ...], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise ValueError(f"TIFF contains no image series: {path}")
        shape = tuple(int(value) for value in tif.series[0].shape)
        dtype = np.dtype(tif.series[0].dtype)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D TIFF, found shape {shape}: {path}")
    if not 0 <= slice_index < shape[0]:
        raise IndexError(f"Slice {slice_index} is outside axis 0 with {shape[0]} slices")
    if dtype.kind != "u" or int(np.iinfo(dtype).max) > 65535:
        raise TypeError(f"Expected an unsigned 8-bit or 16-bit CT TIFF, found {dtype}")
    return shape, dtype


def exact_histogram(path: Path, dtype: np.dtype) -> np.ndarray:
    bins = int(np.iinfo(dtype).max) + 1
    histogram = np.zeros(bins, dtype=np.uint64)
    with tifffile.TiffFile(path) as tif:
        for page in tif.series[0].pages:
            values = page.asarray()
            histogram += np.bincount(values.ravel(), minlength=bins).astype(np.uint64)
    return histogram


def otsu_from_histogram(histogram: np.ndarray) -> int:
    counts = histogram.astype(np.float64)
    levels = np.arange(counts.size, dtype=np.float64)
    cumulative_weight = np.cumsum(counts)
    cumulative_sum = np.cumsum(counts * levels)
    total_weight = cumulative_weight[-1]
    total_sum = cumulative_sum[-1]
    foreground_weight = total_weight - cumulative_weight
    valid = (cumulative_weight > 0) & (foreground_weight > 0)
    score = np.full(counts.size, -np.inf)
    background_mean = np.zeros(counts.size)
    foreground_mean = np.zeros(counts.size)
    background_mean[valid] = cumulative_sum[valid] / cumulative_weight[valid]
    foreground_mean[valid] = (
        total_sum - cumulative_sum[valid]
    ) / foreground_weight[valid]
    score[valid] = (
        cumulative_weight[valid]
        * foreground_weight[valid]
        * (background_mean[valid] - foreground_mean[valid]) ** 2
    )
    if not np.isfinite(score).any():
        raise ValueError("Cannot compute Otsu threshold from a constant histogram")
    return int(np.argmax(score))


def sample_volume(path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
    strides = [max(1, math.ceil(dimension / 128)) for dimension in shape]
    z_stride, y_stride, x_stride = strides
    with tifffile.TiffFile(path) as tif:
        pages = tif.series[0].pages
        sample = np.stack(
            [
                pages[index].asarray()[::y_stride, ::x_stride]
                for index in range(0, shape[0], z_stride)
            ]
        )
    return sample, strides


def component_metrics(mask: np.ndarray) -> dict[str, int | float]:
    structure = ndi.generate_binary_structure(mask.ndim, mask.ndim)
    labels, count = ndi.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    foreground = int(np.count_nonzero(mask))
    if foreground == 0 or not sizes.size:
        return {
            "component_count": 0,
            "largest_component_share": 0.0,
            "small_component_share": 0.0,
        }
    small_cutoff = 8 if mask.ndim == 2 else 27
    return {
        "component_count": int(count),
        "largest_component_share": float(sizes.max() / foreground),
        "small_component_share": float(sizes[sizes < small_cutoff].sum() / foreground),
    }


def candidate_thresholds(
    otsu_threshold: int,
    dtype: np.dtype,
    offsets: list[float],
    explicit: list[int] | None,
) -> list[int]:
    maximum = int(np.iinfo(dtype).max)
    values = explicit or [round(otsu_threshold * (1.0 + offset / 100.0)) for offset in offsets]
    thresholds = sorted({int(np.clip(value, 0, maximum)) for value in values})
    if not 3 <= len(thresholds) <= 7:
        raise ValueError(f"Expected 3 to 7 distinct thresholds, found {thresholds}")
    return thresholds


def save_preview(
    path: Path,
    histogram: np.ndarray,
    source_slice: np.ndarray,
    threshold: int,
    foreground_fraction: float,
) -> None:
    occupied = np.flatnonzero(histogram)
    low, high = int(occupied[0]), int(occupied[-1])
    mask_slice = source_slice >= threshold
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(np.arange(low, high + 1), histogram[low : high + 1], lw=0.8)
    axes[0].axvline(threshold, color="red", label=f"threshold={threshold}")
    axes[0].set_yscale("log")
    axes[0].set(title="Exact whole-volume histogram", xlabel="Intensity", ylabel="Voxels")
    axes[0].legend()
    axes[1].imshow(source_slice, cmap="gray")
    axes[1].set_title("Source review slice")
    axes[2].imshow(mask_slice, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Candidate mask; volume foreground={foreground_fraction:.2%}")
    axes[1].axis("off")
    axes[2].axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_sweep(
    path: Path,
    output_dir: Path,
    slice_index: int,
    offsets: list[float],
    explicit_thresholds: list[int] | None,
) -> dict[str, Any]:
    shape, dtype = inspect_tiff(path, slice_index)
    histogram = exact_histogram(path, dtype)
    if int(histogram.sum()) != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("Histogram voxel total does not match TIFF shape")
    baseline = otsu_from_histogram(histogram)
    thresholds = candidate_thresholds(baseline, dtype, offsets, explicit_thresholds)
    sample, strides = sample_volume(path, shape)
    with tifffile.TiffFile(path) as tif:
        source_slice = tif.series[0].pages[slice_index].asarray()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "intensity_histogram.npy", histogram, allow_pickle=False)
    total = int(histogram.sum())
    baseline_foreground = int(histogram[baseline:].sum())
    baseline_fraction = baseline_foreground / total
    candidates = []
    for threshold in thresholds:
        foreground = int(histogram[threshold:].sum())
        fraction = foreground / total
        slice_mask = source_slice >= threshold
        sample_mask = sample >= threshold
        preview_path = output_dir / f"candidate_threshold_{threshold}.png"
        save_preview(preview_path, histogram, source_slice, threshold, fraction)
        candidates.append(
            {
                "threshold": threshold,
                "is_otsu_baseline": threshold == baseline,
                "foreground_voxels": foreground,
                "foreground_fraction": fraction,
                "foreground_growth_relative_to_otsu": (
                    (fraction - baseline_fraction) / baseline_fraction
                    if baseline_fraction
                    else None
                ),
                "slice_foreground_fraction": float(slice_mask.mean()),
                "slice_components_diagnostic_only": component_metrics(slice_mask),
                "sampled_3d_components": component_metrics(sample_mask),
                "preview_path": str(preview_path.resolve()),
            }
        )

    result: dict[str, Any] = {
        "input_path": str(path),
        "input_file_bytes": int(path.stat().st_size),
        "shape": list(shape),
        "dtype": str(dtype),
        "slice_index": slice_index,
        "otsu_threshold": baseline,
        "otsu_foreground_fraction": baseline_fraction,
        "sample_strides": strides,
        "selection_is_unsupervised": True,
        "candidates": candidates,
    }
    (output_dir / "threshold_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def materialize_mask(
    input_path: Path,
    shape: tuple[int, ...],
    threshold: int,
    mask_output: Path,
    slice_output: Path,
    slice_index: int,
) -> None:
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    slice_output.parent.mkdir(parents=True, exist_ok=True)
    use_bigtiff = int(np.prod(shape, dtype=np.int64)) >= (2**32 - 2**25)

    def mask_pages():
        with tifffile.TiffFile(input_path) as tif:
            for page in tif.series[0].pages:
                yield (page.asarray() >= threshold).astype(np.uint8)

    tifffile.imwrite(
        mask_output,
        mask_pages(),
        shape=shape,
        dtype=np.uint8,
        photometric="minisblack",
        bigtiff=use_bigtiff,
        compression="zlib",
        metadata={"axes": "ZYX", "threshold": threshold},
    )
    with tifffile.TiffFile(mask_output) as tif:
        mask_slice = tif.series[0].pages[slice_index].asarray()
    plt.imsave(slice_output, mask_slice, cmap="gray", vmin=0, vmax=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded Otsu-centered threshold sweep on a 3D CT TIFF."
    )
    parser.add_argument("input", type=Path, help="Input .tif/.tiff CT volume")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=380)
    parser.add_argument("--offset-percent", type=float, action="append")
    parser.add_argument("--threshold", type=int, action="append")
    parser.add_argument("--select-threshold", type=int)
    parser.add_argument("--mask-output", type=Path)
    parser.add_argument("--slice-output", type=Path)
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve(strict=True)
    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        parser.error("input must be a .tif or .tiff file")
    output_dir = args.output_dir.expanduser().resolve()
    offsets = args.offset_percent or list(DEFAULT_OFFSETS)
    result = run_sweep(
        input_path,
        output_dir,
        args.slice_index,
        offsets,
        args.threshold,
    )

    selected = args.select_threshold
    if selected is not None:
        candidate_values = {item["threshold"] for item in result["candidates"]}
        if selected not in candidate_values:
            parser.error(f"selected threshold {selected} is not in {sorted(candidate_values)}")
        if args.mask_output is None or args.slice_output is None:
            parser.error("--mask-output and --slice-output are required with --select-threshold")
        materialize_mask(
            input_path,
            tuple(result["shape"]),
            selected,
            args.mask_output.expanduser().resolve(),
            args.slice_output.expanduser().resolve(),
            args.slice_index,
        )
        result["materialized_threshold"] = selected
        result["mask_output"] = str(args.mask_output.expanduser().resolve())
        result["slice_output"] = str(args.slice_output.expanduser().resolve())
        (output_dir / "threshold_sweep.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
