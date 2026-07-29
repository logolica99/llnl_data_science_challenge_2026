#!/usr/bin/env python3
"""Memory-aware, reproducible threshold segmentation for one CT TIFF volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_triangle, threshold_yen
import tifffile


SLICE_INDEX = 380
METHODS = {
    "otsu": threshold_otsu,
    "triangle": threshold_triangle,
    "yen": threshold_yen,
}


def inspect(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        dtype = np.dtype(series.dtype)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D TIFF, got shape {shape}")
    if shape[0] <= SLICE_INDEX:
        raise ValueError(f"Axis 0 has {shape[0]} slices; slice {SLICE_INDEX} is required")
    if dtype.kind not in "ui":
        raise ValueError(f"Expected integer CT intensities, got {dtype}")
    return shape, dtype


def exact_histogram(path: Path, dtype: np.dtype, cache: Path) -> np.ndarray:
    bins = int(np.iinfo(dtype).max) + 1
    if cache.exists():
        hist = np.load(cache, allow_pickle=False)
        if hist.shape == (bins,):
            return hist
    hist = np.zeros(bins, dtype=np.uint64)
    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            values = page.asarray()
            hist += np.bincount(values.ravel(), minlength=bins).astype(np.uint64)
    np.save(cache, hist, allow_pickle=False)
    return hist


def threshold_from_hist(hist: np.ndarray, method: str) -> int:
    occupied = np.flatnonzero(hist)
    lo, hi = int(occupied[0]), int(occupied[-1])
    counts = hist[lo : hi + 1]
    centers = np.arange(lo, hi + 1)
    if method == "triangle":
        # Histogram form of skimage's Triangle implementation.  Keeping the
        # exact uint16 bins avoids materializing the full compressed volume.
        work = counts.astype(np.float64)
        peak = int(np.argmax(work))
        low, high = np.flatnonzero(work)[[0, -1]]
        if low == high:
            return int(centers[low])
        flip = peak - low < high - peak
        if flip:
            work = work[::-1]
            low = len(work) - high - 1
            peak = len(work) - peak - 1
        width = peak - low
        x = np.arange(width)
        y = work[x + low]
        norm = np.hypot(work[peak], width)
        distance = (work[peak] / norm) * x - (width / norm) * y
        level = int(np.argmax(distance) + low)
        if flip:
            level = len(work) - level - 1
        return int(centers[level])
    return int(METHODS[method](image=None, hist=(counts, centers)))


def component_metrics(mask: np.ndarray, dimensions: int) -> dict[str, float | int]:
    structure = ndi.generate_binary_structure(dimensions, dimensions)
    labels, count = ndi.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())[1:]
    foreground = int(mask.sum())
    if not sizes.size:
        return {
            "component_count": 0,
            "largest_component_voxels": 0,
            "largest_component_share": 0.0,
            "small_component_share": 0.0,
        }
    small_cutoff = 8 if dimensions == 2 else 27
    return {
        "component_count": int(count),
        "largest_component_voxels": int(sizes.max()),
        "largest_component_share": float(sizes.max() / max(foreground, 1)),
        "small_component_share": float(sizes[sizes < small_cutoff].sum() / max(foreground, 1)),
    }


def sampled_volume(path: Path, step: int = 4) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        first = tif.pages[0].asarray()[::step, ::step]
        sample = np.empty(((len(tif.pages) + step - 1) // step, *first.shape), dtype=first.dtype)
        sample[0] = first
        out_i = 1
        for page_i in range(step, len(tif.pages), step):
            sample[out_i] = tif.pages[page_i].asarray()[::step, ::step]
            out_i += 1
    return sample


def candidate(path: Path, outdir: Path, method: str) -> dict:
    started = time.time()
    shape, dtype = inspect(path)
    hist = exact_histogram(path, dtype, outdir / "intensity_histogram.npy")
    threshold = threshold_from_hist(hist, method)
    total = int(np.prod(shape))
    foreground = int(hist[threshold:].sum())
    with tifffile.TiffFile(path) as tif:
        ct_slice = tif.pages[SLICE_INDEX].asarray()
    slice_mask = ct_slice >= threshold
    sample_mask = sampled_volume(path) >= threshold
    metrics = {
        "method": method,
        "threshold": threshold,
        "shape": list(shape),
        "dtype": str(dtype),
        "total_voxels": total,
        "foreground_voxels": foreground,
        "foreground_fraction": foreground / total,
        "slice_380_foreground_fraction": float(slice_mask.mean()),
        "slice_380_components": component_metrics(slice_mask, 2),
        "downsample_step": 4,
        "sampled_3d_components": component_metrics(sample_mask, 3),
        "elapsed_seconds": time.time() - started,
    }
    (outdir / f"candidate_{method}.json").write_text(json.dumps(metrics, indent=2) + "\n")

    occupied = np.flatnonzero(hist)
    lo, hi = int(occupied[0]), int(occupied[-1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(np.arange(lo, hi + 1), hist[lo : hi + 1], lw=0.8)
    axes[0].axvline(threshold, color="red", label=f"{method}: {threshold}")
    axes[0].set_yscale("log")
    axes[0].set_title("Exact intensity histogram")
    axes[0].set_xlabel("uint16 intensity")
    axes[0].legend()
    axes[1].imshow(ct_slice, cmap="gray")
    axes[1].set_title("CT slice 380")
    axes[2].imshow(slice_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(
        f"Mask preview; fg={metrics['slice_380_foreground_fraction']:.2%}"
    )
    for axis in axes[1:]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(outdir / f"candidate_{method}.png", dpi=150)
    plt.close(fig)
    return metrics


def write_final(path: Path, outdir: Path, method: str) -> dict:
    metrics_path = outdir / f"candidate_{method}.json"
    if not metrics_path.exists():
        metrics = candidate(path, outdir, method)
    else:
        metrics = json.loads(metrics_path.read_text())
    threshold = int(metrics["threshold"])
    shape = tuple(metrics["shape"])
    output = outdir / "mask.tif"
    estimated_bytes = int(np.prod(shape))
    use_bigtiff = estimated_bytes >= (2**32 - 2**25)

    def pages():
        with tifffile.TiffFile(path) as tif:
            for page in tif.pages:
                yield (page.asarray() >= threshold).astype(np.uint8)

    tifffile.imwrite(
        output,
        pages(),
        shape=shape,
        dtype=np.uint8,
        photometric="minisblack",
        bigtiff=use_bigtiff,
        compression="zlib",
        metadata={"axes": "ZYX", "threshold_method": method, "threshold": threshold},
    )
    with tifffile.TiffFile(output) as tif:
        mask_slice = tif.pages[SLICE_INDEX].asarray()
    plt.imsave(outdir / "slice_380.png", mask_slice, cmap="gray", vmin=0, vmax=1)
    metrics["mask_bigtiff"] = use_bigtiff
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--method", choices=sorted(METHODS), default="otsu")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    path = args.input.expanduser().resolve(strict=True)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Input must be exactly one existing .tif or .tiff")
    outdir = path.parent / "segmentation"
    outdir.mkdir(exist_ok=True)
    metrics = write_final(path, outdir, args.method) if args.final else candidate(path, outdir, args.method)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
