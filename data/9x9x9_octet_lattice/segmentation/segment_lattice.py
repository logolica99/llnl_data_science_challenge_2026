#!/usr/bin/env python3
"""Segment the octet-lattice CT volume with seeded hysteresis thresholding.

The default invocation writes ``mask.tif`` and the required quality-control
figures next to this script.  ``--preview-only`` evaluates one parameter set
on slice 380 without allocating the full output volume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "9x9x9_octet_lattice.tif"
SLICE_INDEX = 380


def segment_slice(image: np.ndarray, low: int, high: int, close_radius: int = 1) -> np.ndarray:
    """Keep low-threshold pixels only when 8-connected to a high-threshold seed."""
    weak = image >= low
    strong = image >= high
    mask = ndi.binary_propagation(strong, structure=np.ones((3, 3), bool), mask=weak)
    if close_radius:
        yy, xx = np.ogrid[-close_radius : close_radius + 1, -close_radius : close_radius + 1]
        footprint = xx * xx + yy * yy <= close_radius * close_radius
        mask = ndi.binary_closing(mask, structure=footprint)
    # Remove isolated speckle while retaining thin connected struts.
    labels, count = ndi.label(mask, structure=np.ones((3, 3), bool))
    if count:
        sizes = np.bincount(labels.ravel())
        mask = mask & (sizes[labels] >= 6)
    return mask


def save_feedback(image: np.ndarray, mask: np.ndarray, output: Path, title: str) -> None:
    lo, hi = np.percentile(image, (1, 99.7))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=lo, vmax=hi)
    axes[0].set_title("CT slice 380")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"mask (foreground={int(mask.sum()):,})")
    axes[2].imshow(image, cmap="gray", vmin=lo, vmax=hi)
    axes[2].imshow(mask, cmap="autumn", alpha=0.35, vmin=0, vmax=1)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def save_histogram(image: np.ndarray, output: Path, low: int, high: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(image.ravel()[::4], bins=256, color="0.25")
    ax.axvline(low, color="tab:orange", label=f"low={low}")
    ax.axvline(high, color="tab:red", label=f"high={high}")
    ax.set(title="Slice 380 intensity histogram", xlabel="uint16 intensity", ylabel="sampled voxels")
    ax.legend()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--low", type=int, default=40000)
    parser.add_argument("--high", type=int, default=50468)
    parser.add_argument("--close-radius", type=int, default=1)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--iteration", type=int, default=0)
    args = parser.parse_args()

    volume = tifffile.memmap(args.input)
    image = np.asarray(volume[SLICE_INDEX])
    preview = segment_slice(image, args.low, args.high, args.close_radius)
    suffix = f"_iter{args.iteration}" if args.iteration else ""
    save_feedback(image, preview, HERE / f"preview{suffix}.png", f"low={args.low}, high={args.high}, close={args.close_radius}")
    save_histogram(image, HERE / f"histogram{suffix}.png", args.low, args.high)
    if args.preview_only:
        print(f"preview foreground={int(preview.sum())} fraction={float(preview.mean()):.6f}")
        return

    output = HERE / "mask.tif"
    mask_volume = tifffile.memmap(output, shape=volume.shape, dtype=np.uint8, bigtiff=True)
    foreground = 0
    for z in range(volume.shape[0]):
        mask = segment_slice(np.asarray(volume[z]), args.low, args.high, args.close_radius)
        mask_volume[z] = mask
        foreground += int(mask.sum())
        if z % 50 == 0 or z + 1 == volume.shape[0]:
            mask_volume.flush()
            print(f"segmented {z + 1}/{volume.shape[0]} slices")
    mask_volume.flush()
    Image.fromarray((preview * 255).astype(np.uint8), mode="L").save(HERE / "slice_380.png")
    overlay = np.stack([
        np.maximum(((image - image.min()) / max(1, int(image.max()) - int(image.min())) * 255).astype(np.uint8), preview * 255),
        (((image - image.min()) / max(1, int(image.max()) - int(image.min())) * 255).astype(np.uint8) * (~preview)),
        (((image - image.min()) / max(1, int(image.max()) - int(image.min())) * 255).astype(np.uint8) * (~preview)),
    ], axis=-1)
    Image.fromarray(overlay, mode="RGB").save(HERE / "overlay_slice_380.png")
    total = int(np.prod(volume.shape))
    print(f"foreground={foreground} background={total - foreground} fraction={foreground / total:.8f}")


if __name__ == "__main__":
    main()
