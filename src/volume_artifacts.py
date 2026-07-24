"""Deterministic analysis and rendering for MCP-exposed volume artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import ndimage
from skimage import measure


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CHUNK_DEPTH = 16


def _load_3d_npy(filepath: str) -> tuple[Path, np.ndarray]:
    path = Path(filepath).expanduser().resolve()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, found shape {array.shape}: {path}")
    return path, array


def _slabs(depth: int) -> Iterator[tuple[int, int]]:
    for start in range(0, depth, SUMMARY_CHUNK_DEPTH):
        yield start, min(start + SUMMARY_CHUNK_DEPTH, depth)


def compare_segmentation_masks(
    raw_filepath: str,
    mask_filepaths: list[str],
    thresholds: list[float],
) -> dict[str, Any]:
    """Validate threshold masks and return compact foreground statistics."""
    if not mask_filepaths:
        raise ValueError("At least one mask filepath is required.")
    if len(mask_filepaths) != len(thresholds):
        raise ValueError(
            "mask_filepaths and thresholds must contain the same number of items."
        )

    raw_path, raw = _load_3d_npy(raw_filepath)
    candidates: list[dict[str, Any]] = []

    for threshold, mask_filepath in zip(
        thresholds, mask_filepaths, strict=True
    ):
        if not np.isfinite(threshold):
            raise ValueError("Every threshold must be finite.")

        mask_path, mask = _load_3d_npy(mask_filepath)
        if mask.shape != raw.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match raw shape "
                f"{raw.shape}: {mask_path}"
            )
        if mask.dtype.kind not in "bui":
            raise TypeError(
                "Masks must use a boolean or integer dtype, found "
                f"{mask.dtype}: {mask_path}"
            )

        foreground_voxels = sum(
            int(np.count_nonzero(mask[start:end]))
            for start, end in _slabs(mask.shape[0])
        )
        total_voxels = int(mask.size)
        candidates.append(
            {
                "threshold": float(threshold),
                "path": str(mask_path),
                "dtype": str(mask.dtype),
                "foreground_voxels": foreground_voxels,
                "total_voxels": total_voxels,
                "foreground_percent": (
                    100.0 * foreground_voxels / total_voxels
                    if total_voxels
                    else 0.0
                ),
            }
        )

    return {
        "status": "ok",
        "raw_path": str(raw_path),
        "shape": list(raw.shape),
        "candidates": candidates,
    }


def summarize_nde_artifacts(
    raw_filepath: str,
    mask_filepath: str,
    skeleton_filepath: str | None = None,
) -> dict[str, Any]:
    """Return compact intensity, mask, and skeleton metrics for an NDE report."""
    raw_path, raw = _load_3d_npy(raw_filepath)
    mask_path, mask = _load_3d_npy(mask_filepath)
    if raw.dtype.kind not in "biuf":
        raise TypeError(
            f"Raw volume must use a real numeric dtype, found "
            f"{raw.dtype}: {raw_path}"
        )
    if mask.shape != raw.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match raw shape "
            f"{raw.shape}: {mask_path}"
        )
    if mask.dtype.kind not in "bui":
        raise TypeError(
            f"Mask must use a boolean or integer dtype, found "
            f"{mask.dtype}: {mask_path}"
        )

    foreground_voxels = 0
    foreground_intensity_sum = 0.0
    for start, end in _slabs(raw.shape[0]):
        foreground = np.asarray(mask[start:end] > 0)
        chunk_foreground = int(np.count_nonzero(foreground))
        foreground_voxels += chunk_foreground
        if chunk_foreground:
            foreground_intensity_sum += float(
                np.sum(
                    raw[start:end],
                    where=foreground,
                    dtype=np.float64,
                    initial=0.0,
                )
            )

    total_voxels = int(mask.size)
    mean_intensity: float | None = None
    mean_intensity_status = "empty_mask"
    if foreground_voxels:
        candidate_mean = foreground_intensity_sum / foreground_voxels
        if np.isfinite(candidate_mean):
            mean_intensity = candidate_mean
            mean_intensity_status = "computed"
        else:
            mean_intensity_status = "non_finite"

    skeleton_path: Path | None = None
    skeleton_metrics: dict[str, Any] | None = None
    if skeleton_filepath is not None:
        skeleton_path, skeleton = _load_3d_npy(skeleton_filepath)
        if skeleton.shape != raw.shape:
            raise ValueError(
                f"Skeleton shape {skeleton.shape} does not match raw shape "
                f"{raw.shape}: {skeleton_path}"
            )
        if skeleton.dtype.kind not in "bui":
            raise TypeError(
                "Skeleton must use a boolean or integer dtype, found "
                f"{skeleton.dtype}: {skeleton_path}"
            )

        neighborhood = np.ones((3, 3, 3), dtype=np.uint8)
        neighborhood[1, 1, 1] = 0
        skeleton_voxels = 0
        endpoints = 0
        branch_points = 0
        for start, end in _slabs(skeleton.shape[0]):
            halo_start = max(0, start - 1)
            halo_end = min(skeleton.shape[0], end + 1)
            skeleton_slab = np.asarray(
                skeleton[halo_start:halo_end] > 0,
                dtype=np.uint8,
            )
            neighbor_counts = ndimage.convolve(
                skeleton_slab,
                neighborhood,
                mode="constant",
                cval=0,
            )
            core = slice(start - halo_start, end - halo_start)
            core_skeleton = skeleton_slab[core] > 0
            core_counts = neighbor_counts[core]
            skeleton_voxels += int(np.count_nonzero(core_skeleton))
            endpoints += int(
                np.count_nonzero(core_skeleton & (core_counts == 1))
            )
            branch_points += int(
                np.count_nonzero(core_skeleton & (core_counts > 2))
            )

        skeleton_metrics = {
            "path": str(skeleton_path),
            "dtype": str(skeleton.dtype),
            "skeleton_voxels": skeleton_voxels,
            "endpoints_26_connected": endpoints,
            "branch_points_26_connected": branch_points,
        }

    return {
        "status": "ok",
        "shape": list(raw.shape),
        "raw": {
            "path": str(raw_path),
            "dtype": str(raw.dtype),
            "total_voxels": int(raw.size),
        },
        "mask": {
            "path": str(mask_path),
            "dtype": str(mask.dtype),
            "foreground_voxels": foreground_voxels,
            "total_voxels": total_voxels,
            "foreground_percent": (
                100.0 * foreground_voxels / total_voxels
                if total_voxels
                else 0.0
            ),
            "mean_foreground_intensity": mean_intensity,
            "mean_foreground_intensity_status": mean_intensity_status,
        },
        "skeleton": skeleton_metrics,
    }


def render_volume_3d(
    input_filepath: str,
    output_filepath: str,
    surface_level: float = 0.5,
    downsample_factor: int = 2,
    elevation: float = 30.0,
    azimuth: float = 45.0,
    skeleton_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render a normalized 3D isosurface and optional skeleton overlay."""
    if not np.isfinite(surface_level) or not 0.0 < surface_level < 1.0:
        raise ValueError("surface_level must be finite and between 0 and 1.")
    if (
        isinstance(downsample_factor, bool)
        or not isinstance(downsample_factor, int)
        or downsample_factor < 1
    ):
        raise ValueError("downsample_factor must be a positive integer.")
    if not np.isfinite(elevation) or not np.isfinite(azimuth):
        raise ValueError("elevation and azimuth must be finite.")

    input_path, volume = _load_3d_npy(input_filepath)
    output_path = Path(output_filepath).expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"Output file must use the .png extension: {output_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists; choose a new path or enable overwrite: "
            f"{output_path}"
        )

    sampled = np.array(
        volume[
            ::downsample_factor,
            ::downsample_factor,
            ::downsample_factor,
        ],
        dtype=np.float32,
        copy=True,
        order="C",
    )
    minimum = float(np.min(sampled))
    maximum = float(np.max(sampled))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("The downsampled volume contains non-finite extrema.")
    if minimum == maximum:
        raise ValueError("A constant-valued volume has no renderable isosurface.")

    absolute_level = minimum + surface_level * (maximum - minimum)
    vertices, faces, _, _ = measure.marching_cubes(
        sampled,
        level=absolute_level,
    )

    skeleton_path: Path | None = None
    skeleton_voxels: int | None = None
    rendered_skeleton_points = 0
    skeleton_points: np.ndarray | None = None
    if skeleton_filepath is not None:
        skeleton_path, skeleton = _load_3d_npy(skeleton_filepath)
        if skeleton.shape != volume.shape:
            raise ValueError(
                f"Skeleton shape {skeleton.shape} does not match volume shape "
                f"{volume.shape}: {skeleton_path}"
            )
        skeleton_voxels = int(np.count_nonzero(skeleton))
        skeleton_points = np.argwhere(
            skeleton[
                ::downsample_factor,
                ::downsample_factor,
                ::downsample_factor,
            ]
            > 0
        )
        rendered_skeleton_points = int(len(skeleton_points))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10, 8))
    try:
        axes = figure.add_subplot(111, projection="3d")
        axes.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            faces,
            vertices[:, 2],
            cmap="viridis",
            linewidth=0.1,
            edgecolor="none",
            alpha=0.3 if skeleton_points is not None else 1.0,
        )
        if skeleton_points is not None and len(skeleton_points):
            axes.scatter(
                skeleton_points[:, 0],
                skeleton_points[:, 1],
                skeleton_points[:, 2],
                color="red",
                s=1.0,
                alpha=0.8,
                label="Skeleton",
            )
        axes.set_title(f"3D Isosurface (normalized level={surface_level})")
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)

    return {
        "status": "ok",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "skeleton_path": str(skeleton_path) if skeleton_path else None,
        "source_shape": list(volume.shape),
        "render_shape": list(sampled.shape),
        "surface_level": float(surface_level),
        "absolute_level": absolute_level,
        "downsample_factor": downsample_factor,
        "elevation": float(elevation),
        "azimuth": float(azimuth),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "skeleton_voxels": skeleton_voxels,
        "rendered_skeleton_points": rendered_skeleton_points,
    }
