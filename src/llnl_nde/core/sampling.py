"""Shared deterministic XYZ corridor sampling for QA, metrics, and evidence."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def perpendicular_basis(direction_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction_xyz, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Strut endpoints must define a finite non-zero direction")
    unit = direction / length
    reference = (
        np.asarray([1.0, 0.0, 0.0])
        if abs(unit[0]) < 0.8
        else np.asarray([0.0, 1.0, 0.0])
    )
    first = np.cross(unit, reference)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    second /= np.linalg.norm(second)
    return first, second


def radial_disk_offsets(
    radius_voxels: float,
    angular_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if radius_voxels <= 0 or angular_samples < 4:
        raise ValueError("Corridor radius must be positive and angular_samples >= 4")
    integer_radius = int(math.ceil(radius_voxels))
    vectors = [[0.0, 0.0]]
    radius_ids = [0]
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    for radius in range(1, integer_radius + 1):
        effective = min(float(radius), float(radius_voxels))
        for angle in angles:
            vectors.append(
                [effective * math.cos(float(angle)), effective * math.sin(float(angle))]
            )
            radius_ids.append(radius)
    return np.asarray(vectors, dtype=np.float64), np.asarray(radius_ids, dtype=np.int64)


def corridor_coordinates(
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    axial_samples: int,
    radius_voxels: float,
    angular_samples: int,
    axial_padding_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(axial, radial, xyz)`` sample coordinates and their parameters."""

    if axial_samples < 3:
        raise ValueError("axial_samples must be at least 3")
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    direction = end - start
    first, second = perpendicular_basis(direction)
    radial, radius_ids = radial_disk_offsets(radius_voxels, angular_samples)
    axial_t = np.linspace(
        -float(axial_padding_fraction),
        1.0 + float(axial_padding_fraction),
        int(axial_samples),
    )
    base = start[None, :] + axial_t[:, None] * direction[None, :]
    offsets = radial[:, 0, None] * first + radial[:, 1, None] * second
    return base[:, None, :] + offsets[None, :, :], axial_t, radius_ids


def sample_corridor(
    volume_zyx: np.ndarray,
    start_xyz: np.ndarray,
    end_xyz: np.ndarray,
    *,
    threshold: float,
    axial_samples: int,
    radius_voxels: float,
    angular_samples: int,
    axial_padding_fraction: float = 0.0,
) -> dict[str, Any]:
    """Sample using the pinned ``volume[z,y,x]`` indexing convention."""

    coordinates, axial_t, radius_ids = corridor_coordinates(
        start_xyz,
        end_xyz,
        axial_samples=axial_samples,
        radius_voxels=radius_voxels,
        angular_samples=angular_samples,
        axial_padding_fraction=axial_padding_fraction,
    )
    indices = np.rint(coordinates).astype(np.int64)
    shape_xyz = np.asarray(volume_zyx.shape[::-1], dtype=np.int64)
    valid = np.all((indices >= 0) & (indices < shape_xyz), axis=2)
    clipped = np.clip(indices, 0, shape_xyz - 1)
    # Explicit XYZ -> ZYX mapping. Do not replace this with generic tuple use.
    values = np.asarray(
        volume_zyx[
            clipped[..., 2],
            clipped[..., 1],
            clipped[..., 0],
        ],
        dtype=np.float64,
    )
    foreground = (values >= float(threshold)) & valid
    values[~valid] = np.nan
    return {
        "coordinates_xyz": coordinates,
        "axial_t": axial_t,
        "radius_ids": radius_ids,
        "valid": valid,
        "intensity": values,
        "foreground": foreground,
    }


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)
