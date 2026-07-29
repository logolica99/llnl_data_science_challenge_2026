"""Deterministic batched rotated-cuboid measurements for production Stage 2.

This module contains measurement logic only.  It consumes a Stage 1 canonical
segmentation mask and independently localized nominal graph; it never assigns
defect labels or selects an intensity threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy import ndimage

from .artifacts import require_new_path, sha256_file, sha256_json, write_json_atomic
from .lattice import LatticeGraph, load_lattice_json
from .volume import AXIS_MAPPING, load_volume


STRUT_METRICS_SCHEMA_VERSION = "part2-strut-metrics/2.0.0"
CORRIDOR_CALIBRATION_SCHEMA_VERSION = "part2-corridor-calibration/1.0.0"
COMPONENT_STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)
COMPONENT_STRUCTURE_2D = ndimage.generate_binary_structure(2, 2)

DEFAULT_STAGE2_CONFIG: dict[str, Any] = {
    "schema_version": "part2-strut-metrics-config/1.0.0",
    "otsu_threshold": None,
    "axial_padding_fraction_total": 0.20,
    "interpolation_batch_size": 64,
    "junction_mask_radius_voxels": 3.0,
    "transverse_margin_fraction": 0.20,
    "minimum_transverse_margin_voxels": 2.0,
    "collar_fraction": 0.20,
    "endpoint_seed_half_length_voxels": 2.0,
    "collar_half_length_voxels": 1.0,
    "minimum_axial_foreground_fraction": 0.10,
    "centerline_smoothing_passes": 2,
    "minimum_valid_roi_fraction": 0.99,
    "corridor_bootstrap": {
        "sample_count": 24,
        "minimum_valid_samples": 6,
        "minimum_valid_slice_fraction": 0.75,
        "central_fraction": 0.60,
        "calibration_half_width_voxels": 12.0,
        "maximum_axis_distance_voxels": 8.0,
        "radial_extent_quantile": 0.90,
        "radius_multiplier": 1.0,
        "radius_safety_voxels": 0.5,
        "minimum_radius_voxels": 1.0,
        "maximum_radius_voxels": 11.0,
    },
}

METRIC_FIELDS = [
    "strut_id",
    "junction0_id",
    "junction1_id",
    "length_voxels",
    "corridor_radius_voxels",
    "cuboid_half_width_voxels",
    "axial_padding_fraction_total",
    "corridor_foreground_fraction",
    "minimum_foreground_fraction",
    "median_foreground_fraction",
    "maximum_foreground_fraction",
    "maximum_axial_gap_samples",
    "maximum_axial_gap_fraction",
    "endpoint0_support_fraction",
    "endpoint1_support_fraction",
    "a_collar_foreground_fraction",
    "b_collar_foreground_fraction",
    "interior_component_count",
    "largest_component_fraction",
    "same_material_component_connects_a_to_b",
    "same_component_connects_collar_a_to_b",
    "shared_component_voxel_count_in_corridor",
    "endpoint0_to_collar_component_voxel_count_in_corridor",
    "endpoint1_to_collar_component_voxel_count_in_corridor",
    "both_endpoint_segments_observed",
    "junction_masked_collar_shared_component_voxel_count_in_corridor",
    "edt_radius_median_voxels",
    "centerline_curvature_rms_voxels",
    "roi_in_bounds_fraction",
    "roi_valid",
]


@dataclass(frozen=True)
class CuboidGeometry:
    """Auditable local-to-CT affine geometry for one nominal strut."""

    strut_id: int
    junction0_id: int
    junction1_id: int
    node_a_xyz: list[float]
    node_b_xyz: list[float]
    center_xyz: list[float]
    length_voxels: float
    axial_padding_fraction_total: float
    axial_margin_each_end_voxels: float
    half_width_voxels: float
    corridor_radius_voxels: float
    transverse_margin_voxels: float
    basis_x_xyz: list[float]
    basis_y_xyz: list[float]
    basis_z_xyz: list[float]
    local_shape_zyx: list[int]


@dataclass(frozen=True)
class SampledCuboid:
    foreground_probability_zyx: np.ndarray
    valid_zyx: np.ndarray
    geometry: CuboidGeometry
    local_x: np.ndarray
    local_y: np.ndarray
    local_z: np.ndarray


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_stage2_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the strict fragment while retaining direct-core compatibility."""

    raw = config or {}
    fragment = raw.get("stage_2_strut_metrics", raw)
    if not isinstance(fragment, dict):
        raise ValueError("stage_2_strut_metrics must be a JSON object")
    merged = _deep_merge(DEFAULT_STAGE2_CONFIG, fragment)
    # Older direct-core callers used this name.  The strict MCP schema does not.
    if "axial_padding_fraction" in fragment and "axial_padding_fraction_total" not in fragment:
        merged["axial_padding_fraction_total"] = fragment["axial_padding_fraction"]
    numeric_positive = (
        "interpolation_batch_size",
        "junction_mask_radius_voxels",
        "collar_half_length_voxels",
    )
    for field in numeric_positive:
        value = merged[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{field} must be positive")
    if not math.isclose(float(merged["axial_padding_fraction_total"]), 0.20, abs_tol=1e-12):
        raise ValueError("Stage 2 requires exactly 0.20 total axial padding")
    if not 0.0 < float(merged["collar_fraction"]) < 0.5:
        raise ValueError("collar_fraction must be between zero and one half")
    bootstrap = merged["corridor_bootstrap"]
    if not isinstance(bootstrap, dict):
        raise ValueError("corridor_bootstrap must be an object")
    if int(bootstrap["minimum_valid_samples"]) > int(bootstrap["sample_count"]):
        raise ValueError("minimum_valid_samples cannot exceed sample_count")
    if float(bootstrap["minimum_radius_voxels"]) > float(bootstrap["maximum_radius_voxels"]):
        raise ValueError("corridor bootstrap radius bounds are reversed")
    return merged


def stable_frame(
    node_a_xyz: np.ndarray, node_b_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a deterministic right-handed frame whose local z is A to B."""

    edge = np.asarray(node_b_xyz, dtype=np.float64) - np.asarray(
        node_a_xyz, dtype=np.float64
    )
    length = float(np.linalg.norm(edge))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("A strut has coincident or invalid endpoint positions")
    basis_z = edge / length
    world_axes = np.eye(3, dtype=np.float64)
    reference = world_axes[int(np.argmin(np.abs(world_axes @ basis_z)))]
    basis_x = reference - float(np.dot(reference, basis_z)) * basis_z
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(basis_z, basis_x)
    return basis_x, basis_y, basis_z, length


def _prepare_sampling(
    *,
    strut_id: int,
    endpoint_ids: np.ndarray,
    node_a_xyz: np.ndarray,
    node_b_xyz: np.ndarray,
    half_width: float,
    corridor_radius: float,
    axial_padding_fraction_total: float,
) -> tuple[np.ndarray, CuboidGeometry, np.ndarray, np.ndarray, np.ndarray]:
    """Construct flat ZYX coordinates using p=c+x*ex+y*ey+z*ez."""

    basis_x, basis_y, basis_z, length = stable_frame(node_a_xyz, node_b_xyz)
    center = (node_a_xyz + node_b_xyz) / 2.0
    axial_margin = 0.5 * axial_padding_fraction_total * length
    local_x = np.arange(-half_width, half_width + 0.5, 1.0, dtype=np.float64)
    local_y = np.arange(-half_width, half_width + 0.5, 1.0, dtype=np.float64)
    # Preserve A and B as exact sampling planes while extending the strict ROI
    # by ten percent of L on each end.  Constructing one padded linspace would
    # shift the nominal samples and generally omit z=0 and z=L.
    nominal_z = np.linspace(
        0.0, length, int(np.ceil(length)) + 1, dtype=np.float64
    )
    if axial_margin > 0.0:
        margin_samples = int(np.ceil(axial_margin)) + 1
        before_a = np.linspace(
            -axial_margin, 0.0, margin_samples, dtype=np.float64
        )[:-1]
        after_b = np.linspace(
            length, length + axial_margin, margin_samples, dtype=np.float64
        )[1:]
        local_z = np.concatenate((before_a, nominal_z, after_b))
    else:
        local_z = nominal_z
    plane_size = local_y.size * local_x.size
    zz = np.repeat(local_z, plane_size)
    yy = np.tile(np.repeat(local_y, local_x.size), local_z.size)
    xx = np.tile(local_x, local_z.size * local_y.size)
    z_centered = zz - 0.5 * length
    coordinates_zyx = np.empty((3, xx.size), dtype=np.float64)
    for coordinate_index, world_axis in enumerate((2, 1, 0)):
        coordinates_zyx[coordinate_index] = (
            center[world_axis]
            + xx * basis_x[world_axis]
            + yy * basis_y[world_axis]
            + z_centered * basis_z[world_axis]
        )
    shape = (local_z.size, local_y.size, local_x.size)
    geometry = CuboidGeometry(
        strut_id=int(strut_id),
        junction0_id=int(endpoint_ids[0]),
        junction1_id=int(endpoint_ids[1]),
        node_a_xyz=np.asarray(node_a_xyz, dtype=float).tolist(),
        node_b_xyz=np.asarray(node_b_xyz, dtype=float).tolist(),
        center_xyz=center.tolist(),
        length_voxels=length,
        axial_padding_fraction_total=float(axial_padding_fraction_total),
        axial_margin_each_end_voxels=float(axial_margin),
        half_width_voxels=float(half_width),
        corridor_radius_voxels=float(corridor_radius),
        transverse_margin_voxels=float(max(0.0, half_width - corridor_radius)),
        basis_x_xyz=basis_x.tolist(),
        basis_y_xyz=basis_y.tolist(),
        basis_z_xyz=basis_z.tolist(),
        local_shape_zyx=list(shape),
    )
    return coordinates_zyx, geometry, local_x, local_y, local_z


def _sample_batch(
    mask_zyx: np.ndarray,
    graph: LatticeGraph,
    edge_rows: Iterable[int],
    *,
    half_width: float,
    corridor_radius: float,
    axial_padding_fraction_total: float,
) -> list[SampledCuboid]:
    """Concatenate variable-length cuboids into one SciPy interpolation call."""

    prepared = []
    for row in edge_rows:
        first_row, second_row = graph.edge_node_rows[row]
        prepared.append(
            _prepare_sampling(
                strut_id=int(graph.edge_ids[row]),
                endpoint_ids=graph.edge_node_ids[row],
                node_a_xyz=graph.node_positions_xyz[first_row],
                node_b_xyz=graph.node_positions_xyz[second_row],
                half_width=half_width,
                corridor_radius=corridor_radius,
                axial_padding_fraction_total=axial_padding_fraction_total,
            )
        )
    if not prepared:
        return []
    counts = np.asarray([item[0].shape[1] for item in prepared], dtype=np.int64)
    offsets = np.empty(len(prepared) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    coordinates = np.empty((3, int(offsets[-1])), dtype=np.float64)
    for item, start, stop in zip(prepared, offsets[:-1], offsets[1:], strict=True):
        coordinates[:, start:stop] = item[0]
    shape_zyx = np.asarray(mask_zyx.shape, dtype=np.float64)
    valid_flat = np.all(
        (coordinates >= 0.0) & (coordinates <= (shape_zyx[:, None] - 1.0)), axis=0
    )
    sampled = ndimage.map_coordinates(
        mask_zyx,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
        output=np.float32,
    )
    result: list[SampledCuboid] = []
    for item, start, stop in zip(prepared, offsets[:-1], offsets[1:], strict=True):
        _, geometry, local_x, local_y, local_z = item
        shape = tuple(geometry.local_shape_zyx)
        result.append(
            SampledCuboid(
                foreground_probability_zyx=sampled[start:stop].reshape(shape),
                valid_zyx=valid_flat[start:stop].reshape(shape),
                geometry=geometry,
                local_x=local_x,
                local_y=local_y,
                local_z=local_z,
            )
        )
    return result


def _edge_midpoints(graph: LatticeGraph, rows: list[int]) -> np.ndarray:
    endpoints = graph.edge_node_rows[np.asarray(rows, dtype=np.int64)]
    return (
        graph.node_positions_xyz[endpoints[:, 0]]
        + graph.node_positions_xyz[endpoints[:, 1]]
    ) / 2.0


def _spatially_diverse_rows(graph: LatticeGraph, rows: list[int], count: int) -> list[int]:
    if len(rows) < count:
        raise ValueError("Not enough in-bounds graph struts for corridor calibration")
    points = _edge_midpoints(graph, rows)
    first = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    selected = [first]
    nearest = np.linalg.norm(points - points[first], axis=1)
    while len(selected) < count:
        nearest[np.asarray(selected, dtype=int)] = -1.0
        next_index = int(np.argmax(nearest))
        selected.append(next_index)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[next_index], axis=1))
    return [rows[index] for index in selected]


def calibrate_corridor_radius(
    mask_zyx: np.ndarray,
    graph: LatticeGraph,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Derive one scan-level radius without labels or outcome-conditioned sampling."""

    policy = config["corridor_bootstrap"]
    half_width = float(policy["calibration_half_width_voxels"])
    shape_xyz = np.asarray(mask_zyx.shape[::-1], dtype=np.float64)
    candidate_rows: list[int] = []
    for row, node_rows in enumerate(graph.edge_node_rows):
        endpoints = graph.node_positions_xyz[node_rows]
        _, _, _, length = stable_frame(endpoints[0], endpoints[1])
        axial_margin = 0.1 * length
        bound = half_width + axial_margin + 1.0
        if np.all(endpoints >= bound) and np.all(endpoints <= shape_xyz - 1.0 - bound):
            candidate_rows.append(row)
    sample_count = min(int(policy["sample_count"]), len(candidate_rows))
    minimum_valid = int(policy["minimum_valid_samples"])
    if sample_count < minimum_valid:
        raise ValueError(
            f"Corridor bootstrap has {sample_count} in-bounds struts; requires {minimum_valid}"
        )
    selected_rows = _spatially_diverse_rows(graph, candidate_rows, sample_count)
    sampled = _sample_batch(
        mask_zyx,
        graph,
        selected_rows,
        half_width=half_width,
        corridor_radius=0.0,
        axial_padding_fraction_total=0.0,
    )
    per_strut: list[dict[str, Any]] = []
    accepted_extents: list[float] = []
    central_fraction = float(policy["central_fraction"])
    minimum_valid_slice_fraction = float(policy["minimum_valid_slice_fraction"])
    maximum_axis_distance = float(policy["maximum_axis_distance_voxels"])
    for cuboid in sampled:
        binary = cuboid.foreground_probability_zyx >= 0.5
        edge_fraction = 0.5 * (1.0 - central_fraction)
        central_indices = np.flatnonzero(
            (cuboid.local_z >= edge_fraction * cuboid.geometry.length_voxels)
            & (cuboid.local_z <= (1.0 - edge_fraction) * cuboid.geometry.length_voxels)
        )
        extents: list[float] = []
        for z_index in central_indices:
            labels, component_count = ndimage.label(
                binary[z_index] & cuboid.valid_zyx[z_index], structure=COMPONENT_STRUCTURE_2D
            )
            candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
            for component_id in range(1, component_count + 1):
                iy, ix = np.nonzero(labels == component_id)
                if ix.size < 5:
                    continue
                x_values = cuboid.local_x[ix]
                y_values = cuboid.local_y[iy]
                distances = np.hypot(x_values, y_values)
                candidates.append((float(distances.min()), -int(ix.size), x_values, y_values))
            if not candidates:
                continue
            minimum_distance, _, x_values, y_values = min(candidates, key=lambda item: (item[0], item[1]))
            if minimum_distance > maximum_axis_distance:
                continue
            if np.any(np.abs(x_values) >= half_width) or np.any(np.abs(y_values) >= half_width):
                continue
            extents.append(float(np.max(np.hypot(x_values, y_values))))
        valid_fraction = len(extents) / max(1, int(central_indices.size))
        accepted = bool(extents and valid_fraction >= minimum_valid_slice_fraction)
        record: dict[str, Any] = {
            "strut_id": cuboid.geometry.strut_id,
            "central_slice_count": int(central_indices.size),
            "valid_slice_count": len(extents),
            "valid_slice_fraction": float(valid_fraction),
            "accepted": accepted,
        }
        if accepted:
            median_extent = float(np.median(extents))
            record["median_radial_extent_voxels"] = median_extent
            accepted_extents.append(median_extent)
        per_strut.append(record)
    if len(accepted_extents) < minimum_valid:
        raise ValueError(
            f"Corridor bootstrap accepted {len(accepted_extents)} struts; requires {minimum_valid}"
        )
    quantile_extent = float(
        np.quantile(accepted_extents, float(policy["radial_extent_quantile"]))
    )
    raw_radius = (
        quantile_extent * float(policy["radius_multiplier"])
        + float(policy["radius_safety_voxels"])
    )
    radius = float(
        np.clip(
            np.ceil(raw_radius),
            float(policy["minimum_radius_voxels"]),
            float(policy["maximum_radius_voxels"]),
        )
    )
    return {
        "schema_version": CORRIDOR_CALIBRATION_SCHEMA_VERSION,
        "method": "label-blind central-slice nearest-component radial-extent bootstrap",
        "label_blind": True,
        "source": "stage_1_canonical_segmentation_mask",
        "interpolation": "trilinear scipy.ndimage.map_coordinates order=1",
        "foreground_probability_cutoff": 0.5,
        "selection": "deterministic farthest-point sampling of in-bounds graph midpoints",
        "policy": policy,
        "candidate_strut_count": len(candidate_rows),
        "sample_count_used": sample_count,
        "accepted_strut_count": len(accepted_extents),
        "quantile_radial_extent_voxels": quantile_extent,
        "unclipped_radius_voxels": raw_radius,
        "corridor_radius_voxels": radius,
        "per_strut": per_strut,
    }


def _disk(local_x: np.ndarray, local_y: np.ndarray, radius: float) -> np.ndarray:
    yy, xx = np.meshgrid(local_y, local_x, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def _window_indices(local_z: np.ndarray, center: float, half_length: float) -> np.ndarray:
    indices = np.flatnonzero(np.abs(local_z - center) <= half_length)
    if indices.size == 0:
        raise ValueError("An axial measurement window contains no resampled slice")
    return indices


def _shared_components(
    labels: np.ndarray,
    local_z: np.ndarray,
    disk: np.ndarray,
    first_z: float,
    second_z: float,
    first_half_length: float,
    second_half_length: float | None = None,
) -> tuple[bool, int, int]:
    second_half = (
        first_half_length if second_half_length is None else second_half_length
    )
    first = _window_indices(local_z, first_z, first_half_length)
    second = _window_indices(local_z, second_z, second_half)
    first_labels = np.unique(labels[first][:, disk])
    second_labels = np.unique(labels[second][:, disk])
    shared = np.intersect1d(first_labels[first_labels > 0], second_labels[second_labels > 0])
    sizes = np.bincount(labels.ravel())
    voxels = int(sizes[shared].sum()) if shared.size else 0
    return bool(shared.size), int(shared.size), voxels


def _window_foreground_fraction(
    binary: np.ndarray,
    valid: np.ndarray,
    local_z: np.ndarray,
    disk: np.ndarray,
    center: float,
    half_length: float,
) -> float:
    indices = _window_indices(local_z, center, half_length)
    selected_valid = valid[indices][:, disk]
    denominator = int(np.count_nonzero(selected_valid))
    if denominator == 0:
        return 0.0
    return float(np.count_nonzero(binary[indices][:, disk] & selected_valid) / denominator)


def _longest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def _curvature_rms(
    binary: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    axial_indices: np.ndarray,
    disk: np.ndarray,
    smoothing_passes: int,
) -> float:
    centroids: list[list[float]] = []
    valid: list[bool] = []
    yy, xx = np.meshgrid(local_y, local_x, indexing="ij")
    for z_index in axial_indices:
        foreground = binary[z_index] & disk
        if not np.any(foreground):
            centroids.append([math.nan, math.nan])
            valid.append(False)
        else:
            centroids.append([float(np.mean(xx[foreground])), float(np.mean(yy[foreground]))])
            valid.append(True)
    points = np.asarray(centroids, dtype=np.float64)
    valid_array = np.asarray(valid, dtype=bool)
    if np.count_nonzero(valid_array) < 5:
        return 0.0
    positions = np.arange(len(points))
    for axis in range(2):
        points[~valid_array, axis] = np.interp(
            positions[~valid_array], positions[valid_array], points[valid_array, axis]
        )
    for _ in range(max(0, smoothing_passes)):
        smoothed = points.copy()
        smoothed[1:-1] = (points[:-2] + 2.0 * points[1:-1] + points[2:]) / 4.0
        points = smoothed
    chord = np.linspace(points[0], points[-1], len(points))
    return float(np.sqrt(np.mean(np.sum((points - chord) ** 2, axis=1))))


def _analyze_cuboid(
    cuboid: SampledCuboid,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry = cuboid.geometry
    radius = geometry.corridor_radius_voxels
    disk = _disk(cuboid.local_x, cuboid.local_y, radius)
    foreground = cuboid.foreground_probability_zyx >= 0.5
    valid = cuboid.valid_zyx
    corridor = foreground & valid & disk[None, :, :]
    nominal_span = (
        (cuboid.local_z >= 0.0)
        & (cuboid.local_z <= geometry.length_voxels)
    )[:, None, None]
    nominal_corridor = corridor & nominal_span
    yy, xx = np.meshgrid(cuboid.local_y, cuboid.local_x, indexing="ij")
    radial_squared = xx * xx + yy * yy
    junction_radius = float(config["junction_mask_radius_voxels"])
    distance_a = np.sqrt(radial_squared[None, :, :] + cuboid.local_z[:, None, None] ** 2)
    distance_b = np.sqrt(
        radial_squared[None, :, :]
        + (cuboid.local_z[:, None, None] - geometry.length_voxels) ** 2
    )
    junction_clear = (distance_a > junction_radius) & (distance_b > junction_radius)
    interior = nominal_corridor & junction_clear
    labels_interior, component_count = ndimage.label(interior, structure=COMPONENT_STRUCTURE_26)
    sizes = np.bincount(labels_interior.ravel())[1:]
    foreground_count = int(np.count_nonzero(interior))
    largest_fraction = float(sizes.max() / foreground_count) if sizes.size and foreground_count else 0.0

    # Claire's authoritative component domain is the full 20%-padded,
    # unmasked cylindrical corridor. Padding extends inspection but never
    # moves the nominal endpoint windows at z=0 and z=L.
    labels_full, _ = ndimage.label(corridor, structure=COMPONENT_STRUCTURE_26)
    endpoint_half = float(config["endpoint_seed_half_length_voxels"])
    collar_half = float(config["collar_half_length_voxels"])
    # The nominal collar locations are frozen at 0.20L and 0.80L.  Junction
    # masking is supplementary evidence and must not move those windows or
    # redefine Claire's endpoint-to-endpoint connectivity measurement.
    collar_from_end = float(config["collar_fraction"]) * geometry.length_voxels
    collar_a = collar_from_end
    collar_b = geometry.length_voxels - collar_from_end
    direct_connected, _, direct_shared_voxels = _shared_components(
        labels_full,
        cuboid.local_z,
        disk,
        0.0,
        geometry.length_voxels,
        endpoint_half,
    )
    endpoint0_to_collar, _, endpoint0_to_collar_voxels = _shared_components(
        labels_full,
        cuboid.local_z,
        disk,
        0.0,
        collar_a,
        endpoint_half,
        collar_half,
    )
    endpoint1_to_collar, _, endpoint1_to_collar_voxels = _shared_components(
        labels_full,
        cuboid.local_z,
        disk,
        geometry.length_voxels,
        collar_b,
        endpoint_half,
        collar_half,
    )
    collar_connected, _, collar_shared_voxels = _shared_components(
        labels_interior,
        cuboid.local_z,
        disk,
        collar_a,
        collar_b,
        collar_half,
    )

    denominator = np.count_nonzero(valid[:, disk], axis=1)
    numerator = np.count_nonzero(corridor[:, disk], axis=1)
    occupancy = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    design_indices = np.flatnonzero(
        (cuboid.local_z >= 0.0) & (cuboid.local_z <= geometry.length_voxels)
    )
    interior_indices = np.flatnonzero(
        (cuboid.local_z >= collar_a) & (cuboid.local_z <= collar_b)
    )
    design_occupancy = occupancy[design_indices]
    interior_occupancy = occupancy[interior_indices]
    supported = interior_occupancy >= float(config["minimum_axial_foreground_fraction"])
    maximum_gap = _longest_false_run(supported)
    endpoint_count = min(3, max(1, design_occupancy.size // 2))

    radius_profile: list[float] = []
    for z_index in interior_indices:
        distance = ndimage.distance_transform_edt(corridor[z_index])
        values = distance[disk]
        radius_profile.append(float(values.max()) if values.size else 0.0)
    positive_radii = np.asarray([value for value in radius_profile if value > 0.0])
    edt_radius = float(np.median(positive_radii)) if positive_radii.size else 0.0
    curvature = _curvature_rms(
        corridor,
        cuboid.local_x,
        cuboid.local_y,
        interior_indices,
        disk,
        int(config["centerline_smoothing_passes"]),
    )
    in_bounds_fraction = float(np.mean(valid))
    minimum_valid_roi_fraction = float(config["minimum_valid_roi_fraction"])
    row = {
        "strut_id": geometry.strut_id,
        "junction0_id": geometry.junction0_id,
        "junction1_id": geometry.junction1_id,
        "length_voxels": geometry.length_voxels,
        "corridor_radius_voxels": radius,
        "cuboid_half_width_voxels": geometry.half_width_voxels,
        "axial_padding_fraction_total": geometry.axial_padding_fraction_total,
        "corridor_foreground_fraction": float(np.mean(design_occupancy)),
        "minimum_foreground_fraction": float(np.min(interior_occupancy)),
        "median_foreground_fraction": float(np.median(interior_occupancy)),
        "maximum_foreground_fraction": float(np.max(interior_occupancy)),
        "maximum_axial_gap_samples": int(maximum_gap),
        "maximum_axial_gap_fraction": float(maximum_gap / max(1, supported.size)),
        "endpoint0_support_fraction": float(np.mean(design_occupancy[:endpoint_count])),
        "endpoint1_support_fraction": float(np.mean(design_occupancy[-endpoint_count:])),
        "a_collar_foreground_fraction": _window_foreground_fraction(
            foreground, valid, cuboid.local_z, disk, collar_a, collar_half
        ),
        "b_collar_foreground_fraction": _window_foreground_fraction(
            foreground, valid, cuboid.local_z, disk, collar_b, collar_half
        ),
        "interior_component_count": int(component_count),
        "largest_component_fraction": largest_fraction,
        "same_material_component_connects_a_to_b": direct_connected,
        "same_component_connects_collar_a_to_b": collar_connected,
        "shared_component_voxel_count_in_corridor": direct_shared_voxels,
        "endpoint0_to_collar_component_voxel_count_in_corridor": endpoint0_to_collar_voxels,
        "endpoint1_to_collar_component_voxel_count_in_corridor": endpoint1_to_collar_voxels,
        "both_endpoint_segments_observed": bool(
            endpoint0_to_collar and endpoint1_to_collar
        ),
        "junction_masked_collar_shared_component_voxel_count_in_corridor": collar_shared_voxels,
        "edt_radius_median_voxels": edt_radius,
        "centerline_curvature_rms_voxels": curvature,
        "roi_in_bounds_fraction": in_bounds_fraction,
        "roi_valid": bool(
            in_bounds_fraction + 1e-12 >= minimum_valid_roi_fraction
        ),
    }
    profile = {
        "strut_id": geometry.strut_id,
        "geometry": asdict(geometry),
        "collars": {
            "a_z_voxels": collar_a,
            "b_z_voxels": collar_b,
            "half_length_voxels": collar_half,
            "junction_mask_radius_voxels": junction_radius,
            "junction_mask_overlaps_a_collar": bool(
                collar_a - collar_half <= junction_radius
            ),
            "junction_mask_overlaps_b_collar": bool(
                geometry.length_voxels - (collar_b + collar_half)
                <= junction_radius
            ),
        },
        "local_z_from_node_a_voxels": cuboid.local_z.tolist(),
        "axial_t": (cuboid.local_z / geometry.length_voxels).tolist(),
        "foreground_fraction": occupancy.tolist(),
        "occupancy_profile": occupancy.tolist(),
        "interior_edt_radius_voxels": radius_profile,
        "same_material_component_connects_a_to_b": direct_connected,
        "same_component_connects_collar_a_to_b": collar_connected,
        "endpoint0_to_collar_component_observed": endpoint0_to_collar,
        "endpoint1_to_collar_component_observed": endpoint1_to_collar,
        "endpoint0_to_collar_component_voxel_count_in_corridor": endpoint0_to_collar_voxels,
        "endpoint1_to_collar_component_voxel_count_in_corridor": endpoint1_to_collar_voxels,
        "both_endpoint_segments_observed": bool(
            endpoint0_to_collar and endpoint1_to_collar
        ),
        "roi_in_bounds_fraction": in_bounds_fraction,
    }
    return row, profile


def _finite_text(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}" if math.isfinite(float(value)) else ""
    return str(value)


def _write_metrics_csv(path: str | Path, rows: list[dict[str, Any]], *, overwrite: bool) -> dict[str, Any]:
    destination = require_new_path(path, overwrite)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _finite_text(row[field]) for field in METRIC_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": sha256_file(destination), "changed": True}


def _final_artifact_record(
    staged: dict[str, Any], final_path: Path, *, changed: bool = True
) -> dict[str, Any]:
    """Bind a staged artifact hash to its canonical published path."""

    return {
        "path": str(final_path.resolve()),
        "sha256": str(staged["sha256"]),
        "changed": changed,
    }


def _strict_bundle_paths(
    *,
    calibration_path: Path,
    metrics_path: Path,
    profiles_path: Path,
    report_path: Path,
) -> tuple[Path, dict[str, Path]]:
    """Validate and return the canonical four-file Stage 2 bundle."""

    paths = {
        "corridor_calibration": calibration_path.expanduser().resolve(),
        "per_strut_metrics": metrics_path.expanduser().resolve(),
        "per_strut_profiles": profiles_path.expanduser().resolve(),
        "metrics_report": report_path.expanduser().resolve(),
    }
    expected_names = {
        "corridor_calibration": "corridor_calibration.json",
        "per_strut_metrics": "per_strut_metrics.csv",
        "per_strut_profiles": "per_strut_profiles.json",
        "metrics_report": "metrics_report.json",
    }
    if any(paths[role].name != name for role, name in expected_names.items()):
        raise ValueError("Strict Stage 2 outputs must use the canonical filenames")
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1:
        raise ValueError("Strict Stage 2 outputs must share one canonical directory")
    return next(iter(parents)), paths


def _publish_staged_bundle(
    staging_directory: Path,
    output_directory: Path,
    paths: dict[str, Path],
) -> bool:
    """Publish a complete bundle atomically or accept exact idempotent replay."""

    filenames = {path.name for path in paths.values()}
    if output_directory.exists():
        if not output_directory.is_dir():
            raise NotADirectoryError(
                f"Stage 2 output directory is an existing file: {output_directory}"
            )
        existing_names = {path.name for path in output_directory.iterdir()}
        if existing_names != filenames:
            raise FileExistsError(
                "Stage 2 output directory contains a partial or open-ended bundle: "
                f"{output_directory}"
            )
        for final_path in paths.values():
            staged_path = staging_directory / final_path.name
            if not final_path.is_file() or staged_path.read_bytes() != final_path.read_bytes():
                raise FileExistsError(
                    "Stage 2 output bundle already exists with different bytes: "
                    f"{output_directory}"
                )
        return False

    os.replace(staging_directory, output_directory)
    return True


def compute_strut_metrics(
    ct_path: str | Path,
    localized_graph_path: str | Path,
    output_metrics_path: str | Path,
    output_profiles_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    config: dict[str, Any] | None = None,
    registration_qa_path: str | Path | None = None,
    canonical_mask_path: str | Path | None = None,
    output_calibration_path: str | Path | None = None,
    input_provenance: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure every graph strut with batched trilinear rotated cuboids.

    Production MCP calls must provide ``canonical_mask_path`` and
    ``output_calibration_path``.  Direct legacy core callers may omit the mask;
    that compatibility path is deliberately not exposed by the MCP tool.
    """

    merged = normalize_stage2_config(config)
    configured_threshold = merged.get("otsu_threshold")
    if configured_threshold is not None and not math.isclose(
        float(configured_threshold), float(threshold), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("Frozen Stage 2 threshold does not match the Stage 1 Otsu result")
    volume = load_volume(ct_path)
    ct_hash = sha256_file(volume.path)
    if canonical_mask_path is None:
        mask_array = np.asarray(volume.array >= float(threshold), dtype=np.uint8)
        mask_path = None
        mask_hash = None
        mask_source = "legacy_direct_core_threshold"
    else:
        mask_view = load_volume(canonical_mask_path)
        if mask_view.shape != volume.shape:
            raise ValueError("Canonical mask shape does not match CT shape")
        if mask_view.dtype != np.dtype(np.uint8):
            raise ValueError("Canonical segmentation mask must have dtype uint8")
        mask_array = mask_view.array
        mask_path = mask_view.path
        mask_hash = sha256_file(mask_path)
        mask_source = "stage_1_canonical_segmentation_mask"
    graph = load_lattice_json(localized_graph_path)

    explicit_radius = merged.get("corridor_radius_voxels")
    if explicit_radius is None:
        calibration = calibrate_corridor_radius(mask_array, graph, merged)
    else:
        calibration = {
            "schema_version": CORRIDOR_CALIBRATION_SCHEMA_VERSION,
            "method": "legacy direct-core explicit radius",
            "label_blind": True,
            "source": mask_source,
            "corridor_radius_voxels": float(explicit_radius),
            "warning": "Fixed radii are rejected by the strict Stage 2 MCP schema",
        }
    corridor_radius = float(calibration["corridor_radius_voxels"])
    transverse_margin = max(
        float(merged["minimum_transverse_margin_voxels"]),
        float(merged["transverse_margin_fraction"]) * corridor_radius,
    )
    half_width = float(np.ceil(corridor_radius + transverse_margin))
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    batch_size = int(merged["interpolation_batch_size"])
    batch_count = 0
    for start in range(0, graph.counts["edges"], batch_size):
        stop = min(start + batch_size, graph.counts["edges"])
        cuboids = _sample_batch(
            mask_array,
            graph,
            range(start, stop),
            half_width=half_width,
            corridor_radius=corridor_radius,
            axial_padding_fraction_total=float(merged["axial_padding_fraction_total"]),
        )
        batch_count += 1
        for cuboid in cuboids:
            row, profile = _analyze_cuboid(cuboid, merged)
            rows.append(row)
            profiles.append(profile)

    ids = [int(row["strut_id"]) for row in rows]
    expected_ids = [int(value) for value in graph.edge_ids]
    roi_valid_fraction = float(np.mean([bool(row["roi_valid"]) for row in rows]))
    gates = {
        "one_row_per_nominal_strut": len(rows) == graph.counts["edges"],
        "metric_row_count_matches_graph": len(rows) == graph.counts["edges"],
        "strut_ids_unique": len(ids) == len(set(ids)),
        "strut_ids_exhaustive": ids == expected_ids,
        "axis_mapping_is_xyz_to_zyx": AXIS_MAPPING["coordinate_order"] == ["x", "y", "z"]
        and AXIS_MAPPING["array_axes"] == ["z", "y", "x"],
        "corridor_radius_empirically_derived": calibration["method"]
        != "legacy direct-core explicit radius",
        "all_roi_provenance_present": len(profiles) == len(rows),
    }
    strict_production = canonical_mask_path is not None and output_calibration_path is not None
    deterministic_failure = not all(
        value for key, value in gates.items() if key != "corridor_radius_empirically_derived"
    ) or (strict_production and not gates["corridor_radius_empirically_derived"])
    if deterministic_failure:
        gate = "halt"
    elif roi_valid_fraction < 1.0:
        gate = "manual_review"
    else:
        gate = "pass"

    calibration_path = (
        Path(output_calibration_path)
        if output_calibration_path is not None
        else Path(output_report_path).with_name("corridor_calibration.json")
    )
    strict_output_directory: Path | None = None
    strict_paths: dict[str, Path] | None = None
    staging_directory: Path | None = None
    if strict_production:
        if overwrite:
            raise ValueError("Strict Stage 2 artifacts are immutable; overwrite is forbidden")
        strict_output_directory, strict_paths = _strict_bundle_paths(
            calibration_path=calibration_path,
            metrics_path=Path(output_metrics_path),
            profiles_path=Path(output_profiles_path),
            report_path=Path(output_report_path),
        )
        strict_output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging_directory = Path(
            tempfile.mkdtemp(
                dir=strict_output_directory.parent,
                prefix=f".{strict_output_directory.name}.",
                suffix=".tmp",
            )
        )
        calibration_write_path = staging_directory / calibration_path.name
        metrics_write_path = staging_directory / Path(output_metrics_path).name
        profiles_write_path = staging_directory / Path(output_profiles_path).name
        report_write_path = staging_directory / Path(output_report_path).name
    else:
        calibration_write_path = calibration_path
        metrics_write_path = Path(output_metrics_path)
        profiles_write_path = Path(output_profiles_path)
        report_write_path = Path(output_report_path)

    calibration_payload = {
        **calibration,
        "ct_sha256": ct_hash,
        "canonical_mask_sha256": mask_hash,
        "localized_graph_sha256": graph.source_sha256,
        "configuration_sha256": sha256_json(merged),
    }
    try:
        staged_calibration_artifact = write_json_atomic(
            calibration_write_path, calibration_payload, overwrite=False
        )
    except Exception:
        if staging_directory is not None and staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise
    calibration_artifact = (
        _final_artifact_record(
            staged_calibration_artifact, strict_paths["corridor_calibration"]
        )
        if strict_paths is not None
        else staged_calibration_artifact
    )
    try:
        staged_metrics_artifact = _write_metrics_csv(
            metrics_write_path, rows, overwrite=False if strict_production else overwrite
        )
    except Exception:
        if staging_directory is not None and staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise
    metrics_artifact = (
        _final_artifact_record(staged_metrics_artifact, strict_paths["per_strut_metrics"])
        if strict_paths is not None
        else staged_metrics_artifact
    )
    profiles_payload = {
        "schema_version": STRUT_METRICS_SCHEMA_VERSION,
        "measurement_only": True,
        "classification_performed": False,
        "axis_mapping": AXIS_MAPPING,
        "interpolation": "trilinear scipy.ndimage.map_coordinates order=1",
        "foreground_source": mask_source,
        "foreground_probability_cutoff": 0.5,
        "otsu_threshold": float(threshold),
        "corridor_radius_voxels": corridor_radius,
        "junction_spheres_excluded_from_collar_connectivity": True,
        "profiles": profiles,
    }
    try:
        staged_profiles_artifact = write_json_atomic(
            profiles_write_path, profiles_payload, overwrite=False
        )
    except Exception:
        if staging_directory is not None and staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise
    profiles_artifact = (
        _final_artifact_record(
            staged_profiles_artifact, strict_paths["per_strut_profiles"]
        )
        if strict_paths is not None
        else staged_profiles_artifact
    )
    qa_hash = sha256_file(registration_qa_path) if registration_qa_path else None
    report = {
        "schema_version": STRUT_METRICS_SCHEMA_VERSION,
        "stage_number": 2,
        "stage": "strut_metrics",
        "gate": gate,
        "overall_pass": gate == "pass",
        "measurement_only": True,
        "classification_performed": False,
        "registration_mode": registration_mode,
        "requested_analysis_scope": (input_provenance or {}).get(
            "requested_analysis_scope"
        ),
        "direct_dimensional_measurement_performed": False,
        "otsu_threshold": float(threshold),
        "foreground_source": mask_source,
        "axis_mapping": AXIS_MAPPING,
        "method": {
            "local_mapping": "p=c+x*e_x+y*e_y+(z-L/2)*e_z",
            "axial_padding_fraction_total": float(merged["axial_padding_fraction_total"]),
            "minimum_valid_roi_fraction": float(merged["minimum_valid_roi_fraction"]),
            "interpolation": "trilinear scipy.ndimage.map_coordinates order=1",
            "interpolation_batch_size": batch_size,
            "strut_interpolation_batch_count": batch_count,
            "component_connectivity": "independent per-strut 26-neighbor labeling",
            "primary_connectivity": "one unmasked corridor-local component intersects both A and B endpoint windows",
            "primary_connectivity_span": "nominal A-to-B span only; padded slices are excluded",
            "supplementary_connectivity": "junction-masked component evidence at fixed 0.20L and 0.80L collar slabs",
        },
        "counts": {
            **graph.counts,
            "metric_rows": len(rows),
            "valid_rois": int(sum(bool(row["roi_valid"]) for row in rows)),
            "invalid_rois": int(sum(not bool(row["roi_valid"]) for row in rows)),
        },
        "summary": {
            "median_corridor_foreground_fraction": float(
                np.median([row["corridor_foreground_fraction"] for row in rows])
            ),
            "median_edt_radius_voxels": float(
                np.median([row["edt_radius_median_voxels"] for row in rows])
            ),
            "roi_valid_fraction": roi_valid_fraction,
        },
        "gates": gates,
        "artifacts": {
            "corridor_calibration": {
                **calibration_artifact,
                "role": "corridor_calibration",
                "retention": "committed",
            },
            "per_strut_metrics": {
                **metrics_artifact,
                "role": "per_strut_metrics",
                "retention": "committed",
            },
            "per_strut_profiles": {
                **profiles_artifact,
                "role": "per_strut_profiles",
                "retention": "committed",
            },
        },
        "hashes": {
            "ct_sha256": ct_hash,
            "localized_graph_sha256": graph.source_sha256,
            "corridor_calibration_sha256": calibration_artifact["sha256"],
            "per_strut_metrics_sha256": metrics_artifact["sha256"],
            "per_strut_profiles_sha256": profiles_artifact["sha256"],
            **({"canonical_mask_sha256": mask_hash} if mask_hash else {}),
            **({"registration_qa_sha256": qa_hash} if qa_hash else {}),
        },
        "provenance": {
            "config_sha256": sha256_json(merged),
            "input_bindings": input_provenance or {},
            "defect_labels_accessed": False,
            "development_split_accessed": False,
            "sealed_split_accessed": False,
        },
        "warnings": (
            []
            if gate == "pass"
            else [
                f"{len(rows) - int(sum(bool(row['roi_valid']) for row in rows))} padded ROIs require review"
            ]
        ),
    }
    try:
        staged_report_artifact = write_json_atomic(
            report_write_path, report, overwrite=False
        )
    except Exception:
        if staging_directory is not None and staging_directory.exists():
            shutil.rmtree(staging_directory)
        raise
    report_artifact = (
        _final_artifact_record(staged_report_artifact, strict_paths["metrics_report"])
        if strict_paths is not None
        else staged_report_artifact
    )
    if strict_paths is not None:
        assert staging_directory is not None and strict_output_directory is not None
        try:
            published = _publish_staged_bundle(
                staging_directory, strict_output_directory, strict_paths
            )
        finally:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
        for artifact in (calibration_artifact, metrics_artifact, profiles_artifact):
            artifact["changed"] = published
        report_artifact["changed"] = published
        for role in (
            "corridor_calibration",
            "per_strut_metrics",
            "per_strut_profiles",
        ):
            report["artifacts"][role]["changed"] = published
    report["artifacts"]["metrics_report"] = {
        **report_artifact,
        "role": "metrics_report",
        "retention": "committed",
    }
    report["hashes"]["metrics_report_sha256"] = report_artifact["sha256"]
    # Release Windows-backed NumPy mappings before the MCP call returns so a
    # completed run does not keep Stage 1 artifacts locked for later hashing,
    # archival, or test cleanup.
    for array in (mask_array, volume.array):
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    return report


def read_metrics_csv(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != set(METRIC_FIELDS):
        raise ValueError(f"Metrics CSV does not match production schema: {source}")
    integer_fields = {
        "strut_id",
        "junction0_id",
        "junction1_id",
        "maximum_axial_gap_samples",
        "interior_component_count",
        "shared_component_voxel_count_in_corridor",
        "endpoint0_to_collar_component_voxel_count_in_corridor",
        "endpoint1_to_collar_component_voxel_count_in_corridor",
        "junction_masked_collar_shared_component_voxel_count_in_corridor",
    }
    boolean_fields = {
        "same_material_component_connects_a_to_b",
        "same_component_connects_collar_a_to_b",
        "both_endpoint_segments_observed",
        "roi_valid",
    }
    parsed: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in integer_fields:
                item[key] = int(value) if value else None
            elif key in boolean_fields:
                item[key] = value.lower() == "true"
            else:
                item[key] = float(value) if value else None
        parsed.append(item)
    return parsed
