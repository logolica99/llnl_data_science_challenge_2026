#!/usr/bin/env python3
"""Test whether one CT-material component directly connects two registered nodes.

This is deliberately a *connectivity primitive*, not a defect classifier.  It
does not inspect a strut's middle section or decide whether a strut is missing,
thin, bent, or broken.  For four representative interior edges, it implements
the first per-strut operation from LatticeAnalytics:

1. take two adjacent, registered CT-space node positions;
2. construct a rotated cuboid whose longitudinal axis is that edge;
3. resample the CT into the cuboid's common local coordinates;
4. threshold it with the challenge's frozen segmentation rule; and
5. restrict foreground to a scan-calibrated cylindrical corridor; and
6. require one connected-component label to intersect both endpoint windows.

The cuboid is normalized so its local z-axis follows node A -> node B.  This
makes the same test applicable to axis-aligned and diagonal octet struts.

Run from the repository root:

    python scripts/test_strut_node_connectivity.py

For the complete graph while retaining heavy artifacts only for failures:

    python scripts/test_strut_node_connectivity.py --all-struts --failures-only

The default graph was produced by the CT-only registration proof of concept.
Pass --graph to use a different registered graph, such as the supplied aligned
JSON, without changing the implementation.

Optional tube-label validation mode:

    python scripts/test_strut_node_connectivity.py \
        --strut-ids-file poc/tube_emptiness_test/results/deleted_struts_0.5.json

In that mode the tube-emptiness JSON selects a small spatially diverse subset
of intentionally deleted *design* strut IDs plus locally matched, non-labelled
controls.  The labels never change cuboid extraction, thresholding, or the
component-connectivity calculation; they are attached only to summarize the
already-computed measurements.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage


FROZEN_THRESHOLD = 40_129
DEFAULT_ENDPOINT_SEED_HALF_LENGTH_VOXELS = 2.0
DEFAULT_COLLAR_HALF_LENGTH_VOXELS = 1.0
DEFAULT_COLLAR_FRACTION = 0.20
DEFAULT_TRANSVERSE_MARGIN_FRACTION = 0.20
DEFAULT_TEST_COUNT = 4
DEFAULT_LABELLED_TEST_COUNT = 6
DEFAULT_CALIBRATION_COUNT = 24
DEFAULT_INTERPOLATION_BATCH_SIZE = 64
CALIBRATION_HALF_WIDTH_VOXELS = 12.0
CALIBRATION_CENTRAL_FRACTION = 0.60
CALIBRATION_MIN_VALID_SLICE_FRACTION = 0.75
CALIBRATION_RADIUS_QUANTILE = 0.90
CALIBRATION_RADIUS_SAFETY_VOXELS = 0.5
CALIBRATION_MAX_AXIS_DISTANCE_VOXELS = 8.0
ANALYSIS_CONFIG_SCHEMA_VERSION = 1
COMPONENT_STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)
GENERATED_RUN_FILENAMES = (
    "connection_summary.json",
    "connection_metrics.csv",
    "connection_overview.png",
    "README.md",
)


@dataclass(frozen=True)
class CuboidGeometry:
    """The local-to-CT affine geometry for one strut cuboid.

    All vectors are in CT ``xyz`` voxel coordinates.  The resampled array uses
    the conventional image order ``zyx``; this class preserves the distinction
    explicitly so the axis-order conversion remains auditable.
    """

    strut_id: int
    junction0_id: int
    junction1_id: int
    node_a_xyz: list[float]
    node_b_xyz: list[float]
    center_xyz: list[float]
    length_voxels: float
    axial_margin_voxels: float
    half_width_voxels: float
    corridor_radius_voxels: float
    transverse_margin_voxels: float
    basis_x_xyz: list[float]
    basis_y_xyz: list[float]
    basis_z_xyz: list[float]
    local_shape_zyx: list[int]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=root
        / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif",
        help="Input CT TIFF. It is memory-mapped; the whole volume is not loaded.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=root / "poc/ct_registration/test_results/our_registered.json",
        help="Registered graph JSON whose junction positions are in CT xyz voxels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs/strut_node_connectivity_test",
        help="Directory for reproducible test artifacts.",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        help=(
            "Frozen scan-level analysis configuration. Defaults to "
            "<output-dir>/analysis_config.json. An existing compatible file is reused."
        ),
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Replace the frozen corridor-radius calibration for this scan.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_TEST_COUNT,
        help="Number of spatially representative, unlabeled interior struts to test.",
    )
    parser.add_argument(
        "--strut-ids",
        type=int,
        nargs="+",
        help="Explicit graph strut IDs to test, in the supplied order.",
    )
    parser.add_argument(
        "--all-struts",
        action="store_true",
        help="Test every graph edge in deterministic JSON order.",
    )
    parser.add_argument(
        "--strut-id-range",
        type=int,
        nargs=2,
        metavar=("FIRST_ID", "LAST_ID"),
        help="Test one inclusive deterministic graph-ID range; useful for checkpointed full runs.",
    )
    parser.add_argument(
        "--strut-ids-file",
        type=Path,
        help=(
            "Optional tube-emptiness JSON containing deleted_strut_ids. It is "
            "used only to select validation cases and matched controls."
        ),
    )
    parser.add_argument(
        "--max-labeled",
        type=int,
        default=DEFAULT_LABELLED_TEST_COUNT,
        help=(
            "Maximum number of spatially diverse tube-labelled struts to test "
            "when --strut-ids-file is provided (default: 6)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INTERPOLATION_BATCH_SIZE,
        help=(
            "Number of strut cuboids concatenated into each SciPy interpolation "
            "call (default: 64). Connectivity labeling remains per strut."
        ),
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help=(
            "Keep all CSV/JSON measurements, but save cuboid NPZ files and plot "
            "profiles only for struts that are not connected."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=FROZEN_THRESHOLD,
        help=(
            "Frozen CT foreground threshold for this run. The value is recorded "
            "in and checked against the analysis configuration."
        ),
    )
    parser.add_argument(
        "--write-compact-profiles",
        action="store_true",
        help=(
            "Write one compressed all-strut axial-profile NPZ containing only "
            "strut IDs, lengths, and corridor-disk foreground profiles."
        ),
    )
    parser.add_argument(
        "--skip-cuboid-artifacts",
        action="store_true",
        help=(
            "Do not write individual full-cuboid NPZ artifacts. Intended with "
            "--write-compact-profiles for lightweight all-strut feature extraction."
        ),
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Do not render connection_overview.png; useful for all-strut feature runs.",
    )
    return parser.parse_args()


def load_registered_graph(path: Path) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Load junction positions and strut topology without reading any labels."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    nodes = {
        int(node["id"]): np.asarray(node["position"], dtype=np.float64)
        for node in document["junctions"]
    }
    struts = list(document["struts"])
    if not nodes or not struts:
        raise ValueError(f"{path} does not contain non-empty junctions and struts")
    return nodes, struts


def load_tube_label_ids(path: Path) -> set[int]:
    """Read deletion IDs produced by the STL-only tube-emptiness test.

    The IDs are intentionally kept out of all CT measurement functions below.
    This loader exists only to choose a validation cohort after the primitive is
    already defined.
    """
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    ids = document.get("deleted_strut_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"{path} has no non-empty deleted_strut_ids list")
    return {int(value) for value in ids}


def select_struts_by_ids(
    struts: list[dict[str, Any]], requested_ids: list[int]
) -> list[dict[str, Any]]:
    """Resolve explicit IDs without consulting CT evidence or defect labels."""
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("--strut-ids must not contain duplicates")
    by_id = {int(strut["id"]): strut for strut in struts}
    missing = [strut_id for strut_id in requested_ids if strut_id not in by_id]
    if missing:
        raise ValueError(f"Unknown strut IDs: {missing}")
    return [by_id[strut_id] for strut_id in requested_ids]


def stable_frame(node_a: np.ndarray, node_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a deterministic, right-handed local frame for the edge A -> B."""
    edge = node_b - node_a
    length = float(np.linalg.norm(edge))
    if length <= 0:
        raise ValueError("A strut has coincident endpoint positions")
    basis_z = edge / length

    # Use the world axis least parallel to the strut as the reference.  Project
    # it into the transverse plane; this avoids numerical instability near a
    # principal CT axis and gives a reproducible roll angle for every cuboid.
    world_axes = np.eye(3)
    reference = world_axes[int(np.argmin(np.abs(world_axes @ basis_z)))]
    basis_x = reference - np.dot(reference, basis_z) * basis_z
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(basis_z, basis_x)
    return basis_x, basis_y, basis_z, length


def select_representative_struts(
    nodes: dict[int, np.ndarray],
    struts: list[dict[str, Any]],
    volume_shape_zyx: tuple[int, int, int],
    count: int,
) -> list[dict[str, Any]]:
    """Choose interior edges by location only, never by CT evidence or labels.

    Four target locations in normalized graph space make the small test cover
    multiple portions of the part.  A 20-voxel endpoint boundary exclusion
    prevents interpolation outside the CT during the cuboid test.
    """
    volume_extent_xyz = np.asarray(
        [volume_shape_zyx[2], volume_shape_zyx[1], volume_shape_zyx[0]], dtype=np.float64
    )
    records: list[tuple[dict[str, Any], np.ndarray]] = []
    for strut in struts:
        a = nodes[int(strut["junction0"])]
        b = nodes[int(strut["junction1"])]
        if np.min(np.minimum(a, b)) < 20:
            continue
        if np.max(np.maximum(a, b) - volume_extent_xyz) > -20:
            continue
        records.append((strut, (a + b) / 2.0))
    if len(records) < count:
        raise ValueError("Not enough interior struts available for the requested test count")

    points = np.asarray([midpoint for _, midpoint in records])
    low = points.min(axis=0)
    high = points.max(axis=0)
    target_fractions = np.array(
        [
            [0.30, 0.30, 0.30],
            [0.70, 0.30, 0.70],
            [0.30, 0.70, 0.70],
            [0.70, 0.70, 0.30],
        ],
        dtype=np.float64,
    )
    targets = low + target_fractions * (high - low)

    chosen: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for target in targets[:count]:
        order = np.argsort(np.linalg.norm(points - target, axis=1))
        for index in order:
            candidate = records[int(index)][0]
            candidate_id = int(candidate["id"])
            if candidate_id not in used_ids:
                chosen.append(candidate)
                used_ids.add(candidate_id)
                break
    return chosen


def edge_midpoint_and_direction(
    strut: dict[str, Any], nodes: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered midpoint and orientation, with no CT sampling."""
    node_a = nodes[int(strut["junction0"])]
    node_b = nodes[int(strut["junction1"])]
    direction = node_b - node_a
    direction /= np.linalg.norm(direction)
    return (node_a + node_b) / 2.0, direction


def select_spatially_diverse(
    candidates: list[dict[str, Any]], nodes: dict[int, np.ndarray], count: int
) -> list[dict[str, Any]]:
    """Use deterministic farthest-point sampling on graph midpoints only."""
    if len(candidates) < count:
        raise ValueError("Not enough labelled interior struts for the requested validation subset")
    points = np.asarray([edge_midpoint_and_direction(strut, nodes)[0] for strut in candidates])
    first = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen_indices = [first]
    nearest_distance = np.linalg.norm(points - points[first], axis=1)
    while len(chosen_indices) < count:
        next_index = int(np.argmax(nearest_distance))
        chosen_indices.append(next_index)
        nearest_distance = np.minimum(nearest_distance, np.linalg.norm(points - points[next_index], axis=1))
    return [candidates[index] for index in chosen_indices]


def select_matched_controls(
    labelled: list[dict[str, Any]],
    struts: list[dict[str, Any]],
    nodes: dict[int, np.ndarray],
    labelled_ids: set[int],
) -> list[dict[str, Any]]:
    """Match each labelled edge to a nearby, similarly oriented non-labelled edge.

    This is a cohort-selection step only.  It does not use CT intensity or any
    connectivity result, avoiding circular selection of apparently good or bad
    controls.
    """
    available = [strut for strut in struts if int(strut["id"]) not in labelled_ids]
    used_ids: set[int] = set()
    controls: list[dict[str, Any]] = []
    for labelled_strut in labelled:
        labelled_midpoint, labelled_direction = edge_midpoint_and_direction(labelled_strut, nodes)
        best: dict[str, Any] | None = None
        best_score = float("inf")
        for candidate in available:
            candidate_id = int(candidate["id"])
            if candidate_id in used_ids:
                continue
            midpoint, direction = edge_midpoint_and_direction(candidate, nodes)
            distance = float(np.linalg.norm(midpoint - labelled_midpoint))
            orientation_difference = 1.0 - abs(float(np.dot(direction, labelled_direction)))
            score = distance + 40.0 * orientation_difference
            if score < best_score:
                best_score = score
                best = candidate
        if best is None:
            raise ValueError("Could not find a unique non-labelled matched control")
        controls.append(best)
        used_ids.add(int(best["id"]))
    return controls


def select_labelled_validation_struts(
    nodes: dict[int, np.ndarray],
    struts: list[dict[str, Any]],
    volume_shape_zyx: tuple[int, int, int],
    labelled_ids: set[int],
    count: int,
) -> list[tuple[dict[str, Any], str]]:
    """Return labelled test edges and matched controls, both safely interior."""
    volume_extent_xyz = np.asarray(
        [volume_shape_zyx[2], volume_shape_zyx[1], volume_shape_zyx[0]], dtype=np.float64
    )
    interior = []
    for strut in struts:
        node_a = nodes[int(strut["junction0"])]
        node_b = nodes[int(strut["junction1"])]
        if np.min(np.minimum(node_a, node_b)) < 20:
            continue
        if np.max(np.maximum(node_a, node_b) - volume_extent_xyz) > -20:
            continue
        interior.append(strut)
    labelled_candidates = [strut for strut in interior if int(strut["id"]) in labelled_ids]
    labelled = select_spatially_diverse(labelled_candidates, nodes, count)
    controls = select_matched_controls(labelled, interior, nodes, labelled_ids)
    return [(strut, "tube_labelled_deleted_design") for strut in labelled] + [
        (strut, "matched_nonlabelled_control") for strut in controls
    ]


def prepare_cuboid_sampling(
    strut: dict[str, Any],
    nodes: dict[int, np.ndarray],
    half_width: float,
    corridor_radius: float,
) -> tuple[np.ndarray, CuboidGeometry, np.ndarray, np.ndarray, np.ndarray]:
    """Build one cuboid's flattened CT ``zyx`` sampling coordinates.

    Coordinates are constructed directly as three flat arrays. This avoids the
    previous ``(..., 3)`` world-coordinate array while preserving the exact
    centered affine mapping ``p = c + x*e_x + y*e_y + z*e_z``.
    """
    junction0_id = int(strut["junction0"])
    junction1_id = int(strut["junction1"])
    node_a = nodes[junction0_id]
    node_b = nodes[junction1_id]
    basis_x, basis_y, basis_z, length = stable_frame(node_a, node_b)
    center = (node_a + node_b) / 2.0
    transverse_margin = max(0.0, half_width - corridor_radius)

    local_x = np.arange(-half_width, half_width + 1.0, dtype=np.float64)
    local_y = np.arange(-half_width, half_width + 1.0, dtype=np.float64)
    # The paper's normalized cuboid places A in the first slice and B in the
    # last.  Linspace preserves both endpoints exactly while keeping spacing at
    # approximately one CT voxel for non-integral registered edge lengths.
    local_z = np.linspace(0.0, length, int(np.ceil(length)) + 1, dtype=np.float64)
    shape = (local_z.size, local_y.size, local_x.size)
    plane_size = local_y.size * local_x.size
    zz = np.repeat(local_z, plane_size)
    yy = np.tile(np.repeat(local_y, local_x.size), local_z.size)
    xx = np.tile(local_x, local_z.size * local_y.size)

    # Centered local-to-world mapping requested by the pipeline design:
    # p = c + x*e_x + y*e_y + z_centered*e_z.  The stored local z remains the
    # more interpretable distance from A in [0, length].
    z_centered = zz - 0.5 * length
    sample_coordinates_zyx = np.empty((3, xx.size), dtype=np.float64)
    for coordinate_index, world_axis in enumerate((2, 1, 0)):
        sample_coordinates_zyx[coordinate_index] = (
            center[world_axis]
            + xx * basis_x[world_axis]
            + yy * basis_y[world_axis]
            + z_centered * basis_z[world_axis]
        )

    geometry = CuboidGeometry(
        strut_id=int(strut["id"]),
        junction0_id=junction0_id,
        junction1_id=junction1_id,
        node_a_xyz=node_a.tolist(),
        node_b_xyz=node_b.tolist(),
        center_xyz=center.tolist(),
        length_voxels=length,
        axial_margin_voxels=0.0,
        half_width_voxels=half_width,
        corridor_radius_voxels=corridor_radius,
        transverse_margin_voxels=transverse_margin,
        basis_x_xyz=basis_x.tolist(),
        basis_y_xyz=basis_y.tolist(),
        basis_z_xyz=basis_z.tolist(),
        local_shape_zyx=list(shape),
    )
    return sample_coordinates_zyx, geometry, local_x, local_y, local_z


def build_cuboids_batch(
    volume: np.ndarray,
    struts: list[dict[str, Any]],
    nodes: dict[int, np.ndarray],
    half_width: float,
    corridor_radius: float,
) -> list[tuple[np.ndarray, CuboidGeometry, np.ndarray, np.ndarray, np.ndarray]]:
    """Resample multiple independently rotated cuboids in one SciPy call."""
    if not struts:
        return []
    prepared = [
        prepare_cuboid_sampling(strut, nodes, half_width, corridor_radius)
        for strut in struts
    ]
    sample_counts = np.asarray([item[0].shape[1] for item in prepared], dtype=np.int64)
    offsets = np.empty(len(prepared) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sample_counts, out=offsets[1:])
    coordinates = np.empty((3, int(offsets[-1])), dtype=np.float64)
    for item, start, stop in zip(prepared, offsets[:-1], offsets[1:], strict=True):
        coordinates[:, start:stop] = item[0]

    sampled_values = ndimage.map_coordinates(
        volume,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
        output=np.float32,
    )
    results = []
    for item, start, stop in zip(prepared, offsets[:-1], offsets[1:], strict=True):
        _, geometry, local_x, local_y, local_z = item
        sampled = sampled_values[start:stop].reshape(geometry.local_shape_zyx)
        results.append((sampled, geometry, local_x, local_y, local_z))
    return results


def build_cuboid(
    volume: np.ndarray,
    strut: dict[str, Any],
    nodes: dict[int, np.ndarray],
    half_width: float,
    corridor_radius: float,
) -> tuple[np.ndarray, CuboidGeometry, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility wrapper for callers that request one cuboid."""
    return build_cuboids_batch(
        volume, [strut], nodes, half_width, corridor_radius
    )[0]


def disk_mask(local_x: np.ndarray, local_y: np.ndarray, radius: float) -> np.ndarray:
    """Return the in-plane circular corridor used for all axial measurements."""
    yy, xx = np.meshgrid(local_y, local_x, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def file_sha256(path: Path) -> str:
    """Return a content hash for small configuration inputs such as the graph."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interior_struts(
    nodes: dict[int, np.ndarray],
    struts: list[dict[str, Any]],
    volume_shape_zyx: tuple[int, int, int],
    boundary_voxels: float,
) -> list[dict[str, Any]]:
    """Return graph edges whose endpoint-centered calibration windows are in bounds."""
    extent_xyz = np.asarray(
        [volume_shape_zyx[2], volume_shape_zyx[1], volume_shape_zyx[0]], dtype=np.float64
    )
    candidates: list[dict[str, Any]] = []
    for strut in struts:
        node_a = nodes[int(strut["junction0"])]
        node_b = nodes[int(strut["junction1"])]
        if np.min(np.minimum(node_a, node_b)) < boundary_voxels:
            continue
        if np.max(np.maximum(node_a, node_b) - extent_xyz) > -boundary_voxels:
            continue
        candidates.append(strut)
    return candidates


def calibrate_corridor_radius(
    volume: np.ndarray,
    nodes: dict[int, np.ndarray],
    struts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Estimate one label-blind corridor radius from high-continuity graph edges.

    Calibration uses only registered geometry and thresholded CT intensities. It
    samples spatially diverse interior edges, ignores their junction regions,
    and retains edges whose central cross-sections consistently contain a
    foreground component near the registered axis. No deletion or defect label
    is read. The robust radial extent includes both material radius and typical
    centerline registration offset.
    """
    candidates = interior_struts(
        nodes,
        struts,
        volume.shape,
        boundary_voxels=CALIBRATION_HALF_WIDTH_VOXELS + 2.0,
    )
    sample_count = min(DEFAULT_CALIBRATION_COUNT, len(candidates))
    if sample_count < 4:
        raise ValueError("At least four interior struts are required for radius calibration")
    selected = select_spatially_diverse(candidates, nodes, sample_count)

    per_strut: list[dict[str, Any]] = []
    local_structure = ndimage.generate_binary_structure(2, 2)
    calibration_cuboids = build_cuboids_batch(
        volume,
        selected,
        nodes,
        half_width=CALIBRATION_HALF_WIDTH_VOXELS,
        corridor_radius=0.0,
    )
    for strut, cuboid in zip(selected, calibration_cuboids, strict=True):
        sampled, geometry, local_x, local_y, local_z = cuboid
        binary = sampled >= FROZEN_THRESHOLD
        edge_fraction = (1.0 - CALIBRATION_CENTRAL_FRACTION) / 2.0
        central = np.flatnonzero(
            (local_z >= edge_fraction * geometry.length_voxels)
            & (local_z <= (1.0 - edge_fraction) * geometry.length_voxels)
        )
        radial_extents: list[float] = []
        equivalent_radii: list[float] = []
        centroid_offsets: list[float] = []
        for z_index in central:
            labels_yx, component_count = ndimage.label(binary[z_index], structure=local_structure)
            component_candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
            for component_id in range(1, component_count + 1):
                iy, ix = np.nonzero(labels_yx == component_id)
                area = int(ix.size)
                if area < 5:
                    continue
                x_values = local_x[ix]
                y_values = local_y[iy]
                distances = np.hypot(x_values, y_values)
                component_candidates.append((float(distances.min()), -area, x_values, y_values))
            if not component_candidates:
                continue
            minimum_distance, _, x_values, y_values = min(
                component_candidates, key=lambda item: (item[0], item[1])
            )
            if minimum_distance > CALIBRATION_MAX_AXIS_DISTANCE_VOXELS:
                continue
            if (
                np.any(np.abs(x_values) >= CALIBRATION_HALF_WIDTH_VOXELS)
                or np.any(np.abs(y_values) >= CALIBRATION_HALF_WIDTH_VOXELS)
            ):
                continue
            radial_extents.append(float(np.max(np.hypot(x_values, y_values))))
            equivalent_radii.append(float(np.sqrt(x_values.size / np.pi)))
            centroid_offsets.append(float(np.hypot(x_values.mean(), y_values.mean())))

        valid_fraction = len(radial_extents) / max(1, central.size)
        record = {
            "strut_id": int(strut["id"]),
            "central_slice_count": int(central.size),
            "valid_slice_count": len(radial_extents),
            "valid_slice_fraction": float(valid_fraction),
        }
        if valid_fraction >= CALIBRATION_MIN_VALID_SLICE_FRACTION:
            record.update(
                {
                    "accepted": True,
                    "median_radial_extent_voxels": float(np.median(radial_extents)),
                    "median_equivalent_radius_voxels": float(np.median(equivalent_radii)),
                    "median_centroid_offset_voxels": float(np.median(centroid_offsets)),
                }
            )
        else:
            record["accepted"] = False
        per_strut.append(record)

    accepted = [record for record in per_strut if record["accepted"]]
    if len(accepted) < max(4, sample_count // 4):
        raise ValueError(
            f"Radius calibration accepted only {len(accepted)} of {sample_count} sampled struts"
        )
    radial_extents = np.asarray(
        [record["median_radial_extent_voxels"] for record in accepted], dtype=np.float64
    )
    quantile_extent = float(np.quantile(radial_extents, CALIBRATION_RADIUS_QUANTILE))
    radius = float(np.ceil(quantile_extent + CALIBRATION_RADIUS_SAFETY_VOXELS))
    if radius >= CALIBRATION_HALF_WIDTH_VOXELS:
        raise ValueError(
            "Calibrated radius reaches the calibration window boundary; increase the search window"
        )
    return {
        "method": "central-slice nearest-component radial-extent bootstrap",
        "label_blind": True,
        "threshold": FROZEN_THRESHOLD,
        "sample_count_requested": DEFAULT_CALIBRATION_COUNT,
        "sample_count_used": sample_count,
        "accepted_strut_count": len(accepted),
        "central_fraction": CALIBRATION_CENTRAL_FRACTION,
        "minimum_valid_slice_fraction": CALIBRATION_MIN_VALID_SLICE_FRACTION,
        "radial_extent_quantile": CALIBRATION_RADIUS_QUANTILE,
        "radius_safety_voxels": CALIBRATION_RADIUS_SAFETY_VOXELS,
        "calibration_half_width_voxels": CALIBRATION_HALF_WIDTH_VOXELS,
        "quantile_radial_extent_voxels": quantile_extent,
        "corridor_radius_voxels": radius,
        "per_strut": per_strut,
    }


def load_or_create_analysis_config(
    path: Path,
    volume: np.ndarray,
    ct_path: Path,
    graph_path: Path,
    nodes: dict[int, np.ndarray],
    struts: list[dict[str, Any]],
    recalibrate: bool,
) -> dict[str, Any]:
    """Reuse a compatible frozen calibration or create it once for this scan."""
    identity = {
        "ct_path": str(ct_path.resolve()),
        "ct_file_size_bytes": ct_path.stat().st_size,
        "ct_shape_zyx": list(volume.shape),
        "ct_dtype": str(volume.dtype),
        "registered_graph_path": str(graph_path.resolve()),
        "registered_graph_sha256": file_sha256(graph_path),
        "junction_count": len(nodes),
        "strut_count": len(struts),
        "frozen_threshold": FROZEN_THRESHOLD,
    }
    if path.is_file() and not recalibrate:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("schema_version") != ANALYSIS_CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported analysis-config schema in {path}; use --recalibrate")
        if config.get("input_identity") != identity:
            raise ValueError(f"Analysis config {path} belongs to different inputs; use --recalibrate")
        return config

    calibration = calibrate_corridor_radius(volume, nodes, struts)
    radius = float(calibration["corridor_radius_voxels"])
    transverse_margin = max(2.0, DEFAULT_TRANSVERSE_MARGIN_FRACTION * radius)
    config = {
        "schema_version": ANALYSIS_CONFIG_SCHEMA_VERSION,
        "input_identity": identity,
        "corridor_calibration": calibration,
        "normalized_cuboid": {
            "node_a_first_slice": True,
            "node_b_last_slice": True,
            "axial_margin_voxels": 0.0,
            "transverse_margin_fraction": DEFAULT_TRANSVERSE_MARGIN_FRACTION,
            "transverse_margin_voxels": transverse_margin,
            "half_width_voxels": float(np.ceil(radius + transverse_margin)),
            "interpolation": "trilinear (scipy.ndimage.map_coordinates order=1)",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def clear_previous_run_artifacts(output_dir: Path) -> None:
    """Remove only files owned by this script, preserving the frozen config."""
    for filename in GENERATED_RUN_FILENAMES:
        path = output_dir / filename
        if path.is_file():
            path.unlink()
    for path in output_dir.glob("strut_*_cuboid.npz"):
        if path.is_file():
            path.unlink()


def axial_window_indices(local_z: np.ndarray, center: float, half_length: float) -> np.ndarray:
    indices = np.flatnonzero(np.abs(local_z - center) <= half_length)
    if indices.size == 0:
        raise ValueError("Requested axial measurement window contains no resampled slice")
    return indices


def endpoint_connection_measurement(
    binary_zyx: np.ndarray,
    labels_zyx: np.ndarray,
    local_z: np.ndarray,
    disk_yx: np.ndarray,
    endpoint_z: float,
    collar_z: float,
    component_sizes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure whether one node-seeded component reaches its local strut collar.

    A node seed is a short axial disk centered on the registered node.  The
    collar is another disk 20% of the nominal edge length away from that node.
    The result is true only if at least one *foreground connected component*
    intersects both windows.  Thus it is a local node-to-strut connectivity
    observation rather than a whole-strut health score.
    """
    seed_indices = axial_window_indices(
        local_z, endpoint_z, DEFAULT_ENDPOINT_SEED_HALF_LENGTH_VOXELS
    )
    collar_indices = axial_window_indices(
        local_z, collar_z, DEFAULT_COLLAR_HALF_LENGTH_VOXELS
    )
    seed_values = binary_zyx[seed_indices][:, disk_yx]
    collar_values = binary_zyx[collar_indices][:, disk_yx]

    seed_labels = np.unique(labels_zyx[seed_indices][:, disk_yx])
    collar_labels = np.unique(labels_zyx[collar_indices][:, disk_yx])
    seed_labels = seed_labels[seed_labels > 0]
    collar_labels = collar_labels[collar_labels > 0]
    shared_labels = np.intersect1d(seed_labels, collar_labels)

    if component_sizes is None:
        component_sizes = np.bincount(labels_zyx.ravel())
    component_voxels = (
        int(component_sizes[shared_labels].sum()) if shared_labels.size else 0
    )
    return {
        "endpoint_seed_z_voxels": float(endpoint_z),
        "collar_z_voxels": float(collar_z),
        "seed_foreground_fraction": float(seed_values.mean()),
        "collar_foreground_fraction": float(collar_values.mean()),
        "seed_component_count": int(seed_labels.size),
        "collar_component_count": int(collar_labels.size),
        "shared_component_count": int(shared_labels.size),
        "shared_component_voxel_count_in_cuboid": component_voxels,
        "node_to_collar_component_observed": bool(shared_labels.size > 0),
    }


def shared_component_between_windows(
    labels_zyx: np.ndarray,
    local_z: np.ndarray,
    disk_yx: np.ndarray,
    first_z: float,
    second_z: float,
    half_length: float,
    component_sizes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Require an identical corridor-local component label in two axial windows."""
    first_indices = axial_window_indices(local_z, first_z, half_length)
    second_indices = axial_window_indices(local_z, second_z, half_length)
    first_labels = np.unique(labels_zyx[first_indices][:, disk_yx])
    second_labels = np.unique(labels_zyx[second_indices][:, disk_yx])
    first_labels = first_labels[first_labels > 0]
    second_labels = second_labels[second_labels > 0]
    shared_labels = np.intersect1d(first_labels, second_labels)
    if component_sizes is None:
        component_sizes = np.bincount(labels_zyx.ravel())
    return {
        "first_window_z_voxels": float(first_z),
        "second_window_z_voxels": float(second_z),
        "window_half_length_voxels": float(half_length),
        "first_window_component_count": int(first_labels.size),
        "second_window_component_count": int(second_labels.size),
        "shared_component_labels": shared_labels.astype(int).tolist(),
        "shared_component_count": int(shared_labels.size),
        "shared_component_voxel_count_in_corridor": (
            int(component_sizes[shared_labels].sum()) if shared_labels.size else 0
        ),
        "same_component_observed": bool(shared_labels.size > 0),
    }


def analyze_strut(
    volume: np.ndarray,
    strut: dict[str, Any],
    nodes: dict[int, np.ndarray],
    corridor_radius: float,
    half_width: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray], CuboidGeometry]:
    """Determine whether one corridor-local material component connects A to B."""
    sampled, geometry, local_x, local_y, local_z = build_cuboid(
        volume, strut, nodes, half_width, corridor_radius
    )
    return analyze_sampled_cuboid(
        sampled, geometry, local_x, local_y, local_z, corridor_radius
    )


def analyze_sampled_cuboid(
    sampled: np.ndarray,
    geometry: CuboidGeometry,
    local_x: np.ndarray,
    local_y: np.ndarray,
    local_z: np.ndarray,
    corridor_radius: float,
    disk: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], CuboidGeometry]:
    """Label and measure one already-resampled cuboid independently."""
    binary = sampled >= FROZEN_THRESHOLD
    if disk is None:
        disk = disk_mask(local_x, local_y, corridor_radius)
    corridor_binary = binary & disk[None, :, :]
    labels, component_count = ndimage.label(corridor_binary, structure=COMPONENT_STRUCTURE_26)
    component_sizes = np.bincount(labels.ravel())
    axial_occupancy = binary[:, disk].mean(axis=1)

    collar_from_a = DEFAULT_COLLAR_FRACTION * geometry.length_voxels
    collar_from_b = (1.0 - DEFAULT_COLLAR_FRACTION) * geometry.length_voxels
    endpoint_a = endpoint_connection_measurement(
        binary,
        labels,
        local_z,
        disk,
        endpoint_z=0.0,
        collar_z=collar_from_a,
        component_sizes=component_sizes,
    )
    endpoint_b = endpoint_connection_measurement(
        binary,
        labels,
        local_z,
        disk,
        endpoint_z=geometry.length_voxels,
        collar_z=collar_from_b,
        component_sizes=component_sizes,
    )
    node_a_to_b = shared_component_between_windows(
        labels,
        local_z,
        disk,
        first_z=0.0,
        second_z=geometry.length_voxels,
        half_length=DEFAULT_ENDPOINT_SEED_HALF_LENGTH_VOXELS,
        component_sizes=component_sizes,
    )
    collar_a_to_b = shared_component_between_windows(
        labels,
        local_z,
        disk,
        first_z=collar_from_a,
        second_z=collar_from_b,
        half_length=DEFAULT_COLLAR_HALF_LENGTH_VOXELS,
        component_sizes=component_sizes,
    )

    result = {
        "geometry": asdict(geometry),
        "ct_threshold": FROZEN_THRESHOLD,
        "cuboid_foreground_fraction": float(binary.mean()),
        "cuboid_component_count": int(component_count),
        "axial_occupancy": {
            "minimum": float(axial_occupancy.min()),
            "median": float(np.median(axial_occupancy)),
            "maximum": float(axial_occupancy.max()),
        },
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "node_a_to_node_b": node_a_to_b,
        "collar_a_to_collar_b": collar_a_to_b,
        "same_material_component_connects_a_to_b": node_a_to_b["same_component_observed"],
        "both_node_to_collar_components_observed": bool(
            endpoint_a["node_to_collar_component_observed"]
            and endpoint_b["node_to_collar_component_observed"]
        ),
        "scope_note": (
            "The primary result requires one identical component label at A and B "
            "inside the calibrated corridor. It is not a defect classification."
        ),
    }
    arrays = {
        "sampled_intensity_zyx": sampled.astype(np.float32),
        "foreground_mask_zyx": binary,
        "corridor_foreground_mask_zyx": corridor_binary,
        "component_labels_zyx": labels.astype(np.int32),
        "corridor_disk_yx": disk,
        "local_x_voxels": local_x,
        "local_y_voxels": local_y,
        "local_z_from_node_a_voxels": local_z,
        "axial_disk_foreground_fraction": axial_occupancy,
    }
    return result, arrays, geometry


def analyze_struts_batch(
    volume: np.ndarray,
    struts: list[dict[str, Any]],
    nodes: dict[int, np.ndarray],
    corridor_radius: float,
    half_width: float,
) -> list[tuple[dict[str, Any], dict[str, np.ndarray], CuboidGeometry]]:
    """Interpolate a batch once, then preserve per-strut component analysis."""
    cuboids = build_cuboids_batch(volume, struts, nodes, half_width, corridor_radius)
    if not cuboids:
        return []
    shared_disk = disk_mask(cuboids[0][2], cuboids[0][3], corridor_radius)
    return [
        analyze_sampled_cuboid(
            sampled,
            geometry,
            local_x,
            local_y,
            local_z,
            corridor_radius,
            disk=shared_disk,
        )
        for sampled, geometry, local_x, local_y, local_z in cuboids
    ]


def write_overview(
    results: list[dict[str, Any]],
    output_path: Path,
    compact_profiles: dict[int, np.ndarray] | None = None,
) -> None:
    """Render a compact evidence plot; it contains measurements, not labels."""
    if not results:
        figure, axis = plt.subplots(1, 1, figsize=(10, 2.5))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "No not-connected struts were observed.",
            ha="center",
            va="center",
            fontsize=13,
        )
        figure.tight_layout()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        return
    figure, axes = plt.subplots(len(results), 1, figsize=(10, 2.5 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]
    for axis, result in zip(axes, results, strict=True):
        geometry = result["geometry"]
        length = geometry["length_voxels"]
        if "profile_path" in result:
            profile_path = result["profile_path"]
            with np.load(profile_path) as profile_data:
                profile = profile_data["axial_disk_foreground_fraction"]
                z = profile_data["local_z_from_node_a_voxels"]
        elif compact_profiles is not None:
            profile = compact_profiles[int(geometry["strut_id"])]
            z = np.linspace(0.0, length, len(profile), dtype=np.float64)
        else:
            raise ValueError("Overview requires saved individual or compact axial profiles")
        axis.plot(z, profile, color="#2a6fbb", linewidth=1.5, label="disk foreground fraction")
        axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1, label="node A")
        axis.axvline(length, color="#444444", linestyle="--", linewidth=1, label="node B")
        axis.axvline(DEFAULT_COLLAR_FRACTION * length, color="#d95f02", linestyle=":", linewidth=1.5)
        axis.axvline((1.0 - DEFAULT_COLLAR_FRACTION) * length, color="#d95f02", linestyle=":", linewidth=1.5)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("foreground\nfraction")
        status = "CONNECTED" if result["same_material_component_connects_a_to_b"] else "NOT CONNECTED"
        axis.set_title(
            f"Strut {geometry['strut_id']}: one corridor component A to B = {status}"
        )
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("local z distance from node A (CT voxels)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_output_readme(output_dir: Path) -> None:
    """Document every generated artifact next to the test output."""
    text = """# Strut-node connectivity test artifacts

This directory contains a limited, reproducible test of the first
LatticeAnalytics-style per-strut operation: extracting a rotated cuboid between
two registered CT-space nodes and requiring one identical foreground component
to intersect both endpoint windows inside a calibrated cylindrical corridor.

It is **not** a defect detector. It does not inspect the middle of a strut or
infer missing/thin/bent/broken classes. When launched with `--strut-ids-file`,
the tube-emptiness labels select validation cohorts only; they never alter a
cuboid measurement or connectivity decision.

## Files

- `analysis_config.json` - the frozen, label-blind scan-level corridor-radius
  calibration. Compatible reruns reuse it unless `--recalibrate` is supplied.
- `connection_summary.json` - all inputs, geometry, derived statistics, and
  direct A-to-B connectivity observations in one machine-readable document.
- `connection_metrics.csv` - one row per tested strut endpoint, convenient for
  spreadsheet inspection.
- `all_strut_axial_profiles.npz` - written with `--write-compact-profiles`.
  It stores only each strut ID, A-to-B length, and axial corridor-disk
  foreground profile; it is suitable for downstream material-loss features and
  is not a full-cuboid artifact.
- `connection_overview.png` - axial foreground-fraction profiles. Dashed lines
  are registered nodes; orange dotted lines are the 20%-length collar samples.
  With `--failures-only`, it contains only not-connected cases (or a clear
  no-failures message).
- `strut_<id>_cuboid.npz` - one file per tested strut containing the resampled
  intensity cuboid, thresholded foreground mask, corridor-restricted mask,
  corridor-local component labels, local coordinates, and axial profile.
  With `--failures-only`, these files are written only for not-connected cases.

The CT was sampled as `volume[z, y, x]` while the graph coordinates are stored
as `[x, y, z]`. The test explicitly performs this conversion before sampling.
Node A is the first normalized z slice and node B is the last. The cuboid keeps
the calibrated corridor plus a transverse margin of surrounding space.
Multiple independently rotated cuboids are interpolated in configurable
batches; thresholding and connected-component labeling remain independent for
every strut.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def cohort_statistics(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Summarize groups after measurement; no statistic feeds back into testing."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["cohort"], []).append(result)
    summary: dict[str, dict[str, float | int]] = {}
    for cohort, members in grouped.items():
        endpoints = [member[endpoint] for member in members for endpoint in ("endpoint_a", "endpoint_b")]
        collar_values = [endpoint["collar_foreground_fraction"] for endpoint in endpoints]
        connected = [endpoint["node_to_collar_component_observed"] for endpoint in endpoints]
        connected_a_to_b = [member["same_material_component_connects_a_to_b"] for member in members]
        summary[cohort] = {
            "strut_count": len(members),
            "endpoint_count": len(endpoints),
            "node_to_collar_connections_observed": int(sum(connected)),
            "node_to_collar_connection_fraction": float(np.mean(connected)),
            "same_component_a_to_b_count": int(sum(connected_a_to_b)),
            "same_component_a_to_b_fraction": float(np.mean(connected_a_to_b)),
            "mean_collar_foreground_fraction": float(np.mean(collar_values)),
            "minimum_collar_foreground_fraction": float(np.min(collar_values)),
            "maximum_collar_foreground_fraction": float(np.max(collar_values)),
        }
    return summary


def main() -> None:
    global FROZEN_THRESHOLD
    args = parse_args()
    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    FROZEN_THRESHOLD = int(args.threshold)
    selection_mode_count = sum(
        (
            bool(args.strut_ids),
            bool(args.strut_ids_file),
            args.all_struts,
            args.strut_id_range is not None,
        )
    )
    if selection_mode_count > 1:
        raise ValueError(
            "--strut-ids, --strut-ids-file, --strut-id-range, and --all-struts are mutually exclusive"
        )
    if selection_mode_count == 0 and (args.count < 1 or args.count > 4):
        raise ValueError("--count must be from 1 to 4 for the fixed representative selection")
    if args.max_labeled < 1:
        raise ValueError("--max-labeled must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.strut_id_range and args.strut_id_range[0] > args.strut_id_range[1]:
        raise ValueError("--strut-id-range requires FIRST_ID <= LAST_ID")
    if args.skip_cuboid_artifacts and not args.write_compact_profiles:
        raise ValueError("--skip-cuboid-artifacts requires --write-compact-profiles")
    if not args.ct.is_file():
        raise FileNotFoundError(f"CT TIFF not found: {args.ct}")
    if not args.graph.is_file():
        raise FileNotFoundError(f"Registered graph not found: {args.graph}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    volume = tifffile.memmap(args.ct)
    nodes, struts = load_registered_graph(args.graph)
    analysis_config_path = args.analysis_config or (args.output_dir / "analysis_config.json")
    analysis_config = load_or_create_analysis_config(
        analysis_config_path,
        volume,
        args.ct,
        args.graph,
        nodes,
        struts,
        args.recalibrate,
    )
    corridor_radius = float(
        analysis_config["corridor_calibration"]["corridor_radius_voxels"]
    )
    half_width = float(analysis_config["normalized_cuboid"]["half_width_voxels"])
    clear_previous_run_artifacts(args.output_dir)
    if args.strut_ids:
        selected_with_cohorts = [
            (strut, "explicit_id_validation")
            for strut in select_struts_by_ids(struts, args.strut_ids)
        ]
    elif args.all_struts:
        selected_with_cohorts = [(strut, "all_graph_edges") for strut in struts]
    elif args.strut_id_range:
        first_id, last_id = args.strut_id_range
        selected_with_cohorts = [
            (strut, "all_graph_edges_checkpoint")
            for strut in select_struts_by_ids(struts, list(range(first_id, last_id + 1)))
        ]
    elif args.strut_ids_file:
        if not args.strut_ids_file.is_file():
            raise FileNotFoundError(f"Tube-emptiness label JSON not found: {args.strut_ids_file}")
        labelled_ids = load_tube_label_ids(args.strut_ids_file)
        selected_with_cohorts = select_labelled_validation_struts(
            nodes, struts, volume.shape, labelled_ids, args.max_labeled
        )
    else:
        selected_with_cohorts = [
            (strut, "unlabelled_spatial_coverage")
            for strut in select_representative_struts(nodes, struts, volume.shape, args.count)
        ]

    if args.strut_ids:
        selection_method = (
            "explicit graph strut IDs supplied by the user; CT measurements are label-blind"
        )
    elif args.all_struts:
        selection_method = (
            "all graph struts in deterministic JSON order; CT measurements are label-blind"
        )
    elif args.strut_id_range:
        selection_method = (
            f"inclusive deterministic graph-ID checkpoint {args.strut_id_range[0]} "
            f"through {args.strut_id_range[1]}; CT measurements are label-blind"
        )
    elif args.strut_ids_file:
        selection_method = (
            "tube-labelled validation: labelled IDs select the cohort and matched controls; "
            "CT measurements are label-blind"
        )
    else:
        selection_method = (
            "four interior struts selected by spatial coverage only; "
            "no CT evidence or labels used"
        )

    results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    compact_profile_ids: list[int] = []
    compact_profile_lengths: list[float] = []
    compact_profile_values: list[np.ndarray] = []
    for batch_start in range(0, len(selected_with_cohorts), args.batch_size):
        batch = selected_with_cohorts[batch_start : batch_start + args.batch_size]
        batch_analyses = analyze_struts_batch(
            volume,
            [strut for strut, _ in batch],
            nodes,
            corridor_radius,
            half_width,
        )
        for (_, cohort), (result, arrays, _) in zip(batch, batch_analyses, strict=True):
            result["cohort"] = cohort
            strut_id = result["geometry"]["strut_id"]
            if args.write_compact_profiles:
                compact_profile_ids.append(int(strut_id))
                compact_profile_lengths.append(float(result["geometry"]["length_voxels"]))
                compact_profile_values.append(
                    arrays["axial_disk_foreground_fraction"].astype(np.float32, copy=True)
                )
            should_save_profile = (
                not args.skip_cuboid_artifacts
                and (
                    not args.failures_only
                    or not result["same_material_component_connects_a_to_b"]
                )
            )
            if should_save_profile:
                cuboid_path = args.output_dir / f"strut_{strut_id}_cuboid.npz"
                np.savez_compressed(cuboid_path, **arrays)
                result["profile_path"] = str(cuboid_path)
            results.append(result)

            for endpoint_name, endpoint in (
                ("A", result["endpoint_a"]),
                ("B", result["endpoint_b"]),
            ):
                csv_rows.append(
                    {
                        "strut_id": strut_id,
                        "cohort": cohort,
                        "endpoint": endpoint_name,
                        "junction_id": result["geometry"]["junction0_id"]
                        if endpoint_name == "A"
                        else result["geometry"]["junction1_id"],
                        "length_voxels": result["geometry"]["length_voxels"],
                        "seed_foreground_fraction": endpoint["seed_foreground_fraction"],
                        "collar_foreground_fraction": endpoint["collar_foreground_fraction"],
                        "seed_component_count": endpoint["seed_component_count"],
                        "collar_component_count": endpoint["collar_component_count"],
                        "shared_component_count": endpoint["shared_component_count"],
                        "shared_component_voxel_count_in_cuboid": endpoint[
                            "shared_component_voxel_count_in_cuboid"
                        ],
                        "node_to_collar_component_observed": endpoint[
                            "node_to_collar_component_observed"
                        ],
                        "same_material_component_connects_a_to_b": result[
                            "same_material_component_connects_a_to_b"
                        ],
                        "same_component_connects_collar_a_to_b": result[
                            "collar_a_to_collar_b"
                        ]["same_component_observed"],
                    }
                )
        del batch_analyses
        gc.collect()

    compact_profiles_by_id: dict[int, np.ndarray] | None = None
    compact_profile_path: Path | None = None
    if args.write_compact_profiles:
        max_profile_length = max(len(profile) for profile in compact_profile_values)
        compact_profile_array = np.full(
            (len(compact_profile_values), max_profile_length), np.nan, dtype=np.float32
        )
        for index, profile in enumerate(compact_profile_values):
            compact_profile_array[index, : len(profile)] = profile
        compact_profile_path = args.output_dir / "all_strut_axial_profiles.npz"
        np.savez_compressed(
            compact_profile_path,
            strut_id=np.asarray(compact_profile_ids, dtype=np.int32),
            length_voxels=np.asarray(compact_profile_lengths, dtype=np.float32),
            axial_disk_foreground_fraction=compact_profile_array,
        )
        compact_profiles_by_id = {
            strut_id: profile[~np.isnan(profile)]
            for strut_id, profile in zip(
                compact_profile_ids, compact_profile_array, strict=True
            )
        }

    summary = {
        "purpose": (
            "Direct A-to-B strut connectivity: one corridor-local material component "
            "must intersect both endpoint windows; no defect classification."
        ),
        "source_files": {"ct_tiff": str(args.ct), "registered_graph": str(args.graph)},
        "analysis_config": str(analysis_config_path),
        "ct_array_order": "zyx",
        "graph_coordinate_order": "xyz",
        "configuration": {
            "frozen_intensity_threshold": FROZEN_THRESHOLD,
            "cuboid_half_width_voxels": half_width,
            "corridor_radius_voxels": corridor_radius,
            "corridor_radius_source": "frozen label-blind scan calibration",
            "axial_margin_voxels": 0.0,
            "node_a_first_slice": True,
            "node_b_last_slice": True,
            "transverse_margin_fraction": DEFAULT_TRANSVERSE_MARGIN_FRACTION,
            "node_seed_half_length_voxels": DEFAULT_ENDPOINT_SEED_HALF_LENGTH_VOXELS,
            "collar_half_length_voxels": DEFAULT_COLLAR_HALF_LENGTH_VOXELS,
            "collar_location_fraction_from_each_node": DEFAULT_COLLAR_FRACTION,
            "component_connectivity": "26-neighbor inside calibrated cylindrical corridor",
            "primary_decision": (
                "one identical corridor-local component label intersects both node windows"
            ),
            "interpolation_batch_size": args.batch_size,
            "interpolation_calls_for_selected_struts": int(
                np.ceil(len(selected_with_cohorts) / args.batch_size)
            ),
            "artifact_policy": "not-connected struts only"
            if args.failures_only
            else "all tested struts",
            "compact_axial_profiles": str(compact_profile_path)
            if compact_profile_path
            else None,
        },
        "selection": {
            "method": selection_method,
            "tested_strut_ids": [result["geometry"]["strut_id"] for result in results],
            "tube_emptiness_label_file": str(args.strut_ids_file) if args.strut_ids_file else None,
        },
        "cohort_statistics": cohort_statistics(results),
        "results": results,
    }
    summary_path = args.output_dir / "connection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_path = args.output_dir / "connection_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    overview_results = (
        [result for result in results if not result["same_material_component_connects_a_to_b"]]
        if args.failures_only
        else results
    )
    if not args.skip_overview:
        write_overview(
            overview_results,
            args.output_dir / "connection_overview.png",
            compact_profiles=compact_profiles_by_id,
        )
    write_output_readme(args.output_dir)
    print(f"Tested {len(results)} struts; wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
