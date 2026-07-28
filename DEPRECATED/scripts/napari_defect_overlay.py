#!/usr/bin/env python3
"""Overlay Brian Tran's CT scan on the registered ideal 9x9x9 octet lattice.

Brian's measured TIFF is rendered in green.  The ideal lattice topology is read
from the registered JSON and drawn in red in the same TIFF voxel coordinates.
STL-derived missing-strut IDs are first remapped through the validated cube
orientation so they refer to the correct physical locations in the CT scan.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIFF = PROJECT_ROOT / (
    "data/missing_struts/tif_stacks/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
DEFAULT_IDEAL = PROJECT_ROOT / (
    "data/missing_struts/registered_jsons/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
)
DEFAULT_NOMINAL_DESIGN = (
    PROJECT_ROOT / "data/missing_struts/octet_truss_9x9x9.json"
)
DEFAULT_MISSING_STRUTS = PROJECT_ROOT / (
    "data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv"
)

# The project's selected Triangle threshold preserves faint diagonal struts.
DEFAULT_CT_THRESHOLD = 34_963

# The STL deletion labels were extracted in the STL's cube orientation, while
# the registered JSON uses the physical CT orientation.  The intact octet
# lattice is cube-symmetric, so registration against complete geometry cannot
# determine this signed axis permutation.  Testing all 48 cube orientations
# against the intentionally missing CT corridors isolates this mapping:
#
#     centered (x, y, z) -> (-x, -z, -y)
#     absolute (x, y, z) -> (18-x, 18-z, 18-y)
#
# This is a 180-degree cube rotation (determinant +1), not a deformation.
CT_VALIDATED_AXIS_MAP = np.asarray(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
IDENTITY_AXIS_MAP = np.eye(3, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiff", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--ideal", type=Path, default=DEFAULT_IDEAL)
    parser.add_argument(
        "--nominal-design",
        type=Path,
        default=DEFAULT_NOMINAL_DESIGN,
        help="Complete nominal graph used to remap STL-label IDs into CT orientation",
    )
    parser.add_argument(
        "--missing-struts", type=Path, default=DEFAULT_MISSING_STRUTS
    )
    parser.add_argument(
        "--missing-id-orientation",
        choices=("ct-validated", "identity"),
        default="ct-validated",
        help=(
            "Remap STL-derived IDs into the CT-validated cube orientation "
            "(default), or use identity to reproduce the former incorrect overlay"
        ),
    )
    parser.add_argument("--ct-threshold", type=float, default=DEFAULT_CT_THRESHOLD)
    parser.add_argument(
        "--display-downsample",
        type=int,
        default=2,
        help="3-D display downsampling factor (2 keeps GPU memory manageable)",
    )
    parser.add_argument(
        "--ideal-radius",
        type=float,
        default=3.0,
        help="Radius of the red ideal-strut tubes in TIFF voxels",
    )
    parser.add_argument(
        "--tube-sides",
        type=int,
        default=8,
        help="Polygon sides per ideal tube (8 is smooth enough for inspection)",
    )
    return parser.parse_args()


def resolve_file(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=True)


def load_tiff(path: Path) -> np.ndarray:
    """Memory-map the 1 GB scan instead of copying it into RAM."""
    import tifffile

    try:
        volume = tifffile.memmap(path, mode="r")
    except ValueError:
        print("TIFF cannot be memory-mapped; decoding it into memory...")
        volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF, found {volume.shape}: {path}")
    return volume


def load_ideal_vectors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ideal strut IDs and vectors in TIFF (z, y, x) order."""
    document = json.loads(path.read_text(encoding="utf-8"))
    junctions = {
        int(item["id"]): np.asarray(item["position"], dtype=np.float32)
        for item in document["junctions"]
    }

    strut_ids = np.empty(len(document["struts"]), dtype=np.int64)
    vectors_xyz = np.empty((len(document["struts"]), 2, 3), dtype=np.float32)
    for index, strut in enumerate(document["struts"]):
        strut_ids[index] = int(strut["id"])
        start = junctions[int(strut["junction0"])]
        end = junctions[int(strut["junction1"])]
        vectors_xyz[index, 0] = start
        vectors_xyz[index, 1] = end - start

    # JSON positions are (x, y, z); numpy volumes and napari use (z, y, x).
    return strut_ids, vectors_xyz[..., ::-1]


def load_missing_strut_ids(path: Path) -> set[int]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["strut_id"]) for row in csv.DictReader(stream)}


def _edge_key(
    start: np.ndarray, end: np.ndarray
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return a direction-independent, numerically stable edge-position key."""
    rounded = (
        tuple(float(value) for value in np.round(start, decimals=6)),
        tuple(float(value) for value in np.round(end, decimals=6)),
    )
    return tuple(sorted(rounded))


def remap_missing_strut_ids(
    missing_path: Path,
    nominal_design_path: Path,
    axis_map: np.ndarray,
) -> tuple[set[int], set[int], dict[int, int]]:
    """Map STL-orientation deletion IDs onto equivalent CT-orientation edges.

    Strut IDs encode topology in a particular orientation.  Because the octet
    lattice is cube-symmetric, applying a signed axis permutation moves an edge
    to a different strut ID even though the full lattice looks unchanged.
    """
    if axis_map.shape != (3, 3) or not np.allclose(
        axis_map @ axis_map.T, np.eye(3)
    ):
        raise ValueError("axis_map must be a 3x3 orthogonal cube transform")

    source_ids = load_missing_strut_ids(missing_path)
    document = json.loads(nominal_design_path.read_text(encoding="utf-8"))
    junctions = {
        int(item["id"]): np.asarray(item["position"], dtype=np.float64)
        for item in document["junctions"]
    }
    struts = {
        int(item["id"]): (int(item["junction0"]), int(item["junction1"]))
        for item in document["struts"]
    }
    unknown = source_ids - set(struts)
    if unknown:
        preview = ", ".join(map(str, sorted(unknown)[:8]))
        raise ValueError(f"Missing-strut CSV contains unknown strut IDs: {preview}")

    positions = np.stack(list(junctions.values()))
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    edge_lookup: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for strut_id, (junction0, junction1) in struts.items():
        key = _edge_key(junctions[junction0], junctions[junction1])
        if key in edge_lookup:
            raise ValueError(
                f"Nominal design has duplicate physical edge IDs "
                f"{edge_lookup[key]} and {strut_id}"
            )
        edge_lookup[key] = strut_id

    mapping: dict[int, int] = {}
    for source_id in sorted(source_ids):
        junction0, junction1 = struts[source_id]
        start = (junctions[junction0] - center) @ axis_map.T + center
        end = (junctions[junction1] - center) @ axis_map.T + center
        key = _edge_key(start, end)
        if key not in edge_lookup:
            raise ValueError(
                f"Cube transform moved strut {source_id} outside nominal topology"
            )
        mapping[source_id] = edge_lookup[key]

    target_ids = set(mapping.values())
    if len(target_ids) != len(source_ids):
        raise ValueError("Cube transform produced colliding target strut IDs")
    return source_ids, target_ids, mapping


def clean_brian_mask(
    volume: np.ndarray,
    ideal_vectors: np.ndarray,
    threshold: float,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Threshold, crop, and remove disconnected CT noise for a clear overlay."""
    from scipy import ndimage

    starts = ideal_vectors[:, 0]
    ends = starts + ideal_vectors[:, 1]
    endpoints = np.concatenate((starts, ends), axis=0)
    padding = 8
    low = np.maximum(np.floor(endpoints.min(axis=0) - padding), 0).astype(int)
    high = np.minimum(
        np.ceil(endpoints.max(axis=0) + padding + 1), volume.shape
    ).astype(int)
    crop = tuple(
        slice(int(start), int(stop), downsample)
        for start, stop in zip(low, high)
    )
    sampled_mask = np.asarray(volume[crop] >= threshold)

    # Keep measured CT only in a narrow neighborhood of the registered design.
    # This removes the scan fixture/end plate while preserving the actual struts.
    starts_sampled = (starts - low) / downsample
    directions_sampled = ideal_vectors[:, 1] / downsample
    sample_count = int(np.ceil(np.linalg.norm(directions_sampled, axis=1).max())) + 2
    fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    centerline_points = np.rint(
        starts_sampled[:, None, :]
        + directions_sampled[:, None, :] * fractions[None, :, None]
    ).astype(np.int32)
    centerline_points = centerline_points.reshape(-1, 3)
    centerline_points = np.clip(
        centerline_points, 0, np.asarray(sampled_mask.shape) - 1
    )
    corridor = np.zeros_like(sampled_mask, dtype=bool)
    corridor[tuple(centerline_points.T)] = True
    corridor = ndimage.binary_dilation(
        corridor,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=4,
    )
    sampled_mask &= corridor

    # Isolated high-intensity speckles caused the green cloud in the first view.
    # Brian's lattice is connected, so retain its largest 26-connected component.
    labels, component_count = ndimage.label(
        sampled_mask, structure=np.ones((3, 3, 3), dtype=bool)
    )
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    cleaned = np.asarray(labels == largest_label, dtype=np.uint8)
    return cleaned, low, int(component_count)


def tube_surface(
    vectors: np.ndarray, radius: float, sides: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a lightweight red tube mesh instead of projecting every line."""
    starts = vectors[:, 0].astype(np.float32, copy=False)
    directions = vectors[:, 1].astype(np.float32, copy=False)
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = directions / lengths

    reference = np.zeros_like(unit)
    reference[:, 0] = 1.0
    nearly_parallel = np.abs(unit[:, 0]) > 0.9
    reference[nearly_parallel] = (0.0, 1.0, 0.0)
    basis_u = np.cross(unit, reference)
    basis_u /= np.linalg.norm(basis_u, axis=1, keepdims=True)
    basis_v = np.cross(unit, basis_u)

    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False, dtype=np.float32)
    offsets = radius * (
        np.cos(angles)[None, :, None] * basis_u[:, None, :]
        + np.sin(angles)[None, :, None] * basis_v[:, None, :]
    )
    rings = np.stack(
        (starts[:, None, :] + offsets, starts[:, None, :] + directions[:, None, :] + offsets),
        axis=1,
    )
    vertices = np.ascontiguousarray(rings.reshape(-1, 3), dtype=np.float32)

    base = np.arange(len(vectors), dtype=np.int32)[:, None] * (2 * sides)
    side = np.arange(sides, dtype=np.int32)[None, :]
    next_side = (side + 1) % sides
    first = np.stack(
        (base + side, base + next_side, base + sides + next_side), axis=-1
    )
    second = np.stack(
        (base + side, base + sides + next_side, base + sides + side), axis=-1
    )
    faces = np.ascontiguousarray(
        np.concatenate((first, second), axis=1).reshape(-1, 3), dtype=np.int32
    )
    colors = np.empty((len(vertices), 4), dtype=np.float32)
    colors[:] = (1.0, 0.0, 0.0, 1.0)
    return vertices, faces, colors


def mask_surface(
    mask: np.ndarray, origin: np.ndarray, scale: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Brian's cleaned mask to a GPU-reliable green surface mesh."""
    from skimage import measure

    vertices, faces, _normals, _values = measure.marching_cubes(
        mask, level=0.5, step_size=2, allow_degenerate=False
    )
    vertices = np.ascontiguousarray(
        vertices * float(scale) + origin[None, :], dtype=np.float32
    )
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    colors = np.empty((len(vertices), 4), dtype=np.float32)
    colors[:] = (0.0, 1.0, 0.0, 1.0)
    return vertices, faces, colors


def main() -> None:
    args = parse_args()
    import napari

    if args.display_downsample < 1:
        raise ValueError("--display-downsample must be at least 1")
    if args.ideal_radius <= 0 or args.tube_sides < 3:
        raise ValueError("Ideal radius must be positive and tube sides must be at least 3")
    tiff_path = resolve_file(args.tiff)
    ideal_path = resolve_file(args.ideal)
    nominal_design_path = resolve_file(args.nominal_design)
    missing_path = resolve_file(args.missing_struts)
    volume = load_tiff(tiff_path)
    strut_ids, ideal_vectors = load_ideal_vectors(ideal_path)
    axis_map = (
        CT_VALIDATED_AXIS_MAP
        if args.missing_id_orientation == "ct-validated"
        else IDENTITY_AXIS_MAP
    )
    source_missing_ids, missing_ids, missing_id_map = remap_missing_strut_ids(
        missing_path, nominal_design_path, axis_map
    )
    missing_vectors = ideal_vectors[np.isin(strut_ids, list(missing_ids))]
    if len(missing_vectors) != len(missing_ids):
        raise ValueError(
            f"Mapped {len(missing_vectors)} of {len(missing_ids)} missing strut IDs"
        )
    brian_mask, crop_origin, component_count = clean_brian_mask(
        volume, ideal_vectors, args.ct_threshold, args.display_downsample
    )
    ideal_surface = tube_surface(
        missing_vectors, radius=args.ideal_radius, sides=args.tube_sides
    )
    brian_surface = mask_surface(
        brian_mask, crop_origin, scale=args.display_downsample
    )

    print(f"Brian CT: {tiff_path}")
    print(f"  shape={volume.shape}, dtype={volume.dtype}")
    print(f"  threshold={args.ct_threshold:g}, display downsample={args.display_downsample}x")
    print(f"  removed disconnected noise from {component_count:,} components")
    print(f"Ideal lattice: {ideal_path}")
    print(f"  struts={len(ideal_vectors):,}")
    print(f"Missing-strut labels: {missing_path}")
    print(
        f"  source IDs={len(source_missing_ids):,}, "
        f"orientation={args.missing_id_orientation}, "
        f"remapped IDs={len(missing_ids):,}, "
        f"unchanged IDs={sum(source == target for source, target in missing_id_map.items()):,}"
    )
    print(f"Known missing ideal struts shown in red: {len(missing_vectors):,}")

    viewer = napari.Viewer(title="Defect overlay — Brian green, ideal red")
    viewer.add_surface(
        (brian_surface[0], brian_surface[1]),
        name="Brian segmented CT — green",
        vertex_colors=brian_surface[2],
        opacity=1.0,
        blending="translucent",
        shading="smooth",
    )
    viewer.add_surface(
        (ideal_surface[0], ideal_surface[1]),
        name=f"Missing ideal struts ({len(missing_vectors)}) — red",
        vertex_colors=ideal_surface[2],
        opacity=1.0,
        blending="additive",
        shading="flat",
    )

    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (32.0, -28.0, 42.0)
    viewer.reset_view()
    viewer.camera.zoom *= 0.70
    napari.run()


if __name__ == "__main__":
    main()
