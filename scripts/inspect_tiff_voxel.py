"""Inspect CT voxels and test whether JSON positions are in a TIFF voxel frame.

TIFF arrays are indexed as ``volume[z, y, x]``. JSON geometry positions are
interpreted as ``[x, y, z]``. This utility performs that conversion explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", required=True, type=Path, help="Path to a TIFF stack")
    parser.add_argument("--json", type=Path, help="Optional lattice JSON path")
    parser.add_argument("--junction-id", type=int, help="Read the position of this JSON junction")
    parser.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"), help="Voxel coordinate")
    parser.add_argument(
        "--test-json",
        action="store_true",
        help="Sample all JSON junction positions and compare them with a global TIFF sample",
    )
    return parser.parse_args()


def tiff_shape(path: Path) -> tuple[int, int, int]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "ZYX":
            raise ValueError(f"Expected a grayscale ZYX TIFF, found axes={series.axes!r}")
        return tuple(int(value) for value in series.shape)


def nearest_voxels(path: Path, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbor voxel values for continuous JSON [x, y, z] points."""
    zyx = np.rint(xyz[:, ::-1]).astype(int)
    z_size, y_size, x_size = tiff_shape(path)
    valid = (
        (zyx[:, 0] >= 0)
        & (zyx[:, 0] < z_size)
        & (zyx[:, 1] >= 0)
        & (zyx[:, 1] < y_size)
        & (zyx[:, 2] >= 0)
        & (zyx[:, 2] < x_size)
    )
    values = np.full(len(xyz), np.nan)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in np.flatnonzero(valid):
        grouped[int(zyx[index, 0])].append(int(index))

    with tifffile.TiffFile(path) as tif:
        for z, indices in grouped.items():
            page = tif.pages[z].asarray()
            indices_array = np.asarray(indices)
            values[indices_array] = page[zyx[indices_array, 1], zyx[indices_array, 2]]
    return values, valid


def global_sample(path: Path, stride: int = 16) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        return np.concatenate(
            [tif.pages[z].asarray()[::stride, ::stride].ravel() for z in range(0, len(tif.pages), stride)]
        )


def junction_position(json_path: Path, junction_id: int) -> np.ndarray:
    data = json.loads(json_path.read_text())
    for junction in data["junctions"]:
        if junction["id"] == junction_id:
            return np.asarray(junction["position"], dtype=float)
    raise ValueError(f"No junction with id={junction_id}")


def print_metadata(path: Path) -> None:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        print("TIFF voxel frame")
        print(f"  array shape / axes: {series.shape} / {series.axes}")
        print(f"  valid JSON-style xyz: [0, 0, 0] through [{series.shape[2]-1}, {series.shape[1]-1}, {series.shape[0]-1}]")
        print(f"  dtype: {series.dtype}")
        print("  lookup convention: JSON [x, y, z] -> TIFF volume[z, y, x]")


def main() -> None:
    args = parse_args()
    print_metadata(args.tif)

    if args.junction_id is not None:
        if args.json is None:
            raise SystemExit("--junction-id requires --json")
        xyz = junction_position(args.json, args.junction_id)
        args.xyz = xyz.tolist()
        print(f"junction {args.junction_id} JSON position: {xyz.tolist()}")

    if args.xyz is not None:
        xyz = np.asarray([args.xyz], dtype=float)
        values, valid = nearest_voxels(args.tif, xyz)
        print(f"requested xyz: {xyz[0].tolist()}")
        print(f"nearest TIFF index zyx: {np.rint(xyz[0, ::-1]).astype(int).tolist()}")
        print(f"inside TIFF frame: {bool(valid[0])}")
        print(f"uint16 intensity: {None if not valid[0] else int(values[0])}")

    if args.test_json:
        if args.json is None:
            raise SystemExit("--test-json requires --json")
        data = json.loads(args.json.read_text())
        positions = np.asarray([junction["position"] for junction in data["junctions"]], dtype=float)
        values, valid = nearest_voxels(args.tif, positions)
        values = values[valid]
        background = global_sample(args.tif)
        print("\nDirect JSON-to-TIFF registration test")
        print(f"  junctions inside voxel frame: {len(values)} / {len(positions)}")
        print(f"  JSON position bounds xyz: {positions.min(axis=0).tolist()} to {positions.max(axis=0).tolist()}")
        print(f"  JSON-position median intensity: {np.median(values):.1f}")
        print(f"  global sampled median intensity: {np.median(background):.1f}")
        print(f"  JSON-position 90th percentile: {np.quantile(values, 0.9):.1f}")
        print(f"  global sampled 90th percentile: {np.quantile(background, 0.9):.1f}")
        if np.median(values) > np.quantile(background, 0.9):
            print("  result: positions strongly coincide with high-density CT material; direct registration is plausible.")
        else:
            print("  result: positions do not strongly coincide with CT material; do not treat these coordinates as registered.")


if __name__ == "__main__":
    main()
