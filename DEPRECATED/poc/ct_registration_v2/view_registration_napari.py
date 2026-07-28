#!/usr/bin/env python3
"""Interactively view a registered lattice graph over its CT volume in Napari."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import napari
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=repo_root
        / "data/missing_struts/tif_stacks/"
        "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=script_dir / "results/current/our_registered.json",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=script_dir / "results/current/fitted_transform.json",
        help="Transform JSON used to obtain the CT threshold.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="CT downsampling factor. Graph coordinates remain in full-resolution voxels.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Override the fitted per-scan foreground threshold.",
    )
    return parser.parse_args()


def load_registered_graph(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    positions_xyz = {
        int(node["id"]): np.asarray(node["position"], dtype=np.float32)
        for node in document["junctions"]
    }
    unique_xyz = np.unique(np.stack(list(positions_xyz.values())), axis=0)
    unique_zyx = unique_xyz[:, ::-1]
    segments_zyx = [
        np.stack(
            (
                positions_xyz[int(strut["junction0"])][::-1],
                positions_xyz[int(strut["junction1"])][::-1],
            )
        )
        for strut in document["struts"]
    ]
    return unique_zyx, segments_zyx


def fitted_threshold(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["per_scan_threshold"])


def main() -> int:
    args = parse_args()
    if args.downsample < 1:
        raise ValueError("--downsample must be at least 1")

    ct_path = args.ct.expanduser().resolve()
    registration_path = args.registration.expanduser().resolve()
    transform_path = args.transform.expanduser().resolve()
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else fitted_threshold(transform_path)
    )

    volume = tifffile.memmap(ct_path)
    sampled_foreground = np.asarray(
        volume[:: args.downsample, :: args.downsample, :: args.downsample]
        >= threshold
    )
    del volume
    nodes_zyx, segments_zyx = load_registered_graph(registration_path)

    viewer = napari.Viewer(
        title=f"CT registration v2 — {registration_path.name}",
        ndisplay=3,
    )
    viewer.add_image(
        sampled_foreground,
        name=f"CT foreground (≥ {threshold:g})",
        scale=(args.downsample,) * 3,
        colormap="gray",
        rendering="iso",
        iso_threshold=0.5,
        opacity=0.32,
    )
    viewer.add_shapes(
        segments_zyx,
        name=f"Registered struts ({len(segments_zyx):,})",
        shape_type="line",
        edge_color="#ff9f1c",
        edge_width=1.4,
        opacity=0.9,
    )
    viewer.add_points(
        nodes_zyx,
        name=f"Registered junctions ({len(nodes_zyx):,} unique)",
        size=4.0,
        face_color="#00e5ff",
        border_color="#082f49",
        border_width=0.15,
        opacity=0.95,
    )
    viewer.camera.angles = (25.0, -35.0, 120.0)
    viewer.reset_view()
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
