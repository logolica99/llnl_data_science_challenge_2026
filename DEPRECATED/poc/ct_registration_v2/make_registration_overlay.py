#!/usr/bin/env python3
"""Render an axial CT slab with v2 and post-fit reference graph overlays."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from registration_core import load_ct_volume


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
        "--reference",
        type=Path,
        default=repo_root
        / "data/missing_struts/registered_jsons/"
        "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json",
    )
    parser.add_argument(
        "--ct-only",
        action="store_true",
        help="Render without opening or comparing a supplied registered graph.",
    )
    parser.add_argument("--z", type=int, default=380)
    parser.add_argument("--half-slab", type=float, default=6.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir
        / "results/current/registration_overlay_z380.png",
    )
    parser.add_argument(
        "--jpeg",
        type=Path,
        default=script_dir
        / "results/current/registration_overlay_z380.jpg",
    )
    parser.add_argument(
        "--html-fragment",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_graph(path: Path) -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(path.read_text())
    nodes = {
        int(node["id"]): np.asarray(node["position"], dtype=np.float64)
        for node in document["junctions"]
    }
    max_id = max(nodes)
    positions = np.empty((max_id + 1, 3), dtype=np.float64)
    for node_id, position in nodes.items():
        positions[node_id] = position
    edges = np.asarray(
        [
            [int(strut["junction0"]), int(strut["junction1"])]
            for strut in document["struts"]
        ],
        dtype=np.int64,
    )
    return positions, edges


def clipped_xy_segments(
    positions: np.ndarray,
    edges: np.ndarray,
    z_low: float,
    z_high: float,
) -> np.ndarray:
    start = positions[edges[:, 0]]
    end = positions[edges[:, 1]]
    delta = end - start
    segments: list[np.ndarray] = []
    for p0, d in zip(start, delta):
        if abs(d[2]) < 1e-12:
            if z_low <= p0[2] <= z_high:
                segments.append(np.stack([p0[:2], (p0 + d)[:2]]))
            continue
        t0 = (z_low - p0[2]) / d[2]
        t1 = (z_high - p0[2]) / d[2]
        enter = max(0.0, min(t0, t1))
        leave = min(1.0, max(t0, t1))
        if enter <= leave:
            segments.append(
                np.stack(
                    [
                        (p0 + enter * d)[:2],
                        (p0 + leave * d)[:2],
                    ]
                )
            )
    return np.asarray(segments, dtype=np.float64)


def draw_base(
    axis: plt.Axes,
    slab: np.ndarray,
    norm: PowerNorm,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    axis.imshow(
        slab,
        cmap="gray",
        norm=norm,
        origin="upper",
        interpolation="nearest",
    )
    axis.set_xlim(*x_limits)
    axis.set_ylim(y_limits[1], y_limits[0])
    axis.set_aspect("equal")
    axis.set_xlabel("CT X voxel")
    axis.set_ylabel("CT Y voxel")


def add_graph(
    axis: plt.Axes,
    segments: np.ndarray,
    positions: np.ndarray,
    node_mask: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float,
    alpha: float,
    linestyle: str = "solid",
    nodes: bool = False,
) -> None:
    axis.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            linestyles=linestyle,
            label=label,
            zorder=3,
        )
    )
    if nodes:
        axis.scatter(
            positions[node_mask, 0],
            positions[node_mask, 1],
            s=6,
            c=color,
            alpha=min(1.0, alpha + 0.1),
            linewidths=0,
            zorder=4,
        )


def write_html_fragment(
    path: Path,
    jpeg_path: Path,
    *,
    z_center: int,
    ct_only: bool,
) -> None:
    encoded = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    comparison = (
        "The third panel is a center crop because no registered reference exists "
        "for this dataset."
        if ct_only
        else "The third panel compares v2 against the supplied aligned reference."
    )
    fragment = f"""<div id="ct-registration-overlay-view">
  <img src="data:image/jpeg;base64,{encoded}" alt="Axial CT slab at Z {z_center} shown raw and with the v2 CT-only lattice registration. {comparison}" style="display:block;width:100%;height:auto;">
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment)


def main() -> int:
    args = parse_args()
    volume = load_ct_volume(args.ct.resolve())
    z_center = int(args.z)
    z0 = max(0, int(np.floor(z_center - args.half_slab)))
    z1 = min(volume.shape[0], int(np.ceil(z_center + args.half_slab)) + 1)
    slab = np.asarray(volume[z0:z1]).max(axis=0)

    v2_positions, edges = load_graph(args.registration.resolve())
    reference_positions = None
    reference_segments = np.empty((0, 2, 2), dtype=np.float64)
    if not args.ct_only:
        reference_positions, reference_edges = load_graph(
            args.reference.resolve()
        )
        if not np.array_equal(edges, reference_edges):
            raise RuntimeError("V2 and reference graph topology differ")

    z_low = z_center - float(args.half_slab)
    z_high = z_center + float(args.half_slab)
    v2_segments = clipped_xy_segments(
        v2_positions, edges, z_low, z_high
    )
    if reference_positions is not None:
        reference_segments = clipped_xy_segments(
            reference_positions, edges, z_low, z_high
        )
    v2_nodes = (
        (v2_positions[:, 2] >= z_low)
        & (v2_positions[:, 2] <= z_high)
    )

    finite = slab[np.isfinite(slab)]
    vmin = float(np.quantile(finite, 0.01))
    vmax = float(np.quantile(finite, 0.997))
    norm = PowerNorm(gamma=0.8, vmin=vmin, vmax=vmax, clip=True)
    graph_low = np.min(v2_positions[:, :2], axis=0)
    graph_high = np.max(v2_positions[:, :2], axis=0)
    graph_span = graph_high - graph_low
    padding = np.maximum(12.0, graph_span * 0.035)
    x_limits = (
        max(0.0, float(graph_low[0] - padding[0])),
        min(float(slab.shape[1] - 1), float(graph_high[0] + padding[0])),
    )
    y_limits = (
        max(0.0, float(graph_low[1] - padding[1])),
        min(float(slab.shape[0] - 1), float(graph_high[1] + padding[1])),
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16.5, 5.7),
        constrained_layout=True,
        facecolor="#10141a",
    )
    for axis in axes:
        axis.set_facecolor("#10141a")
        axis.tick_params(colors="#d9e2ec", labelsize=8)
        axis.xaxis.label.set_color("#d9e2ec")
        axis.yaxis.label.set_color("#d9e2ec")
        for spine in axis.spines.values():
            spine.set_color("#52606d")

    draw_base(axes[0], slab, norm, x_limits, y_limits)
    axes[0].set_title(
        f"Raw CT maximum projection\nZ={z0}–{z1 - 1}",
        color="#f0f4f8",
    )

    draw_base(axes[1], slab, norm, x_limits, y_limits)
    add_graph(
        axes[1],
        v2_segments,
        v2_positions,
        v2_nodes,
        color="#ff9f1c",
        label="V2 CT-only graph",
        linewidth=1.15,
        alpha=0.9,
        nodes=True,
    )
    axes[1].set_title(
        "V2 registration over CT", color="#f0f4f8"
    )
    axes[1].legend(
        loc="lower right",
        framealpha=0.8,
        fontsize=8,
    )

    draw_base(axes[2], slab, norm, x_limits, y_limits)
    if reference_positions is not None:
        add_graph(
            axes[2],
            reference_segments,
            reference_positions,
            np.zeros(len(reference_positions), dtype=bool),
            color="#24d6e5",
            label="Supplied alignment",
            linewidth=1.8,
            alpha=0.8,
            linestyle="dashed",
        )
    add_graph(
        axes[2],
        v2_segments,
        v2_positions,
        v2_nodes,
        color="#ff9f1c",
        label="V2 CT-only",
        linewidth=1.0,
        alpha=0.95,
    )
    axes[2].set_title(
        (
            "Center crop — CT-only evidence"
            if args.ct_only
            else "Post-fit comparison"
        ),
        color="#f0f4f8",
    )
    axes[2].legend(
        loc="lower right",
        framealpha=0.8,
        fontsize=8,
    )

    center = (graph_low + graph_high) / 2.0
    zoom_half = max(80.0, float(np.max(graph_span) * 0.13))
    if args.ct_only:
        axes[2].set_xlim(center[0] - zoom_half, center[0] + zoom_half)
        axes[2].set_ylim(center[1] + zoom_half, center[1] - zoom_half)
    else:
        zoom = inset_axes(
            axes[2],
            width="39%",
            height="39%",
            loc="upper right",
            borderpad=0.8,
        )
        draw_base(zoom, slab, norm, x_limits, y_limits)
        add_graph(
            zoom,
            reference_segments,
            reference_positions,
            np.zeros(len(reference_positions), dtype=bool),
            color="#24d6e5",
            label="",
            linewidth=2.0,
            alpha=0.85,
            linestyle="dashed",
        )
        add_graph(
            zoom,
            v2_segments,
            v2_positions,
            v2_nodes,
            color="#ff9f1c",
            label="",
            linewidth=1.2,
            alpha=0.95,
        )
        zoom.set_xlim(center[0] - zoom_half, center[0] + zoom_half)
        zoom.set_ylim(center[1] + zoom_half, center[1] - zoom_half)
        zoom.set_xticks([])
        zoom.set_yticks([])
        zoom.set_xlabel("")
        zoom.set_ylabel("")
        zoom.set_title("Center zoom", color="#f0f4f8", fontsize=8)
        mark_inset(
            axes[2],
            zoom,
            loc1=2,
            loc2=4,
            fc="none",
            ec="#d9e2ec",
            lw=0.8,
        )

    figure.suptitle(
        "Axial CT registration check — thin slab avoids projection clutter",
        color="#f0f4f8",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.resolve(), dpi=170, facecolor=figure.get_facecolor())
    figure.savefig(
        args.jpeg.resolve(),
        dpi=150,
        facecolor=figure.get_facecolor(),
        pil_kwargs={"quality": 88, "optimize": True},
    )
    plt.close(figure)
    write_html_fragment(
        args.html_fragment.resolve(),
        args.jpeg.resolve(),
        z_center=z_center,
        ct_only=args.ct_only,
    )
    print(
        json.dumps(
            {
                "png": str(args.output.resolve()),
                "jpeg": str(args.jpeg.resolve()),
                "html_fragment": str(args.html_fragment.resolve()),
                "slice_z": z_center,
                "slab_z": [z0, z1 - 1],
                "v2_segments": int(len(v2_segments)),
                "reference_segments": int(len(reference_segments)),
                "v2_nodes": int(np.count_nonzero(v2_nodes)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
