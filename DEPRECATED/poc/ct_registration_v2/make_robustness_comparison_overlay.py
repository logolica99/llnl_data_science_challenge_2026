#!/usr/bin/env python3
"""Render baseline, failed coarse alternative, and refined recovery over CT."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

from make_registration_overlay import (
    add_graph,
    clipped_xy_segments,
    draw_base,
    load_graph,
)
from registration_core import load_ct_volume


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    result_dir = script_dir / "results/pacificvis_8x8x8_v2run"
    comparison_dir = result_dir / "edt_2p25_comparison"
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=repo_root
        / "data/pacificvis/8x8x8 octet lattice with defects/"
        "five_defects_1200_xray_recon.npy",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=result_dir / "our_registered.json",
    )
    parser.add_argument(
        "--alternative-raw",
        type=Path,
        default=comparison_dir / "alternative_raw_registered.json",
    )
    parser.add_argument(
        "--alternative-refined",
        type=Path,
        default=comparison_dir / "alternative_refined_registered.json",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=comparison_dir / "comparison.json",
    )
    parser.add_argument("--z", type=int, default=599)
    parser.add_argument("--half-slab", type=float, default=7.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=comparison_dir / "edt_2p25_true_test_overlay_z599.png",
    )
    parser.add_argument(
        "--jpeg",
        type=Path,
        default=comparison_dir / "edt_2p25_true_test_overlay_z599.jpg",
    )
    parser.add_argument("--html-fragment", type=Path, required=True)
    return parser.parse_args()


def write_html_fragment(path: Path, jpeg_path: Path) -> None:
    encoded = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    fragment = f"""<div id="ct-registration-robustness-true-test">
  <img src="data:image/jpeg;base64,{encoded}" alt="The same axial PacificVis CT slab with three registration views: the accepted baseline aligned to strut centers, the raw EDT 2.25 alternative visibly offset and scaled incorrectly, and the same alternative after full-resolution image refinement nearly overlapping the baseline." style="display:block;width:100%;height:auto;">
</div>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment, encoding="utf-8")


def main() -> int:
    args = parse_args()
    volume = load_ct_volume(args.ct.resolve())
    z_center = int(args.z)
    z0 = max(0, int(np.floor(z_center - args.half_slab)))
    z1 = min(volume.shape[0], int(np.ceil(z_center + args.half_slab)) + 1)
    slab = np.asarray(volume[z0:z1]).max(axis=0)

    baseline, edges = load_graph(args.baseline.resolve())
    raw, raw_edges = load_graph(args.alternative_raw.resolve())
    refined, refined_edges = load_graph(args.alternative_refined.resolve())
    if not np.array_equal(edges, raw_edges) or not np.array_equal(
        edges, refined_edges
    ):
        raise RuntimeError("Comparison graph topologies differ")

    z_low = z_center - float(args.half_slab)
    z_high = z_center + float(args.half_slab)
    baseline_segments = clipped_xy_segments(baseline, edges, z_low, z_high)
    raw_segments = clipped_xy_segments(raw, edges, z_low, z_high)
    refined_segments = clipped_xy_segments(refined, edges, z_low, z_high)
    baseline_nodes = (baseline[:, 2] >= z_low) & (baseline[:, 2] <= z_high)
    raw_nodes = (raw[:, 2] >= z_low) & (raw[:, 2] <= z_high)
    refined_nodes = (refined[:, 2] >= z_low) & (refined[:, 2] <= z_high)

    finite = slab[np.isfinite(slab)]
    norm = PowerNorm(
        gamma=0.8,
        vmin=float(np.quantile(finite, 0.01)),
        vmax=float(np.quantile(finite, 0.997)),
        clip=True,
    )
    graph_low = np.min(baseline[:, :2], axis=0)
    graph_high = np.max(baseline[:, :2], axis=0)
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

    comparison = json.loads(args.comparison.read_text())
    base_metrics = comparison["baseline"]["image_metrics"]
    raw_metrics = comparison["alternative_raw"]["image_metrics"]
    refined_metrics = comparison["alternative_refined"]["image_metrics"]
    refined_difference = comparison["alternative_refined"][
        "difference_from_baseline"
    ]

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
        draw_base(axis, slab, norm, x_limits, y_limits)

    add_graph(
        axes[0],
        baseline_segments,
        baseline,
        baseline_nodes,
        color="#ff9f1c",
        label="Baseline final",
        linewidth=1.1,
        alpha=0.95,
        nodes=True,
    )
    axes[0].set_title(
        "Baseline final — PASS\n"
        f"holdout median {base_metrics['holdout_candidate_median_distance_voxels']:.2f} vox  ·  "
        f"corridor {base_metrics['corridor_median_foreground_fraction']:.3f}",
        color="#f0f4f8",
    )

    add_graph(
        axes[1],
        raw_segments,
        raw,
        raw_nodes,
        color="#f03e8c",
        label="Raw EDT 2.25",
        linewidth=1.1,
        alpha=0.95,
        nodes=True,
    )
    axes[1].set_title(
        "Raw EDT 2.25 — REJECTED\n"
        f"holdout median {raw_metrics['holdout_candidate_median_distance_voxels']:.2f} vox  ·  "
        f"corridor {raw_metrics['corridor_median_foreground_fraction']:.3f}",
        color="#f0f4f8",
    )

    add_graph(
        axes[2],
        baseline_segments,
        baseline,
        np.zeros(len(baseline), dtype=bool),
        color="#ff9f1c",
        label="Baseline",
        linewidth=1.8,
        alpha=0.75,
        linestyle="dashed",
    )
    add_graph(
        axes[2],
        refined_segments,
        refined,
        refined_nodes,
        color="#25d0a6",
        label="EDT 2.25 after image refinement",
        linewidth=1.0,
        alpha=0.95,
        nodes=True,
    )
    axes[2].set_title(
        "EDT 2.25 after full-resolution refinement — PASS\n"
        f"P95 vs baseline {refined_difference['p95_voxels']:.2f} vox  ·  "
        f"corridor {refined_metrics['corridor_median_foreground_fraction']:.3f}",
        color="#f0f4f8",
    )

    for axis in axes:
        axis.legend(loc="lower right", framealpha=0.85, fontsize=8)
    figure.suptitle(
        f"End-to-end robustness test on the same axial CT slab  ·  Z={z0}–{z1 - 1}",
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
    write_html_fragment(args.html_fragment.resolve(), args.jpeg.resolve())
    print(
        json.dumps(
            {
                "png": str(args.output.resolve()),
                "jpeg": str(args.jpeg.resolve()),
                "html_fragment": str(args.html_fragment.resolve()),
                "slice_z": z_center,
                "slab_z": [z0, z1 - 1],
                "baseline_segments": int(len(baseline_segments)),
                "raw_alternative_segments": int(len(raw_segments)),
                "refined_alternative_segments": int(len(refined_segments)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
