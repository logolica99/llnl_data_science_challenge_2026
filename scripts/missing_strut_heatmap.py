#!/usr/bin/env python3
"""Detect and visualize missing nodes/struts by comparing a lattice STL to its design graph."""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


STL_TRIANGLE_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def triangle_centroids(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        handle.seek(80)
        triangle_count = struct.unpack("<I", handle.read(4))[0]
    triangles = np.memmap(path, dtype=STL_TRIANGLE_DTYPE, mode="r", offset=84, shape=(triangle_count,))
    return np.asarray(triangles["vertices"].mean(axis=1), dtype=np.float32)


def load_design(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open() as handle:
        design = json.load(handle)
    junction_positions = np.asarray([item["position"] for item in design["junctions"]], dtype=float)
    strut_ids = np.asarray([item["id"] for item in design["struts"]], dtype=int)
    endpoints = np.asarray(
        [[item["junction0"], item["junction1"]] for item in design["struts"]], dtype=int
    )
    return junction_positions, endpoints, strut_ids


def stl_coordinates(points: np.ndarray, center: float, scale: float) -> np.ndarray:
    return (points - center) * scale


def physical_graph(
    junction_positions: np.ndarray, endpoints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    unique_positions, inverse = np.unique(junction_positions, axis=0, return_inverse=True)
    return unique_positions, inverse[endpoints]


def write_missing_struts_csv(
    path: Path,
    strut_ids: np.ndarray,
    endpoint_positions: np.ndarray,
    endpoint_positions_stl: np.ndarray,
    midpoint_gaps: np.ndarray,
    missing_mask: np.ndarray,
) -> None:
    columns = [
        "strut_id",
        "midpoint_surface_gap_mm",
        "x0_json",
        "y0_json",
        "z0_json",
        "x1_json",
        "y1_json",
        "z1_json",
        "x0_stl_mm",
        "y0_stl_mm",
        "z0_stl_mm",
        "x1_stl_mm",
        "y1_stl_mm",
        "z1_stl_mm",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for strut_id, json_ends, stl_ends, gap in zip(
            strut_ids[missing_mask],
            endpoint_positions[missing_mask],
            endpoint_positions_stl[missing_mask],
            midpoint_gaps[missing_mask],
        ):
            writer.writerow(
                [
                    int(strut_id),
                    f"{gap:.6f}",
                    *[f"{value:.3f}" for value in json_ends.ravel()],
                    *[f"{value:.3f}" for value in stl_ends.ravel()],
                ]
            )


def make_static_heatmap(
    path: Path,
    nodes_stl: np.ndarray,
    physical_edges: np.ndarray,
    missing_edges: np.ndarray,
    missing_gaps: np.ndarray,
    missing_nodes: np.ndarray,
) -> None:
    projections = [
        (0, 1, "STL X (mm)", "STL Y (mm)", "XY projection"),
        (0, 2, "STL X (mm)", "STL Z (mm)", "XZ projection"),
        (1, 2, "STL Y (mm)", "STL Z (mm)", "YZ projection"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True)
    full_segments = nodes_stl[physical_edges]
    hot_segments = nodes_stl[missing_edges]
    minimum = float(missing_gaps.min())
    maximum = float(missing_gaps.max())

    for axis, (horizontal, vertical, xlabel, ylabel, title) in zip(axes, projections):
        background = LineCollection(
            full_segments[:, :, [horizontal, vertical]],
            colors="#a8b0bb",
            linewidths=0.20,
            alpha=0.16,
            rasterized=True,
        )
        hot = LineCollection(
            hot_segments[:, :, [horizontal, vertical]],
            array=missing_gaps,
            cmap="inferno",
            norm=plt.Normalize(minimum, maximum),
            linewidths=2.2,
            alpha=0.95,
        )
        axis.add_collection(background)
        axis.add_collection(hot)
        axis.scatter(
            missing_nodes[:, horizontal],
            missing_nodes[:, vertical],
            marker="X",
            s=90,
            c="#00d6ff",
            edgecolors="#16212b",
            linewidths=0.8,
            label="missing node",
            zorder=4,
        )
        axis.autoscale()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.15, linewidth=0.5)

    colorbar = fig.colorbar(hot, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Centerline-to-STL surface gap (mm)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        f"0.5.stl missing-geometry heatmap — {len(missing_edges)} struts, {len(missing_nodes)} nodes",
        fontsize=14,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_inline_html(
    path: Path,
    nodes_stl: np.ndarray,
    physical_edges: np.ndarray,
    strut_ids: np.ndarray,
    missing_indices: np.ndarray,
    midpoint_gaps: np.ndarray,
    missing_node_indices: np.ndarray,
    node_surface_gaps: np.ndarray,
    node_missing_incident: np.ndarray,
    node_degree: np.ndarray,
) -> None:
    rounded_nodes = np.round(nodes_stl, 2).tolist()
    edge_pairs = physical_edges.astype(int).tolist()
    missing_records = [
        {
            "edge": int(index),
            "id": int(strut_ids[index]),
            "gap": round(float(midpoint_gaps[index]), 3),
        }
        for index in missing_indices
    ]
    affected_indices = np.flatnonzero(node_missing_incident)
    affected_records = [
        {
            "node": int(index),
            "missing": int(node_missing_incident[index]),
            "degree": int(node_degree[index]),
            "gap": round(float(node_surface_gaps[index]), 3),
        }
        for index in affected_indices
    ]
    payload = json.dumps(
        {
            "nodes": rounded_nodes,
            "edges": edge_pairs,
            "missing": missing_records,
            "affected": affected_records,
            "missingNodes": missing_node_indices.astype(int).tolist(),
        },
        separators=(",", ":"),
    )

    fragment = f'''<div id="missing-geometry-heatmap">
  <style>
    #missing-geometry-heatmap {{ width: 100%; color: var(--foreground); }}
    #missing-geometry-heatmap .heatmap-toolbar {{ justify-content: space-between; margin-bottom: 8px; }}
    #missing-geometry-heatmap .heatmap-legend {{ gap: 12px; }}
    #missing-geometry-heatmap .heatmap-key {{ display: inline-flex; align-items: center; gap: 6px; }}
    #missing-geometry-heatmap .heatmap-dot {{ width: 10px; height: 10px; border-radius: 50%; background: var(--destructive); }}
    #missing-geometry-heatmap .heatmap-node {{ width: 10px; height: 10px; transform: rotate(45deg); background: var(--viz-series-2); }}
    #missing-geometry-heatmap .heatmap-plot {{ width: 100%; height: clamp(390px, 72vw, 640px); }}
    #missing-geometry-heatmap .heatmap-detail {{ min-height: 1.5em; margin-top: 6px; color: var(--muted-foreground); }}
    @media (max-width: 520px) {{
      #missing-geometry-heatmap .heatmap-toolbar {{ align-items: flex-start; }}
      #missing-geometry-heatmap .heatmap-plot {{ height: 420px; }}
    }}
  </style>
  <div class="viz-row heatmap-toolbar">
    <div class="viz-row heatmap-legend text-small" aria-label="Legend">
      <span class="heatmap-key"><span class="heatmap-dot" aria-hidden="true"></span>93 missing struts</span>
      <span class="heatmap-key"><span class="heatmap-node" aria-hidden="true"></span>2 missing nodes</span>
    </div>
    <div class="viz-row" role="group" aria-label="Camera view">
      <button type="button" class="btn btn-primary" data-view="iso" aria-pressed="true">3D</button>
      <button type="button" class="btn" data-view="xy" aria-pressed="false">XY</button>
      <button type="button" class="btn" data-view="xz" aria-pressed="false">XZ</button>
      <button type="button" class="btn" data-view="yz" aria-pressed="false">YZ</button>
    </div>
  </div>
  <div id="missing-geometry-plot" class="heatmap-plot" role="img" aria-label="Interactive 3D lattice showing missing struts as hot lines and missing nodes as diamond markers"></div>
  <div id="missing-geometry-detail" class="heatmap-detail text-small" aria-live="polite">Select a hot midpoint or diamond node for coordinates and gap size.</div>
  <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.0.1/plotly.min.js"></script>
  <script>
    (() => {{
      const root = document.getElementById('missing-geometry-heatmap');
      const plot = document.getElementById('missing-geometry-plot');
      const detail = document.getElementById('missing-geometry-detail');
      const data = {payload};

      const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
      const buildLineArrays = (indices) => {{
        const x = [], y = [], z = [];
        indices.forEach((edgeIndex) => {{
          const edge = data.edges[edgeIndex];
          const a = data.nodes[edge[0]], b = data.nodes[edge[1]];
          x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null);
        }});
        return {{x, y, z}};
      }};
      const all = buildLineArrays(data.edges.map((_, index) => index));
      const missingIndices = data.missing.map((item) => item.edge);
      const hot = buildLineArrays(missingIndices);
      const midpoints = data.missing.map((item) => {{
        const edge = data.edges[item.edge], a = data.nodes[edge[0]], b = data.nodes[edge[1]];
        return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      }});
      const missingNodes = data.missingNodes.map((index) => data.nodes[index]);
      const missingNodeRecords = data.missingNodes.map((index) => data.affected.find((item) => item.node === index));

      const colors = () => ({{
        foreground: css('--foreground'), muted: css('--muted-foreground'), border: css('--border'),
        structure: css('--muted-foreground'), cool: css('--viz-series-2'), warm: css('--viz-series-1'),
        hot: css('--destructive'), background: css('--background')
      }});
      const traces = () => {{
        const c = colors();
        return [
          {{type:'scatter3d', mode:'lines', x:all.x, y:all.y, z:all.z, hoverinfo:'skip',
            line:{{color:c.structure, width:1}}, opacity:0.12, name:'Design lattice', showlegend:false}},
          {{type:'scatter3d', mode:'lines', x:hot.x, y:hot.y, z:hot.z, hoverinfo:'skip',
            line:{{color:c.hot, width:7}}, opacity:0.9, name:'Missing struts', showlegend:false}},
          {{type:'scatter3d', mode:'markers', x:midpoints.map(p=>p[0]), y:midpoints.map(p=>p[1]), z:midpoints.map(p=>p[2]),
            customdata:data.missing.map((item, i)=>[item.id,item.gap,midpoints[i][0],midpoints[i][1],midpoints[i][2]]),
            hovertemplate:'Strut %{{customdata[0]}}<br>gap %{{customdata[1]:.3f}} mm<br>STL (%{{customdata[2]:.2f}}, %{{customdata[3]:.2f}}, %{{customdata[4]:.2f}})<extra></extra>',
            marker:{{size:4,color:data.missing.map(item=>item.gap),cmin:0.5,cmax:1.2,
              colorscale:[[0,c.warm],[1,c.hot]],colorbar:{{title:{{text:'Gap (mm)'}},thickness:12,len:0.42,y:0.28}}}},
            name:'Strut gap', showlegend:false}},
          {{type:'scatter3d', mode:'markers', x:missingNodes.map(p=>p[0]), y:missingNodes.map(p=>p[1]), z:missingNodes.map(p=>p[2]),
            customdata:missingNodeRecords.map((item,i)=>[item.gap,missingNodes[i][0],missingNodes[i][1],missingNodes[i][2]]),
            hovertemplate:'Missing node<br>gap %{{customdata[0]:.3f}} mm<br>STL (%{{customdata[1]:.2f}}, %{{customdata[2]:.2f}}, %{{customdata[3]:.2f}})<extra></extra>',
            marker:{{size:9,color:c.cool,symbol:'diamond',line:{{color:c.foreground,width:1}}}}, name:'Missing nodes', showlegend:false}}
        ];
      }};
      const layout = () => {{
        const c = colors();
        const axis = {{title:{{font:{{color:c.foreground}}}},tickfont:{{color:c.muted}},gridcolor:c.border,zerolinecolor:c.border,backgroundcolor:'rgba(0,0,0,0)'}};
        return {{margin:{{l:0,r:0,t:0,b:0}},paper_bgcolor:'rgba(0,0,0,0)',font:{{color:c.foreground}},showlegend:false,
          scene:{{xaxis:{{...axis,title:{{text:'STL X (mm)',font:{{color:c.foreground}}}}}},yaxis:{{...axis,title:{{text:'STL Y (mm)',font:{{color:c.foreground}}}}}},zaxis:{{...axis,title:{{text:'STL Z (mm)',font:{{color:c.foreground}}}}}},
            aspectmode:'cube',camera:{{eye:{{x:1.45,y:1.45,z:1.15}},projection:{{type:'perspective'}}}}}}}};
      }};
      Plotly.newPlot(plot, traces(), layout(), {{responsive:true,displaylogo:false,scrollZoom:true,modeBarButtonsToRemove:['toImage','lasso3d','select2d']}});

      const cameras = {{
        iso:{{eye:{{x:1.45,y:1.45,z:1.15}},up:{{x:0,y:0,z:1}},projection:{{type:'perspective'}}}},
        xy:{{eye:{{x:0,y:0,z:2.5}},up:{{x:0,y:1,z:0}},projection:{{type:'orthographic'}}}},
        xz:{{eye:{{x:0,y:2.5,z:0}},up:{{x:0,y:0,z:1}},projection:{{type:'orthographic'}}}},
        yz:{{eye:{{x:2.5,y:0,z:0}},up:{{x:0,y:0,z:1}},projection:{{type:'orthographic'}}}}
      }};
      root.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => {{
        root.querySelectorAll('[data-view]').forEach((item) => {{ item.classList.remove('btn-primary'); item.setAttribute('aria-pressed','false'); }});
        button.classList.add('btn-primary'); button.setAttribute('aria-pressed','true');
        Plotly.relayout(plot, {{'scene.camera':cameras[button.dataset.view]}});
      }}));
      plot.on('plotly_click', (event) => {{
        const point = event.points && event.points[0];
        if (!point || !point.customdata) return;
        if (point.curveNumber === 2) {{
          const [id,gap,x,y,z] = point.customdata;
          detail.textContent = `Strut ${{id}} — centerline gap ${{Number(gap).toFixed(3)}} mm at STL midpoint (${{Number(x).toFixed(2)}}, ${{Number(y).toFixed(2)}}, ${{Number(z).toFixed(2)}}).`;
        }} else if (point.curveNumber === 3) {{
          const [gap,x,y,z] = point.customdata;
          detail.textContent = `Missing node — surface gap ${{Number(gap).toFixed(3)}} mm at STL coordinates (${{Number(x).toFixed(2)}}, ${{Number(y).toFixed(2)}}, ${{Number(z).toFixed(2)}}).`;
        }}
      }});
    }})();
  </script>
</div>
'''
    path.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--reference-stl", type=Path)
    parser.add_argument("--center", type=float, default=9.0)
    parser.add_argument("--scale", type=float, default=2.28)
    parser.add_argument("--gap-threshold", type=float, default=0.5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    junction_positions, endpoints, strut_ids = load_design(args.design)
    physical_nodes, physical_edges = physical_graph(junction_positions, endpoints)
    nodes_stl = stl_coordinates(physical_nodes, args.center, args.scale)
    endpoint_positions = junction_positions[endpoints]
    endpoint_positions_stl = stl_coordinates(endpoint_positions, args.center, args.scale)
    midpoint_positions_stl = endpoint_positions_stl.mean(axis=1)

    tree = cKDTree(triangle_centroids(args.stl), leafsize=32)
    midpoint_gaps, _ = tree.query(midpoint_positions_stl, workers=-1)
    node_surface_gaps, _ = tree.query(nodes_stl, workers=-1)
    missing_mask = midpoint_gaps > args.gap_threshold
    missing_node_mask = node_surface_gaps > args.gap_threshold
    missing_indices = np.flatnonzero(missing_mask)
    missing_node_indices = np.flatnonzero(missing_node_mask)

    node_degree = np.bincount(physical_edges.ravel(), minlength=len(physical_nodes))
    node_missing_incident = np.bincount(
        physical_edges[missing_mask].ravel(), minlength=len(physical_nodes)
    )

    reference_max_gap = None
    if args.reference_stl:
        reference_tree = cKDTree(triangle_centroids(args.reference_stl), leafsize=32)
        reference_gaps, _ = reference_tree.query(midpoint_positions_stl, workers=-1)
        reference_max_gap = float(reference_gaps.max())

    write_missing_struts_csv(
        args.output_dir / "missing_struts.csv",
        strut_ids,
        endpoint_positions,
        endpoint_positions_stl,
        midpoint_gaps,
        missing_mask,
    )
    missing_nodes_payload = [
        {
            "physical_node_index": int(index),
            "json_position": physical_nodes[index].tolist(),
            "stl_position_mm": np.round(nodes_stl[index], 6).tolist(),
            "surface_gap_mm": round(float(node_surface_gaps[index]), 6),
            "missing_incident_struts": int(node_missing_incident[index]),
            "degree": int(node_degree[index]),
        }
        for index in missing_node_indices
    ]
    summary = {
        "input_stl": str(args.stl.resolve()),
        "design_json": str(args.design.resolve()),
        "method": "KD-tree nearest-surface gap at design strut midpoints and physical node centers",
        "json_to_stl_transform": f"stl_mm = (json_coordinate - {args.center}) * {args.scale}",
        "gap_threshold_mm": args.gap_threshold,
        "design_strut_count": int(len(endpoints)),
        "missing_strut_count": int(missing_mask.sum()),
        "missing_strut_percent": float(100 * missing_mask.mean()),
        "physical_node_count": int(len(physical_nodes)),
        "missing_node_count": int(missing_node_mask.sum()),
        "affected_node_count": int(np.count_nonzero(node_missing_incident)),
        "reference_stl_max_midpoint_gap_mm": reference_max_gap,
        "missing_nodes": missing_nodes_payload,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    make_static_heatmap(
        args.output_dir / "missing_geometry_heatmap.png",
        nodes_stl,
        physical_edges,
        physical_edges[missing_mask],
        midpoint_gaps[missing_mask],
        nodes_stl[missing_node_mask],
    )
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        make_inline_html(
            args.html,
            nodes_stl,
            physical_edges,
            strut_ids,
            missing_indices,
            midpoint_gaps,
            missing_node_indices,
            node_surface_gaps,
            node_missing_incident,
            node_degree,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
