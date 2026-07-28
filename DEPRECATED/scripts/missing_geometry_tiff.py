#!/usr/bin/env python3
"""Create CT-aligned TIFF labels and annotated slices for missing lattice geometry."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def read_missing_strut_ids(path: Path) -> list[int]:
    with path.open(newline="") as handle:
        return [int(row["strut_id"]) for row in csv.DictReader(handle)]


def registered_missing_segments(registered: dict, missing_ids: list[int]) -> dict[int, np.ndarray]:
    junctions = {int(item["id"]): np.asarray(item["position"], dtype=float) for item in registered["junctions"]}
    struts = {int(item["id"]): item for item in registered["struts"]}
    return {
        strut_id: np.stack(
            [
                junctions[int(struts[strut_id]["junction0"])],
                junctions[int(struts[strut_id]["junction1"])],
            ]
        )
        for strut_id in missing_ids
    }


def registered_missing_nodes(nominal: dict, registered: dict, summary: dict) -> np.ndarray:
    nominal_positions = np.asarray([item["position"] for item in nominal["junctions"]], dtype=float)
    registered_positions = np.asarray([item["position"] for item in registered["junctions"]], dtype=float)
    node_positions = []
    for node in summary["missing_nodes"]:
        target = np.asarray(node["json_position"], dtype=float)
        matches = np.all(np.isclose(nominal_positions, target, atol=1e-8), axis=1)
        if not matches.any():
            raise ValueError(f"Could not map missing node {target.tolist()} into registered JSON")
        node_positions.append(registered_positions[matches].mean(axis=0))
    return np.asarray(node_positions)


def build_slice_stamps(
    shape: tuple[int, int, int],
    segments: dict[int, np.ndarray],
    missing_nodes: np.ndarray,
    strut_radius: float,
    node_radius: float,
) -> tuple[list[list[tuple[float, float, float, int]]], list[set[int]], list[set[int]]]:
    depth, _, _ = shape
    stamps: list[list[tuple[float, float, float, int]]] = [[] for _ in range(depth)]
    slice_struts: list[set[int]] = [set() for _ in range(depth)]
    slice_nodes: list[set[int]] = [set() for _ in range(depth)]

    def add_sphere(center: np.ndarray, radius: float, label: int, identity: int | None = None) -> None:
        x, y, z = center
        z0 = max(0, int(math.floor(z - radius)))
        z1 = min(depth - 1, int(math.ceil(z + radius)))
        for zi in range(z0, z1 + 1):
            dz = zi - z
            radius_xy = math.sqrt(max(radius * radius - dz * dz, 0.0))
            stamps[zi].append((x, y, radius_xy, label))
            if label == 1 and identity is not None:
                slice_struts[zi].add(identity)
            elif label == 2 and identity is not None:
                slice_nodes[zi].add(identity)

    for strut_id, segment in segments.items():
        length = float(np.linalg.norm(segment[1] - segment[0]))
        samples = max(2, int(math.ceil(length / 0.75)) + 1)
        for center in np.linspace(segment[0], segment[1], samples):
            add_sphere(center, strut_radius, 1, strut_id)

    for node_index, center in enumerate(missing_nodes):
        add_sphere(center, node_radius, 2, node_index)

    return stamps, slice_struts, slice_nodes


def render_label_slice(
    shape_yx: tuple[int, int], stamps: list[tuple[float, float, float, int]]
) -> np.ndarray:
    height, width = shape_yx
    result = np.zeros((height, width), dtype=np.uint8)
    for x, y, radius, label in stamps:
        x0 = max(0, int(math.floor(x - radius)))
        x1 = min(width - 1, int(math.ceil(x + radius)))
        y0 = max(0, int(math.floor(y - radius)))
        y1 = min(height - 1, int(math.ceil(y + radius)))
        if x0 > x1 or y0 > y1:
            continue
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
        region = result[y0 : y1 + 1, x0 : x1 + 1]
        region[disk] = np.maximum(region[disk], label)
    return result


def contrast_to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.7])
    if high <= low:
        high = low + 1
    return np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def overlay_labels(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    gray = contrast_to_uint8(image)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    colors = {1: np.asarray([255, 72, 20], dtype=np.float32), 2: np.asarray([0, 218, 255], dtype=np.float32)}
    for label, color in colors.items():
        mask = labels == label
        if mask.any():
            rgb[mask] = (0.25 * rgb[mask].astype(np.float32) + 0.75 * color).astype(np.uint8)
    return rgb


def select_slices(
    pixel_counts: np.ndarray,
    missing_nodes: np.ndarray,
    count: int,
    minimum_spacing: int,
) -> list[int]:
    selected = [int(round(node[2])) for node in missing_nodes]
    selected = [index for index in selected if 0 <= index < len(pixel_counts)]
    selected = list(dict.fromkeys(selected))
    for index in np.argsort(pixel_counts)[::-1]:
        index = int(index)
        if pixel_counts[index] == 0:
            break
        if all(abs(index - existing) >= minimum_spacing for existing in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for index in np.argsort(pixel_counts)[::-1]:
            index = int(index)
            if pixel_counts[index] and index not in selected:
                selected.append(index)
            if len(selected) >= count:
                break
    return sorted(selected)


def write_contact_sheet(
    path: Path,
    overlays: list[np.ndarray],
    source_slices: list[int],
    slice_struts: list[set[int]],
    slice_nodes: list[set[int]],
) -> None:
    columns = 4
    rows = math.ceil(len(overlays) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.55 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, overlay, source_slice in zip(axes, overlays, source_slices):
        axis.imshow(overlay)
        node_note = f" · node {sorted(slice_nodes[source_slice])}" if slice_nodes[source_slice] else ""
        axis.set_title(f"CT z={source_slice} · {len(slice_struts[source_slice])} struts{node_note}")
        axis.set_axis_off()
    for axis in axes[len(overlays) :]:
        axis.set_axis_off()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_inline_preview(template_path: Path, output_path: Path, contact_sheet_path: Path) -> None:
    image = Image.open(contact_sheet_path).convert("RGB")
    image.thumbnail((1400, 1800), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=78, optimize=True, progressive=True)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")
    fragment = template_path.read_text().replace("__CONTACT_SHEET_DATA_URI__", data_uri)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(fragment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--registered-design", type=Path, required=True)
    parser.add_argument("--nominal-design", type=Path, required=True)
    parser.add_argument("--missing-struts-csv", type=Path, required=True)
    parser.add_argument("--heatmap-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strut-radius", type=float, default=4.0)
    parser.add_argument("--node-radius", type=float, default=8.0)
    parser.add_argument("--overlay-slices", type=int, default=24)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nominal = load_json(args.nominal_design)
    registered = load_json(args.registered_design)
    heatmap_summary = load_json(args.heatmap_summary)
    missing_ids = read_missing_strut_ids(args.missing_struts_csv)
    segments = registered_missing_segments(registered, missing_ids)
    missing_nodes = registered_missing_nodes(nominal, registered, heatmap_summary)

    with tifffile.TiffFile(args.ct) as source:
        shape = tuple(int(value) for value in source.series[0].shape)
        dtype = str(source.series[0].dtype)
    if len(shape) != 3:
        raise ValueError(f"Expected a ZYX CT stack, found shape {shape}")

    stamps, slice_struts, slice_nodes = build_slice_stamps(
        shape, segments, missing_nodes, args.strut_radius, args.node_radius
    )
    pixel_counts = np.zeros(shape[0], dtype=np.int64)

    def label_pages():
        for z_index in range(shape[0]):
            labels = render_label_slice(shape[1:], stamps[z_index])
            pixel_counts[z_index] = int(np.count_nonzero(labels))
            yield labels

    label_path = args.output_dir / "0_5_missing_geometry_labels.tif"
    tifffile.imwrite(
        label_path,
        label_pages(),
        shape=shape,
        dtype=np.uint8,
        bigtiff=True,
        compression="zlib",
        compressionargs={"level": 6},
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "labels": {"0": "background", "1": "missing strut", "2": "missing node"},
            "source_ct": str(args.ct.resolve()),
        },
    )

    selected = select_slices(pixel_counts, missing_nodes, args.overlay_slices, minimum_spacing=12)
    overlays: list[np.ndarray] = []
    with tifffile.TiffFile(args.ct) as source:
        for z_index in selected:
            ct_slice = source.pages[z_index].asarray()
            labels = render_label_slice(shape[1:], stamps[z_index])
            overlays.append(overlay_labels(ct_slice, labels))

    overlay_path = args.output_dir / "0_5_ct_defect_overlay_slices.tif"
    tifffile.imwrite(
        overlay_path,
        np.stack(overlays),
        bigtiff=True,
        compression="zlib",
        photometric="rgb",
        metadata={
            "axes": "ZYXS",
            "source_z_indices": selected,
            "legend": {"orange": "missing strut", "cyan": "missing node"},
            "source_ct": str(args.ct.resolve()),
        },
    )

    index_path = args.output_dir / "overlay_slice_index.csv"
    with index_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["overlay_page", "source_ct_z", "missing_strut_count", "missing_strut_ids", "missing_node_indices"])
        for page, z_index in enumerate(selected):
            writer.writerow(
                [
                    page,
                    z_index,
                    len(slice_struts[z_index]),
                    " ".join(map(str, sorted(slice_struts[z_index]))),
                    " ".join(map(str, sorted(slice_nodes[z_index]))),
                ]
            )

    preview_slices = selected[: min(12, len(selected))]
    preview_overlays = [overlays[selected.index(index)] for index in preview_slices]
    preview_path = args.output_dir / "overlay_contact_sheet.png"
    write_contact_sheet(preview_path, preview_overlays, preview_slices, slice_struts, slice_nodes)
    if args.html:
        write_inline_preview(
            Path(__file__).with_name("templates") / "missing_geometry_slices.fragment.html",
            args.html,
            preview_path,
        )

    summary = {
        "source_ct": str(args.ct.resolve()),
        "source_ct_shape_zyx": list(shape),
        "source_ct_dtype": dtype,
        "label_tiff": str(label_path.resolve()),
        "label_values": {"0": "background", "1": "missing strut", "2": "missing node"},
        "missing_strut_count": len(missing_ids),
        "missing_node_count": int(len(missing_nodes)),
        "strut_display_radius_voxels": args.strut_radius,
        "node_display_radius_voxels": args.node_radius,
        "overlay_tiff": str(overlay_path.resolve()),
        "overlay_source_z_indices": selected,
        "registered_missing_node_xyz": np.round(missing_nodes, 4).tolist(),
    }
    (args.output_dir / "tiff_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
