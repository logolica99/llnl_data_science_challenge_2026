#!/usr/bin/env python3
"""Register the smooth lattice CAD STL into the CT/TIFF coordinate frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cad_stl", type=Path)
    parser.add_argument("ideal_json", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("ct_tiff", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cell-size-mm", type=float, default=4.56)
    parser.add_argument(
        "--source-envelope-mm",
        type=float,
        default=20.9,
        help="Discard source triangles with any coordinate outside +/- this value",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    paths = [
        args.cad_stl.expanduser().resolve(strict=True),
        args.ideal_json.expanduser().resolve(strict=True),
        args.registered_json.expanduser().resolve(strict=True),
        args.ct_tiff.expanduser().resolve(strict=True),
    ]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cell_size_mm <= 0 or args.source_envelope_mm <= 0:
        raise ValueError("Cell size and source envelope must be positive")
    return *paths, output_dir


def load_junction_positions(path: Path) -> dict[int, np.ndarray]:
    data = json.loads(path.read_text())
    if "junctions" not in data or not isinstance(data["junctions"], list):
        raise ValueError(f"{path} does not contain a junctions list")
    return {
        int(item["id"]): np.asarray(item["position"], dtype=np.float64)
        for item in data["junctions"]
    }


def fit_json_registration(
    ideal: dict[int, np.ndarray], registered: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = sorted(set(ideal) & set(registered))
    if len(ids) < 4:
        raise ValueError("At least four matching junction IDs are required")
    source = np.stack([ideal[item_id] for item_id in ids])
    target = np.stack([registered[item_id] for item_id in ids])
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    linear = coefficients[:3].T
    translation = coefficients[3]
    residual = np.linalg.norm(design @ coefficients - target, axis=1)
    return linear, translation, residual, source


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform_triangles(
    triangles_mm: np.ndarray,
    axis_map: np.ndarray,
    model_center: np.ndarray,
    model_unit_mm: float,
    linear: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    oriented_mm = triangles_mm @ axis_map.T
    model_coordinates = oriented_mm / model_unit_mm + model_center
    return model_coordinates @ linear.T + translation


def write_stl(path: Path, triangles_xyz: np.ndarray) -> tuple[np.ndarray, int]:
    vertices = np.ascontiguousarray(triangles_xyz.reshape(-1, 3), dtype=np.float64)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    bounds = mesh.bounds.copy()
    face_count = len(mesh.faces)
    mesh.export(path, file_type="stl")
    return bounds, face_count


def write_overlay(
    path: Path,
    tiff_path: Path,
    triangles_voxel_xyz: np.ndarray,
) -> None:
    centroids = triangles_voxel_xyz.mean(axis=1)
    with tifffile.TiffFile(tiff_path) as tif:
        slice_count = len(tif.pages)
        requested = [slice_count // 4, slice_count // 2, (3 * slice_count) // 4]
        images = [tif.pages[index].asarray() for index in requested]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, image, slice_index in zip(axes, images, requested):
        lo, hi = np.percentile(image, (1, 99.8))
        axis.imshow(image, cmap="gray", vmin=lo, vmax=hi)
        near = np.abs(centroids[:, 2] - slice_index) <= 0.75
        points = centroids[near]
        axis.scatter(points[:, 0], points[:, 1], s=0.08, c="#ff3b30", alpha=0.45)
        axis.set_title(f"Qualitative CAD-centroid overlay — CT slice {slice_index}")
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cad_path, ideal_path, registered_path, tiff_path, output_dir = resolve_inputs(args)
    started = time.time()

    ideal = load_junction_positions(ideal_path)
    registered = load_junction_positions(registered_path)
    linear, translation, residual, ideal_points = fit_json_registration(ideal, registered)
    singular_values = np.linalg.svd(linear, compute_uv=False)
    registration_scale = float(np.mean(singular_values))
    model_unit_mm = args.cell_size_mm / 2.0
    voxel_size_mm = model_unit_mm / registration_scale
    model_center = (ideal_points.min(axis=0) + ideal_points.max(axis=0)) / 2.0

    # The full octet topology is cube-symmetric, so the JSON graph alone cannot
    # distinguish every signed axis permutation. This orientation was the best
    # CT-scored candidate, but missing-strut orientation still requires labeled
    # validation before it can be used for defect classification.
    axis_map = np.diag([1.0, -1.0, 1.0])

    source_mesh = trimesh.load_mesh(cad_path, process=False)
    source_triangles = np.asarray(source_mesh.vertices, dtype=np.float64).reshape(-1, 3, 3)
    keep = np.all(np.abs(source_triangles) <= args.source_envelope_mm, axis=(1, 2))
    filtered_triangles = source_triangles[keep]
    registered_voxel = transform_triangles(
        filtered_triangles,
        axis_map,
        model_center,
        model_unit_mm,
        linear,
        translation,
    )

    voxel_stl = output_dir / "brian_tran_cad_registered_voxel.stl"
    mm_stl = output_dir / "brian_tran_cad_registered_mm.stl"
    preview = output_dir / "brian_tran_cad_registration_overlay.png"
    voxel_bounds, voxel_faces = write_stl(voxel_stl, registered_voxel)
    registered_mm = registered_voxel * voxel_size_mm
    mm_bounds, mm_faces = write_stl(mm_stl, registered_mm)
    write_overlay(preview, tiff_path, registered_voxel)

    metadata = {
        "method": "smooth source CAD transformed by the exact ideal-JSON to registered-JSON affine fit",
        "source_cad_stl": str(cad_path),
        "ideal_json": str(ideal_path),
        "registered_json": str(registered_path),
        "ct_tiff": str(tiff_path),
        "cell_size_mm": args.cell_size_mm,
        "model_unit_mm": model_unit_mm,
        "model_center": model_center.tolist(),
        "source_stl_to_json_axis_map": axis_map.tolist(),
        "axis_map_selection": "best CT-scored cube-symmetry candidate; not defect-ground-truth validated",
        "json_to_tiff_linear_xyz": linear.tolist(),
        "json_to_tiff_translation_xyz": translation.tolist(),
        "json_fit_singular_values_voxel_per_model_unit": singular_values.tolist(),
        "json_fit_rms_residual_voxels": float(np.sqrt(np.mean(residual**2))),
        "json_fit_max_residual_voxels": float(residual.max()),
        "voxel_size_mm": voxel_size_mm,
        "source_faces": int(len(source_triangles)),
        "discarded_outlier_faces": int(np.count_nonzero(~keep)),
        "registered_faces": voxel_faces,
        "voxel_stl_bounds_xyz": voxel_bounds.tolist(),
        "millimetre_stl_bounds_xyz": mm_bounds.tolist(),
        "millimetre_stl_extents_xyz": np.ptp(mm_bounds, axis=0).tolist(),
        "limitations": [
            "The ideal and Brian registered JSONs have identical topology; this registration does not identify missing struts.",
            "The CAD represents design intent, not measured CT segmentation.",
            "Cube symmetry leaves the missing-strut orientation heuristic until labeled validation is available.",
            "The PNG plots near-slice triangle centroids, not exact mesh-plane intersections, and must not be interpreted as a defect classification map.",
        ],
        "outputs": {
            "voxel_registered_stl": str(voxel_stl),
            "millimetre_registered_stl": str(mm_stl),
            "registration_overlay": str(preview),
        },
        "sha256": {
            "voxel_registered_stl": sha256(voxel_stl),
            "millimetre_registered_stl": sha256(mm_stl),
            "registration_overlay": sha256(preview),
        },
        "elapsed_seconds": time.time() - started,
    }
    if voxel_faces != mm_faces:
        raise RuntimeError("Voxel and millimetre STL face counts differ")
    metadata_path = output_dir / "cad_registration_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
