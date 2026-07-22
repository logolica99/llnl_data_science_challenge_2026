"""Register a nominal lattice JSON to a TIFF using a verified reference pair.

This workflow is appropriate when ``reference_tif`` is the same CT volume as
``tif`` and ``reference_registered_json`` contains the same lattice graph in
that CT voxel frame. The script verifies both conditions before writing output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tiff_signature(path: Path) -> tuple[tuple[int, ...], str, str]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(series.shape), str(series.dtype), series.axes


def junction_map(data: dict) -> dict[int, dict]:
    return {int(junction["id"]): junction for junction in data["junctions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tif", required=True, type=Path, help="Target CT TIFF")
    parser.add_argument("--nominal-json", required=True, type=Path)
    parser.add_argument("--reference-tif", required=True, type=Path)
    parser.add_argument("--reference-registered-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    args = parser.parse_args()

    target_hash = sha256(args.tif)
    reference_hash = sha256(args.reference_tif)
    if target_hash != reference_hash:
        raise SystemExit("Reference TIFF differs from target TIFF; refusing to transfer its registration.")
    if tiff_signature(args.tif) != tiff_signature(args.reference_tif):
        raise SystemExit("Reference TIFF metadata differs from target TIFF; refusing to transfer registration.")

    nominal = json.loads(args.nominal_json.read_text())
    reference = json.loads(args.reference_registered_json.read_text())
    nominal_by_id = junction_map(nominal)
    reference_by_id = junction_map(reference)
    if nominal_by_id.keys() != reference_by_id.keys():
        raise SystemExit("Nominal and registered reference JSONs have different junction IDs.")

    ids = sorted(nominal_by_id)
    source = np.asarray([nominal_by_id[item_id]["position"] for item_id in ids], dtype=float)
    target = np.asarray([reference_by_id[item_id]["position"] for item_id in ids], dtype=float)
    design = np.column_stack((source, np.ones(len(source))))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    errors = np.linalg.norm(design @ coefficients - target, axis=1)
    for junction in nominal["junctions"]:
        junction["position"] = (np.append(junction["position"], 1.0) @ coefficients).tolist()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(nominal, indent=2), encoding="utf-8")
    report = {
        "status": "registered_by_verified_reference_transfer",
        "target_tiff_sha256": target_hash,
        "reference_tiff_sha256": reference_hash,
        "tiff_signature": {"shape": tiff_signature(args.tif)[0], "dtype": tiff_signature(args.tif)[1], "axes": tiff_signature(args.tif)[2]},
        "junction_count": len(ids),
        "affine_coefficients": coefficients.tolist(),
        "mean_reference_fit_error_voxels": float(errors.mean()),
        "max_reference_fit_error_voxels": float(errors.max()),
    }
    args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Registered JSON written to {args.output_json}")
    print(f"Verified reference transfer. Mean fit error: {errors.mean():.3e} voxels")


if __name__ == "__main__":
    main()
