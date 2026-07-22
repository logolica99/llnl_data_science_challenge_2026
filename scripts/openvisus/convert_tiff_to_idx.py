#!/usr/bin/env python3
"""Convert a multipage CT TIFF volume to an OpenVisus IDX dataset."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_openvisus() -> None:
    pkgs = _repo_root() / ".python_pkgs"
    if pkgs.is_dir():
        sys.path.insert(0, str(pkgs))
    try:
        import OpenVisus  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "OpenVisus is not installed. From the repo root run:\n"
            "  pip install --target .python_pkgs OpenVisus\n"
            "or: pip install OpenVisus"
        ) from exc


def convert_tiff_to_idx(
    tiff_path: Path,
    idx_path: Path,
    *,
    arco: str = "4mb",
    compress: bool = True,
    overwrite: bool = False,
) -> Path:
    _ensure_openvisus()
    import tifffile
    from OpenVisus import CreateIdx, Field, NormalizeArcoArg, StringTree

    tiff_path = tiff_path.resolve()
    idx_path = idx_path.resolve()
    out_dir = idx_path.parent

    if idx_path.exists() and not overwrite:
        print(f"IDX already exists: {idx_path} (use --overwrite to rebuild)")
        return idx_path

    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(tiff_path) as tif:
        series = tif.series[0]
        shape = series.shape  # Z, Y, X for this dataset
        dtype = series.dtype
        if len(shape) != 3:
            raise SystemExit(f"Expected a 3D volume, got shape {shape}")
        depth, height, width = (int(v) for v in shape)
        print(f"Source: {tiff_path}")
        print(f"Shape ZYX={shape}, dtype={dtype}")

        field = Field("data", str(dtype), "row_major")
        arco_bytes = NormalizeArcoArg(arco)
        if arco_bytes:
            bitsperblock = int(math.log2(arco_bytes // field.dtype.getByteSize()))
        else:
            bitsperblock = 16

        db = CreateIdx(
            url=str(idx_path),
            dims=[width, height, depth],
            fields=[field],
            bitsperblock=bitsperblock,
            arco=arco_bytes,
        )

        access = db.createAccessForBlockQuery(StringTree())
        access.disableWriteLocks()
        access.disableCompression()

        def slabs():
            # Stream page-by-page to avoid loading the full ~1GB volume.
            for i, page in enumerate(tif.pages):
                if i % 50 == 0 or i == depth - 1:
                    print(f"  writing slab {i + 1}/{depth}")
                arr = page.asarray()
                if arr.shape != (height, width):
                    raise RuntimeError(
                        f"Unexpected page shape {arr.shape}, expected {(height, width)}"
                    )
                yield arr

        t0 = time.time()
        db.writeSlabs(slabs(), access=access)
        print(f"Wrote IDX slabs in {time.time() - t0:.1f}s")

    if compress:
        print("Compressing dataset (zip)...")
        t1 = time.time()
        db.compressDataset(["zip"])
        print(f"Compression done in {time.time() - t1:.1f}s")

    print(f"IDX ready: {idx_path}")
    return idx_path


def main() -> None:
    default_tiff = (
        _repo_root()
        / "data/missing_struts/tif_stacks"
        / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
    )
    default_idx = (
        _repo_root()
        / "data/missing_struts/tif_stacks/openvisus_idx"
        / "0point5dash1"
        / "visus.idx"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiff", type=Path, default=default_tiff, help="Input multipage TIFF")
    parser.add_argument("--idx", type=Path, default=default_idx, help="Output visus.idx path")
    parser.add_argument("--arco", default="4mb", help="OpenVisus ARCO block size (e.g. 4mb)")
    parser.add_argument("--no-compress", action="store_true", help="Skip zip compression pass")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild IDX if it already exists")
    args = parser.parse_args()

    if not args.tiff.is_file():
        raise SystemExit(f"TIFF not found: {args.tiff}")

    convert_tiff_to_idx(
        args.tiff,
        args.idx,
        arco=args.arco,
        compress=not args.no_compress,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
