#!/usr/bin/env python3
"""Emit manifest-ready metadata for NumPy and TIFF CT volumes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from volume_metadata import VolumeMetadataError, inspect_volumes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="root used to resolve and constrain paths",
    )
    parser.add_argument(
        "--header-only",
        action="store_true",
        help="read headers and hash files without decoding voxel values",
    )
    parser.add_argument(
        "--skip-hash",
        action="store_true",
        help="skip the streaming file hash for a metadata-only preview",
    )
    parser.add_argument(
        "--chunk-voxels",
        type=int,
        default=8 * 1024 * 1024,
        help="maximum voxels processed per statistics chunk",
    )
    parser.add_argument(
        "--retention",
        choices=("committed", "external", "regenerable"),
        default="external",
        help="retention policy emitted in the manifest artifact fragment",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = inspect_volumes(
            args.files,
            repository_root=args.repository_root,
            header_only=args.header_only,
            include_sha256=not args.skip_hash,
            chunk_voxels=args.chunk_voxels,
            retention=args.retention,
        )
    except (OSError, TypeError, ValueError, VolumeMetadataError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
