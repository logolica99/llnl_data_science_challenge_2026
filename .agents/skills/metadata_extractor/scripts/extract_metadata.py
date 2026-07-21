#!/usr/bin/env python3
"""Print basic metadata for a .npy volume, mask, or skeleton."""

import argparse
import sys

import numpy as np


def extract_metadata(path: str) -> None:
    array = np.load(path)
    print(f"path: {path}")
    print(f"shape: {array.shape}")
    print(f"dtype: {array.dtype}")
    print(f"min: {array.min()}")
    print(f"max: {array.max()}")
    print(f"mean: {array.mean()}")
    print(f"size: {array.size}")

    nonzero = int(np.count_nonzero(array))
    print(f"nonzero: {nonzero}")
    if array.size > 0:
        print(f"nonzero_fraction: {nonzero / array.size:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npy_path", help="Path to a .npy file")
    args = parser.parse_args()

    try:
        extract_metadata(args.npy_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
