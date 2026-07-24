"""Deterministic per-scan exact-histogram Otsu replay for Part 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from part2_core.otsu import (
    deterministic_histogram as _deterministic_histogram,
    histogram_diagnostics,
    histogram_sha256,
    otsu_from_histogram,
)
from part2_core.volume import load_volume

from specimen_manifest import (
    REPOSITORY_ROOT,
    load_json,
    sha256_file,
    validate_manifest,
)


def load_ct_volume(path: Path, format_name: str) -> np.ndarray:
    """Compatibility adapter over the shared Part 2 volume loader."""
    view = load_volume(path)
    if view.format != format_name:
        raise ValueError(
            f"Manifest declares CT format {format_name!r}, found {view.format!r}"
        )
    return view.array


def deterministic_histogram(
    volume: np.ndarray,
    *,
    chunk_depth: int,
    edge_slices_excluded: int,
    encoding: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility adapter using the manifest's depth-based chunk setting."""
    plane_voxels = int(np.prod(volume.shape[1:], dtype=np.int64))
    return _deterministic_histogram(
        volume,
        chunk_voxels=max(1, int(chunk_depth)) * plane_voxels,
        edge_slices_excluded=edge_slices_excluded,
        encoding=encoding,
    )


def replay_manifest(
    manifest_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay a manifest's segmentation recipe and enforce its frozen result."""
    validate_manifest(manifest_path)
    manifest = load_json(manifest_path)
    artifact = manifest["inputs"]["ct"]
    ct_path = repository_root / artifact["path"]
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT input is unavailable: {ct_path}")
    actual_input_hash = sha256_file(ct_path)
    if actual_input_hash != artifact["sha256"]:
        raise ValueError(
            f"CT SHA-256 mismatch: expected {artifact['sha256']}, "
            f"found {actual_input_hash}"
        )

    started = time.perf_counter()
    metadata = manifest["inputs"]["ct_metadata"]
    recipe = manifest["analysis_parameters"]["segmentation"]
    volume = load_ct_volume(ct_path, metadata["format"])
    if list(volume.shape) != metadata["shape"]:
        raise ValueError(
            f"CT shape mismatch: expected {metadata['shape']}, found {volume.shape}"
        )
    histogram, encoding = deterministic_histogram(
        volume,
        chunk_depth=recipe["chunk_depth"],
        edge_slices_excluded=recipe["edge_slices_excluded"],
        encoding=recipe["histogram_encoding"],
    )
    threshold_bin, separability = otsu_from_histogram(histogram)
    result = histogram_diagnostics(histogram, threshold_bin, separability, recipe)
    if encoding["encoding"] == "native_uint16":
        threshold: int | float = threshold_bin
    else:
        threshold = (
            encoding["native_min"]
            + threshold_bin * encoding["native_units_per_bin"]
        )
    result["threshold"] = threshold
    result["threshold_histogram_bin"] = threshold_bin
    result["histogram_encoding"] = encoding
    result["elapsed_seconds"] = time.perf_counter() - started

    expected = manifest["derived"]["segmentation_result"]["values"]
    mismatches: list[str] = []
    exact_fields = (
        "threshold",
        "voxel_count",
        "foreground_voxel_count",
        "significant_modes",
        "histogram_sha256",
        "overall_pass",
    )
    for field in exact_fields:
        if result[field] != expected[field]:
            mismatches.append(
                f"{field}: expected {expected[field]!r}, found {result[field]!r}"
            )
    float_fields = (
        "foreground_fraction",
        "otsu_separability",
        "background_mean",
        "foreground_mean",
        "class_mean_separation_sigma",
    )
    for field in float_fields:
        if not math.isclose(
            result[field], expected[field], rel_tol=1e-12, abs_tol=1e-12
        ):
            mismatches.append(
                f"{field}: expected {expected[field]!r}, found {result[field]!r}"
            )
    if mismatches:
        raise ValueError("Segmentation replay mismatch:\n- " + "\n- ".join(mismatches))
    if not result["overall_pass"]:
        failed = [name for name, passed in result["gates"].items() if not passed]
        raise ValueError("Histogram rejection gates failed: " + ", ".join(failed))

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "exact_histogram_uint16.npy", histogram)
        with (output_dir / "histogram_report.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="root used to resolve manifest input paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optionally persist the regenerable histogram and report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = replay_manifest(
            args.manifest,
            repository_root=args.repository_root.resolve(),
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(
        "PASS "
        f"threshold={result['threshold']} "
        f"foreground={result['foreground_voxel_count']}/"
        f"{result['voxel_count']} "
        f"histogram_sha256={result['histogram_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
