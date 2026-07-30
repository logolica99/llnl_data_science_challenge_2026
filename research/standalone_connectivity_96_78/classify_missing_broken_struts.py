"""Classify failed connectivity candidates as missing, broken, or review.

This is downstream post-processing only. It does not alter CT thresholding,
cuboid extraction, or the A-to-B connectivity result. It reads the saved axial
foreground profiles for failed struts and should normally be run after
``postprocess_missing_struts.py`` so known intentional edge crops are excluded.

Definitions used by the baseline:
* missing: at most 10% of central (20%-80%) axial slices contain material;
* broken: substantial material independently reaches both endpoint collars, but
  no foreground component connects A to B;
* review: a failed connection not confidently matching either pattern.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/true_missing_struts.csv"
DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap"


def longest_run(values: np.ndarray, target: bool) -> int:
    longest = current = 0
    for value in values:
        if bool(value) is target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def profile_path_by_id(artifact_root: Path) -> dict[int, Path]:
    matches: dict[int, Path] = {}
    for path in artifact_root.glob("strut_node_connectivity_*/strut_*_cuboid.npz"):
        strut_id = int(path.name.removeprefix("strut_").removesuffix("_cuboid.npz"))
        if strut_id in matches:
            raise ValueError(f"Duplicate saved profile for strut {strut_id}: {matches[strut_id]} and {path}")
        matches[strut_id] = path
    return matches


def classify_profile(
    profile: np.ndarray,
    z: np.ndarray,
    row: dict[str, str],
    present_fraction: float,
    missing_max_present_fraction: float,
    min_collar_foreground_fraction: float,
    min_shared_component_voxels: int,
) -> dict[str, float | int | str]:
    central = (z >= 0.2 * z[-1]) & (z <= 0.8 * z[-1])
    values = profile[central]
    present = values >= present_fraction
    count = len(present)
    if count == 0:
        raise ValueError("Saved profile has no central axial slices")

    side_width = max(1, count // 3)
    left_present = float(present[:side_width].mean())
    right_present = float(present[-side_width:].mean())
    central_present = float(present.mean())
    empty_run = longest_run(present, target=False)
    a_collar = float(row["a_collar_foreground_fraction"])
    b_collar = float(row["b_collar_foreground_fraction"])
    a_shared_voxels = int(row["a_shared_component_voxel_count_in_cuboid"])
    b_shared_voxels = int(row["b_shared_component_voxel_count_in_cuboid"])
    both_endpoint_segments = (
        min(a_collar, b_collar) >= min_collar_foreground_fraction
        and min(a_shared_voxels, b_shared_voxels) >= min_shared_component_voxels
    )

    if central_present <= missing_max_present_fraction:
        classification = "missing"
    elif both_endpoint_segments:
        classification = "broken"
    else:
        classification = "review"

    return {
        "classification": classification,
        "central_slice_count": count,
        "central_material_slice_fraction": central_present,
        "a_side_material_slice_fraction": left_present,
        "b_side_material_slice_fraction": right_present,
        "longest_empty_run_slices": empty_run,
        "central_mean_foreground_fraction": float(values.mean()),
        "both_endpoint_segments_observed": both_endpoint_segments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Candidate CSV to classify.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT, help="Root containing saved cuboid NPZ files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--present-fraction", type=float, default=0.05, help="Axial disk fraction at which one slice counts as material-bearing.")
    parser.add_argument("--missing-max-present-fraction", type=float, default=0.10)
    parser.add_argument(
        "--min-collar-foreground-fraction",
        type=float,
        default=0.05,
        help="Minimum foreground fraction required in each endpoint collar for a broken label.",
    )
    parser.add_argument(
        "--min-shared-component-voxels",
        type=int,
        default=500,
        help="Minimum A- and B-side shared-component size for a broken label.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.present_fraction <= 1:
        raise ValueError("--present-fraction must be between zero and one")
    if not 0 <= args.missing_max_present_fraction <= 1:
        raise ValueError("--missing-max-present-fraction must be between zero and one")
    if not 0 <= args.min_collar_foreground_fraction <= 1:
        raise ValueError("--min-collar-foreground-fraction must be between zero and one")
    if args.min_shared_component_voxels < 1:
        raise ValueError("--min-shared-component-voxels must be positive")

    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.input}")
        candidate_rows = list(reader)
        base_fields = reader.fieldnames

    endpoint_fields = {
        "a_collar_foreground_fraction",
        "b_collar_foreground_fraction",
        "a_shared_component_voxel_count_in_cuboid",
        "b_shared_component_voxel_count_in_cuboid",
    }
    if missing := endpoint_fields.difference(base_fields):
        raise ValueError(
            "Input CSV is missing endpoint metrics. Run append_connectivity_metrics.py first: "
            + ", ".join(sorted(missing))
        )

    profiles = profile_path_by_id(args.artifact_root)
    classified: list[dict[str, object]] = []
    for row in candidate_rows:
        strut_id = int(row["strut_id"])
        if strut_id not in profiles:
            raise FileNotFoundError(f"No saved failed-strut profile for ID {strut_id}")
        with np.load(profiles[strut_id]) as data:
            metrics = classify_profile(
                data["axial_disk_foreground_fraction"],
                data["local_z_from_node_a_voxels"],
                row,
                args.present_fraction,
                args.missing_max_present_fraction,
                args.min_collar_foreground_fraction,
                args.min_shared_component_voxels,
            )
        classified.append({**row, **metrics})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = [
        "classification",
        "central_slice_count",
        "central_material_slice_fraction",
        "a_side_material_slice_fraction",
        "b_side_material_slice_fraction",
        "longest_empty_run_slices",
        "central_mean_foreground_fraction",
        "both_endpoint_segments_observed",
    ]
    write_csv(args.output_dir / "missing_broken_classification.csv", [*base_fields, *metric_fields], classified)
    for label, name in (("missing", "missing_strut_candidates.csv"), ("broken", "broken_strut_candidates.csv"), ("review", "review_strut_candidates.csv")):
        write_csv(args.output_dir / name, base_fields, [row for row in classified if row["classification"] == label])

    counts = Counter(str(row["classification"]) for row in classified)
    print(f"Classified candidates: {len(classified)}")
    for label in ("missing", "broken", "review"):
        print(f"{label}: {counts[label]}")
    print(f"Wrote: {args.output_dir / 'missing_broken_classification.csv'}")


if __name__ == "__main__":
    main()
