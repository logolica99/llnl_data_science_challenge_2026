"""Classify all struts from compact axial profiles without changing connectivity.

Missing labels are preserved from the existing missing-strut candidate CSV.
Broken is a material-loss label: an otherwise material-bearing strut has a
substantial set of central axial slices below half of its own robust healthy
reference level. It therefore includes connected struts with a thin remaining
bridge as well as disconnected two-sided fragments.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_GLOB = "outputs/strut_node_connectivity_profiles_*/all_strut_axial_profiles.npz"
DEFAULT_MISSING = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/missing_strut_candidates.csv"
DEFAULT_ALL_NONCONNECTED = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv"
DEFAULT_TRUE_CANDIDATES = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/true_missing_struts.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-glob", default=DEFAULT_PROFILE_GLOB)
    parser.add_argument("--missing-labels", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--all-nonconnected", type=Path, default=DEFAULT_ALL_NONCONNECTED)
    parser.add_argument("--true-candidates", type=Path, default=DEFAULT_TRUE_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deficit-ratio", type=float, default=0.50)
    parser.add_argument("--minimum-deficit-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-deficit-run-slices", type=int, default=3)
    parser.add_argument("--minimum-collar-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-shared-component-voxels", type=int, default=500)
    return parser.parse_args()


def read_ids(path: Path) -> set[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {int(row["strut_id"]) for row in csv.DictReader(handle)}


def longest_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def load_profiles(profile_paths: list[Path]) -> dict[int, tuple[float, np.ndarray]]:
    profiles: dict[int, tuple[float, np.ndarray]] = {}
    for path in profile_paths:
        with np.load(path) as data:
            for strut_id, length, profile in zip(
                data["strut_id"],
                data["length_voxels"],
                data["axial_disk_foreground_fraction"],
                strict=True,
            ):
                key = int(strut_id)
                if key in profiles:
                    raise ValueError(f"Duplicate compact profile for strut {key}")
                profiles[key] = (float(length), profile[np.isfinite(profile)])
    return profiles


def load_metrics(profile_paths: list[Path]) -> dict[int, dict[str, dict[str, str]]]:
    metrics: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    for profile_path in profile_paths:
        metric_path = profile_path.with_name("connection_metrics.csv")
        with metric_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                strut_id = int(row["strut_id"])
                endpoint = row["endpoint"]
                if endpoint in metrics[strut_id]:
                    raise ValueError(f"Duplicate endpoint-{endpoint} metrics for strut {strut_id}")
                metrics[strut_id][endpoint] = row
    return metrics


def main() -> None:
    args = parse_args()
    if not 0 < args.deficit_ratio < 1:
        raise ValueError("--deficit-ratio must be between zero and one")
    if not 0 < args.minimum_deficit_fraction <= 1:
        raise ValueError("--minimum-deficit-fraction must be in (0, 1]")
    if args.minimum_deficit_run_slices < 1:
        raise ValueError("--minimum-deficit-run-slices must be positive")

    profile_paths = sorted(REPO_ROOT.glob(args.profile_glob))
    if not profile_paths:
        raise FileNotFoundError(f"No compact profile files match {args.profile_glob}")
    profiles = load_profiles(profile_paths)
    metrics = load_metrics(profile_paths)
    if set(profiles) != set(metrics):
        raise ValueError("Compact-profile IDs and metric IDs do not match")

    missing_ids = read_ids(args.missing_labels)
    intentional_crop_ids = read_ids(args.all_nonconnected).difference(read_ids(args.true_candidates))
    if not missing_ids.issubset(profiles):
        raise ValueError("Missing-label IDs are not all present in the compact profiles")

    feature_rows: list[dict[str, object]] = []
    for strut_id in sorted(profiles):
        length, profile = profiles[strut_id]
        z = np.linspace(0.0, length, len(profile), dtype=np.float64)
        core = profile[(z >= 0.2 * length) & (z <= 0.8 * length)]
        if len(core) == 0:
            raise ValueError(f"Strut {strut_id} has no central profile samples")
        reference = float(np.quantile(core, 0.9))
        smoothed_core = np.convolve(core, np.ones(3, dtype=np.float64) / 3.0, mode="valid")
        minimum_raw = float(core.min())
        minimum_smoothed = float(smoothed_core.min())
        minimum_relative = (
            minimum_smoothed / reference if reference > 0 else float("nan")
        )
        material_loss_fraction = float(
            np.clip(1.0 - core / reference, 0.0, 1.0).mean()
        ) if reference > 0 else 1.0
        deficient = core < (args.deficit_ratio * reference)
        deficit_fraction = float(deficient.mean())
        deficit_run = longest_run(deficient)
        endpoint_a = metrics[strut_id]["A"]
        endpoint_b = metrics[strut_id]["B"]
        minimum_collar = min(
            float(endpoint_a["collar_foreground_fraction"]),
            float(endpoint_b["collar_foreground_fraction"]),
        )
        minimum_shared = min(
            int(endpoint_a["shared_component_voxel_count_in_cuboid"]),
            int(endpoint_b["shared_component_voxel_count_in_cuboid"]),
        )
        endpoint_evidence = (
            minimum_collar >= args.minimum_collar_fraction
            and minimum_shared >= args.minimum_shared_component_voxels
        )
        material_loss = (
            deficit_fraction >= args.minimum_deficit_fraction
            or deficit_run >= args.minimum_deficit_run_slices
        )
        connected = endpoint_a["same_material_component_connects_a_to_b"] == "True"

        if strut_id in intentional_crop_ids:
            label = "intentional_edge_crop"
        elif strut_id in missing_ids:
            label = "missing"
        elif material_loss and endpoint_evidence:
            label = "broken"
        elif not connected:
            label = "not_connected_review"
        else:
            label = "connected_no_broken_evidence"

        feature_rows.append(
            {
                "strut_id": strut_id,
                "classification": label,
                "same_material_component_connects_a_to_b": connected,
                "length_voxels": length,
                "central_reference_foreground_fraction_p90": reference,
                "central_minimum_foreground_fraction": minimum_raw,
                "central_minimum_smoothed_foreground_fraction": minimum_smoothed,
                "central_minimum_relative_to_reference": minimum_relative,
                "central_material_loss_fraction": material_loss_fraction,
                "central_deficit_fraction": deficit_fraction,
                "longest_deficit_run_slices": deficit_run,
                "minimum_endpoint_collar_foreground_fraction": minimum_collar,
                "minimum_endpoint_shared_component_voxels": minimum_shared,
                "both_endpoint_segments_observed": endpoint_evidence,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(feature_rows[0])
    with (args.output_dir / "all_strut_material_loss_features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)
    with (args.output_dir / "broken_strut_candidates_all.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in feature_rows if row["classification"] == "broken")

    with args.true_candidates.open("r", newline="", encoding="utf-8") as handle:
        true_reader = csv.DictReader(handle)
        if not true_reader.fieldnames:
            raise ValueError(f"True-candidate CSV has no header: {args.true_candidates}")
        true_rows = list(true_reader)
        true_fieldnames = true_reader.fieldnames
    classifications = {str(row["strut_id"]): str(row["classification"]) for row in feature_rows}
    true_missing_broken = [
        row
        for row in true_rows
        if classifications.get(row["strut_id"]) in {"missing", "broken"}
    ]
    with (args.output_dir / "true_missing_broken_struts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=true_fieldnames)
        writer.writeheader()
        writer.writerows(true_missing_broken)

    labels = defaultdict(int)
    for row in feature_rows:
        labels[str(row["classification"])] += 1
    print(f"Classified {len(feature_rows)} struts from {len(profile_paths)} compact-profile checkpoints.")
    for label in sorted(labels):
        print(f"{label}: {labels[label]}")
    print(f"true_missing_broken: {len(true_missing_broken)}")


if __name__ == "__main__":
    main()
