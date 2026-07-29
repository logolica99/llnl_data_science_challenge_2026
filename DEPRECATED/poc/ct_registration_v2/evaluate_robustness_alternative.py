#!/usr/bin/env python3
"""Image-space comparison of the baseline fit and one robustness alternative."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from registration_core import (
    SimilarityTransform,
    corridor_image_validation,
    load_design,
    load_json,
    refine_transform,
    sha256_file,
    write_json,
    write_registered_json,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct",
        type=Path,
        default=repo_root
        / "data/pacificvis/8x8x8 octet lattice with defects/"
        "five_defects_1200_xray_recon.npy",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=repo_root / "data/octet_truss_8x8x8/octet_truss_8x8x8.json",
    )
    parser.add_argument(
        "--fit-dir",
        type=Path,
        default=script_dir / "results/pacificvis_8x8x8_v2run",
    )
    parser.add_argument("--case", default="edt_2.25")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir
        / "results/pacificvis_8x8x8_v2run/edt_2p25_comparison",
    )
    return parser.parse_args()


def transform_from_payload(payload: dict[str, Any]) -> SimilarityTransform:
    return SimilarityTransform(
        scale=float(payload["scale"]),
        rotation=np.asarray(payload["rotation_matrix"], dtype=np.float64),
        translation=np.asarray(payload["translation"], dtype=np.float64),
    )


def compact_image_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "junction_mean_foreground_fraction": report["junctions"][
            "mean_5x5x5_foreground_fraction"
        ],
        "junction_any_foreground_fraction": report["junctions"][
            "nodes_with_any_foreground_fraction"
        ],
        "holdout_candidate_median_distance_voxels": report[
            "candidate_holdout"
        ]["median_distance_to_predicted_unique_node_voxels"],
        "holdout_candidate_p95_distance_voxels": report["candidate_holdout"][
            "p95_distance_to_predicted_unique_node_voxels"
        ],
        "corridor_median_foreground_fraction": report["corridors"][
            "median_foreground_fraction"
        ],
        "corridor_p10_foreground_fraction": report["corridors"][
            "p10_foreground_fraction"
        ],
        "complete_corridor_gap_fraction": report["corridors"][
            "edges_with_complete_axial_gap_fraction"
        ],
        "measured_strut_radius_voxels": report["corridors"][
            "measured_strut_radius_voxels"
        ],
        "spatial_bin_median_ranges": {
            axis: report["spatial_bias"][axis]["median_range"]
            for axis in "xyz"
        },
        "gates": report["gates"],
        "overall_pass": report["overall_pass"],
    }


def prediction_difference(
    source: np.ndarray,
    first: SimilarityTransform,
    second: SimilarityTransform,
) -> dict[str, float]:
    distances = np.linalg.norm(
        first.apply(source) - second.apply(source),
        axis=1,
    )
    return {
        "median_voxels": float(np.median(distances)),
        "p95_voxels": float(np.quantile(distances, 0.95)),
        "maximum_voxels": float(np.max(distances)),
    }


def main() -> int:
    args = parse_args()
    ct_path = args.ct.resolve()
    design_path = args.design.resolve()
    fit_dir = args.fit_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    design = load_design(design_path)
    config = load_json(fit_dir / "config.snapshot.json")
    fitted_payload = load_json(fit_dir / "fitted_transform.json")
    baseline = transform_from_payload(fitted_payload)
    threshold = float(fitted_payload["per_scan_threshold"])
    robustness = load_json(fit_dir / "robustness_report.json")
    matching = [case for case in robustness["cases"] if case["name"] == args.case]
    if len(matching) != 1:
        raise ValueError(f"Expected one robustness case named {args.case!r}")
    alternative_case = matching[0]
    alternative_raw = transform_from_payload(
        alternative_case.get(
            "pre_refinement_transform",
            alternative_case["transform"],
        )
    )
    evaluation_holdout = np.load(fit_dir / "holdout_candidates.npy")

    baseline_report, baseline_occupancy = corridor_image_validation(
        ct_path,
        design,
        baseline,
        threshold,
        evaluation_holdout,
        config,
    )
    alternative_raw_report, alternative_raw_occupancy = (
        corridor_image_validation(
            ct_path,
            design,
            alternative_raw,
            threshold,
            evaluation_holdout,
            config,
        )
    )

    alternative_refined = None
    refinement_history: list[dict[str, Any]] | None = None
    refinement_error = None
    alternative_refined_report = None
    alternative_refined_occupancy = None
    try:
        alternative_refined, refinement_history = refine_transform(
            ct_path,
            design.unique_positions,
            alternative_raw,
            threshold,
            config,
        )
        alternative_refined_report, alternative_refined_occupancy = (
            corridor_image_validation(
                ct_path,
                design,
                alternative_refined,
                threshold,
                evaluation_holdout,
                config,
            )
        )
    except (ValueError, RuntimeError) as error:
        refinement_error = f"{type(error).__name__}: {error}"

    write_json(output_dir / "baseline_image_validation.json", baseline_report)
    write_json(
        output_dir / "alternative_raw_image_validation.json",
        alternative_raw_report,
    )
    np.save(output_dir / "baseline_edge_occupancy.npy", baseline_occupancy)
    np.save(
        output_dir / "alternative_raw_edge_occupancy.npy",
        alternative_raw_occupancy,
    )
    write_registered_json(
        output_dir / "alternative_raw_registered.json",
        design,
        alternative_raw,
    )

    refined_payload = None
    if alternative_refined is not None and alternative_refined_report is not None:
        write_json(
            output_dir / "alternative_refined_image_validation.json",
            alternative_refined_report,
        )
        np.save(
            output_dir / "alternative_refined_edge_occupancy.npy",
            alternative_refined_occupancy,
        )
        write_registered_json(
            output_dir / "alternative_refined_registered.json",
            design,
            alternative_refined,
        )
        refined_payload = {
            "transform": alternative_refined.to_dict(),
            "refinement_history": refinement_history,
            "difference_from_baseline": prediction_difference(
                design.unique_positions,
                baseline,
                alternative_refined,
            ),
            "image_metrics": compact_image_metrics(
                alternative_refined_report
            ),
        }

    comparison = {
        "schema_version": 1,
        "purpose": (
            "No-ground-truth image-space discrimination between the baseline "
            "fit and a robustness alternative"
        ),
        "inputs": {
            "ct": {"path": str(ct_path), "sha256": sha256_file(ct_path)},
            "design": {
                "path": str(design_path),
                "sha256": sha256_file(design_path),
            },
            "ground_truth_registration": None,
            "evaluation_holdout": str(fit_dir / "holdout_candidates.npy"),
        },
        "case": args.case,
        "baseline": {
            "transform": baseline.to_dict(),
            "image_metrics": compact_image_metrics(baseline_report),
        },
        "alternative_raw": {
            "transform": alternative_raw.to_dict(),
            "difference_from_baseline": prediction_difference(
                design.unique_positions,
                baseline,
                alternative_raw,
            ),
            "image_metrics": compact_image_metrics(alternative_raw_report),
        },
        "alternative_refined": refined_payload,
        "refinement_error": refinement_error,
        "interpretation_boundary": (
            "This comparison can reject a transform that fits the CT image "
            "poorly; without a registered reference it cannot establish "
            "absolute transform error."
        ),
    }
    write_json(output_dir / "comparison.json", comparison)
    print(
        f"Wrote baseline and {args.case} image-space comparison to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
