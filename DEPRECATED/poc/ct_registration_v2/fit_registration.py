#!/usr/bin/env python3
"""Fit design-to-CT registration without access to ground-truth registration."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from registration_core import (
    coarse_initialization,
    compute_per_scan_threshold,
    corridor_image_validation,
    detect_ct_nodes,
    downstream_tolerance_gate,
    load_design,
    load_json,
    multistart_fit,
    refine_transform,
    run_robustness_suite,
    run_synthetic_suite,
    sha256_file,
    split_candidates,
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
        / "data/missing_struts/tif_stacks/"
        "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=repo_root / "data/missing_struts/octet_truss_9x9x9.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config.default.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results/current",
    )
    parser.add_argument(
        "--external-evidence",
        type=Path,
        help=(
            "Optional external-validation summary. Classification remains blocked "
            "unless this file exists and reports overall_pass=true."
        ),
    )
    return parser.parse_args()


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    ct_path = args.ct.resolve()
    design_path = args.design.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (ct_path, design_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_json(config_path)
    config_snapshot = output_dir / "config.snapshot.json"
    write_json(config_snapshot, config)
    design = load_design(design_path)
    print(
        f"Loaded design: {len(design.node_ids):,} node records, "
        f"{len(design.unique_positions):,} unique positions, "
        f"{len(design.strut_ids):,} struts",
        flush=True,
    )

    synthetic = run_synthetic_suite(design.unique_positions, config)
    synthetic_path = output_dir / "synthetic_recovery.json"
    write_json(synthetic_path, synthetic)
    print(
        f"Synthetic recovery: {synthetic['pass_fraction']:.1%} "
        f"({'PASS' if synthetic['overall_pass'] else 'FAIL'})",
        flush=True,
    )

    threshold, histogram, histogram_report = compute_per_scan_threshold(
        ct_path, config
    )
    np.save(output_dir / "exact_histogram_uint16.npy", histogram)
    histogram_path = output_dir / "histogram_report.json"
    write_json(histogram_path, histogram_report)
    print(
        f"Per-scan Otsu={threshold}; foreground="
        f"{histogram_report['foreground_voxel_count']:,} "
        f"({histogram_report['foreground_fraction']:.2%}); histogram PASS",
        flush=True,
    )

    detected, detection_metadata = detect_ct_nodes(
        ct_path, threshold, config
    )
    fit_candidates, holdout_candidates, fit_indices, holdout_indices = (
        split_candidates(
            detected,
            float(config["detection"]["candidate_holdout_fraction"]),
            int(config["random_seed"]),
        )
    )
    detected_path = output_dir / "detected_nodes.npy"
    fit_candidates_path = output_dir / "fit_candidates.npy"
    holdout_candidates_path = output_dir / "holdout_candidates.npy"
    np.save(detected_path, detected)
    np.save(fit_candidates_path, fit_candidates)
    np.save(holdout_candidates_path, holdout_candidates)
    np.save(output_dir / "fit_candidate_indices.npy", fit_indices)
    np.save(output_dir / "holdout_candidate_indices.npy", holdout_indices)
    detection_metadata["fit_candidate_count"] = int(len(fit_candidates))
    detection_metadata["holdout_candidate_count"] = int(
        len(holdout_candidates)
    )
    detection_metadata["holdout_fraction"] = float(
        config["detection"]["candidate_holdout_fraction"]
    )
    detection_path = output_dir / "detection_report.json"
    write_json(detection_path, detection_metadata)
    print(
        f"Detected {len(detected):,} candidates; fit={len(fit_candidates):,}, "
        f"held out={len(holdout_candidates):,}",
        flush=True,
    )

    base = coarse_initialization(
        design.unique_positions,
        detection_metadata["detection_foreground_bounds_xyz"],
    )
    icp_transform, multistart = multistart_fit(
        design.unique_positions,
        fit_candidates,
        base,
        config,
    )
    multistart_path = output_dir / "multistart_report.json"
    write_json(multistart_path, multistart)
    print(
        f"Multi-start ICP: objective={multistart['best_objective']:.4f}, "
        f"spread={multistart['near_optimal_p95_prediction_spread_voxels']:.4f} "
        f"({'PASS' if multistart['overall_pass'] else 'FAIL'})",
        flush=True,
    )

    fitted, refinement_history = refine_transform(
        ct_path,
        design.unique_positions,
        icp_transform,
        threshold,
        config,
    )
    print(
        f"CT-only transform frozen: scale={fitted.scale:.6f}, "
        f"rotation={fitted.rotation_deg:.5f} deg, "
        f"translation={fitted.translation.tolist()}",
        flush=True,
    )

    robustness = run_robustness_suite(
        ct_path,
        design,
        threshold,
        fit_candidates,
        detection_metadata,
        fitted,
        config,
    )
    robustness_path = output_dir / "robustness_report.json"
    write_json(robustness_path, robustness)
    print(
        f"End-to-end robustness: worst P95 shift="
        f"{robustness['maximum_case_p95_prediction_shift_voxels']:.3f} "
        f"({'PASS' if robustness['overall_pass'] else 'FAIL'})",
        flush=True,
    )

    image_validation, edge_occupancy = corridor_image_validation(
        ct_path,
        design,
        fitted,
        threshold,
        holdout_candidates,
        config,
    )
    image_validation_path = output_dir / "image_validation.json"
    edge_occupancy_path = output_dir / "edge_corridor_occupancy.npy"
    write_json(image_validation_path, image_validation)
    np.save(edge_occupancy_path, edge_occupancy)
    print(
        f"Image validation: median corridor="
        f"{image_validation['corridors']['median_foreground_fraction']:.3f}, "
        f"radius={image_validation['corridors']['measured_strut_radius_voxels']:.2f} "
        f"({'PASS' if image_validation['overall_pass'] else 'FAIL'})",
        flush=True,
    )

    downstream = downstream_tolerance_gate(
        multistart, robustness, image_validation, config
    )
    downstream_path = output_dir / "downstream_tolerance.json"
    write_json(downstream_path, downstream)

    transform_payload = {
        **fitted.to_dict(),
        "fit_inputs": {
            "ct": str(ct_path),
            "design": str(design_path),
            "ground_truth_registration": None,
        },
        "ground_truth_used_for_fit": False,
        "per_scan_threshold": threshold,
        "coarse_initialization": base.to_dict(),
        "multistart_best_pre_refinement": icp_transform.to_dict(),
        "full_resolution_refinement": refinement_history,
    }
    transform_path = output_dir / "fitted_transform.json"
    registered_path = output_dir / "our_registered.json"
    write_json(transform_path, transform_payload)
    write_registered_json(registered_path, design, fitted)

    internal_gates = {
        "histogram": histogram_report["overall_pass"],
        "synthetic_recovery": synthetic["overall_pass"],
        "multistart": multistart["overall_pass"],
        "robustness": robustness["overall_pass"],
        "independent_image_validation": image_validation["overall_pass"],
        "downstream_tolerance": downstream["overall_pass"],
    }
    external_evidence: dict[str, Any]
    if args.external_evidence:
        evidence_path = args.external_evidence.resolve()
        external_evidence = {
            "path": str(evidence_path),
            "present": evidence_path.is_file(),
            "overall_pass": False,
        }
        if evidence_path.is_file():
            evidence = load_json(evidence_path)
            external_evidence["overall_pass"] = bool(
                evidence.get("overall_pass", False)
            )
            external_evidence["sha256"] = sha256_file(evidence_path)
    else:
        external_evidence = {
            "path": None,
            "present": False,
            "overall_pass": False,
        }
    internal_pass = bool(all(internal_gates.values()))
    classification_allowed = bool(
        internal_pass and external_evidence["overall_pass"]
    )

    artifact_paths = [
        config_snapshot,
        synthetic_path,
        histogram_path,
        output_dir / "exact_histogram_uint16.npy",
        detected_path,
        fit_candidates_path,
        holdout_candidates_path,
        detection_path,
        multistart_path,
        robustness_path,
        transform_path,
        registered_path,
        image_validation_path,
        edge_occupancy_path,
        downstream_path,
    ]
    manifest = {
        "schema_version": 1,
        "pipeline": "ct_registration_v2",
        "fit_completed_before_ground_truth_access": True,
        "ground_truth_used_for_fit": False,
        "input_artifacts": {
            "ct": artifact_record(ct_path),
            "design": artifact_record(design_path),
            "config": artifact_record(config_path),
        },
        "per_scan_threshold": threshold,
        "internal_gates": internal_gates,
        "internal_pass": internal_pass,
        "external_evidence": external_evidence,
        "classification_allowed": classification_allowed,
        "trust_status": (
            "trusted_for_classification"
            if classification_allowed
            else (
                "external_validation_pending"
                if internal_pass
                else "internal_gates_failed"
            )
        ),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "artifacts": [artifact_record(path) for path in artifact_paths],
    }
    manifest_path = output_dir / "fit_manifest.json"
    write_json(manifest_path, manifest)
    # A completion marker is written last. The validator requires it.
    completion = {
        "fit_manifest": artifact_record(manifest_path),
        "fitted_transform": artifact_record(transform_path),
        "our_registered": artifact_record(registered_path),
        "ground_truth_used_for_fit": False,
        "fit_completed_before_ground_truth_access": True,
    }
    write_json(output_dir / "FIT_COMPLETE.json", completion)

    print(
        f"Internal gates: {'PASS' if internal_pass else 'FAIL'}; "
        f"classification_allowed={classification_allowed}; "
        f"trust_status={manifest['trust_status']}",
        flush=True,
    )
    print(f"Wrote CT-only artifacts to {output_dir}", flush=True)
    return 0 if internal_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
