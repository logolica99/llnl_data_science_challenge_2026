"""Deterministic challenge and autonomous-v2 lattice registration.

The numerical routines in this module are promoted from
``poc/ct_registration_v2/registration_core.py``.  No supplied aligned graph is
opened on the autonomous path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .artifacts import sha256_file, sha256_json, write_json_atomic
from .lattice import (
    compare_topology,
    graph_bounds,
    load_lattice_json,
    positions_in_volume,
)
from .otsu import replay_exact_otsu
from .volume import AXIS_MAPPING, load_volume

REGISTRATION_SCHEMA_VERSION = "part2-registration/1.0.0"
RegistrationMode = Literal["challenge_aligned_json", "autonomous_v2"]

DEFAULT_REGISTRATION_CONFIG: dict[str, Any] = {
    "random_seed": 20260723,
    "detection": {
        "downsample_factor": 2,
        "central_z_margin_fraction": 0.065,
        "edt_peak_threshold_downsampled_voxels": 2.0,
        "minimum_component_voxels": 2,
        "maximum_component_voxels": 999,
        "candidate_holdout_fraction": 0.2,
    },
    "fitting": {
        "icp_keep_fraction": 0.7,
        "icp_max_iterations": 60,
        "icp_convergence_epsilon": 1e-5,
        "scale_multipliers": [0.99, 1.0, 1.01],
        "rotation_perturbations_degrees_xyz": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        "near_optimal_objective_fraction": 0.05,
        "minimum_near_optimal_starts": 3,
        "maximum_multistart_p95_spread_voxels": 1.0,
        "minimum_converged_start_fraction": 0.5,
    },
    "gates": {
        "maximum_holdout_median_distance_voxels": 8.0,
        "minimum_candidate_to_unique_node_ratio": 0.25,
    },
    "synthetic": {
        "case_count": 12,
        "noise_sigma_voxels": 0.35,
        "missing_fraction": 0.2,
        "outlier_fraction": 0.25,
        "maximum_relative_scale_error": 0.01,
        "maximum_rotation_error_degrees": 0.5,
        "maximum_translation_error_voxels": 2.0,
        "minimum_pass_fraction": 0.9,
    },
}


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    @property
    def rotation_deg(self) -> float:
        cosine = np.clip((np.trace(self.rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": float(self.scale),
            "rotation_matrix": np.asarray(self.rotation).tolist(),
            "translation_xyz": np.asarray(self.translation).tolist(),
            "rotation_deg": self.rotation_deg,
            "convention": (
                "ct_xyz = scale * (design_xyz @ rotation_matrix.T) + translation_xyz"
            ),
        }


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    import copy

    result = copy.deepcopy(DEFAULT_REGISTRATION_CONFIG)
    if config is None:
        return result
    for section, value in config.items():
        if isinstance(value, dict) and isinstance(result.get(section), dict):
            result[section].update(value)
        else:
            result[section] = value
    return result


def solve_similarity(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> SimilarityTransform:
    """Least-squares 7-DOF similarity fit with reflection rejection."""

    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity inputs must be matching N x 3 arrays")
    if (
        len(source) < 4
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Similarity fitting requires at least four finite point pairs")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ np.diag(signs) @ right_transpose
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= 0:
        raise ValueError("Degenerate source points have zero variance")
    scale = float(np.sum(singular_values * signs) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity fit produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    delta = np.asarray(first) @ np.asarray(second).T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def coarse_initialization(
    source_xyz: np.ndarray,
    bounds_xyz: Iterable[Iterable[float]],
) -> SimilarityTransform:
    source = np.asarray(source_xyz, dtype=np.float64)
    low, high = np.asarray(bounds_xyz, dtype=np.float64)
    span = np.ptp(source, axis=0)
    valid = span > 0
    if not np.any(valid):
        raise ValueError("Cannot initialize from a zero-span design")
    scale = float(np.median((high[valid] - low[valid]) / span[valid]))
    rotation = np.eye(3)
    translation = (low + high) / 2.0 - scale * (rotation @ source.mean(axis=0))
    return SimilarityTransform(scale, rotation, translation)


def trimmed_icp(
    source_xyz: np.ndarray,
    detected_xyz: np.ndarray,
    initial: SimilarityTransform,
    *,
    keep_fraction: float,
    max_iterations: int,
    convergence_epsilon: float,
) -> tuple[SimilarityTransform, list[dict[str, Any]], float]:
    """Deterministic nearest-neighbor trimmed similarity ICP."""

    source = np.asarray(source_xyz, dtype=np.float64)
    detected = np.asarray(detected_xyz, dtype=np.float64)
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    if len(source) < 4 or len(detected) < 4:
        raise ValueError("ICP requires at least four source and target points")
    tree = cKDTree(detected)
    transform = initial
    previous_mean = math.inf
    history: list[dict[str, Any]] = []
    for iteration in range(int(max_iterations)):
        transformed = transform.apply(source)
        distances, indices = tree.query(transformed, k=1, workers=1)
        keep_count = max(4, int(math.ceil(len(distances) * keep_fraction)))
        order = np.argsort(distances, kind="stable")
        keep = order[:keep_count]
        mean_residual = float(np.mean(distances[keep]))
        history.append(
            {
                "iteration": iteration,
                "kept_pairs": keep_count,
                "trimmed_mean_residual_voxels": mean_residual,
                "all_pair_median_residual_voxels": float(np.median(distances)),
                "cutoff_voxels": float(distances[keep[-1]]),
            }
        )
        transform = solve_similarity(source[keep], detected[indices[keep]])
        if abs(previous_mean - mean_residual) < convergence_epsilon:
            break
        previous_mean = mean_residual
    final_distances, _ = tree.query(transform.apply(source), k=1, workers=1)
    keep_count = max(4, int(math.ceil(len(final_distances) * keep_fraction)))
    objective = float(np.mean(np.sort(final_distances, kind="stable")[:keep_count]))
    return transform, history, objective


def _multistarts(
    source_xyz: np.ndarray,
    base: SimilarityTransform,
    config: dict[str, Any],
) -> list[SimilarityTransform]:
    params = config["fitting"]
    target_center = base.apply(source_xyz).mean(axis=0)
    source_center = np.asarray(source_xyz).mean(axis=0)
    starts: list[SimilarityTransform] = []
    for multiplier in params["scale_multipliers"]:
        for angles in params["rotation_perturbations_degrees_xyz"]:
            rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
            scale = base.scale * float(multiplier)
            translation = target_center - scale * (rotation @ source_center)
            starts.append(SimilarityTransform(scale, rotation, translation))
    return starts


def multistart_fit(
    source_xyz: np.ndarray,
    detected_xyz: np.ndarray,
    base: SimilarityTransform,
    config: dict[str, Any] | None = None,
) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Run the stable v2 21-start fit and agreement gates."""

    merged = _merged_config(config)
    params = merged["fitting"]
    transforms: list[SimilarityTransform] = []
    entries: list[dict[str, Any]] = []
    for index, initial in enumerate(_multistarts(source_xyz, base, merged)):
        fitted, history, objective = trimmed_icp(
            source_xyz,
            detected_xyz,
            initial,
            keep_fraction=float(params["icp_keep_fraction"]),
            max_iterations=int(params["icp_max_iterations"]),
            convergence_epsilon=float(params["icp_convergence_epsilon"]),
        )
        transforms.append(fitted)
        entries.append(
            {
                "start_index": index,
                "objective": objective,
                "iterations": len(history),
                "transform": fitted.to_dict(),
            }
        )
    objectives = np.asarray([entry["objective"] for entry in entries])
    best_index = int(np.argmin(objectives))
    best = transforms[best_index]
    # A relative-only objective window collapses at exact/near-exact synthetic
    # fits because harmless SVD roundoff dominates a ~1e-15 best objective.
    limit = float(
        max(
            objectives[best_index]
            * (1.0 + float(params["near_optimal_objective_fraction"])),
            objectives[best_index] + float(params["icp_convergence_epsilon"]),
        )
    )
    near = np.flatnonzero(objectives <= limit)
    per_start_p95 = []
    all_shifts: list[float] = []
    for index in near:
        shifts = np.linalg.norm(
            transforms[int(index)].apply(source_xyz) - best.apply(source_xyz),
            axis=1,
        )
        per_start_p95.append(float(np.quantile(shifts, 0.95)))
        all_shifts.extend(shifts.tolist())
    p95_spread = float(np.quantile(all_shifts, 0.95)) if all_shifts else math.inf
    tolerance = float(params["maximum_multistart_p95_spread_voxels"])
    converged_fraction = (
        float(np.mean(np.asarray(per_start_p95) <= tolerance)) if per_start_p95 else 0.0
    )
    gates = {
        "enough_near_optimal_starts": bool(
            len(near) >= int(params["minimum_near_optimal_starts"])
        ),
        "near_optimal_starts_agree": bool(p95_spread <= tolerance),
        "converged_start_fraction_sufficient": bool(
            converged_fraction >= float(params["minimum_converged_start_fraction"])
        ),
    }
    return best, {
        "best_start_index": best_index,
        "best_objective": float(objectives[best_index]),
        "start_count": len(entries),
        "near_optimal_start_indices": near.tolist(),
        "near_optimal_p95_prediction_spread_voxels": p95_spread,
        "near_optimal_converged_fraction": converged_fraction,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
        "starts": entries,
    }


def split_candidates(
    candidates_xyz: np.ndarray,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates_xyz, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or len(candidates) < 5:
        raise ValueError("Candidate split requires at least five XYZ points")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(len(candidates))
    holdout_count = max(1, int(round(len(candidates) * holdout_fraction)))
    holdout_indices = np.sort(order[:holdout_count])
    fit_indices = np.sort(order[holdout_count:])
    return (
        candidates[fit_indices],
        candidates[holdout_indices],
        fit_indices,
        holdout_indices,
    )


def detect_ct_nodes(
    ct_path: str | Path,
    threshold: float,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect EDT node candidates on the configured factor-two CT sample."""

    merged = _merged_config(config)
    params = merged["detection"]
    factor = int(params["downsample_factor"])
    if factor < 1:
        raise ValueError("downsample_factor must be positive")
    volume = load_volume(ct_path)
    source_shape = np.asarray(volume.shape, dtype=int)
    mask = np.asarray(volume.array[::factor, ::factor, ::factor] >= threshold)
    margin = int(round(source_shape[0] * float(params["central_z_margin_fraction"])))
    z_start = math.ceil(margin / factor)
    z_stop = (source_shape[0] - margin - 1) // factor + 1
    detection_mask = mask[z_start:z_stop]
    if not detection_mask.any():
        raise ValueError("Thresholded central CT sample is empty")
    edt = ndimage.distance_transform_edt(detection_mask)
    high_radius = edt >= float(params["edt_peak_threshold_downsampled_voxels"])
    labels, component_count = ndimage.label(high_radius)
    sizes = np.bincount(labels.ravel())[1:]
    kept = (
        np.flatnonzero(
            (sizes >= int(params["minimum_component_voxels"]))
            & (sizes <= int(params["maximum_component_voxels"]))
        )
        + 1
    )
    centers_zyx = np.asarray(
        ndimage.center_of_mass(high_radius, labels, kept),
        dtype=np.float64,
    ).reshape(-1, 3)
    if not len(centers_zyx):
        raise ValueError("CT node detection produced no accepted EDT components")
    detected_xyz = centers_zyx[:, ::-1] * factor
    detected_xyz[:, 2] += z_start * factor
    return detected_xyz, {
        "source_shape_zyx": source_shape.tolist(),
        "axis_mapping": AXIS_MAPPING,
        "threshold": float(threshold),
        "downsample_factor": factor,
        "central_z_margin_full_resolution_voxels": margin,
        "all_component_count": int(component_count),
        "detected_node_count": int(len(detected_xyz)),
        "detected_bounds_xyz": {
            "minimum": detected_xyz.min(axis=0).tolist(),
            "maximum": detected_xyz.max(axis=0).tolist(),
        },
    }


def transform_errors(
    fitted: SimilarityTransform,
    expected: SimilarityTransform,
) -> dict[str, float]:
    return {
        "relative_scale": abs(fitted.scale / expected.scale - 1.0),
        "rotation_degrees": rotation_difference_deg(fitted.rotation, expected.rotation),
        "translation_norm_voxels": float(
            np.linalg.norm(fitted.translation - expected.translation)
        ),
    }


def run_synthetic_suite(
    source_xyz: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise autonomous fitting against seeded missing/outlier cases."""

    merged = _merged_config(config)
    params = merged["synthetic"]
    rng = np.random.default_rng(int(merged["random_seed"]) + 202)
    cases: list[dict[str, Any]] = []
    for case_index in range(int(params["case_count"])):
        rotation = Rotation.from_euler(
            "xyz", rng.uniform(-2.0, 2.0, size=3), degrees=True
        ).as_matrix()
        expected = SimilarityTransform(
            scale=float(rng.uniform(30.0, 50.0)),
            rotation=rotation,
            translation=rng.uniform(20.0, 80.0, size=3),
        )
        target = expected.apply(source_xyz)
        target += rng.normal(
            0.0, float(params["noise_sigma_voxels"]), size=target.shape
        )
        target = target[rng.random(len(target)) >= float(params["missing_fraction"])]
        outlier_count = int(round(len(target) * float(params["outlier_fraction"])))
        outliers = rng.uniform(
            np.quantile(target, 0.05, axis=0),
            np.quantile(target, 0.95, axis=0),
            size=(outlier_count, 3),
        )
        detected = np.concatenate((target, outliers), axis=0)
        base = coarse_initialization(
            source_xyz,
            [
                np.quantile(detected, 0.01, axis=0),
                np.quantile(detected, 0.99, axis=0),
            ],
        )
        fitted, diagnostics = multistart_fit(source_xyz, detected, base, merged)
        errors = transform_errors(fitted, expected)
        gates = {
            "scale": errors["relative_scale"]
            <= float(params["maximum_relative_scale_error"]),
            "rotation": errors["rotation_degrees"]
            <= float(params["maximum_rotation_error_degrees"]),
            "translation": errors["translation_norm_voxels"]
            <= float(params["maximum_translation_error_voxels"]),
            "multistart": diagnostics["overall_pass"],
        }
        cases.append(
            {
                "case": case_index,
                "errors": errors,
                "gates": gates,
                "pass": bool(all(gates.values())),
            }
        )
    pass_fraction = float(np.mean([case["pass"] for case in cases]))
    gate = bool(pass_fraction >= float(params["minimum_pass_fraction"]))
    return {
        "case_count": len(cases),
        "pass_fraction": pass_fraction,
        "minimum_pass_fraction": float(params["minimum_pass_fraction"]),
        "gates": {"synthetic_recovery_pass_fraction": gate},
        "overall_pass": gate,
        "cases": cases,
    }


def register_lattice_to_ct(
    nominal_graph_path: str | Path,
    output_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    mode: RegistrationMode,
    ct_path: str | Path | None = None,
    aligned_graph_path: str | Path | None = None,
    threshold: float | None = None,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Register a lattice using exactly one declared production mode."""

    if mode not in ("challenge_aligned_json", "autonomous_v2"):
        raise ValueError(f"Unsupported registration mode: {mode}")
    nominal = load_lattice_json(nominal_graph_path)
    merged = _merged_config(config)
    output_graph = Path(output_graph_path).expanduser().resolve()
    output_report = Path(output_report_path).expanduser().resolve()
    if output_graph == output_report:
        raise ValueError("Registration graph and report paths must differ")
    warnings: list[str] = []

    if mode == "challenge_aligned_json":
        if aligned_graph_path is None:
            raise ValueError("challenge_aligned_json mode requires aligned_graph_path")
        aligned = load_lattice_json(aligned_graph_path)
        topology = compare_topology(nominal, aligned)
        bounds_gate = True
        volume_hash: str | None = None
        if ct_path is not None:
            volume = load_volume(ct_path)
            in_bounds = positions_in_volume(aligned.node_positions_xyz, volume.shape)
            bounds_gate = bool(np.all(in_bounds))
            volume_hash = sha256_file(volume.path)
        gates = {**topology["gates"], "all_nodes_in_ct_bounds": bounds_gate}
        registered_positions = aligned.node_positions_xyz
        transform_payload = None
        mode_details: dict[str, Any] = {
            "authorized_aligned_graph_sha256": aligned.source_sha256,
            "topology": topology,
            "coordinate_bounds_xyz": graph_bounds(aligned),
        }
        if ct_path is None:
            warnings.append("CT bounds were not checked because ct_path was omitted")
    else:
        if aligned_graph_path is not None:
            raise ValueError(
                "autonomous_v2 forbids aligned_graph_path before fit artifacts freeze"
            )
        if ct_path is None:
            raise ValueError("autonomous_v2 mode requires ct_path")
        volume = load_volume(ct_path)
        threshold_report: dict[str, Any] | None = None
        if threshold is None:
            threshold_report, _ = replay_exact_otsu(volume.path)
            if not threshold_report["overall_pass"]:
                failed = sorted(
                    name
                    for name, passed in threshold_report["gates"].items()
                    if not passed
                )
                raise ValueError(
                    "Per-scan histogram rejected before registration: "
                    + ", ".join(failed)
                )
            threshold = float(threshold_report["threshold"])
        candidates, detection = detect_ct_nodes(volume.path, threshold, merged)
        unique_source = np.unique(nominal.node_positions_xyz, axis=0)
        fit, holdout, fit_indices, holdout_indices = split_candidates(
            candidates,
            float(merged["detection"]["candidate_holdout_fraction"]),
            int(merged["random_seed"]),
        )
        base = coarse_initialization(
            unique_source,
            [
                np.quantile(fit, 0.01, axis=0),
                np.quantile(fit, 0.99, axis=0),
            ],
        )
        transform, multistart = multistart_fit(unique_source, fit, base, merged)
        registered_positions = transform.apply(nominal.node_positions_xyz)
        unique_predictions = transform.apply(unique_source)
        holdout_distances, _ = cKDTree(unique_predictions).query(
            holdout, k=1, workers=1
        )
        in_bounds = positions_in_volume(registered_positions, volume.shape)
        candidate_ratio = len(candidates) / len(unique_source)
        gates = {
            **multistart["gates"],
            "candidate_population_sufficient": bool(
                candidate_ratio
                >= float(merged["gates"]["minimum_candidate_to_unique_node_ratio"])
            ),
            "holdout_supported": bool(
                np.median(holdout_distances)
                <= float(merged["gates"]["maximum_holdout_median_distance_voxels"])
            ),
            "all_nodes_in_ct_bounds": bool(np.all(in_bounds)),
            "fit_holdout_disjoint": bool(
                not np.intersect1d(fit_indices, holdout_indices).size
            ),
        }
        transform_payload = transform.to_dict()
        volume_hash = sha256_file(volume.path)
        mode_details = {
            "threshold": float(threshold),
            "threshold_report_sha256": (
                sha256_json(threshold_report) if threshold_report else None
            ),
            "detection": detection,
            "candidate_count": int(len(candidates)),
            "fit_candidate_count": int(len(fit)),
            "holdout_candidate_count": int(len(holdout)),
            "candidate_to_unique_node_ratio": float(candidate_ratio),
            "holdout_distance_voxels": {
                "median": float(np.median(holdout_distances)),
                "p95": float(np.quantile(holdout_distances, 0.95)),
                "maximum": float(np.max(holdout_distances)),
            },
            "multistart": multistart,
            "coordinate_bounds_xyz": {
                "minimum": registered_positions.min(axis=0).tolist(),
                "maximum": registered_positions.max(axis=0).tolist(),
            },
        }

    if not all(gates.values()):
        gate = "halt"
    elif mode == "challenge_aligned_json" and ct_path is None:
        gate = "manual_review"
    else:
        gate = "pass"
    registered_document = nominal.document_with_positions(registered_positions)
    graph_artifact = write_json_atomic(
        output_graph,
        registered_document,
        overwrite=overwrite,
    )
    hashes = {
        "nominal_graph_sha256": nominal.source_sha256,
        "registered_graph_sha256": graph_artifact["sha256"],
    }
    if mode == "challenge_aligned_json":
        hashes["aligned_graph_sha256"] = mode_details["authorized_aligned_graph_sha256"]
    if volume_hash:
        hashes["ct_sha256"] = volume_hash
    report = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "mode": mode,
        "gate": gate,
        "counts": nominal.counts,
        "axis_mapping": AXIS_MAPPING,
        "transform": transform_payload,
        "gates": gates,
        "overall_pass": gate != "halt",
        "mode_details": mode_details,
        "artifacts": {
            "registered_graph": {
                **graph_artifact,
                "role": "registered_lattice_graph",
                "retention": "regenerable",
            }
        },
        "hashes": hashes,
        "provenance": {
            "registration_mode": mode,
            "aligned_graph_used_for_fit": mode == "challenge_aligned_json",
            "sealed_labels_read": False,
            "config_sha256": sha256_json(merged),
        },
        "warnings": warnings,
    }
    report_artifact = write_json_atomic(
        output_report,
        report,
        overwrite=overwrite,
    )
    report["artifacts"]["registration_report"] = {
        **report_artifact,
        "role": "registration_report",
        "retention": "committed",
    }
    report["hashes"]["registration_report_sha256"] = report_artifact["sha256"]
    return report
