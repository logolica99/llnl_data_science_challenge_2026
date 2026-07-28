"""Independent full-resolution CT localization of registered lattice nodes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json
from .volume import AXIS_MAPPING, load_volume

LOCALIZATION_SCHEMA_VERSION = "part2-node-localization/1.1.0"
DEFAULT_LOCALIZATION_CONFIG: dict[str, Any] = {
    "patch_radius_voxels": 10,
    "search_radius_voxels": 8.0,
    "maximum_shift_voxels": 8.0,
    "smoothing_sigma_voxels": 1.25,
    "mean_shift_bandwidth_voxels": 4.0,
    "mean_shift_max_iterations": 12,
    "mean_shift_tolerance_voxels": 0.05,
    "seed_perturbation_voxels": 2.0,
    "seed_cluster_radius_voxels": 2.0,
    "minimum_seed_consensus_fraction": 0.7,
    "minimum_candidate_support": 0.05,
    "minimum_relative_support_improvement": 0.01,
    "incident_sample_distances_voxels": [3.0, 5.0, 7.0],
    "core_support_weight": 0.7,
    "incident_support_weight": 0.3,
    "minimum_accepted_fraction": 0.95,
    "maximum_ambiguous_fraction": 0.05,
}


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_LOCALIZATION_CONFIG)
    if config:
        result.update(config)
    return result


def _localize_one(
    volume: np.ndarray,
    prediction_xyz: np.ndarray,
    incident_directions_xyz: np.ndarray,
    threshold: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    patch_radius = int(config["patch_radius_voxels"])
    search_radius = float(config["search_radius_voxels"])
    shape_xyz = np.asarray(volume.shape[::-1], dtype=np.int64)
    center = np.rint(prediction_xyz).astype(np.int64)
    low = np.maximum(center - patch_radius, 0)
    high = np.minimum(center + patch_radius + 1, shape_xyz)
    boundary_truncated = bool(
        np.any(low != center - patch_radius)
        or np.any(high != center + patch_radius + 1)
    )
    patch = np.asarray(
        volume[
            low[2] : high[2],
            low[1] : high[1],
            low[0] : high[0],
        ]
        >= threshold,
        dtype=bool,
    )
    if not patch.size or not patch.any():
        return prediction_xyz.copy(), {
            "accepted": False,
            "reason": "no_foreground",
            "localization_status": "fallback",
            "seed_consensus_fraction": 0.0,
            "stability_uncertainty_voxels": None,
            "coarse_support": 0.0,
            "candidate_support": 0.0,
            "selected_support": 0.0,
            "relative_support_improvement": 0.0,
            "shift_voxels": 0.0,
            "boundary_truncated": boundary_truncated,
        }
    smoothed = ndimage.gaussian_filter(
        patch.astype(np.float64),
        sigma=float(config["smoothing_sigma_voxels"]),
        mode="constant",
        cval=0.0,
    )
    z_index, y_index, x_index = np.indices(patch.shape)
    global_x = x_index + low[0]
    global_y = y_index + low[1]
    global_z = z_index + low[2]
    distance_squared = (
        (global_x - prediction_xyz[0]) ** 2
        + (global_y - prediction_xyz[1]) ** 2
        + (global_z - prediction_xyz[2]) ** 2
    )
    search = distance_squared <= search_radius**2
    coordinates_xyz = np.stack((global_x, global_y, global_z), axis=-1)
    bandwidth = float(config["mean_shift_bandwidth_voxels"])

    def mean_shift(start_xyz: np.ndarray) -> np.ndarray | None:
        position = np.asarray(start_xyz, dtype=np.float64)
        for _ in range(int(config["mean_shift_max_iterations"])):
            kernel_distance_squared = np.sum(
                (coordinates_xyz - position) ** 2,
                axis=-1,
            )
            weights = (
                smoothed**2
                * np.exp(-kernel_distance_squared / (2.0 * bandwidth**2))
                * search
            )
            total_weight = float(weights.sum())
            if total_weight <= 0.0:
                return None
            updated = np.sum(
                weights[..., None] * coordinates_xyz,
                axis=(0, 1, 2),
            ) / total_weight
            displacement = updated - prediction_xyz
            displacement_norm = float(np.linalg.norm(displacement))
            if displacement_norm > search_radius:
                updated = (
                    prediction_xyz
                    + displacement * (search_radius / displacement_norm)
                )
            if (
                float(np.linalg.norm(updated - position))
                <= float(config["mean_shift_tolerance_voxels"])
            ):
                return updated
            position = updated
        return position

    perturbation = float(config["seed_perturbation_voxels"])
    seed_offsets = perturbation * np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    solutions = [
        solution
        for offset in seed_offsets
        if (solution := mean_shift(prediction_xyz + offset)) is not None
    ]
    if not solutions:
        return prediction_xyz.copy(), {
            "accepted": False,
            "reason": "no_converged_seed",
            "localization_status": "fallback",
            "seed_consensus_fraction": 0.0,
            "stability_uncertainty_voxels": None,
            "coarse_support": 0.0,
            "candidate_support": 0.0,
            "selected_support": 0.0,
            "relative_support_improvement": 0.0,
            "shift_voxels": 0.0,
            "boundary_truncated": boundary_truncated,
        }
    solution_array = np.asarray(solutions, dtype=np.float64)
    pairwise = np.linalg.norm(
        solution_array[:, None, :] - solution_array[None, :, :],
        axis=2,
    )
    cluster_radius = float(config["seed_cluster_radius_voxels"])
    neighbor_counts = np.count_nonzero(pairwise <= cluster_radius, axis=1)
    medoid = int(np.argmax(neighbor_counts))
    consensus = pairwise[medoid] <= cluster_radius
    consensus_fraction = float(np.mean(consensus))
    location = mean_shift(np.median(solution_array[consensus], axis=0))
    if location is None:
        location = solution_array[medoid]
    consensus_distances = np.linalg.norm(
        solution_array[consensus] - location,
        axis=1,
    )
    stability_uncertainty = float(np.quantile(consensus_distances, 0.95))

    def support(position_xyz: np.ndarray) -> float:
        kernel_distance_squared = np.sum(
            (coordinates_xyz - position_xyz) ** 2,
            axis=-1,
        )
        kernel = np.exp(-kernel_distance_squared / (2.0 * bandwidth**2)) * search
        core_support = float(np.sum(smoothed * kernel) / max(float(kernel.sum()), 1e-12))
        incident_support = core_support
        if incident_directions_xyz.size:
            distances = np.asarray(
                config["incident_sample_distances_voxels"],
                dtype=np.float64,
            )
            samples_xyz = (
                position_xyz[None, None, :]
                + incident_directions_xyz[:, None, :] * distances[None, :, None]
            )
            sample_coordinates = np.vstack(
                (
                    samples_xyz[..., 2].ravel() - low[2],
                    samples_xyz[..., 1].ravel() - low[1],
                    samples_xyz[..., 0].ravel() - low[0],
                )
            )
            sampled = ndimage.map_coordinates(
                smoothed,
                sample_coordinates,
                order=1,
                mode="constant",
                cval=0.0,
            ).reshape(len(incident_directions_xyz), len(distances))
            incident_support = float(np.median(np.mean(sampled, axis=1)))
        return (
            float(config["core_support_weight"]) * core_support
            + float(config["incident_support_weight"]) * incident_support
        )

    coarse_support = support(prediction_xyz)
    candidate_support = support(location)
    relative_improvement = (
        (candidate_support - coarse_support) / max(coarse_support, 1e-12)
        if candidate_support > 0.0 or coarse_support > 0.0
        else 0.0
    )
    localization_status = (
        "localized"
        if relative_improvement
        >= float(config["minimum_relative_support_improvement"])
        else "stable_coarse"
    )
    selected = location if localization_status == "localized" else prediction_xyz.copy()
    selected_support = max(candidate_support, coarse_support)
    shift = float(np.linalg.norm(selected - prediction_xyz))
    reasons = []
    if consensus_fraction < float(config["minimum_seed_consensus_fraction"]):
        reasons.append("unstable_multistart")
    if shift > float(config["maximum_shift_voxels"]):
        reasons.append("shift_exceeds_limit")
    if selected_support < float(config["minimum_candidate_support"]):
        reasons.append("insufficient_ct_support")
    accepted = not reasons
    return (selected if accepted else prediction_xyz.copy()), {
        "accepted": accepted,
        "reason": "accepted" if accepted else ",".join(reasons),
        "localization_status": localization_status if accepted else "fallback",
        "seed_consensus_fraction": consensus_fraction,
        "stability_uncertainty_voxels": stability_uncertainty,
        "coarse_support": coarse_support,
        "candidate_support": candidate_support,
        "selected_support": selected_support,
        "relative_support_improvement": relative_improvement,
        "shift_voxels": shift,
        "boundary_truncated": boundary_truncated,
    }


def _incident_directions(graph: Any) -> list[np.ndarray]:
    """Return normalized registered-edge directions for every explicit node row."""

    directions: list[list[np.ndarray]] = [[] for _ in graph.node_ids]
    for first, second in graph.edge_node_rows:
        first_row, second_row = int(first), int(second)
        delta = graph.node_positions_xyz[second_row] - graph.node_positions_xyz[first_row]
        norm = float(np.linalg.norm(delta))
        if norm <= 0.0:
            continue
        unit = delta / norm
        directions[first_row].append(unit)
        directions[second_row].append(-unit)
    return [
        np.asarray(values, dtype=np.float64).reshape((-1, 3))
        for values in directions
    ]


def localize_lattice_nodes(
    ct_path: str | Path,
    registered_graph_path: str | Path,
    output_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    config: dict[str, Any] | None = None,
    registration_report_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recenter nodes independently and never collapse them to a global fit."""

    if registration_mode not in ("challenge_aligned_json", "autonomous_v2"):
        raise ValueError(f"Unsupported registration mode: {registration_mode}")
    if not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    merged = _config(config)
    volume = load_volume(ct_path)
    graph = load_lattice_json(registered_graph_path)
    incident_directions = _incident_directions(graph)
    localized = np.empty_like(graph.node_positions_xyz)
    records: list[dict[str, Any]] = []
    for row, (node_id, prediction) in enumerate(
        zip(graph.node_ids, graph.node_positions_xyz)
    ):
        location, details = _localize_one(
            volume.array,
            prediction,
            incident_directions[row],
            float(threshold),
            merged,
        )
        localized[row] = location
        records.append(
            {
                "node_id": int(node_id),
                "coarse_xyz": prediction.tolist(),
                "localized_xyz": location.tolist(),
                **details,
            }
        )

    accepted = np.asarray([record["accepted"] for record in records], dtype=bool)
    ambiguous = np.asarray(
        ["unstable_multistart" in record["reason"] for record in records],
        dtype=bool,
    )
    localized_nodes = np.asarray(
        [record["localization_status"] == "localized" for record in records],
        dtype=bool,
    )
    stable_coarse_nodes = np.asarray(
        [record["localization_status"] == "stable_coarse" for record in records],
        dtype=bool,
    )
    boundary = np.asarray(
        [record["boundary_truncated"] for record in records], dtype=bool
    )
    accepted_fraction = float(np.mean(accepted))
    ambiguous_fraction = float(np.mean(ambiguous))
    gates = {
        "accepted_fraction_sufficient": bool(
            accepted_fraction >= float(merged["minimum_accepted_fraction"])
        ),
        "ambiguity_fraction_within_limit": bool(
            ambiguous_fraction <= float(merged["maximum_ambiguous_fraction"])
        ),
        "all_localized_positions_finite": bool(np.isfinite(localized).all()),
    }
    if not gates["all_localized_positions_finite"] or not accepted.any():
        gate = "halt"
    elif not all(gates.values()) or not np.all(accepted) or np.any(boundary):
        gate = "manual_review"
    else:
        gate = "pass"

    localized_document = graph.document_with_positions(localized)
    localized_document["part2_provenance"] = {
        "schema_version": "part2-coordinate-provenance/1.0.0",
        "registration_mode": registration_mode,
        "input_registered_graph_sha256": graph.source_sha256,
        "ct_sha256": sha256_file(volume.path),
        "independent_positions_retained": True,
        "global_refit_performed": False,
        "config_sha256": sha256_json(merged),
    }
    graph_artifact = write_json_atomic(
        output_graph_path,
        localized_document,
        overwrite=overwrite,
    )
    persistent_graph_artifact = {
        key: value for key, value in graph_artifact.items() if key != "changed"
    }
    registration_report_hash = (
        sha256_file(registration_report_path)
        if registration_report_path is not None
        else None
    )
    absolute_registration_uncertainty = None
    absolute_uncertainty_source = "unavailable_in_challenge_aligned_json"
    if registration_report_path is not None and registration_mode == "autonomous_v2":
        registration_report = read_json_object(registration_report_path)
        bounded = registration_report.get("mode_details", {}).get(
            "bounded_robustness", {}
        )
        candidate = bounded.get("p95_prediction_spread_voxels")
        if isinstance(candidate, (int, float)) and math.isfinite(float(candidate)):
            absolute_registration_uncertainty = float(candidate)
            absolute_uncertainty_source = (
                "autonomous_registration_bounded_robustness_p95_prediction_spread"
            )
    hashes = {
        "ct_sha256": sha256_file(volume.path),
        "input_registered_graph_sha256": graph.source_sha256,
        "localized_graph_sha256": graph_artifact["sha256"],
    }
    if registration_report_hash:
        hashes["registration_report_sha256"] = registration_report_hash
    report = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate != "halt",
        "registration_mode": registration_mode,
        "threshold": float(threshold),
        "axis_mapping": AXIS_MAPPING,
        "counts": {
            **graph.counts,
            "accepted_nodes": int(np.count_nonzero(accepted)),
            "fallback_nodes": int(np.count_nonzero(~accepted)),
            "ambiguous_nodes": int(np.count_nonzero(ambiguous)),
            "localized_nodes": int(np.count_nonzero(localized_nodes)),
            "stable_coarse_nodes": int(np.count_nonzero(stable_coarse_nodes)),
            "boundary_truncated_nodes": int(np.count_nonzero(boundary)),
        },
        "localization": {
            "accepted_fraction": accepted_fraction,
            "ambiguous_fraction": ambiguous_fraction,
            "accepted_shift_voxels": {
                "median": (
                    float(
                        np.median(
                            [
                                record["shift_voxels"]
                                for record in records
                                if record["accepted"]
                            ]
                        )
                    )
                    if accepted.any()
                    else None
                ),
                "p95": (
                    float(
                        np.quantile(
                            [
                                record["shift_voxels"]
                                for record in records
                                if record["accepted"]
                            ],
                            0.95,
                        )
                    )
                    if accepted.any()
                    else None
                ),
            },
            "stability_uncertainty_voxels": {
                "median": (
                    float(
                        np.median(
                            [
                                record["stability_uncertainty_voxels"]
                                for record in records
                                if record["accepted"]
                                and record["stability_uncertainty_voxels"] is not None
                            ]
                        )
                    )
                    if accepted.any()
                    else None
                ),
                "p95": (
                    float(
                        np.quantile(
                            [
                                record["stability_uncertainty_voxels"]
                                for record in records
                                if record["accepted"]
                                and record["stability_uncertainty_voxels"] is not None
                            ],
                            0.95,
                        )
                    )
                    if accepted.any()
                    else None
                ),
                "kind": "deterministic_multistart_repeatability",
                "absolute_registration_accuracy_claimed": False,
            },
            "absolute_registration_uncertainty_voxels": (
                absolute_registration_uncertainty
            ),
            "absolute_registration_uncertainty_source": absolute_uncertainty_source,
            "search_radius_voxels": float(merged["search_radius_voxels"]),
            "estimator": "topology_supported_multistart_mean_shift",
            "independent_positions_retained": True,
            "global_refit_performed": False,
        },
        "gates": gates,
        "records": records,
        "artifacts": {
            "localized_graph": {
                **persistent_graph_artifact,
                "role": "independently_localized_lattice_graph",
                "retention": "regenerable",
            }
        },
        "hashes": hashes,
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "sealed_labels_read": False,
        },
        "warnings": (
            []
            if gate == "pass"
            else [
                f"{int(np.count_nonzero(~accepted))} nodes retained coarse coordinates"
            ]
        ),
    }
    report_artifact = write_json_atomic(
        output_report_path,
        report,
        overwrite=overwrite,
    )
    report["artifacts"]["localization_report"] = {
        **report_artifact,
        "role": "node_localization_report",
        "retention": "committed",
    }
    report["artifacts"]["localized_graph"]["changed"] = graph_artifact["changed"]
    report["hashes"]["localization_report_sha256"] = report_artifact["sha256"]
    return report
