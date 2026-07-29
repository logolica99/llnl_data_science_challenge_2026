"""Independent full-resolution CT localization of registered lattice nodes."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json
from .otsu import replay_exact_otsu
from .volume import AXIS_MAPPING, load_volume

LOCALIZATION_SCHEMA_VERSION = "part2-node-localization/1.2.0"
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
    "minimum_primary_or_stable_coarse_fraction": 0.95,
    "maximum_fallback_fraction": 0.05,
    "maximum_ambiguous_fraction": 0.05,
    "maximum_rejected_fraction": 0.0,
    "maximum_boundary_limited_fraction": 0.0,
}

_FRACTION_CONFIG_FIELDS = (
    "minimum_seed_consensus_fraction",
    "minimum_candidate_support",
    "minimum_relative_support_improvement",
    "core_support_weight",
    "incident_support_weight",
    "minimum_primary_or_stable_coarse_fraction",
    "maximum_fallback_fraction",
    "maximum_ambiguous_fraction",
    "maximum_rejected_fraction",
    "maximum_boundary_limited_fraction",
)

_REASON_CODES = {
    "no_foreground": "LOCALIZATION_NO_FOREGROUND",
    "no_converged_seed": "LOCALIZATION_NO_CONVERGED_SEED",
    "unstable_multistart": "LOCALIZATION_AMBIGUOUS_MULTISTART",
    "shift_exceeds_limit": "LOCALIZATION_SHIFT_EXCEEDS_LIMIT",
    "insufficient_ct_support": "LOCALIZATION_INSUFFICIENT_CT_SUPPORT",
    "boundary_limited": "LOCALIZATION_BOUNDARY_LIMITED",
}

_LOCALIZATION_POLICY_FIELDS = (
    "patch_radius_voxels",
    "search_radius_voxels",
    "maximum_shift_voxels",
    "smoothing_sigma_voxels",
    "mean_shift_bandwidth_voxels",
    "mean_shift_max_iterations",
    "mean_shift_tolerance_voxels",
    "seed_perturbation_voxels",
    "seed_cluster_radius_voxels",
    "minimum_seed_consensus_fraction",
    "minimum_candidate_support",
    "minimum_relative_support_improvement",
    "incident_sample_distances_voxels",
    "core_support_weight",
    "incident_support_weight",
    "minimum_primary_or_stable_coarse_fraction",
    "maximum_fallback_fraction",
    "maximum_ambiguous_fraction",
    "maximum_rejected_fraction",
    "maximum_boundary_limited_fraction",
)

_SEGMENTATION_POLICY_FIELDS = (
    "method",
    "method_version",
    "comparison",
    "histogram_bins",
    "histogram_encoding",
    "edge_slices_excluded",
    "chunk_depth",
    "coarse_bins",
    "peak_smoothing_sigma_bins",
    "peak_prominence_fraction",
    "minimum_significant_peaks",
    "minimum_foreground_fraction",
    "maximum_foreground_fraction",
    "minimum_otsu_separability",
    "minimum_class_mean_separation_sigma",
)

_ARTIFACT_SCHEMA_VERSIONS = {
    "specimen_manifest": "2.1.0",
    "node_localization": "1.2.0",
    "registration_qa": "1.2.0",
    "per_strut_metrics": "1.0.0",
    "classified_struts": "1.0.0",
    "nde_report": "1.0.0",
}


def _validate_segmentation_policy(policy: dict[str, Any]) -> None:
    if set(policy) != set(_SEGMENTATION_POLICY_FIELDS):
        raise ValueError("analysis_parameters.segmentation must use its closed schema")
    if (
        policy.get("method") != "exact_histogram_otsu"
        or policy.get("method_version") != "2.0.0"
        or policy.get("comparison") != "value >= threshold"
        or policy.get("histogram_bins") != 65_536
        or policy.get("histogram_encoding")
        not in {"native_uint16", "full_volume_affine_uint16"}
    ):
        raise ValueError("Unsupported hashed segmentation policy")
    integer_minimums = {
        "edge_slices_excluded": 0,
        "chunk_depth": 1,
        "coarse_bins": 2,
        "minimum_significant_peaks": 1,
    }
    for name, minimum in integer_minimums.items():
        value = policy[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(f"segmentation.{name} must be an integer >= {minimum}")
    if (
        policy["coarse_bins"] > 65_536
        or 65_536 % policy["coarse_bins"] != 0
    ):
        raise ValueError("segmentation.coarse_bins must divide 65,536")
    fraction_fields = (
        "peak_prominence_fraction",
        "minimum_foreground_fraction",
        "maximum_foreground_fraction",
        "minimum_otsu_separability",
    )
    for name in fraction_fields:
        value = policy[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"segmentation.{name} must be a finite fraction in [0, 1]")
    for name in (
        "peak_smoothing_sigma_bins",
        "minimum_class_mean_separation_sigma",
    ):
        value = policy[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"segmentation.{name} must be finite and non-negative")
    if (
        float(policy["minimum_foreground_fraction"])
        > float(policy["maximum_foreground_fraction"])
    ):
        raise ValueError(
            "segmentation minimum_foreground_fraction exceeds its maximum"
        )


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_LOCALIZATION_CONFIG)
    if config:
        supplied = dict(config)
        legacy_minimum = supplied.pop("minimum_accepted_fraction", None)
        if legacy_minimum is not None:
            canonical = supplied.get("minimum_primary_or_stable_coarse_fraction")
            if canonical is not None and canonical != legacy_minimum:
                raise ValueError(
                    "Conflicting minimum_accepted_fraction and "
                    "minimum_primary_or_stable_coarse_fraction"
                )
            supplied["minimum_primary_or_stable_coarse_fraction"] = legacy_minimum
        unknown = sorted(set(supplied) - set(DEFAULT_LOCALIZATION_CONFIG))
        if unknown:
            raise ValueError(
                "Unknown localization config fields: " + ", ".join(unknown)
            )
        result.update(supplied)
    for name in _FRACTION_CONFIG_FIELDS:
        value = result[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    for name in ("patch_radius_voxels", "mean_shift_max_iterations"):
        value = result[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "search_radius_voxels",
        "maximum_shift_voxels",
        "smoothing_sigma_voxels",
        "mean_shift_bandwidth_voxels",
        "mean_shift_tolerance_voxels",
        "seed_cluster_radius_voxels",
    ):
        value = result[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    seed_perturbation = result["seed_perturbation_voxels"]
    if (
        not isinstance(seed_perturbation, (int, float))
        or isinstance(seed_perturbation, bool)
        or not math.isfinite(float(seed_perturbation))
        or float(seed_perturbation) < 0.0
    ):
        raise ValueError("seed_perturbation_voxels must be finite and non-negative")
    incident_distances = result["incident_sample_distances_voxels"]
    if (
        not isinstance(incident_distances, list)
        or not incident_distances
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0
            for value in incident_distances
        )
        or any(
            float(second) <= float(first)
            for first, second in zip(
                incident_distances, incident_distances[1:], strict=False
            )
        )
    ):
        raise ValueError(
            "incident_sample_distances_voxels must be strictly increasing"
        )
    if not math.isclose(
        float(result["core_support_weight"])
        + float(result["incident_support_weight"]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Localization support weights must sum to 1")
    return result


def _load_localization_policy(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    document = read_json_object(resolved)
    if (
        document.get("schema_version") != "2.1.0"
        or not isinstance(document.get("inputs"), dict)
        or not isinstance(document["inputs"].get("ct"), dict)
        or not isinstance(document["inputs"]["ct"].get("sha256"), str)
    ):
        raise ValueError(
            "Analysis-policy artifact must be a specimen manifest at schema 2.1.0"
        )
    analysis_parameters = document.get("analysis_parameters")
    if not isinstance(analysis_parameters, dict):
        raise ValueError(
            "Analysis-policy artifact must contain canonical analysis_parameters"
        )
    expected_hash = document.get("analysis_parameters_sha256")
    actual_hash = sha256_json(analysis_parameters)
    if expected_hash != actual_hash:
        raise ValueError(
            "Analysis-policy artifact analysis_parameters_sha256 does not match "
            "canonical analysis_parameters"
        )
    requested_scope = analysis_parameters.get("requested_analysis_scope")
    if requested_scope not in {"roi_screening", "direct_metrology"}:
        raise ValueError(
            "analysis_parameters.requested_analysis_scope must be roi_screening "
            "or direct_metrology"
        )
    top_level_scope = document.get("requested_analysis_scope")
    if top_level_scope is not None and top_level_scope != requested_scope:
        raise ValueError(
            "Analysis-policy artifact has contradictory requested_analysis_scope values"
        )
    registration_policy = analysis_parameters.get("registration")
    if (
        not isinstance(registration_policy, dict)
        or set(registration_policy) != {"mode", "local_recenter_required"}
        or registration_policy.get("mode")
        not in {"challenge_aligned_json", "autonomous_v2"}
        or registration_policy.get("local_recenter_required") is not True
    ):
        raise ValueError(
            "analysis_parameters.registration must be closed and require local recentering"
        )
    specimen_id = document.get("specimen_id")
    design_id = document.get("design_id")
    if not isinstance(specimen_id, str) or not specimen_id:
        raise ValueError("Analysis-policy artifact must identify specimen_id")
    if not isinstance(design_id, str) or not design_id:
        raise ValueError("Analysis-policy artifact must identify design_id")
    policy = analysis_parameters.get("localization_policy")
    if not isinstance(policy, dict):
        raise ValueError("analysis_parameters.localization_policy must be an object")
    if policy.get("schema_version") != "stage2-localization-policy/1.1.0":
        raise ValueError(
            "Unsupported analysis_parameters.localization_policy schema_version"
        )
    if set(policy) != {"schema_version", *_LOCALIZATION_POLICY_FIELDS}:
        raise ValueError(
            "analysis_parameters.localization_policy must use its closed schema"
        )
    missing = [name for name in _LOCALIZATION_POLICY_FIELDS if name not in policy]
    if missing:
        raise ValueError(
            "analysis_parameters.localization_policy is missing: "
            + ", ".join(missing)
        )
    segmentation_policy = analysis_parameters.get("segmentation")
    if not isinstance(segmentation_policy, dict):
        raise ValueError("analysis_parameters.segmentation must be an object")
    _validate_segmentation_policy(segmentation_policy)
    artifact_versions = analysis_parameters.get("artifact_schema_versions")
    if artifact_versions != _ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError("Hashed artifact_schema_versions are incompatible")
    _config({name: policy[name] for name in _LOCALIZATION_POLICY_FIELDS})
    return {
        "source_artifact_path": str(resolved),
        "source_artifact_sha256": sha256_file(resolved),
        "analysis_parameters_sha256": actual_hash,
        "localization_policy_sha256": sha256_json(policy),
        "requested_analysis_scope": requested_scope,
        "registration_mode": registration_policy["mode"],
        "specimen_id": specimen_id,
        "design_id": design_id,
        "declared_ct_sha256": document["inputs"]["ct"]["sha256"],
        "segmentation_policy_sha256": sha256_json(segmentation_policy),
        "segmentation_policy": dict(segmentation_policy),
        "policy": {name: policy[name] for name in _LOCALIZATION_POLICY_FIELDS},
    }


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
    if boundary_truncated:
        reasons.append("boundary_limited")
    accepted = not reasons
    rejected_status = (
        "ambiguous" if "unstable_multistart" in reasons else "fallback"
    )
    return (selected if accepted else prediction_xyz.copy()), {
        "accepted": accepted,
        "reason": "accepted" if accepted else ",".join(reasons),
        "localization_status": localization_status if accepted else rejected_status,
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


def _quality_reason_codes(record: dict[str, Any]) -> list[str]:
    reasons = {
        item
        for item in str(record.get("reason", "")).split(",")
        if item and item != "accepted"
    }
    if record.get("boundary_truncated"):
        reasons.add("boundary_limited")
    return sorted(_REASON_CODES.get(item, f"LOCALIZATION_{item.upper()}") for item in reasons)


def _decorate_node_quality(
    record: dict[str, Any],
    *,
    input_registered_graph_sha256: str,
) -> None:
    boundary_limited = bool(record.get("boundary_truncated"))
    accepted = bool(record.get("accepted")) and not boundary_limited
    raw_status = str(record.get("localization_status", "fallback"))
    reason_codes = _quality_reason_codes(record)
    if accepted and raw_status == "localized":
        match_class = "primary"
    elif accepted and raw_status == "stable_coarse":
        match_class = "stable_coarse"
    elif raw_status == "ambiguous" or "LOCALIZATION_AMBIGUOUS_MULTISTART" in reason_codes:
        match_class = "ambiguous"
    else:
        match_class = "fallback"

    fallback_used = match_class in {"fallback", "ambiguous"}
    low_confidence = fallback_used
    rejected = not np.isfinite(
        np.asarray(record.get("localized_xyz", []), dtype=np.float64)
    ).all()
    quality_flags = set(reason_codes)
    if match_class == "stable_coarse":
        quality_flags.add("STABLE_COARSE_ASSIGNMENT")
    elif match_class == "fallback":
        quality_flags.add("FALLBACK_ASSIGNMENT")
    elif match_class == "ambiguous":
        quality_flags.add("AMBIGUOUS_ASSIGNMENT")
    if not accepted:
        quality_flags.add("LOW_CONFIDENCE_ASSIGNMENT")

    record.update(
        {
            "accepted": accepted,
            "match_class": match_class,
            "primary_match": match_class == "primary",
            "low_confidence": low_confidence,
            "rejected": rejected,
            "rejected_or_low_confidence": rejected or low_confidence,
            "boundary_limited": boundary_limited,
            "assigned_coordinate_source": (
                "independent_ct_localization"
                if match_class == "primary"
                else "registered_coarse"
            ),
            "quality_flags": sorted(quality_flags),
            "reason_codes": reason_codes,
            "usable_for_roi_screening": not boundary_limited
            and match_class != "ambiguous",
            "usable_for_direct_metrology": accepted and not boundary_limited,
            "fallback_provenance": {
                "used": fallback_used,
                "coordinate_source": "registered_coarse" if fallback_used else None,
                "input_registered_graph_sha256": (
                    input_registered_graph_sha256 if fallback_used else None
                ),
                "reason_codes": reason_codes if fallback_used else [],
            },
        }
    )


def _edge_quality_records(graph: Any, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node_id = {int(record["node_id"]): record for record in records}
    result: list[dict[str, Any]] = []
    for edge_id, endpoint_ids in zip(graph.edge_ids, graph.edge_node_ids, strict=True):
        endpoint_records = [by_node_id[int(identifier)] for identifier in endpoint_ids]
        endpoint_classes = [str(record["match_class"]) for record in endpoint_records]
        if "ambiguous" in endpoint_classes:
            match_class = "ambiguous"
        elif "fallback" in endpoint_classes:
            match_class = "fallback"
        elif "stable_coarse" in endpoint_classes:
            match_class = "stable_coarse"
        else:
            match_class = "primary"
        quality_flags = {
            flag
            for record in endpoint_records
            for flag in record.get("quality_flags", [])
        }
        if match_class != "primary":
            quality_flags.add(f"{match_class.upper()}_ENDPOINT_QUALITY")
        result.append(
            {
                "edge_id": int(edge_id),
                "endpoint_node_ids": [int(identifier) for identifier in endpoint_ids],
                "endpoint_match_classes": endpoint_classes,
                "match_class": match_class,
                "quality_flags": sorted(quality_flags),
                "fallback_endpoint_node_ids": [
                    int(record["node_id"])
                    for record in endpoint_records
                    if record["match_class"] in {"fallback", "ambiguous"}
                ],
                "boundary_limited": any(
                    bool(record["boundary_limited"]) for record in endpoint_records
                ),
                "rejected_or_low_confidence": any(
                    bool(record["rejected_or_low_confidence"])
                    for record in endpoint_records
                ),
                "rejected": any(
                    bool(record["rejected"]) for record in endpoint_records
                ),
                "low_confidence": any(
                    bool(record["low_confidence"]) for record in endpoint_records
                ),
                "usable_for_roi_screening": all(
                    bool(record["usable_for_roi_screening"])
                    for record in endpoint_records
                ),
                "usable_for_direct_metrology": all(
                    bool(record["usable_for_direct_metrology"])
                    for record in endpoint_records
                ),
            }
        )
    return result


def localize_lattice_nodes(
    ct_path: str | Path,
    registered_graph_path: str | Path,
    output_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    analysis_policy_artifact_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    registration_report_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Recenter nodes independently and never collapse them to a global fit."""

    if registration_mode not in ("challenge_aligned_json", "autonomous_v2"):
        raise ValueError(f"Unsupported registration mode: {registration_mode}")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("threshold must be finite")
    policy_source = (
        _load_localization_policy(analysis_policy_artifact_path)
        if analysis_policy_artifact_path is not None
        else None
    )
    if policy_source is not None and registration_report_path is None:
        raise ValueError(
            "Hashed-policy localization requires its registration report artifact"
        )
    supplied_config = dict(config or {})
    if policy_source is not None:
        if registration_mode != policy_source["registration_mode"]:
            raise ValueError(
                "registration_mode does not match the hashed specimen manifest"
            )
        unknown_config = sorted(
            set(supplied_config) - set(_LOCALIZATION_POLICY_FIELDS)
        )
        if unknown_config:
            raise ValueError(
                "Free-form localization config contains fields outside the closed "
                "hashed policy: " + ", ".join(unknown_config)
            )
        policy_config = dict(policy_source["policy"])
        for name, value in policy_config.items():
            if name in supplied_config and supplied_config[name] != value:
                raise ValueError(
                    f"Free-form localization config conflicts with hashed policy: {name}"
                )
        supplied_config = {**supplied_config, **policy_config}
    merged = _config(supplied_config)
    volume = load_volume(ct_path)
    ct_sha256 = sha256_file(volume.path)
    segmentation_binding: dict[str, Any] | None = None
    if policy_source is not None:
        if ct_sha256 != policy_source["declared_ct_sha256"]:
            raise ValueError("CT SHA-256 does not match the hashed specimen manifest")
        segmentation_policy = dict(policy_source["segmentation_policy"])
        otsu_recipe = {
            "histogram_encoding": segmentation_policy["histogram_encoding"],
            "edge_slices_excluded": segmentation_policy["edge_slices_excluded"],
            "chunk_voxels": int(segmentation_policy["chunk_depth"])
            * int(volume.shape[1])
            * int(volume.shape[2]),
            "coarse_bins": segmentation_policy["coarse_bins"],
            "peak_smoothing_sigma_bins": segmentation_policy[
                "peak_smoothing_sigma_bins"
            ],
            "peak_prominence_fraction": segmentation_policy[
                "peak_prominence_fraction"
            ],
            "minimum_significant_peaks": segmentation_policy[
                "minimum_significant_peaks"
            ],
            "minimum_foreground_fraction": segmentation_policy[
                "minimum_foreground_fraction"
            ],
            "maximum_foreground_fraction": segmentation_policy[
                "maximum_foreground_fraction"
            ],
            "minimum_otsu_separability": segmentation_policy[
                "minimum_otsu_separability"
            ],
            "minimum_class_mean_separation_sigma": segmentation_policy[
                "minimum_class_mean_separation_sigma"
            ],
        }
        exact_otsu, _ = replay_exact_otsu(volume, recipe=otsu_recipe)
        if not exact_otsu.get("overall_pass"):
            raise ValueError("Hashed exact-Otsu segmentation gates did not pass")
        if float(threshold) != float(exact_otsu["threshold"]):
            raise ValueError("Localization threshold does not match exact Otsu replay")
        segmentation_binding = {
            "method": exact_otsu["method"],
            "method_version": exact_otsu["method_version"],
            "threshold": exact_otsu["threshold"],
            "threshold_comparison": exact_otsu["threshold_comparison"],
            "ct_sha256": ct_sha256,
            "segmentation_policy_sha256": policy_source[
                "segmentation_policy_sha256"
            ],
            "overall_pass": True,
        }
    graph = load_lattice_json(registered_graph_path)
    registration_report_hash = None
    registration_report_artifact: dict[str, Any] | None = None
    registration_report: dict[str, Any] | None = None
    if registration_report_path is not None:
        resolved_registration_report = (
            Path(registration_report_path).expanduser().resolve()
        )
        resolved_output_report = Path(output_report_path).expanduser().resolve()
        if resolved_registration_report.parent != resolved_output_report.parent:
            raise ValueError(
                "Registration and localization reports must share the bounded "
                "Stage 2 registration directory"
            )
        bounded_root = Path(
            os.path.commonpath(
                [
                    str(Path(volume.path).resolve()),
                    str(Path(registered_graph_path).expanduser().resolve()),
                    str(Path(output_graph_path).expanduser().resolve()),
                    str(resolved_output_report),
                    *(
                        [str(Path(analysis_policy_artifact_path).expanduser().resolve())]
                        if analysis_policy_artifact_path is not None
                        else []
                    ),
                ]
            )
        )
        if (
            bounded_root == Path(bounded_root.anchor)
            or not resolved_registration_report.is_relative_to(bounded_root)
        ):
            raise ValueError("Registration report path is outside the bounded run root")
        registration_report_hash = sha256_file(resolved_registration_report)
        registration_report = read_json_object(resolved_registration_report)
        report_hashes = registration_report.get("hashes", {})
        if (
            registration_report.get("schema_version")
            != "part2-registration/1.0.0"
            or registration_report.get("gate") != "pass"
        ):
            raise ValueError("Registration report schema or gate is incompatible")
        if report_hashes.get("registered_graph_sha256") != graph.source_sha256:
            raise ValueError(
                "Registration report registered_graph_sha256 does not match "
                "the localization input graph"
            )
        if report_hashes.get("ct_sha256") != ct_sha256:
            raise ValueError(
                "Registration report ct_sha256 does not match the localization CT"
            )
        if registration_report.get("mode") != registration_mode:
            raise ValueError(
                "Registration report mode does not match localization registration_mode"
            )
        registration_report_artifact = {
            "path": str(resolved_registration_report),
            "sha256": registration_report_hash,
            "role": "registration_report",
            "retention": "committed",
        }
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
        details = {
            **details,
            "accepted": bool(details.get("accepted")),
            "boundary_truncated": bool(details.get("boundary_truncated")),
        }
        if details.get("boundary_truncated") and details.get("accepted"):
            location = prediction.copy()
            details = {
                **details,
                "accepted": False,
                "reason": "boundary_limited",
                "localization_status": "fallback",
                "shift_voxels": 0.0,
            }
        localized[row] = location
        record = {
            "node_id": int(node_id),
            "coarse_xyz": prediction.tolist(),
            "localized_xyz": location.tolist(),
            **details,
        }
        _decorate_node_quality(
            record,
            input_registered_graph_sha256=graph.source_sha256,
        )
        records.append(record)

    accepted = np.asarray([record["accepted"] for record in records], dtype=bool)
    ambiguous = np.asarray(
        [record["match_class"] == "ambiguous" for record in records],
        dtype=bool,
    )
    primary_nodes = np.asarray(
        [record["match_class"] == "primary" for record in records],
        dtype=bool,
    )
    stable_coarse_nodes = np.asarray(
        [record["match_class"] == "stable_coarse" for record in records],
        dtype=bool,
    )
    fallback_nodes = np.asarray(
        [record["match_class"] == "fallback" for record in records],
        dtype=bool,
    )
    rejected_or_low_confidence = np.asarray(
        [record["rejected_or_low_confidence"] for record in records],
        dtype=bool,
    )
    rejected = np.asarray([record["rejected"] for record in records], dtype=bool)
    boundary = np.asarray(
        [record["boundary_limited"] for record in records], dtype=bool
    )
    accepted_fraction = float(np.mean(accepted))
    primary_fraction = float(np.mean(primary_nodes))
    stable_coarse_fraction = float(np.mean(stable_coarse_nodes))
    fallback_fraction = float(np.mean(fallback_nodes))
    ambiguous_fraction = float(np.mean(ambiguous))
    rejected_or_low_confidence_fraction = float(
        np.mean(rejected_or_low_confidence)
    )
    boundary_limited_fraction = float(np.mean(boundary))
    gates = {
        "primary_or_stable_coarse_fraction_sufficient": bool(
            accepted_fraction
            >= float(merged["minimum_primary_or_stable_coarse_fraction"])
        ),
        "fallback_fraction_within_limit": bool(
            fallback_fraction <= float(merged["maximum_fallback_fraction"])
        ),
        "ambiguity_fraction_within_limit": bool(
            ambiguous_fraction <= float(merged["maximum_ambiguous_fraction"])
        ),
        "rejected_fraction_within_limit": bool(
            float(np.mean(rejected)) <= float(merged["maximum_rejected_fraction"])
        ),
        "boundary_limited_fraction_within_limit": bool(
            boundary_limited_fraction
            <= float(merged["maximum_boundary_limited_fraction"])
        ),
        "all_localized_positions_finite": bool(np.isfinite(localized).all()),
    }
    if not gates["all_localized_positions_finite"] or not accepted.any():
        gate = "halt"
    elif not all(gates.values()):
        gate = "manual_review"
    else:
        gate = "pass"

    localized_document = graph.document_with_positions(localized)
    node_quality_by_id = {
        int(record["node_id"]): {
            key: record[key]
            for key in (
                "match_class",
                "primary_match",
                "assigned_coordinate_source",
                "quality_flags",
                "reason_codes",
                "boundary_limited",
                "rejected",
                "low_confidence",
                "rejected_or_low_confidence",
                "usable_for_roi_screening",
                "usable_for_direct_metrology",
                "fallback_provenance",
            )
        }
        for record in records
    }
    edge_records = _edge_quality_records(graph, records)
    edge_quality_by_id = {int(record["edge_id"]): record for record in edge_records}
    for node in localized_document["junctions"]:
        node["part2_localization_quality"] = node_quality_by_id[int(node["id"])]
    for edge in localized_document["struts"]:
        edge["part2_localization_quality"] = edge_quality_by_id[int(edge["id"])]
    localized_document["part2_provenance"] = {
        "schema_version": "part2-coordinate-provenance/1.0.0",
        "registration_mode": registration_mode,
        "input_registered_graph_sha256": graph.source_sha256,
        "ct_sha256": ct_sha256,
        "independent_positions_retained": True,
        "global_refit_performed": False,
        "config_sha256": sha256_json(merged),
        **(
            {
                "analysis_policy_artifact_sha256": policy_source[
                    "source_artifact_sha256"
                ],
                "analysis_parameters_sha256": policy_source[
                    "analysis_parameters_sha256"
                ],
                "localization_policy_sha256": policy_source[
                    "localization_policy_sha256"
                ],
                "segmentation_policy_sha256": policy_source[
                    "segmentation_policy_sha256"
                ],
                "requested_analysis_scope": policy_source[
                    "requested_analysis_scope"
                ],
                "specimen_id": policy_source["specimen_id"],
                "design_id": policy_source["design_id"],
            }
            if policy_source is not None
            else {}
        ),
    }
    graph_artifact = write_json_atomic(
        output_graph_path,
        localized_document,
        overwrite=overwrite,
    )
    persistent_graph_artifact = {
        key: value for key, value in graph_artifact.items() if key != "changed"
    }
    absolute_registration_uncertainty = None
    absolute_uncertainty_source = "unavailable_in_challenge_aligned_json"
    if registration_report is not None and registration_mode == "autonomous_v2":
        bounded = registration_report.get("mode_details", {}).get(
            "bounded_robustness", {}
        )
        candidate = bounded.get("p95_prediction_spread_voxels")
        if (
            registration_report.get("gate") == "pass"
            and bounded.get("overall_pass") is True
            and isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
        ):
            absolute_registration_uncertainty = float(candidate)
            absolute_uncertainty_source = (
                "autonomous_registration_bounded_robustness_p95_prediction_spread"
            )
    hashes = {
        "ct_sha256": ct_sha256,
        "input_registered_graph_sha256": graph.source_sha256,
        "localized_graph_sha256": graph_artifact["sha256"],
    }
    if registration_report_hash:
        hashes["registration_report_sha256"] = registration_report_hash
    if policy_source is not None:
        hashes.update(
            {
                "analysis_policy_artifact_sha256": policy_source[
                    "source_artifact_sha256"
                ],
                "analysis_parameters_sha256": policy_source[
                    "analysis_parameters_sha256"
                ],
                "localization_policy_sha256": policy_source[
                    "localization_policy_sha256"
                ],
                "segmentation_policy_sha256": policy_source[
                    "segmentation_policy_sha256"
                ],
            }
        )
    reason_codes = {
        code for record in records for code in record.get("reason_codes", [])
    }
    reason_codes.update(
        f"LOCALIZATION_GATE_FAILED_{name.upper()}"
        for name, passed in gates.items()
        if not passed
    )
    warnings: list[str] = []
    if np.any(fallback_nodes):
        warnings.append(
            f"{int(np.count_nonzero(fallback_nodes))} nodes use explicit coarse-registration fallback assignments"
        )
    if np.any(ambiguous):
        warnings.append(
            f"{int(np.count_nonzero(ambiguous))} nodes have ambiguous localization assignments"
        )
    if np.any(boundary):
        warnings.append(
            f"{int(np.count_nonzero(boundary))} nodes were boundary-limited and were not classified as primary matches"
        )
    report = {
        "schema_version": LOCALIZATION_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate == "pass",
        "registration_mode": registration_mode,
        "specimen_id": (
            policy_source["specimen_id"] if policy_source is not None else None
        ),
        "design_id": (
            policy_source["design_id"] if policy_source is not None else None
        ),
        "requested_analysis_scope": (
            policy_source["requested_analysis_scope"]
            if policy_source is not None
            else None
        ),
        "analysis_policy_source": (
            {
                key: value
                for key, value in policy_source.items()
                if key not in {"policy", "segmentation_policy"}
            }
            if policy_source is not None
            else None
        ),
        "threshold": float(threshold),
        "segmentation_binding": segmentation_binding,
        "axis_mapping": AXIS_MAPPING,
        "counts": {
            **graph.counts,
            "accepted_nodes": int(np.count_nonzero(accepted)),
            "primary_nodes": int(np.count_nonzero(primary_nodes)),
            "fallback_nodes": int(np.count_nonzero(fallback_nodes)),
            "ambiguous_nodes": int(np.count_nonzero(ambiguous)),
            "rejected_or_low_confidence_nodes": int(
                np.count_nonzero(rejected_or_low_confidence)
            ),
            "rejected_nodes": int(np.count_nonzero(rejected)),
            "low_confidence_nodes": int(
                np.count_nonzero(rejected_or_low_confidence)
            ),
            "localized_nodes": int(np.count_nonzero(primary_nodes)),
            "stable_coarse_nodes": int(np.count_nonzero(stable_coarse_nodes)),
            "boundary_truncated_nodes": int(np.count_nonzero(boundary)),
            "boundary_limited_nodes": int(np.count_nonzero(boundary)),
            "primary_edges": int(
                sum(record["match_class"] == "primary" for record in edge_records)
            ),
            "stable_coarse_edges": int(
                sum(
                    record["match_class"] == "stable_coarse"
                    for record in edge_records
                )
            ),
            "fallback_edges": int(
                sum(record["match_class"] == "fallback" for record in edge_records)
            ),
            "ambiguous_edges": int(
                sum(record["match_class"] == "ambiguous" for record in edge_records)
            ),
            "roi_screening_usable_edges": int(
                sum(record["usable_for_roi_screening"] for record in edge_records)
            ),
            "direct_metrology_usable_edges": int(
                sum(record["usable_for_direct_metrology"] for record in edge_records)
            ),
        },
        "localization": {
            "accepted_fraction": accepted_fraction,
            "primary_fraction": primary_fraction,
            "stable_coarse_fraction": stable_coarse_fraction,
            "fallback_fraction": fallback_fraction,
            "ambiguous_fraction": ambiguous_fraction,
            "rejected_or_low_confidence_fraction": (
                rejected_or_low_confidence_fraction
            ),
            "rejected_fraction": float(np.mean(rejected)),
            "low_confidence_fraction": rejected_or_low_confidence_fraction,
            "boundary_limited_fraction": boundary_limited_fraction,
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
        "quantitative_policy": {
            "minimum_primary_or_stable_coarse_fraction": float(
                merged["minimum_primary_or_stable_coarse_fraction"]
            ),
            "maximum_fallback_fraction": float(
                merged["maximum_fallback_fraction"]
            ),
            "maximum_ambiguous_fraction": float(
                merged["maximum_ambiguous_fraction"]
            ),
            "maximum_rejected_fraction": float(
                merged["maximum_rejected_fraction"]
            ),
            "maximum_boundary_limited_fraction": float(
                merged["maximum_boundary_limited_fraction"]
            ),
        },
        "gates": gates,
        "reason_codes": sorted(reason_codes),
        "records": records,
        "edge_quality_records": edge_records,
        "artifacts": {
            "localized_graph": {
                **persistent_graph_artifact,
                "role": "independently_localized_lattice_graph",
                "retention": "regenerable",
            },
            **(
                {"registration_report": registration_report_artifact}
                if registration_report_artifact is not None
                else {}
            ),
        },
        "hashes": hashes,
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "policy_binding": (
                "hashed_analysis_parameters"
                if policy_source is not None
                else "core_default_or_unit_override"
            ),
            "sealed_labels_read": False,
        },
        "warnings": warnings,
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
