"""Registration QA with separate padded-ROI and metrology gates."""

from __future__ import annotations

import math
from pathlib import Path
import os
import tempfile
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import read_json_object, sha256_file, sha256_json, write_json_atomic
from .lattice import load_lattice_json, positions_in_volume
from .localization import (
    _config as _validate_localization_config,
    _validate_segmentation_policy,
)
from .otsu import replay_exact_otsu
from .sampling import sample_corridor
from .volume import AXIS_MAPPING, load_volume

REGISTRATION_QA_SCHEMA_VERSION = "part2-registration-qa/1.2.0"
REQUESTED_ANALYSIS_SCOPES = {"roi_screening", "direct_metrology"}
ROI_AUTHORIZED_OUTPUTS = [
    "segmentation",
    "registration",
    "node_localization",
    "coarse_region_screening",
    "padded_roi_definition",
]
DIRECT_METROLOGY_OUTPUTS = [
    "absolute_metrology",
    "direct_dimensional_measurement",
]
_QA_POLICY_FIELDS = (
    "junction_patch_radius_voxels",
    "corridor_axial_samples",
    "corridor_radius_voxels",
    "corridor_angular_samples",
    "roi_padding_fraction",
    "spatial_bins_per_axis",
    "radial_foreground_probability",
    "minimum_mean_junction_foreground_fraction",
    "minimum_median_corridor_foreground_fraction",
    "maximum_spatial_bin_median_range",
    "minimum_roi_in_bounds_fraction",
    "maximum_uncertainty_to_radius_ratio",
)
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
DEFAULT_QA_CONFIG: dict[str, Any] = {
    "junction_patch_radius_voxels": 2,
    "corridor_axial_samples": 9,
    "corridor_radius_voxels": 6.0,
    "corridor_angular_samples": 8,
    "roi_padding_fraction": 0.2,
    "spatial_bins_per_axis": 5,
    "minimum_mean_junction_foreground_fraction": 0.85,
    "minimum_median_corridor_foreground_fraction": 0.08,
    "maximum_spatial_bin_median_range": 0.25,
    "minimum_roi_in_bounds_fraction": 0.99,
    "radial_foreground_probability": 0.5,
    "maximum_uncertainty_to_radius_ratio": 1.0,
}


def _config(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_QA_CONFIG)
    if config:
        result.update(config)
    integer_minimums = {
        "junction_patch_radius_voxels": 1,
        "corridor_axial_samples": 2,
        "corridor_angular_samples": 3,
        "spatial_bins_per_axis": 1,
    }
    for name, minimum in integer_minimums.items():
        value = result[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name in ("corridor_radius_voxels",):
        value = result[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    for name in ("roi_padding_fraction", "maximum_uncertainty_to_radius_ratio"):
        value = result[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "radial_foreground_probability",
        "minimum_mean_junction_foreground_fraction",
        "minimum_median_corridor_foreground_fraction",
        "maximum_spatial_bin_median_range",
        "minimum_roi_in_bounds_fraction",
    ):
        value = result[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _load_requested_analysis_scope(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    document = read_json_object(resolved)
    source_artifact_kind = "specimen_manifest"
    if (
        document.get("schema_version") != "2.1.0"
        or not isinstance(document.get("inputs"), dict)
        or not isinstance(document["inputs"].get("ct"), dict)
        or not isinstance(document["inputs"]["ct"].get("sha256"), str)
    ):
        raise ValueError(
            "Analysis-scope artifact must be a specimen manifest at schema 2.1.0"
        )
    analysis_parameters = document.get("analysis_parameters")
    if not isinstance(analysis_parameters, dict):
        raise ValueError(
            "Analysis-scope artifact must contain canonical analysis_parameters"
        )
    expected_hash = document.get("analysis_parameters_sha256")
    actual_hash = sha256_json(analysis_parameters)
    if expected_hash != actual_hash:
        raise ValueError(
            "Analysis-scope artifact analysis_parameters_sha256 does not match "
            "canonical analysis_parameters"
        )
    scope = analysis_parameters.get("requested_analysis_scope")
    if scope not in REQUESTED_ANALYSIS_SCOPES:
        raise ValueError(
            "analysis_parameters.requested_analysis_scope must be one of "
            f"{sorted(REQUESTED_ANALYSIS_SCOPES)}"
        )
    top_level_scope = document.get("requested_analysis_scope")
    if top_level_scope is not None and top_level_scope != scope:
        raise ValueError(
            "Analysis-scope artifact has contradictory requested_analysis_scope values"
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
        raise ValueError("Analysis-scope artifact must identify specimen_id")
    if not isinstance(design_id, str) or not design_id:
        raise ValueError("Analysis-scope artifact must identify design_id")
    qa_policy = analysis_parameters.get("qa_policy")
    if not isinstance(qa_policy, dict):
        raise ValueError("analysis_parameters.qa_policy must be an object")
    if qa_policy.get("schema_version") != "stage2-qa-policy/1.1.0":
        raise ValueError("Unsupported analysis_parameters.qa_policy schema_version")
    if set(qa_policy) != {"schema_version", *_QA_POLICY_FIELDS}:
        raise ValueError("analysis_parameters.qa_policy must use its closed schema")
    missing = [name for name in _QA_POLICY_FIELDS if name not in qa_policy]
    if missing:
        raise ValueError(
            "analysis_parameters.qa_policy is missing: " + ", ".join(missing)
        )
    localization_policy = analysis_parameters.get("localization_policy")
    if not isinstance(localization_policy, dict):
        raise ValueError("analysis_parameters.localization_policy must be an object")
    if (
        localization_policy.get("schema_version")
        != "stage2-localization-policy/1.1.0"
    ):
        raise ValueError(
            "Unsupported analysis_parameters.localization_policy schema_version"
        )
    if set(localization_policy) != {
        "schema_version",
        *_LOCALIZATION_POLICY_FIELDS,
    }:
        raise ValueError(
            "analysis_parameters.localization_policy must use its closed schema"
        )
    missing = [
        name for name in _LOCALIZATION_POLICY_FIELDS if name not in localization_policy
    ]
    if missing:
        raise ValueError(
            "analysis_parameters.localization_policy is missing: "
            + ", ".join(missing)
        )
    support_weights = (
        localization_policy["core_support_weight"],
        localization_policy["incident_support_weight"],
    )
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in support_weights
    ) or not math.isclose(
        float(support_weights[0]) + float(support_weights[1]),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Localization support weights must sum to 1")
    incident_distances = localization_policy["incident_sample_distances_voxels"]
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
    _validate_localization_config(
        {name: localization_policy[name] for name in _LOCALIZATION_POLICY_FIELDS}
    )
    segmentation_policy = analysis_parameters.get("segmentation")
    if not isinstance(segmentation_policy, dict):
        raise ValueError("analysis_parameters.segmentation must be an object")
    _validate_segmentation_policy(segmentation_policy)
    artifact_versions = analysis_parameters.get("artifact_schema_versions")
    if artifact_versions != _ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError("Hashed artifact_schema_versions are incompatible")
    return {
        "requested_analysis_scope": scope,
        "registration_mode": registration_policy["mode"],
        "source_artifact_kind": source_artifact_kind,
        "source_artifact_path": str(resolved),
        "source_artifact_sha256": sha256_file(resolved),
        "analysis_parameters_sha256": actual_hash,
        "qa_policy_sha256": sha256_json(qa_policy),
        "qa_policy": {name: qa_policy[name] for name in _QA_POLICY_FIELDS},
        "localization_policy_sha256": sha256_json(localization_policy),
        "localization_policy": {
            name: localization_policy[name]
            for name in _LOCALIZATION_POLICY_FIELDS
        },
        "segmentation_policy_sha256": sha256_json(segmentation_policy),
        "segmentation_policy": dict(segmentation_policy),
        "specimen_id": specimen_id,
        "design_id": design_id,
        "declared_ct_sha256": document["inputs"]["ct"]["sha256"],
    }


def compute_registration_qa(
    ct_path: str | Path,
    localized_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    analysis_scope_artifact_path: str | Path,
    localization_report_path: str | Path | None = None,
    slice_output_path: str | Path | None = None,
    bias_output_path: str | Path | None = None,
    slice_index: int = 380,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute all-node/all-edge image support and independent trust gates."""

    scope_source = _load_requested_analysis_scope(analysis_scope_artifact_path)
    requested_analysis_scope = str(scope_source["requested_analysis_scope"])
    if registration_mode != scope_source["registration_mode"]:
        raise ValueError(
            "registration_mode does not match the hashed specimen manifest"
        )
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("threshold must be finite")
    supplied_config = dict(config or {})
    unknown_config = sorted(set(supplied_config) - set(_QA_POLICY_FIELDS))
    if unknown_config:
        raise ValueError(
            "Free-form QA config contains fields outside the closed hashed policy: "
            + ", ".join(unknown_config)
        )
    qa_policy = dict(scope_source["qa_policy"])
    for name, value in qa_policy.items():
        if name in supplied_config and supplied_config[name] != value:
            raise ValueError(f"Free-form QA config conflicts with hashed policy: {name}")
    merged = _config({**supplied_config, **qa_policy})
    volume = load_volume(ct_path)
    ct_sha256 = sha256_file(volume.path)
    if ct_sha256 != scope_source["declared_ct_sha256"]:
        raise ValueError("CT SHA-256 does not match the hashed specimen manifest")
    segmentation_policy = dict(scope_source["segmentation_policy"])
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
    segmentation_binding_gates = {
        "exact_otsu_replay_passed": bool(exact_otsu.get("overall_pass")),
        "threshold_matches_exact_otsu": bool(
            float(threshold) == float(exact_otsu["threshold"])
        ),
    }
    graph = load_lattice_json(localized_graph_path)
    binding_root_candidates = [
        Path(volume.path).resolve(),
        Path(localized_graph_path).expanduser().resolve(),
        Path(output_report_path).expanduser().resolve(),
        Path(analysis_scope_artifact_path).expanduser().resolve(),
    ]
    if localization_report_path is not None:
        binding_root_candidates.append(
            Path(localization_report_path).expanduser().resolve()
        )
    run_artifact_root = Path(
        os.path.commonpath([str(path) for path in binding_root_candidates])
    )
    if run_artifact_root == Path(run_artifact_root.anchor):
        raise ValueError("QA inputs do not share a bounded artifact root")
    patch_radius = int(merged["junction_patch_radius_voxels"])
    junction_fractions = np.zeros(len(graph.node_ids), dtype=np.float64)
    for row, position in enumerate(graph.node_positions_xyz):
        x, y, z = np.rint(position).astype(int)
        z0, z1 = max(0, z - patch_radius), min(volume.shape[0], z + patch_radius + 1)
        y0, y1 = max(0, y - patch_radius), min(volume.shape[1], y + patch_radius + 1)
        x0, x1 = max(0, x - patch_radius), min(volume.shape[2], x + patch_radius + 1)
        patch = np.asarray(volume.array[z0:z1, y0:y1, x0:x1] >= threshold)
        junction_fractions[row] = float(patch.mean()) if patch.size else 0.0

    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    midpoints = (starts + ends) / 2.0
    occupancies = np.zeros(len(graph.edge_ids), dtype=np.float64)
    radial_foreground: np.ndarray | None = None
    radial_samples: np.ndarray | None = None
    roi_contained = np.zeros(len(graph.edge_ids), dtype=bool)
    padding = float(merged["roi_padding_fraction"])
    for row, (start, end) in enumerate(zip(starts, ends)):
        sample = sample_corridor(
            volume.array,
            start,
            end,
            threshold=threshold,
            axial_samples=int(merged["corridor_axial_samples"]),
            radius_voxels=float(merged["corridor_radius_voxels"]),
            angular_samples=int(merged["corridor_angular_samples"]),
            axial_padding_fraction=padding,
        )
        foreground = sample["foreground"]
        valid = sample["valid"]
        occupancies[row] = (
            float(np.count_nonzero(foreground) / np.count_nonzero(valid))
            if np.count_nonzero(valid)
            else 0.0
        )
        ids = sample["radius_ids"]
        if radial_foreground is None:
            radial_foreground = np.zeros(int(ids.max()) + 1, dtype=np.float64)
            radial_samples = np.zeros(int(ids.max()) + 1, dtype=np.int64)
        for radius in range(len(radial_foreground)):
            selected = ids == radius
            radial_foreground[radius] += float(foreground[:, selected].sum())
            radial_samples[radius] += int(valid[:, selected].sum())
        roi_contained[row] = bool(np.all(valid))

    assert radial_foreground is not None and radial_samples is not None
    radial_probability = radial_foreground / np.maximum(radial_samples, 1)
    eligible = np.flatnonzero(
        radial_probability >= float(merged["radial_foreground_probability"])
    )
    measured_radius = float(eligible.max()) if eligible.size else 0.0
    spatial: dict[str, Any] = {}
    ranges = []
    for axis, name in enumerate("xyz"):
        order = np.argsort(midpoints[:, axis], kind="stable")
        bins = np.array_split(order, int(merged["spatial_bins_per_axis"]))
        medians = [
            float(np.median(occupancies[index])) if len(index) else 0.0
            for index in bins
        ]
        median_range = float(max(medians) - min(medians))
        ranges.append(median_range)
        spatial[name] = {
            "bin_median_corridor_foreground_fraction": medians,
            "median_range": median_range,
        }

    image_gates = {
        **segmentation_binding_gates,
        "junction_foreground_sufficient": bool(
            np.mean(junction_fractions)
            >= float(merged["minimum_mean_junction_foreground_fraction"])
        ),
        "corridor_foreground_sufficient": bool(
            np.median(occupancies)
            >= float(merged["minimum_median_corridor_foreground_fraction"])
        ),
        "spatial_bias_within_limit": bool(
            max(ranges) <= float(merged["maximum_spatial_bin_median_range"])
        ),
        "all_graph_nodes_inside_volume": bool(
            np.all(positions_in_volume(graph.node_positions_xyz, volume.shape))
        ),
    }
    independent_positions = False
    localization_graph_hash_matches = False
    localization_gate = "unknown"
    localization_hash = None
    localization_records: list[dict[str, Any]] = []
    local_search_radius_voxels = None
    capture_displacement_p95_voxels = None
    stability_uncertainty_p95_voxels = None
    absolute_registration_uncertainty_voxels = None
    absolute_registration_uncertainty_source = "unavailable"
    localization_registration_mode = "unknown"
    localization_specimen_id = None
    localization_design_id = None
    localization_requested_scope = None
    localization_threshold = None
    localization: dict[str, Any] = {}
    resolved_localization_report_path: Path | None = None
    registration_report_hash = None
    registration_report_artifact_verified = False
    registration_uncertainty_matches_artifact = False
    registration_artifact_binding_closed = False
    registration_report_path_within_run_root = False
    registration_report_colocated = False
    registration_report_matches_inputs = False
    registration_artifact_document: dict[str, Any] | None = None
    localization_quality_counts: dict[str, int] = {}
    localization_edge_quality_records: list[dict[str, Any]] = []
    if localization_report_path is not None:
        resolved_localization_report_path = (
            Path(localization_report_path).expanduser().resolve()
        )
        localization = read_json_object(resolved_localization_report_path)
        localization_summary = localization.get("localization", {})
        localization_registration_mode = str(
            localization.get("registration_mode", "unknown")
        )
        localization_specimen_id = localization.get("specimen_id")
        localization_design_id = localization.get("design_id")
        localization_requested_scope = localization.get(
            "requested_analysis_scope"
        )
        localization_threshold = localization.get("threshold")
        registration_report_hash = localization.get("hashes", {}).get(
            "registration_report_sha256"
        )
        registration_artifact = localization.get("artifacts", {}).get(
            "registration_report", {}
        )
        registration_artifact_binding_closed = bool(
            isinstance(registration_artifact, dict)
            and set(registration_artifact)
            == {"path", "sha256", "role", "retention"}
            and registration_artifact.get("role") == "registration_report"
            and registration_artifact.get("retention") == "committed"
        )
        if (
            registration_artifact_binding_closed
            and isinstance(registration_artifact.get("path"), str)
            and isinstance(registration_artifact.get("sha256"), str)
        ):
            registration_artifact_path = Path(
                registration_artifact["path"]
            ).expanduser().resolve()
            registration_report_path_within_run_root = bool(
                registration_artifact_path.is_relative_to(run_artifact_root)
            )
            registration_report_colocated = bool(
                resolved_localization_report_path is not None
                and registration_artifact_path.parent
                == resolved_localization_report_path.parent
            )
            if (
                registration_report_path_within_run_root
                and registration_report_colocated
                and registration_artifact_path.is_file()
            ):
                artifact_hash = sha256_file(registration_artifact_path)
                registration_report_artifact_verified = bool(
                    artifact_hash == registration_artifact["sha256"]
                    and artifact_hash == registration_report_hash
                )
                if registration_report_artifact_verified:
                    registration_artifact_document = read_json_object(
                        registration_artifact_path
                    )
                    registration_report_matches_inputs = bool(
                        registration_artifact_document.get("schema_version")
                        == "part2-registration/1.0.0"
                        and registration_artifact_document.get("mode")
                        == localization_registration_mode
                        and registration_artifact_document.get("gate") == "pass"
                        and registration_artifact_document.get("hashes", {}).get(
                            "ct_sha256"
                        )
                        == ct_sha256
                        and registration_artifact_document.get("hashes", {}).get(
                            "registered_graph_sha256"
                        )
                        == localization.get("hashes", {}).get(
                            "input_registered_graph_sha256"
                        )
                    )
        localization_records = [
            record
            for record in localization.get("records", [])
            if isinstance(record, dict)
        ]
        independent_positions = bool(
            localization_summary.get(
                "independent_positions_retained", False
            )
            and not localization_summary.get(
                "global_refit_performed", True
            )
        )
        localization_graph_hash_matches = (
            localization.get("hashes", {}).get("localized_graph_sha256")
            == graph.source_sha256
        )
        localization_gate = str(localization.get("gate", "unknown"))
        localization_hash = sha256_file(resolved_localization_report_path)
        local_search_radius_voxels = localization_summary.get("search_radius_voxels")
        capture_displacement_p95_voxels = localization_summary.get(
            "accepted_shift_voxels", {}
        ).get("p95")
        stability_uncertainty_p95_voxels = localization_summary.get(
            "stability_uncertainty_voxels", {}
        ).get("p95")
        absolute_registration_uncertainty_voxels = localization_summary.get(
            "absolute_registration_uncertainty_voxels"
        )
        absolute_registration_uncertainty_source = str(
            localization_summary.get(
                "absolute_registration_uncertainty_source", "unavailable"
            )
        )
        if registration_artifact_document is not None:
            bounded = registration_artifact_document.get(
                "mode_details", {}
            ).get("bounded_robustness", {})
            bounded_value = bounded.get("p95_prediction_spread_voxels")
            registration_uncertainty_matches_artifact = bool(
                registration_artifact_document.get("mode") == "autonomous_v2"
                and registration_artifact_document.get("gate") == "pass"
                and bounded.get("overall_pass") is True
                and isinstance(bounded_value, (int, float))
                and not isinstance(bounded_value, bool)
                and math.isfinite(float(bounded_value))
                and isinstance(
                    absolute_registration_uncertainty_voxels, (int, float)
                )
                and not isinstance(absolute_registration_uncertainty_voxels, bool)
                and float(bounded_value)
                == float(absolute_registration_uncertainty_voxels)
                and registration_artifact_document.get("hashes", {}).get(
                    "ct_sha256"
                )
                == ct_sha256
                and registration_artifact_document.get("hashes", {}).get(
                    "registered_graph_sha256"
                )
                == localization.get("hashes", {}).get(
                    "input_registered_graph_sha256"
                )
            )
        raw_counts = localization.get("counts", {})
        localization_quality_counts = {
            name: int(raw_counts.get(name, 0))
            for name in (
                "primary_nodes",
                "stable_coarse_nodes",
                "fallback_nodes",
                "ambiguous_nodes",
                "rejected_or_low_confidence_nodes",
                "boundary_limited_nodes",
                "primary_edges",
                "stable_coarse_edges",
                "fallback_edges",
                "ambiguous_edges",
                "roi_screening_usable_edges",
                "direct_metrology_usable_edges",
            )
        }
        localization_edge_quality_records = [
            record
            for record in localization.get("edge_quality_records", [])
            if isinstance(record, dict)
        ]
    localization_hashes = localization.get("hashes", {})
    if not isinstance(localization_hashes, dict):
        localization_hashes = {}
    localization_policy_source = localization.get("analysis_policy_source", {})
    if not isinstance(localization_policy_source, dict):
        localization_policy_source = {}
    localization_provenance = localization.get("provenance", {})
    if not isinstance(localization_provenance, dict):
        localization_provenance = {}
    observed_quantitative_policy = localization.get("quantitative_policy", {})
    observed_localization_segmentation = localization.get(
        "segmentation_binding", {}
    )
    localized_graph_artifact = localization.get("artifacts", {}).get(
        "localized_graph", {}
    )
    localized_graph_artifact_closed = bool(
        isinstance(localized_graph_artifact, dict)
        and set(localized_graph_artifact)
        == {"path", "sha256", "role", "retention"}
        and localized_graph_artifact.get("role")
        == "independently_localized_lattice_graph"
        and localized_graph_artifact.get("retention") == "regenerable"
        and Path(str(localized_graph_artifact.get("path", "")))
        .expanduser()
        .resolve()
        == Path(localized_graph_path).expanduser().resolve()
        and localized_graph_artifact.get("sha256") == graph.source_sha256
    )
    expected_localization_policy = dict(scope_source["localization_policy"])
    expected_quantitative_policy = {
        name: expected_localization_policy[name]
        for name in (
            "minimum_primary_or_stable_coarse_fraction",
            "maximum_fallback_fraction",
            "maximum_ambiguous_fraction",
            "maximum_rejected_fraction",
            "maximum_boundary_limited_fraction",
        )
    }
    expected_localization_config_sha256 = sha256_json(
        expected_localization_policy
    )
    expected_localization_segmentation = {
        "method": exact_otsu["method"],
        "method_version": exact_otsu["method_version"],
        "threshold": exact_otsu["threshold"],
        "threshold_comparison": exact_otsu["threshold_comparison"],
        "ct_sha256": ct_sha256,
        "segmentation_policy_sha256": scope_source[
            "segmentation_policy_sha256"
        ],
        "overall_pass": True,
    }
    expected_policy_source = {
        "source_artifact_path": scope_source["source_artifact_path"],
        "source_artifact_sha256": scope_source["source_artifact_sha256"],
        "analysis_parameters_sha256": scope_source["analysis_parameters_sha256"],
        "localization_policy_sha256": scope_source[
            "localization_policy_sha256"
        ],
        "requested_analysis_scope": requested_analysis_scope,
        "registration_mode": scope_source["registration_mode"],
        "specimen_id": scope_source["specimen_id"],
        "design_id": scope_source["design_id"],
        "declared_ct_sha256": ct_sha256,
        "segmentation_policy_sha256": scope_source[
            "segmentation_policy_sha256"
        ],
    }
    localization_binding_gates = {
        "schema_version_supported": bool(
            localization.get("schema_version") == "part2-node-localization/1.2.0"
        ),
        "specimen_id_matches_scope": bool(
            localization_specimen_id == scope_source["specimen_id"]
        ),
        "design_id_matches_scope": bool(
            localization_design_id == scope_source["design_id"]
        ),
        "requested_analysis_scope_matches": bool(
            localization_requested_scope == requested_analysis_scope
        ),
        "registration_mode_matches": bool(
            localization_registration_mode == registration_mode
        ),
        "ct_sha256_matches": bool(
            localization_hashes.get("ct_sha256") == ct_sha256
        ),
        "localized_graph_sha256_matches": bool(
            localization_hashes.get("localized_graph_sha256")
            == graph.source_sha256
        ),
        "threshold_matches_exact_otsu": bool(
            isinstance(localization_threshold, (int, float))
            and not isinstance(localization_threshold, bool)
            and float(localization_threshold) == float(exact_otsu["threshold"])
        ),
        "analysis_parameters_sha256_matches": bool(
            localization_hashes.get("analysis_parameters_sha256")
            == scope_source["analysis_parameters_sha256"]
        ),
        "localization_policy_sha256_matches": bool(
            localization_hashes.get("localization_policy_sha256")
            == scope_source["localization_policy_sha256"]
        ),
        "localization_policy_values_match": bool(
            observed_quantitative_policy == expected_quantitative_policy
        ),
        "localization_config_sha256_matches_policy": bool(
            localization_provenance.get("config_sha256")
            == expected_localization_config_sha256
        ),
        "segmentation_binding_matches_exact_otsu": bool(
            observed_localization_segmentation
            == expected_localization_segmentation
        ),
        "analysis_policy_artifact_sha256_matches": bool(
            localization_hashes.get("analysis_policy_artifact_sha256")
            == scope_source["source_artifact_sha256"]
            and localization_policy_source == expected_policy_source
        ),
        "policy_binding_declared": bool(
            localization_provenance.get("policy_binding")
            == "hashed_analysis_parameters"
        ),
        "localized_graph_artifact_binding_closed": localized_graph_artifact_closed,
        "registration_report_artifact_binding_closed": (
            registration_artifact_binding_closed
        ),
        "registration_report_path_within_run_root": (
            registration_report_path_within_run_root
        ),
        "registration_report_colocated_with_localization": (
            registration_report_colocated
        ),
        "registration_report_matches_inputs": registration_report_matches_inputs,
    }
    localization_binding_passed = bool(all(localization_binding_gates.values()))
    numeric_capture = (
        isinstance(local_search_radius_voxels, (int, float))
        and isinstance(capture_displacement_p95_voxels, (int, float))
    )
    roi_fraction = float(np.mean(roi_contained))
    coarse_gates = {
        "accepted_displacement_within_local_capture_radius": bool(
            numeric_capture
            and float(capture_displacement_p95_voxels)
            <= float(local_search_radius_voxels)
        ),
        "independent_node_positions_retained": independent_positions,
        "localization_graph_hash_matches": localization_graph_hash_matches,
        "localization_report_binding_complete": localization_binding_passed,
        "localization_not_halted": localization_gate != "halt",
        "localization_node_quality_counts_complete": bool(
            sum(
                localization_quality_counts.get(name, 0)
                for name in (
                    "primary_nodes",
                    "stable_coarse_nodes",
                    "fallback_nodes",
                    "ambiguous_nodes",
                )
            )
            == graph.counts["nodes"]
        ),
        "localization_edge_quality_records_complete": bool(
            len(localization_edge_quality_records) == graph.counts["edges"]
            and len(
                {
                    record.get("edge_id")
                    for record in localization_edge_quality_records
                }
            )
            == graph.counts["edges"]
        ),
    }
    roi_gates = {
        "padded_rois_in_bounds": bool(
            roi_fraction >= float(merged["minimum_roi_in_bounds_fraction"])
        ),
    }
    numeric_absolute_uncertainty = isinstance(
        absolute_registration_uncertainty_voxels, (int, float)
    )
    metrology_uncertainty = (
        float(absolute_registration_uncertainty_voxels)
        + float(stability_uncertainty_p95_voxels or 0.0)
        if numeric_absolute_uncertainty
        else None
    )
    metrology_ratio_value = (
        metrology_uncertainty / measured_radius
        if metrology_uncertainty is not None and measured_radius > 0
        else None
    )
    metrology_gates = {
        "measured_radius_positive": bool(measured_radius > 0),
        "absolute_registration_uncertainty_available": bool(
            numeric_absolute_uncertainty
        ),
        "absolute_registration_uncertainty_artifact_backed": bool(
            numeric_absolute_uncertainty
            and absolute_registration_uncertainty_source
            == "autonomous_registration_bounded_robustness_p95_prediction_spread"
            and localization_registration_mode == "autonomous_v2"
            and registration_mode == "autonomous_v2"
            and registration_report_artifact_verified
            and registration_uncertainty_matches_artifact
        ),
        "uncertainty_within_measured_radius": bool(
            metrology_ratio_value is not None
            and metrology_ratio_value
            <= float(merged["maximum_uncertainty_to_radius_ratio"])
        ),
    }
    localization_quantitative_gate_passed = localization_gate == "pass"
    roi_gate_results = {
        "production_image_qa_pass": bool(all(image_gates.values())),
        "coarse_capture_pass": bool(all(coarse_gates.values())),
        "localization_binding_pass": localization_binding_passed,
        "localization_quantitative_gate_pass": localization_quantitative_gate_passed,
        "padded_roi_capture_pass": bool(all(roi_gates.values())),
    }
    hard_roi_integrity_passed = bool(
        roi_gate_results["production_image_qa_pass"]
        and roi_gate_results["coarse_capture_pass"]
        and roi_gate_results["padded_roi_capture_pass"]
    )
    roi_gate_passed = bool(
        hard_roi_integrity_passed
        and roi_gate_results["localization_quantitative_gate_pass"]
    )
    metrology_evidence_passed = bool(all(metrology_gates.values()))
    if requested_analysis_scope == "roi_screening":
        metrology_status = "not_authorized"
    else:
        metrology_status = "pass" if metrology_evidence_passed else "fail"

    reason_codes: list[str] = []
    reason_codes.extend(
        f"ROI_IMAGE_GATE_FAILED_{name.upper()}"
        for name, passed in image_gates.items()
        if not passed
    )
    reason_codes.extend(
        f"ROI_COARSE_GATE_FAILED_{name.upper()}"
        for name, passed in coarse_gates.items()
        if not passed
    )
    reason_codes.extend(
        f"ROI_PADDED_GATE_FAILED_{name.upper()}"
        for name, passed in roi_gates.items()
        if not passed
    )
    reason_codes.extend(
        f"LOCALIZATION_BINDING_FAILED_{name.upper()}"
        for name, passed in localization_binding_gates.items()
        if not passed
    )
    if not localization_quantitative_gate_passed:
        reason_codes.append("LOCALIZATION_QUANTITATIVE_GATES_FAILED")
    if roi_gate_passed:
        reason_codes.append("ROI_GATES_PASS")
    if requested_analysis_scope == "roi_screening":
        reason_codes.append("METROLOGY_NOT_AUTHORIZED")
    elif not metrology_gates["absolute_registration_uncertainty_available"]:
        reason_codes.append("METROLOGY_EVIDENCE_MISSING")
    elif not metrology_gates[
        "absolute_registration_uncertainty_artifact_backed"
    ]:
        reason_codes.append("METROLOGY_EVIDENCE_UNVERIFIED")
    elif not metrology_gates["measured_radius_positive"]:
        reason_codes.append("METROLOGY_MEASURED_RADIUS_UNAVAILABLE")
    elif not metrology_gates["uncertainty_within_measured_radius"]:
        reason_codes.append("METROLOGY_UNCERTAINTY_EXCESSIVE")
    elif metrology_evidence_passed:
        reason_codes.append("METROLOGY_GATES_PASS")

    if not hard_roi_integrity_passed:
        gate = "halt"
    elif not localization_quantitative_gate_passed:
        gate = "manual_review"
    elif requested_analysis_scope == "direct_metrology" and not metrology_evidence_passed:
        gate = "manual_review"
    else:
        gate = "pass"

    authorized_outputs = list(ROI_AUTHORIZED_OUTPUTS) if roi_gate_passed else []
    if (
        requested_analysis_scope == "direct_metrology"
        and gate == "pass"
        and metrology_status == "pass"
    ):
        authorized_outputs.extend(DIRECT_METROLOGY_OUTPUTS)
    unauthorized_outputs = [
        output
        for output in DIRECT_METROLOGY_OUTPUTS
        if output not in authorized_outputs
    ]

    figure_artifacts: dict[str, Any] = {}

    def publish_figure(destination_value: str | Path, draw: Any) -> dict[str, Any]:
        destination = Path(destination_value).expanduser().resolve()
        if destination.suffix.lower() != ".png":
            raise ValueError(f"QA figure output must be PNG: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            draw(temporary)
            if destination.exists():
                if destination.read_bytes() == temporary.read_bytes():
                    return {"path": str(destination), "sha256": sha256_file(destination), "changed": False}
                raise FileExistsError(f"QA figure already exists with different bytes: {destination}")
            os.replace(temporary, destination)
            return {"path": str(destination), "sha256": sha256_file(destination), "changed": True}
        finally:
            temporary.unlink(missing_ok=True)

    if slice_output_path is not None:
        if not 0 <= int(slice_index) < volume.shape[0]:
            raise IndexError(f"Slice {slice_index} is outside CT depth {volume.shape[0]}")

        def draw_slice(path: Path) -> None:
            figure, axes = plt.subplots(figsize=(8, 8))
            try:
                axes.imshow(np.asarray(volume.array[int(slice_index)]), cmap="gray")
                record_by_id = {
                    int(record["node_id"]): record
                    for record in localization_records
                    if isinstance(record.get("node_id"), int)
                }
                styles = {
                    "primary": ("#34c759", "primary"),
                    "stable_coarse": ("#ffcc00", "stable coarse"),
                    "fallback": ("#ff9500", "fallback"),
                    "ambiguous": ("#ff3b30", "ambiguous/review"),
                }
                for status, (color, label) in styles.items():
                    rows = np.asarray(
                        [
                            row
                            for row, node_id in enumerate(graph.node_ids)
                            if abs(graph.node_positions_xyz[row, 2] - int(slice_index))
                            <= 2.0
                            and record_by_id.get(int(node_id), {}).get(
                                "match_class", "fallback"
                            )
                            == status
                        ],
                        dtype=np.int64,
                    )
                    if rows.size:
                        axes.scatter(
                            graph.node_positions_xyz[rows, 0],
                            graph.node_positions_xyz[rows, 1],
                            s=11,
                            facecolors="none",
                            edgecolors=color,
                            linewidths=0.9,
                            label=label,
                        )
                if localization_records:
                    axes.legend(loc="lower right", fontsize=7, framealpha=0.8)
                axes.set_title(f"CT-only localization status, z={slice_index}")
                axes.axis("off")
                figure.tight_layout()
                figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "part2-core"})
            finally:
                plt.close(figure)

        figure_artifacts["junction_overlay"] = {
            **publish_figure(slice_output_path, draw_slice),
            "role": "junction_overlay",
            "retention": "committed",
        }

    if bias_output_path is not None:
        def draw_bias(path: Path) -> None:
            figure, axes = plt.subplots(figsize=(8, 5))
            try:
                for axis_name, color in zip("xyz", ("#007aff", "#34c759", "#ff9500"), strict=True):
                    values = spatial[axis_name]["bin_median_corridor_foreground_fraction"]
                    axes.plot(range(len(values)), values, marker="o", label=axis_name.upper(), color=color)
                axes.set_xlabel("Stable spatial bin")
                axes.set_ylabel("Median corridor foreground fraction")
                axes.set_title("Registration QA spatial bias by XYZ")
                axes.grid(alpha=0.25)
                axes.legend()
                figure.tight_layout()
                figure.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": "part2-core"})
            finally:
                plt.close(figure)

        figure_artifacts["spatial_bias_figure"] = {
            **publish_figure(bias_output_path, draw_bias),
            "role": "spatial_bias_figure",
            "retention": "committed",
        }

    persistent_figure_artifacts = {
        name: {
            key: value for key, value in metadata.items() if key != "changed"
        }
        for name, metadata in figure_artifacts.items()
    }
    report = {
        "schema_version": REGISTRATION_QA_SCHEMA_VERSION,
        "gate": gate,
        "overall_pass": gate == "pass",
        "registration_mode": registration_mode,
        "specimen_id": scope_source["specimen_id"],
        "design_id": scope_source["design_id"],
        "requested_analysis_scope": requested_analysis_scope,
        "scope_source": {
            key: value
            for key, value in scope_source.items()
            if key
            not in {"qa_policy", "localization_policy", "segmentation_policy"}
        },
        "authorized_outputs": authorized_outputs,
        "unauthorized_outputs": unauthorized_outputs,
        "reason_codes": sorted(set(reason_codes)),
        "threshold": float(threshold),
        "segmentation_binding": {
            "method": exact_otsu["method"],
            "method_version": exact_otsu["method_version"],
            "threshold": exact_otsu["threshold"],
            "threshold_comparison": exact_otsu["threshold_comparison"],
            "ct_sha256": ct_sha256,
            "segmentation_policy_sha256": scope_source[
                "segmentation_policy_sha256"
            ],
            "gates": segmentation_binding_gates,
            "overall_pass": bool(all(segmentation_binding_gates.values())),
        },
        "axis_mapping": AXIS_MAPPING,
        "counts": graph.counts,
        "localization_quality_counts": localization_quality_counts,
        "localization_binding": {
            "artifact": {
                "path": (
                    str(resolved_localization_report_path)
                    if resolved_localization_report_path is not None
                    else None
                ),
                "sha256": localization_hash,
                "role": "localization_report",
            },
            "expected": {
                "specimen_id": scope_source["specimen_id"],
                "design_id": scope_source["design_id"],
                "requested_analysis_scope": requested_analysis_scope,
                "registration_mode": registration_mode,
                "ct_sha256": ct_sha256,
                "localized_graph_sha256": graph.source_sha256,
                "analysis_parameters_sha256": scope_source[
                    "analysis_parameters_sha256"
                ],
                "localization_policy_sha256": scope_source[
                    "localization_policy_sha256"
                ],
                "analysis_policy_artifact_sha256": scope_source[
                    "source_artifact_sha256"
                ],
            },
            "observed": {
                "specimen_id": localization_specimen_id,
                "design_id": localization_design_id,
                "requested_analysis_scope": localization_requested_scope,
                "registration_mode": localization_registration_mode,
                "ct_sha256": localization_hashes.get("ct_sha256"),
                "localized_graph_sha256": localization_hashes.get(
                    "localized_graph_sha256"
                ),
                "analysis_parameters_sha256": localization_hashes.get(
                    "analysis_parameters_sha256"
                ),
                "localization_policy_sha256": localization_hashes.get(
                    "localization_policy_sha256"
                ),
                "analysis_policy_artifact_sha256": localization_hashes.get(
                    "analysis_policy_artifact_sha256"
                ),
            },
            "gates": localization_binding_gates,
            "overall_pass": localization_binding_passed,
        },
        "roi_gate_results": {
            **roi_gate_results,
            "overall_pass": roi_gate_passed,
        },
        "production_image_qa": {
            "junctions": {
                "record_count": int(len(junction_fractions)),
                "mean_foreground_fraction": float(np.mean(junction_fractions)),
                "median_foreground_fraction": float(np.median(junction_fractions)),
            },
            "corridors": {
                "edge_count": int(len(occupancies)),
                "median_foreground_fraction": float(np.median(occupancies)),
                "p10_foreground_fraction": float(np.quantile(occupancies, 0.1)),
                "p90_foreground_fraction": float(np.quantile(occupancies, 0.9)),
                "radial_foreground_probability": radial_probability.tolist(),
                "measured_strut_radius_voxels": measured_radius,
            },
            "spatial_bias": spatial,
            "gates": image_gates,
            "overall_pass": bool(all(image_gates.values())),
        },
        "coarse_capture": {
            "accepted_displacement_p95_voxels": capture_displacement_p95_voxels,
            "local_search_radius_voxels": local_search_radius_voxels,
            "estimator_stability_p95_voxels": stability_uncertainty_p95_voxels,
            "localization_report_gate": localization_gate,
            "gates": coarse_gates,
            "overall_pass": bool(all(coarse_gates.values())),
        },
        "padded_roi_capture": {
            "padding_fraction": padding,
            "in_bounds_fraction": roi_fraction,
            "gates": roi_gates,
            "overall_pass": bool(all(roi_gates.values())),
        },
        "metrology": {
            "status": metrology_status,
            "absolute_registration_uncertainty_voxels": (
                absolute_registration_uncertainty_voxels
            ),
            "absolute_registration_uncertainty_source": (
                absolute_registration_uncertainty_source
            ),
            "estimator_stability_p95_voxels": stability_uncertainty_p95_voxels,
            "combined_metrology_uncertainty_voxels": metrology_uncertainty,
            "measured_strut_radius_voxels": measured_radius,
            "uncertainty_to_measured_radius_ratio": metrology_ratio_value,
            "direct_narrow_corridor_allowed": bool(
                requested_analysis_scope == "direct_metrology"
                and metrology_status == "pass"
            ),
            "required_resolution": (
                "none"
                if metrology_status in {"pass", "not_authorized"}
                else "manual_review"
            ),
            "gates": metrology_gates,
            "overall_pass": (
                metrology_evidence_passed
                if requested_analysis_scope == "direct_metrology"
                else None
            ),
        },
        "artifacts": persistent_figure_artifacts,
        "hashes": {
            "ct_sha256": ct_sha256,
            "localized_graph_sha256": graph.source_sha256,
            "analysis_scope_artifact_sha256": scope_source[
                "source_artifact_sha256"
            ],
            "analysis_parameters_sha256": scope_source[
                "analysis_parameters_sha256"
            ],
            "qa_policy_sha256": scope_source["qa_policy_sha256"],
            "localization_policy_sha256": scope_source[
                "localization_policy_sha256"
            ],
            "segmentation_policy_sha256": scope_source[
                "segmentation_policy_sha256"
            ],
            **(
                {"registration_report_sha256": registration_report_hash}
                if isinstance(registration_report_hash, str)
                else {}
            ),
            **(
                {"localization_report_sha256": localization_hash}
                if localization_hash
                else {}
            ),
        },
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "requested_analysis_scope": requested_analysis_scope,
            "analysis_scope_artifact_sha256": scope_source[
                "source_artifact_sha256"
            ],
            "analysis_parameters_sha256": scope_source[
                "analysis_parameters_sha256"
            ],
            "qa_policy_sha256": scope_source["qa_policy_sha256"],
            "localization_policy_sha256": scope_source[
                "localization_policy_sha256"
            ],
            "segmentation_policy_sha256": scope_source[
                "segmentation_policy_sha256"
            ],
            "policy_binding": "hashed_analysis_parameters",
            "sealed_labels_read": False,
        },
        "warnings": (
            ["Direct dimensional metrology is not authorized for roi_screening"]
            if requested_analysis_scope == "roi_screening"
            else (
                ["Direct dimensional metrology failed its uncertainty gates"]
                if gate == "manual_review"
                else (
                    ["Registration QA or padded-ROI capture gate failed"]
                    if gate == "halt"
                    else []
                )
            )
        ),
    }
    artifact = write_json_atomic(
        output_report_path,
        report,
        overwrite=overwrite,
    )
    report["artifacts"]["registration_qa"] = {
        **artifact,
        "role": "registration_qa",
        "retention": "committed",
    }
    for name, metadata in figure_artifacts.items():
        report["artifacts"][name]["changed"] = metadata["changed"]
    report["hashes"]["registration_qa_sha256"] = artifact["sha256"]
    return report
