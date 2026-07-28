"""Registration QA with separate padded-ROI and metrology gates."""

from __future__ import annotations

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
from .sampling import sample_corridor
from .volume import AXIS_MAPPING, load_volume

REGISTRATION_QA_SCHEMA_VERSION = "part2-registration-qa/1.1.0"
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
    return result


def compute_registration_qa(
    ct_path: str | Path,
    localized_graph_path: str | Path,
    output_report_path: str | Path,
    *,
    threshold: float,
    registration_mode: str,
    localization_report_path: str | Path | None = None,
    slice_output_path: str | Path | None = None,
    bias_output_path: str | Path | None = None,
    slice_index: int = 380,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Compute all-node/all-edge image support and independent trust gates."""

    merged = _config(config)
    volume = load_volume(ct_path)
    graph = load_lattice_json(localized_graph_path)
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
    if localization_report_path is not None:
        localization = read_json_object(localization_report_path)
        localization_summary = localization.get("localization", {})
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
        localization_hash = sha256_file(localization_report_path)
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
        "localization_not_halted": localization_gate != "halt",
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
        "uncertainty_within_measured_radius": bool(
            metrology_ratio_value is not None
            and metrology_ratio_value
            <= float(merged["maximum_uncertainty_to_radius_ratio"])
        ),
    }
    if (
        not all(image_gates.values())
        or not all(coarse_gates.values())
        or not all(roi_gates.values())
    ):
        gate = "halt"
    elif not all(metrology_gates.values()):
        gate = "manual_review"
    else:
        gate = "pass"

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
                    "localized": ("#34c759", "localized"),
                    "stable_coarse": ("#ffcc00", "stable coarse"),
                    "fallback": ("#ff3b30", "fallback/review"),
                }
                for status, (color, label) in styles.items():
                    rows = np.asarray(
                        [
                            row
                            for row, node_id in enumerate(graph.node_ids)
                            if abs(graph.node_positions_xyz[row, 2] - int(slice_index))
                            <= 2.0
                            and record_by_id.get(int(node_id), {}).get(
                                "localization_status", "fallback"
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
        "overall_pass": gate != "halt",
        "registration_mode": registration_mode,
        "threshold": float(threshold),
        "axis_mapping": AXIS_MAPPING,
        "counts": graph.counts,
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
            "direct_narrow_corridor_allowed": bool(all(metrology_gates.values())),
            "required_resolution": (
                "none"
                if all(metrology_gates.values())
                else "explicit_roi_only_authorization"
            ),
            "gates": metrology_gates,
            "overall_pass": bool(all(metrology_gates.values())),
        },
        "artifacts": persistent_figure_artifacts,
        "hashes": {
            "ct_sha256": sha256_file(volume.path),
            "localized_graph_sha256": graph.source_sha256,
            **(
                {"localization_report_sha256": localization_hash}
                if localization_hash
                else {}
            ),
        },
        "provenance": {
            "registration_mode": registration_mode,
            "config_sha256": sha256_json(merged),
            "sealed_labels_read": False,
        },
        "warnings": (
            []
            if gate == "pass"
            else [
                "metrology/direct narrow-corridor use is blocked"
                if gate == "manual_review"
                else "registration QA or padded-ROI capture gate failed"
            ]
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
