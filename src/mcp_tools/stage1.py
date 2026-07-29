"""Stage 1 data-preparation MCP adapters."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np

from part2_core import (
    compare_segmentation_masks as _compare_masks_core,
    compute_registration_qa as _compute_registration_qa,
    load_volume as _load_volume,
    localize_lattice_nodes as _localize_lattice_nodes,
    register_lattice_to_ct as _register_lattice_to_ct,
    replay_exact_otsu as _replay_exact_otsu,
    segment_ct_dataset as _segment_ct_dataset_core,
    success_response as _success_response,
    visualize_slice as _visualize_slice_core,
    volume_metadata as _volume_metadata,
    write_otsu_artifacts as _write_otsu_artifacts,
)
from part2_core.artifacts import write_json_atomic
from specimen_manifest import (
    canonical_json_sha256 as _canonical_json_sha256,
    load_json as _load_json,
    validate_manifest as _validate_specimen_manifest,
)

from . import common
from .common import (
    MCPResponseEnvelope,
    _config_sha256,
    _core_response,
    _repository_output_directory,
    _repository_path,
    _run_structured_tool,
    _sha256_file,
)
from .registry import mcp

SEGMENTATION_VERIFICATION_EVIDENCE_SCHEMA_VERSION = (
    "segmentation-verification-mcp-evidence/1.0.0"
)


@mcp.tool()
def volume_info(
    input_filepath: str,
    include_sha256: bool = True,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
) -> MCPResponseEnvelope:
    """Return compact shared-loader metadata for a TIFF or NPY CT volume."""

    def operation() -> dict[str, Any]:
        path, relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        volume = _load_volume(path)
        result = _volume_metadata(volume)
        result["path"] = relative
        result["registration_mode"] = registration_mode
        digest = _sha256_file(path) if include_sha256 else ""
        warnings = [] if include_sha256 else ["input SHA-256 was explicitly omitted"]
        return _success_response(
            tool="volume_info",
            gate="pass",
            summary=f"Loaded 3-D {result['format']} volume {relative}",
            result=result,
            artifacts={
                "input": {
                    "path": relative,
                    "role": "ct_volume",
                    "retention": "external",
                }
            },
            hashes={
                **({"input_sha256": digest} if digest else {}),
                "config_sha256": _config_sha256(
                    {
                        "include_sha256": include_sha256,
                        "registration_mode": registration_mode,
                    }
                ),
            },
            warnings=warnings,
        )

    return _run_structured_tool("volume_info", operation)


@mcp.tool()
def replay_exact_otsu(
    input_filepath: str,
    output_directory: str,
    histogram_encoding: Literal[
        "auto", "native_uint16", "full_volume_affine_uint16"
    ] = "auto",
    edge_slices_excluded: int = 0,
    chunk_voxels: int = 8 * 1024 * 1024,
    coarse_bins: int = 1024,
    peak_smoothing_sigma_bins: float = 2.0,
    peak_prominence_fraction: float = 0.003,
    minimum_significant_peaks: int = 2,
    minimum_foreground_fraction: float = 0.01,
    maximum_foreground_fraction: float = 0.35,
    minimum_otsu_separability: float = 0.45,
    minimum_class_mean_separation_sigma: float = 0.75,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    enforce_reference_replay: bool = False,
    reference_threshold: int = 40054,
    reference_foreground_voxels: int = 58653410,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Replay per-scan exact Otsu and persist its histogram and diagnostics."""

    def operation() -> dict[str, Any]:
        source, source_relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        output, _ = _repository_output_directory(output_directory)
        recipe = {
            "histogram_encoding": histogram_encoding,
            "edge_slices_excluded": edge_slices_excluded,
            "chunk_voxels": chunk_voxels,
            "coarse_bins": coarse_bins,
            "peak_smoothing_sigma_bins": peak_smoothing_sigma_bins,
            "peak_prominence_fraction": peak_prominence_fraction,
            "minimum_significant_peaks": minimum_significant_peaks,
            "minimum_foreground_fraction": minimum_foreground_fraction,
            "maximum_foreground_fraction": maximum_foreground_fraction,
            "minimum_otsu_separability": minimum_otsu_separability,
            "minimum_class_mean_separation_sigma": (
                minimum_class_mean_separation_sigma
            ),
        }
        result, histogram = _replay_exact_otsu(source, recipe=recipe)
        result["source_path"] = source_relative
        reference_gates = {
            "reference_threshold_matches": result["threshold"] == reference_threshold,
            "reference_foreground_count_matches": result["foreground_voxel_count"]
            == reference_foreground_voxels,
        }
        if enforce_reference_replay:
            result["gates"].update(reference_gates)
            result["overall_pass"] = bool(all(result["gates"].values()))
        config_hash = _config_sha256(
            {
                "recipe": recipe,
                "registration_mode": registration_mode,
                "enforce_reference_replay": enforce_reference_replay,
                "reference_threshold": reference_threshold,
                "reference_foreground_voxels": reference_foreground_voxels,
            }
        )
        input_hash = _sha256_file(source)
        result["registration_mode"] = registration_mode
        result["reference_replay"] = {
            "enforced": enforce_reference_replay,
            "expected_threshold": reference_threshold,
            "expected_foreground_voxels": reference_foreground_voxels,
            "gates": reference_gates,
        }
        result["hashes"] = {
            "input_sha256": input_hash,
            "config_sha256": config_hash,
        }
        result["provenance"] = {
            "registration_mode": registration_mode,
            "threshold_selected_per_scan": True,
            "target_foreground_fraction_used": False,
            "defect_labels_read": False,
        }
        artifacts = _write_otsu_artifacts(
            output,
            result,
            histogram,
            overwrite=overwrite,
        )
        for artifact in artifacts.values():
            artifact_path = Path(artifact["path"])
            artifact["path"] = artifact_path.relative_to(common.REPOSITORY_ROOT).as_posix()
        failed_gates = sorted(
            name for name, passed in result["gates"].items() if not passed
        )
        gate: Literal["pass", "halt"] = "pass" if result["overall_pass"] else "halt"
        warnings = (
            []
            if not failed_gates
            else ["histogram rejection gates failed: " + ", ".join(failed_gates)]
        )
        return _success_response(
            tool="replay_exact_otsu",
            gate=gate,
            summary=(
                f"Replayed Otsu threshold {result['threshold']} for {source_relative}"
            ),
            result=result,
            artifacts=artifacts,
            hashes={
                "input_sha256": input_hash,
                "histogram_sha256": result["histogram_sha256"],
                "histogram_artifact_sha256": artifacts["histogram"]["sha256"],
                "report_artifact_sha256": artifacts["report"]["sha256"],
                "config_sha256": config_hash,
            },
            warnings=warnings,
        )

    return _run_structured_tool("replay_exact_otsu", operation)


@mcp.tool()
def register_lattice_to_ct(
    nominal_graph_filepath: str,
    output_graph_filepath: str,
    output_report_filepath: str,
    registration_mode: Literal["autonomous_v2"],
    ct_filepath: str | None = None,
    aligned_graph_filepath: str | None = None,
    threshold: float | None = None,
    config: dict[str, Any] | None = None,
    analysis_config_filepath: str | None = None,
    freeze_receipt_filepath: str | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Register through challenge aligned-JSON or isolated autonomous-v2 mode."""

    def operation() -> dict[str, Any]:
        nominal, _ = _repository_path(
            nominal_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        output_graph, _ = _repository_path(
            output_graph_filepath, must_exist=False, expected_suffixes={".json"}
        )
        output_report, _ = _repository_path(
            output_report_filepath, must_exist=False, expected_suffixes={".json"}
        )
        ct = (
            _repository_path(
                ct_filepath,
                must_exist=True,
                expected_suffixes={".npy", ".tif", ".tiff"},
            )[0]
            if ct_filepath
            else None
        )
        aligned = (
            _repository_path(
                aligned_graph_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
            if aligned_graph_filepath
            else None
        )
        analysis_config = (
            _repository_path(
                analysis_config_filepath, must_exist=False, expected_suffixes={".json"}
            )[0]
            if analysis_config_filepath
            else None
        )
        freeze_receipt = (
            _repository_path(
                freeze_receipt_filepath, must_exist=False, expected_suffixes={".json"}
            )[0]
            if freeze_receipt_filepath
            else None
        )
        payload = _register_lattice_to_ct(
            nominal,
            output_graph,
            output_report,
            mode=registration_mode,
            ct_path=ct,
            aligned_graph_path=aligned,
            threshold=threshold,
            config=config,
            analysis_config_path=analysis_config,
            freeze_receipt_path=freeze_receipt,
            overwrite=overwrite,
        )
        return _core_response(
            "register_lattice_to_ct",
            f"Completed {registration_mode} registration with gate {payload['gate']}",
            payload,
        )

    return _run_structured_tool("register_lattice_to_ct", operation)


@mcp.tool()
def localize_lattice_nodes(
    ct_filepath: str,
    registered_graph_filepath: str,
    output_graph_filepath: str,
    output_report_filepath: str,
    threshold: float,
    registration_mode: Literal["autonomous_v2"],
    analysis_policy_artifact_filepath: str,
    registration_report_filepath: str | None = None,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Independently recenter registered nodes inside bounded CT windows."""

    def operation() -> dict[str, Any]:
        ct, _ = _repository_path(
            ct_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        graph, _ = _repository_path(
            registered_graph_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output_graph, _ = _repository_path(
            output_graph_filepath, must_exist=False, expected_suffixes={".json"}
        )
        output_report, _ = _repository_path(
            output_report_filepath, must_exist=False, expected_suffixes={".json"}
        )
        registration_report = (
            _repository_path(
                registration_report_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
            if registration_report_filepath
            else None
        )
        analysis_policy_artifact = _repository_path(
            analysis_policy_artifact_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )[0]
        payload = _localize_lattice_nodes(
            ct,
            graph,
            output_graph,
            output_report,
            threshold=threshold,
            registration_mode=registration_mode,
            analysis_policy_artifact_path=analysis_policy_artifact,
            config=config,
            registration_report_path=registration_report,
            overwrite=overwrite,
        )
        compact_payload = dict(payload)
        compact_payload.pop("records", None)
        compact_payload["localization"] = {
            **dict(payload["localization"]),
            "record_count": int(payload["counts"]["nodes"]),
            "records_persisted_only": True,
        }
        return _core_response(
            "localize_lattice_nodes",
            (
                f"Localized {payload['counts']['accepted_nodes']} of "
                f"{payload['counts']['nodes']} nodes"
            ),
            compact_payload,
        )

    return _run_structured_tool("localize_lattice_nodes", operation)


@mcp.tool()
def compute_registration_qa(
    ct_filepath: str,
    localized_graph_filepath: str,
    output_report_filepath: str,
    threshold: float,
    registration_mode: Literal["autonomous_v2"],
    localization_report_filepath: str,
    analysis_scope_artifact_filepath: str,
    slice_output_filepath: str | None = None,
    bias_output_filepath: str | None = None,
    slice_index: int = 380,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Compute image, padded-ROI capture, and stricter metrology QA gates."""

    def operation() -> dict[str, Any]:
        ct, _ = _repository_path(
            ct_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        graph, _ = _repository_path(
            localized_graph_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output, _ = _repository_path(
            output_report_filepath, must_exist=False, expected_suffixes={".json"}
        )
        localization_report = (
            _repository_path(
                localization_report_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
            if localization_report_filepath
            else None
        )
        analysis_scope_artifact = _repository_path(
            analysis_scope_artifact_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )[0]
        slice_output = (
            _repository_path(
                slice_output_filepath, must_exist=False, expected_suffixes={".png"}
            )[0]
            if slice_output_filepath
            else None
        )
        bias_output = (
            _repository_path(
                bias_output_filepath, must_exist=False, expected_suffixes={".png"}
            )[0]
            if bias_output_filepath
            else None
        )
        payload = _compute_registration_qa(
            ct,
            graph,
            output,
            threshold=threshold,
            registration_mode=registration_mode,
            analysis_scope_artifact_path=analysis_scope_artifact,
            localization_report_path=localization_report,
            slice_output_path=slice_output,
            bias_output_path=bias_output,
            slice_index=slice_index,
            config=config,
            overwrite=overwrite,
        )
        return _core_response(
            "compute_registration_qa",
            f"Registration QA completed with gate {payload['gate']}",
            payload,
        )

    return _run_structured_tool("compute_registration_qa", operation)


@mcp.tool()
def segment_ct_dataset(
    input_filepath: str,
    output_filepath: str,
    threshold: float,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    retention: Literal["committed", "regenerable"] = "committed",
    chunk_depth: int = 16,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Write the canonical uint8 TIFF/NPY threshold mask in bounded slabs."""

    def operation() -> dict[str, Any]:
        input_path, _ = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        output_path, _ = _repository_path(
            output_filepath, must_exist=False, expected_suffixes={".npy"}
        )
        payload = _segment_ct_dataset_core(
            input_path,
            output_path,
            threshold=threshold,
            registration_mode=registration_mode,
            retention=retention,
            chunk_depth=chunk_depth,
            overwrite=overwrite,
        )
        message = (
            f"Saved {payload['result']['foreground_voxels']} foreground voxels "
            f"out of {payload['result']['total_voxels']} total voxels"
        )
        payload["result"]["message"] = message
        return _core_response("segment_ct_dataset", message, payload)

    return _run_structured_tool("segment_ct_dataset", operation)


@mcp.tool()
def visualize_slice(
    input_filepath: str,
    output_filepath: str,
    slice_index: int,
    axis: int = 0,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Render one TIFF/NPY slice and return only compact artifact metadata."""

    def operation() -> dict[str, Any]:
        input_path, _ = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        output_path, _ = _repository_path(
            output_filepath, must_exist=False, expected_suffixes={".png"}
        )
        payload = _visualize_slice_core(
            input_path,
            output_path,
            slice_index=slice_index,
            axis=axis,
            registration_mode=registration_mode,
            overwrite=overwrite,
        )
        return _core_response(
            "visualize_slice",
            f"Saved axis {axis}, slice {slice_index}",
            payload,
        )

    return _run_structured_tool("visualize_slice", operation)


@mcp.tool()
def compare_segmentation_masks(
    raw_filepath: str,
    mask_filepaths: list[str],
    thresholds: list[float],
    output_report_filepath: str | None = None,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Compare threshold masks without returning voxel arrays.

    The mask and threshold lists are positional pairs. Every mask must be a
    three-dimensional boolean or integer NPY array with the same shape as the
    raw volume.
    """
    def operation() -> dict[str, Any]:
        raw, _ = _repository_path(
            raw_filepath, must_exist=True, expected_suffixes={".npy", ".tif", ".tiff"}
        )
        masks = [
            _repository_path(path, must_exist=True, expected_suffixes={".npy", ".tif", ".tiff"})[0]
            for path in mask_filepaths
        ]
        report_path = (
            _repository_path(
                output_report_filepath, must_exist=False, expected_suffixes={".json"}
            )[0]
            if output_report_filepath
            else None
        )
        payload = _compare_masks_core(
            raw,
            masks,
            thresholds,
            registration_mode=registration_mode,
            output_report_path=report_path,
            overwrite=overwrite,
            repository_root=common.REPOSITORY_ROOT,
        )
        stats = dict(payload["result"])
        stats["candidates"] = [dict(item) for item in stats["candidates"]]
        payload["result"] = stats
        return _core_response(
            "compare_segmentation_masks",
            f"Compared {len(masks)} aligned segmentation mask(s)",
            payload,
        )

    return _run_structured_tool("compare_segmentation_masks", operation)


@mcp.tool()
def verify_canonical_segmentation(
    specimen_id: str,
    design_id: str,
    analysis_policy_artifact_filepath: str,
    exact_otsu_report_filepath: str,
    canonical_mask_filepath: str,
    mask_comparison_report_filepath: str,
    output_filepath: str,
    registration_mode: Literal["autonomous_v2"],
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Persist closed evidence that the canonical mask uses the exact Otsu result.

    This verifier independently replays the frozen exact-Otsu recipe and mask
    comparison, then requires the persisted Stage 2 reports to match that
    replay before it mints evidence.
    """

    def operation() -> dict[str, Any]:
        policy_path, policy_relative = _repository_path(
            analysis_policy_artifact_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        otsu_path, otsu_relative = _repository_path(
            exact_otsu_report_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        mask_path, mask_relative = _repository_path(
            canonical_mask_filepath,
            must_exist=True,
            expected_suffixes={".npy"},
        )
        comparison_path, comparison_relative = _repository_path(
            mask_comparison_report_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output_path, output_relative = _repository_path(
            output_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )

        _validate_specimen_manifest(
            policy_path,
            repository_root=common.REPOSITORY_ROOT,
            verify_files=False,
        )
        manifest = _load_json(policy_path)
        if (
            manifest.get("specimen_id") != specimen_id
            or manifest.get("design_id") != design_id
        ):
            raise ValueError(
                "Segmentation verification identity differs from the specimen manifest"
            )
        specimen_root = (common.REPOSITORY_ROOT / "analysis" / specimen_id).resolve()
        expected_policy_path = specimen_root / "config" / "specimen_manifest.json"
        expected_segmentation_root = specimen_root / "segmentation"
        expected_paths = {
            policy_path: expected_policy_path,
            otsu_path: expected_segmentation_root / "histogram_report.json",
            mask_path: expected_segmentation_root / "canonical_mask.npy",
            comparison_path: expected_segmentation_root / "mask_comparison.json",
            output_path: expected_segmentation_root
            / "segmentation_verification_mcp_response.json",
        }
        if any(
            actual != expected.resolve() for actual, expected in expected_paths.items()
        ):
            raise ValueError(
                "Segmentation verification inputs and output must use the fixed specimen-scoped paths"
            )

        analysis_parameters = manifest.get("analysis_parameters")
        if not isinstance(analysis_parameters, dict):
            raise ValueError("Specimen manifest has no analysis_parameters object")
        analysis_parameters_sha256 = _canonical_json_sha256(analysis_parameters)
        if manifest.get("analysis_parameters_sha256") != analysis_parameters_sha256:
            raise ValueError("Specimen manifest analysis_parameters hash is stale")
        if analysis_parameters.get("requested_analysis_scope") not in {
            "roi_screening",
            "direct_metrology",
        }:
            raise ValueError("Specimen manifest requested analysis scope is invalid")
        if analysis_parameters.get("registration", {}).get("mode") != registration_mode:
            raise ValueError("Segmentation verification registration mode is stale")
        segmentation_policy = analysis_parameters.get("segmentation")
        if not isinstance(segmentation_policy, dict):
            raise ValueError("Specimen manifest has no segmentation policy")
        ct_artifact = manifest.get("inputs", {}).get("ct")
        ct_metadata = manifest.get("inputs", {}).get("ct_metadata")
        if not isinstance(ct_artifact, dict) or not isinstance(ct_metadata, dict):
            raise ValueError("Specimen manifest CT binding is incomplete")
        ct_path, ct_relative = _repository_path(
            str(ct_artifact.get("path", "")),
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        ct_sha256 = _sha256_file(ct_path)
        if ct_artifact.get("sha256") != ct_sha256:
            raise ValueError("Specimen manifest CT SHA-256 is stale")

        expected_recipe = {
            "histogram_encoding": segmentation_policy.get("histogram_encoding"),
            "edge_slices_excluded": segmentation_policy.get("edge_slices_excluded"),
            "chunk_voxels": int(segmentation_policy.get("chunk_depth", 0))
            * int(ct_metadata.get("shape", [0, 0, 0])[1])
            * int(ct_metadata.get("shape", [0, 0, 0])[2]),
            "coarse_bins": segmentation_policy.get("coarse_bins"),
            "peak_smoothing_sigma_bins": segmentation_policy.get(
                "peak_smoothing_sigma_bins"
            ),
            "peak_prominence_fraction": segmentation_policy.get(
                "peak_prominence_fraction"
            ),
            "minimum_significant_peaks": segmentation_policy.get(
                "minimum_significant_peaks"
            ),
            "minimum_foreground_fraction": segmentation_policy.get(
                "minimum_foreground_fraction"
            ),
            "maximum_foreground_fraction": segmentation_policy.get(
                "maximum_foreground_fraction"
            ),
            "minimum_otsu_separability": segmentation_policy.get(
                "minimum_otsu_separability"
            ),
            "minimum_class_mean_separation_sigma": segmentation_policy.get(
                "minimum_class_mean_separation_sigma"
            ),
        }
        independently_replayed_otsu, _ = _replay_exact_otsu(
            ct_path,
            recipe=expected_recipe,
        )
        otsu = _load_json(otsu_path)
        threshold = independently_replayed_otsu.get("threshold")
        otsu_hashes = otsu.get("hashes")
        otsu_provenance = otsu.get("provenance")
        exact_otsu_fields = {
            "schema_version",
            "method",
            "method_version",
            "threshold",
            "threshold_histogram_bin",
            "threshold_comparison",
            "histogram_encoding",
            "recipe",
            "voxel_count",
            "foreground_voxel_count",
            "significant_modes",
            "histogram_sha256",
            "overall_pass",
        }
        floating_otsu_fields = {
            "foreground_fraction",
            "otsu_separability",
            "background_mean",
            "foreground_mean",
            "class_mean_separation_sigma",
        }

        def exact_json_equal(left: Any, right: Any) -> bool:
            """Compare JSON values without Python's bool/int/float coercions."""

            return _config_sha256({"value": left}) == _config_sha256(
                {"value": right}
            )

        exact_otsu_mismatch = any(
            not exact_json_equal(
                otsu.get(field), independently_replayed_otsu.get(field)
            )
            for field in exact_otsu_fields
        ) or any(
            not isinstance(otsu.get(field), (int, float))
            or isinstance(otsu.get(field), bool)
            or type(otsu.get(field))
            is not type(independently_replayed_otsu.get(field))
            or not math.isfinite(float(otsu[field]))
            or not math.isclose(
                float(otsu[field]),
                float(independently_replayed_otsu[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for field in floating_otsu_fields
        )
        independent_gates = independently_replayed_otsu.get("gates")
        persisted_gates = otsu.get("gates")
        expected_persisted_gates = (
            dict(independent_gates) if isinstance(independent_gates, dict) else {}
        )
        reference_replay = otsu.get("reference_replay")
        reference_replay_valid = reference_replay is None
        expected_config = {
            "recipe": expected_recipe,
            "registration_mode": registration_mode,
            "enforce_reference_replay": False,
        }
        if isinstance(reference_replay, dict):
            reference_gates = reference_replay.get("gates")
            expected_threshold = reference_replay.get("expected_threshold")
            expected_foreground_voxels = reference_replay.get(
                "expected_foreground_voxels"
            )
            enforced = reference_replay.get("enforced")
            expected_reference_gates = {
                "reference_threshold_matches": (
                    type(expected_threshold) is int
                    and exact_json_equal(expected_threshold, threshold)
                ),
                "reference_foreground_count_matches": (
                    type(expected_foreground_voxels) is int
                    and exact_json_equal(
                        expected_foreground_voxels,
                        independently_replayed_otsu.get("foreground_voxel_count"),
                    )
                ),
            }
            reference_replay_valid = (
                set(reference_replay)
                == {
                    "enforced",
                    "expected_threshold",
                    "expected_foreground_voxels",
                    "gates",
                }
                and type(enforced) is bool
                and type(expected_threshold) is int
                and expected_threshold >= 0
                and type(expected_foreground_voxels) is int
                and expected_foreground_voxels >= 0
                and isinstance(reference_gates, dict)
                and set(reference_gates) == set(expected_reference_gates)
                and all(
                    type(reference_gates.get(name)) is bool
                    and reference_gates.get(name) is expected
                    for name, expected in expected_reference_gates.items()
                )
            )
            expected_config = {
                "recipe": expected_recipe,
                "registration_mode": registration_mode,
                "enforce_reference_replay": enforced,
                "reference_threshold": expected_threshold,
                "reference_foreground_voxels": expected_foreground_voxels,
            }
            if enforced is True:
                expected_persisted_gates.update(expected_reference_gates)
                reference_replay_valid = reference_replay_valid and all(
                    expected_reference_gates.values()
                )
        expected_otsu_fields = (
            exact_otsu_fields
            | floating_otsu_fields
            | {
                "gates",
                "source_path",
                "registration_mode",
                "hashes",
                "provenance",
            }
        )
        if reference_replay is not None:
            expected_otsu_fields.add("reference_replay")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not np.isfinite(float(threshold))
            or set(otsu) != expected_otsu_fields
            or otsu.get("schema_version") != "exact-otsu-replay/1.0.0"
            or otsu.get("method") != segmentation_policy.get("method")
            or otsu.get("method_version") != segmentation_policy.get("method_version")
            or otsu.get("threshold_comparison")
            != segmentation_policy.get("comparison")
            or otsu.get("source_path") != ct_relative
            or otsu.get("registration_mode") != registration_mode
            or otsu.get("recipe") != expected_recipe
            or otsu.get("overall_pass") is not True
            or exact_otsu_mismatch
            or not reference_replay_valid
            or not isinstance(persisted_gates, dict)
            or persisted_gates != expected_persisted_gates
            or not all(value is True for value in persisted_gates.values())
            or not isinstance(otsu_hashes, dict)
            or set(otsu_hashes) != {"input_sha256", "config_sha256"}
            or otsu_hashes.get("input_sha256") != ct_sha256
            or otsu_hashes.get("config_sha256")
            != _config_sha256(expected_config)
            or not isinstance(otsu_provenance, dict)
            or set(otsu_provenance)
            != {
                "registration_mode",
                "threshold_selected_per_scan",
                "target_foreground_fraction_used",
                "defect_labels_read",
            }
            or otsu_provenance.get("registration_mode") != registration_mode
            or otsu_provenance.get("threshold_selected_per_scan") is not True
            or otsu_provenance.get("target_foreground_fraction_used") is not False
            or otsu_provenance.get("defect_labels_read") is not False
        ):
            raise ValueError(
                "Exact-Otsu report is not bound to the frozen specimen policy"
            )

        independent_comparison_payload = _compare_masks_core(
            ct_path,
            [mask_path],
            [float(threshold)],
            registration_mode=registration_mode,
            output_report_path=None,
            overwrite=False,
            repository_root=common.REPOSITORY_ROOT,
        )
        if independent_comparison_payload.get("gate") != "pass":
            raise ValueError(
                "Independent canonical-mask comparison did not pass"
            )
        independent_comparison = independent_comparison_payload.get("result")
        comparison = _load_json(comparison_path)
        if set(comparison) != {
            "status",
            "raw_path",
            "shape",
            "candidates",
            "registration_mode",
            "config_sha256",
            "overall_pass",
        }:
            raise ValueError("Segmentation comparison report schema is open or incomplete")
        candidates = comparison.get("candidates")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 1
            or not isinstance(candidates[0], dict)
        ):
            raise ValueError("Segmentation comparison must contain one canonical mask")
        candidate = candidates[0]
        if set(candidate) != {
            "threshold",
            "path",
            "dtype",
            "foreground_voxels",
            "expected_foreground_voxels",
            "total_voxels",
            "foreground_percent",
            "mismatched_voxels",
            "false_positive_voxels",
            "false_negative_voxels",
            "exact_threshold_match",
            "sha256",
        }:
            raise ValueError("Segmentation comparison candidate schema is open or incomplete")
        mask_sha256 = _sha256_file(mask_path)
        mask = _load_volume(mask_path)
        expected_shape = ct_metadata.get("shape")
        if (
            comparison.get("status") != "ok"
            or comparison.get("raw_path") != ct_relative
            or comparison.get("shape") != expected_shape
            or comparison.get("registration_mode") != registration_mode
            or comparison.get("overall_pass") is not True
            or not exact_json_equal(comparison, independent_comparison)
            or candidate.get("threshold") != threshold
            or candidate.get("path") != mask_relative
            or candidate.get("sha256") != mask_sha256
            or candidate.get("dtype") != "uint8"
            or list(mask.shape) != expected_shape
            or str(mask.dtype) != "uint8"
            or candidate.get("foreground_voxels")
            != otsu.get("foreground_voxel_count")
            or candidate.get("expected_foreground_voxels")
            != otsu.get("foreground_voxel_count")
            or candidate.get("total_voxels") != otsu.get("voxel_count")
            or candidate.get("mismatched_voxels") != 0
            or candidate.get("false_positive_voxels") != 0
            or candidate.get("false_negative_voxels") != 0
            or candidate.get("exact_threshold_match") is not True
        ):
            raise ValueError(
                "Canonical mask comparison is not bound to the exact-Otsu result"
            )

        request_binding = {
            "specimen_id": specimen_id,
            "design_id": design_id,
            "analysis_policy_artifact_filepath": policy_relative,
            "exact_otsu_report_filepath": otsu_relative,
            "canonical_mask_filepath": mask_relative,
            "mask_comparison_report_filepath": comparison_relative,
            "output_filepath": output_relative,
            "registration_mode": registration_mode,
            "overwrite": bool(overwrite),
        }
        segmentation_policy_sha256 = _canonical_json_sha256(segmentation_policy)
        evidence = {
            "schema_version": SEGMENTATION_VERIFICATION_EVIDENCE_SCHEMA_VERSION,
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "verify_canonical_segmentation",
            "status": "ok",
            "gate": "pass",
            "summary": "Persisted canonical segmentation verification",
            "specimen_id": specimen_id,
            "design_id": design_id,
            "requested_analysis_scope": analysis_parameters[
                "requested_analysis_scope"
            ],
            "registration_mode": registration_mode,
            "request": request_binding,
            "policy": {
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
            },
            "result": {
                "threshold": threshold,
                "threshold_comparison": otsu["threshold_comparison"],
                "shape": expected_shape,
                "dtype": "uint8",
                "voxel_count": independently_replayed_otsu["voxel_count"],
                "foreground_voxel_count": independently_replayed_otsu[
                    "foreground_voxel_count"
                ],
                "foreground_fraction": independently_replayed_otsu[
                    "foreground_fraction"
                ],
                "otsu_separability": independently_replayed_otsu[
                    "otsu_separability"
                ],
                "background_mean": independently_replayed_otsu["background_mean"],
                "foreground_mean": independently_replayed_otsu["foreground_mean"],
                "class_mean_separation_sigma": independently_replayed_otsu[
                    "class_mean_separation_sigma"
                ],
                "significant_modes": independently_replayed_otsu[
                    "significant_modes"
                ],
                "histogram_sha256": independently_replayed_otsu[
                    "histogram_sha256"
                ],
                "mismatched_voxels": 0,
                "false_positive_voxels": 0,
                "false_negative_voxels": 0,
                "exact_threshold_match": True,
                "overall_pass": True,
            },
            "bindings": {
                "analysis_policy_artifact": {
                    "path": policy_relative,
                    "sha256": _sha256_file(policy_path),
                    "role": "specimen_manifest",
                },
                "ct_volume": {
                    "path": ct_relative,
                    "sha256": ct_sha256,
                    "role": "ct_volume",
                },
                "exact_otsu_report": {
                    "path": otsu_relative,
                    "sha256": _sha256_file(otsu_path),
                    "role": "otsu_report",
                },
                "canonical_mask": {
                    "path": mask_relative,
                    "sha256": mask_sha256,
                    "role": "canonical_segmentation_mask",
                    "dtype": "uint8",
                    "shape": expected_shape,
                    "array_axes": ["z", "y", "x"],
                },
                "mask_comparison_report": {
                    "path": comparison_relative,
                    "sha256": _sha256_file(comparison_path),
                    "role": "segmentation_mask_comparison",
                },
            },
            "hashes": {
                "request_sha256": _canonical_json_sha256(request_binding),
                "analysis_policy_artifact_sha256": _sha256_file(policy_path),
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
                "ct_sha256": ct_sha256,
                "exact_otsu_report_sha256": _sha256_file(otsu_path),
                "canonical_mask_sha256": mask_sha256,
                "mask_comparison_report_sha256": _sha256_file(comparison_path),
            },
            "warnings": [],
            "error": None,
        }
        artifact = write_json_atomic(output_path, evidence, overwrite=overwrite)
        artifact["path"] = output_relative
        compact_result = {
            "schema_version": evidence["schema_version"],
            "specimen_id": specimen_id,
            "design_id": design_id,
            "requested_analysis_scope": evidence["requested_analysis_scope"],
            "registration_mode": registration_mode,
            **evidence["result"],
        }
        return _success_response(
            tool="verify_canonical_segmentation",
            gate="pass",
            summary=evidence["summary"],
            result=compact_result,
            artifacts={
                "segmentation_verification": {
                    **artifact,
                    "role": "segmentation_verification_mcp_response",
                    "retention": "committed",
                }
            },
            hashes={
                **evidence["hashes"],
                "segmentation_verification_sha256": artifact["sha256"],
            },
            warnings=[],
        )

    return _run_structured_tool("verify_canonical_segmentation", operation)
