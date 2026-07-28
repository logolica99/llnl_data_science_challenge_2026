from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Literal

import numpy as np
from fastmcp import FastMCP

try:
    from .part2_core import (
        classify_struts as _classify_struts,
        compare_segmentation_masks as _compare_masks_core,
        compute_detection_metrics as _compute_detection_metrics,
        compute_registration_qa as _compute_registration_qa,
        compute_strut_metrics as _compute_strut_metrics,
        error_response as _error_response,
        get_strut_report as _get_strut_report,
        label_deleted_edges as _label_deleted_edges,
        load_volume as _load_volume,
        localize_lattice_nodes as _localize_lattice_nodes,
        normalize_lattice_graph as _normalize_lattice_graph,
        register_lattice_to_ct as _register_lattice_to_ct,
        render_strut_evidence as _render_strut_evidence,
        replay_exact_otsu as _replay_exact_otsu,
        resolve_cad_graph_orientation as _resolve_cad_graph_orientation,
        success_response as _success_response,
        segment_ct_dataset as _segment_ct_dataset_core,
        volume_metadata as _volume_metadata,
        visualize_slice as _visualize_slice_core,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from .skeletonization import skeletonize_mask
    from .volume_artifacts import (
        render_volume_3d as _render_volume_3d,
        summarize_nde_artifacts as _summarize_nde_artifacts,
    )
    from .volume_metadata import inspect_volume_envelope
except ImportError:
    from part2_core import (
        classify_struts as _classify_struts,
        compare_segmentation_masks as _compare_masks_core,
        compute_detection_metrics as _compute_detection_metrics,
        compute_registration_qa as _compute_registration_qa,
        compute_strut_metrics as _compute_strut_metrics,
        error_response as _error_response,
        get_strut_report as _get_strut_report,
        label_deleted_edges as _label_deleted_edges,
        load_volume as _load_volume,
        localize_lattice_nodes as _localize_lattice_nodes,
        normalize_lattice_graph as _normalize_lattice_graph,
        register_lattice_to_ct as _register_lattice_to_ct,
        render_strut_evidence as _render_strut_evidence,
        replay_exact_otsu as _replay_exact_otsu,
        resolve_cad_graph_orientation as _resolve_cad_graph_orientation,
        success_response as _success_response,
        segment_ct_dataset as _segment_ct_dataset_core,
        volume_metadata as _volume_metadata,
        visualize_slice as _visualize_slice_core,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from skeletonization import skeletonize_mask
    from volume_artifacts import (
        render_volume_3d as _render_volume_3d,
        summarize_nde_artifacts as _summarize_nde_artifacts,
    )
    from volume_metadata import inspect_volume_envelope


# Initialize the MCP server
mcp = FastMCP("CT Segmentation")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@mcp.tool()
def inspect_volume_metadata(
    input_filepath: str,
    header_only: bool = True,
    include_sha256: bool = True,
    retention: Literal["committed", "external", "regenerable"] = "external",
) -> dict[str, Any]:
    """Inspect one repository CT volume and return manifest-ready metadata.

    Use header-only mode for specimen intake. It reads the NPY/TIFF header and
    streams the file for SHA-256 without decoding voxel intensities. Set
    include_sha256 to false only for a non-authoritative preview. Inputs are
    constrained to this repository and are never modified.
    """
    return inspect_volume_envelope(
        Path(input_filepath),
        repository_root=REPOSITORY_ROOT,
        header_only=header_only,
        include_sha256=include_sha256,
        retention=retention,
    )


def _input_npy_path(filepath: str) -> Path:
    """Resolve and validate an input NumPy volume path."""
    path = Path(filepath).expanduser().resolve()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Input file must use the .npy extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path


def _repository_path(
    filepath: str,
    *,
    must_exist: bool,
    expected_suffixes: set[str] | None = None,
) -> tuple[Path, str]:
    """Resolve one new Part 2 tool path without allowing repository escape."""

    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Input file does not exist: {relative.as_posix()}")
    if expected_suffixes and resolved.suffix.lower() not in expected_suffixes:
        choices = ", ".join(sorted(expected_suffixes))
        raise ValueError(f"Expected one of [{choices}], found {relative.as_posix()}")
    return resolved, relative.as_posix()


def _repository_output_directory(filepath: str) -> tuple[Path, str]:
    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(
            f"Output directory is an existing file: {relative.as_posix()}"
        )
    return resolved, relative.as_posix()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _structured_failure(tool: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        code = "input_not_found"
    elif isinstance(exc, FileExistsError):
        code = "artifact_exists"
    elif isinstance(exc, (ValueError, TypeError, IndexError)):
        code = "invalid_input"
    else:
        code = "tool_execution_failed"
    return _error_response(
        tool=tool,
        code=code,
        error_type=type(exc).__name__,
        message=str(exc),
    )


def _run_structured_tool(
    tool: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return operation()
    except Exception as exc:
        return _structured_failure(tool, exc)


def _relative_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Convert core artifact paths to repository-relative MCP paths."""

    result: dict[str, Any] = {}
    for name, metadata in artifacts.items():
        if not isinstance(metadata, dict):
            result[name] = metadata
            continue
        item = dict(metadata)
        if "path" in item:
            artifact_path = Path(item["path"])
            item["path"] = (
                artifact_path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
                if artifact_path.is_absolute()
                else artifact_path.as_posix()
            )
        result[name] = item
    return result


def _core_response(
    tool: str,
    summary: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Expose a deterministic core result without adding computation."""

    result = (
        dict(payload["result"])
        if isinstance(payload.get("result"), dict)
        else {
            key: value
            for key, value in payload.items()
            if key not in {"artifacts", "hashes", "warnings"}
        }
    )
    return _success_response(
        tool=tool,
        gate=payload["gate"],
        summary=summary,
        result=result,
        artifacts=_relative_artifacts(payload.get("artifacts", {})),
        hashes=dict(payload.get("hashes", {})),
        warnings=list(payload.get("warnings", [])),
    )


def _output_path(filepath: str, expected_suffix: str | None = None) -> Path:
    """Resolve an output path and create its parent directory."""
    path = Path(filepath).expanduser().resolve()
    if expected_suffix and path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"Output file must use the {expected_suffix} extension: {path}"
        )
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@mcp.tool()
def volume_info(
    input_filepath: str,
    include_sha256: bool = True,
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"] = "autonomous_v2",
) -> dict[str, Any]:
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
def load_lattice_graph(
    input_filepath: str,
    output_filepath: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize a lattice JSON to NPZ with explicit node/edge/cell ID maps."""

    def operation() -> dict[str, Any]:
        source, source_relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output, output_relative = _repository_path(
            output_filepath,
            must_exist=False,
            expected_suffixes={".npz"},
        )
        result = _normalize_lattice_graph(
            source,
            output,
            overwrite=overwrite,
        )
        result["source_path"] = source_relative
        result["output_path"] = output_relative
        result["config_sha256"] = _config_sha256(
            {"normalization_schema": "normalized-lattice-graph/1.0.0"}
        )
        warnings = list(result["warnings"])
        gate: Literal["pass", "manual_review"] = (
            "pass" if not warnings else "manual_review"
        )
        return _success_response(
            tool="load_lattice_graph",
            gate=gate,
            summary=(
                f"Normalized {result['counts']['nodes']} nodes, "
                f"{result['counts']['edges']} edges, and "
                f"{result['counts']['cells']} cells"
            ),
            result=result,
            artifacts={
                "normalized_graph": {
                    "path": output_relative,
                    "sha256": result["artifact_sha256"],
                    "role": "normalized_lattice_graph",
                    "retention": "regenerable",
                }
            },
            hashes={
                "input_sha256": result["source_sha256"],
                "artifact_sha256": result["artifact_sha256"],
                "config_sha256": result["config_sha256"],
            },
            warnings=warnings,
        )

    return _run_structured_tool("load_lattice_graph", operation)


@mcp.tool()
def resolve_cad_graph_orientation(
    nominal_graph_filepath: str,
    full_design_stl_filepath: str,
    output_filepath: str,
    sample_count: int = 9,
    scale_candidates: list[float] | None = None,
    ambiguity_absolute_mm: float = 0.0001,
    ambiguity_relative_fraction: float = 0.001,
    config_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve CAD/graph orientation from design geometry without CT access."""

    def operation() -> dict[str, Any]:
        graph, _ = _repository_path(
            nominal_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        stl, _ = _repository_path(
            full_design_stl_filepath, must_exist=True, expected_suffixes={".stl"}
        )
        output, _ = _repository_path(
            output_filepath, must_exist=False, expected_suffixes={".json"}
        )
        payload = _resolve_cad_graph_orientation(
            graph,
            stl,
            output,
            sample_count=sample_count,
            scale_candidates=scale_candidates,
            ambiguity_absolute_mm=ambiguity_absolute_mm,
            ambiguity_relative_fraction=ambiguity_relative_fraction,
            config_sha256=config_sha256,
            overwrite=overwrite,
        )
        artifact = dict(payload["artifact"])
        artifact["path"] = Path(artifact["path"]).relative_to(REPOSITORY_ROOT).as_posix()
        result = {
            key: value
            for key, value in payload.items()
            if key not in {"artifact", "hashes", "warnings"}
        }
        return _success_response(
            tool="resolve_cad_graph_orientation",
            gate=payload["gate"],
            summary=(
                "Resolved one CAD/graph orientation"
                if payload["gate"] == "pass"
                else "CAD/graph orientation requires bounded review"
            ),
            result=result,
            artifacts={"cad_graph_orientation": artifact},
            hashes={
                **payload["hashes"],
                "orientation_artifact_sha256": artifact["sha256"],
            },
            warnings=list(payload.get("warnings", [])),
        )

    return _run_structured_tool("resolve_cad_graph_orientation", operation)


@mcp.tool()
def label_deleted_edges(
    nominal_graph_filepath: str,
    baseline_stl_filepath: str,
    variant_stl_filepaths: dict[str, str],
    orientation_filepath: str,
    output_directory: str,
    development_split_filepath: str | None = None,
    sealed_split_filepath: str | None = None,
    label_report_filepath: str | None = None,
    expected_deletions: dict[str, int] | None = None,
    sample_count: int = 9,
    split_seed: int = 20260723,
    config_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Label all nominal edges by memory-aware, tube-emptiness CAD analysis."""

    def operation() -> dict[str, Any]:
        graph, _ = _repository_path(
            nominal_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        baseline, _ = _repository_path(
            baseline_stl_filepath, must_exist=True, expected_suffixes={".stl"}
        )
        orientation, _ = _repository_path(
            orientation_filepath, must_exist=True, expected_suffixes={".json"}
        )
        variants = {
            name: _repository_path(path, must_exist=True, expected_suffixes={".stl"})[0]
            for name, path in variant_stl_filepaths.items()
        }
        output, _ = _repository_output_directory(output_directory)

        def optional_output(
            path: str | None, suffixes: set[str] | None = None
        ) -> Path | None:
            return (
                _repository_path(
                    path,
                    must_exist=False,
                    expected_suffixes=suffixes or {".json"},
                )[0]
                if path
                else None
            )

        payload = _label_deleted_edges(
            graph,
            baseline,
            variants,
            orientation,
            output,
            development_split_path=optional_output(development_split_filepath),
            sealed_split_path=optional_output(sealed_split_filepath),
            label_report_path=optional_output(label_report_filepath, {".md", ".json"}),
            expected_deletions=expected_deletions,
            sample_count=sample_count,
            split_seed=split_seed,
            config_sha256=config_sha256,
            overwrite=overwrite,
        )
        return _core_response(
            "label_deleted_edges",
            f"Labeled every nominal edge with gate {payload['gate']}",
            payload,
        )

    return _run_structured_tool("label_deleted_edges", operation)


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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"] = "autonomous_v2",
    enforce_reference_replay: bool = False,
    reference_threshold: int = 40054,
    reference_foreground_voxels: int = 58653410,
    overwrite: bool = False,
) -> dict[str, Any]:
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
            artifact["path"] = artifact_path.relative_to(REPOSITORY_ROOT).as_posix()
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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"],
    ct_filepath: str | None = None,
    aligned_graph_filepath: str | None = None,
    threshold: float | None = None,
    config: dict[str, Any] | None = None,
    analysis_config_filepath: str | None = None,
    freeze_receipt_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"],
    registration_report_filepath: str | None = None,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
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
        payload = _localize_lattice_nodes(
            ct,
            graph,
            output_graph,
            output_report,
            threshold=threshold,
            registration_mode=registration_mode,
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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"],
    localization_report_filepath: str,
    slice_output_filepath: str | None = None,
    bias_output_filepath: str | None = None,
    slice_index: int = 380,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
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
def compute_strut_metrics(
    ct_filepath: str,
    localized_graph_filepath: str,
    output_metrics_filepath: str,
    output_profiles_filepath: str,
    output_report_filepath: str,
    threshold: float,
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"],
    registration_qa_filepath: str | None = None,
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write per-ID padded-ROI occupancy, gap, connectivity, radius, and curvature."""

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
        metrics, _ = _repository_path(
            output_metrics_filepath, must_exist=False, expected_suffixes={".csv"}
        )
        profiles, _ = _repository_path(
            output_profiles_filepath, must_exist=False, expected_suffixes={".json"}
        )
        report, _ = _repository_path(
            output_report_filepath, must_exist=False, expected_suffixes={".json"}
        )
        qa = (
            _repository_path(
                registration_qa_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
            if registration_qa_filepath
            else None
        )
        payload = _compute_strut_metrics(
            ct,
            graph,
            metrics,
            profiles,
            report,
            threshold=threshold,
            registration_mode=registration_mode,
            config=config,
            registration_qa_path=qa,
            overwrite=overwrite,
        )
        return _core_response(
            "compute_strut_metrics",
            f"Wrote metrics for {payload['counts']['metric_rows']} struts",
            payload,
        )

    return _run_structured_tool("compute_strut_metrics", operation)


@mcp.tool()
def classify_struts(
    metrics_filepath: str,
    output_classifications_filepath: str,
    output_thresholds_filepath: str,
    thresholds: dict[str, Any] | None = None,
    thresholds_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply frozen cutoffs with missing > broken > thin > present precedence."""

    def operation() -> dict[str, Any]:
        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        classifications, _ = _repository_path(
            output_classifications_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        output_thresholds, _ = _repository_path(
            output_thresholds_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        if (thresholds is None) == (thresholds_filepath is None):
            raise ValueError("Provide exactly one of thresholds or thresholds_filepath")
        policy: dict[str, Any] | Path
        if thresholds_filepath:
            policy = _repository_path(
                thresholds_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
        else:
            policy = thresholds or {}
        payload = _classify_struts(
            metrics,
            policy,
            classifications,
            output_thresholds,
            overwrite=overwrite,
        )
        return _core_response(
            "classify_struts",
            f"Classified {payload['counts']['total']} struts",
            payload,
        )

    return _run_structured_tool("classify_struts", operation)


@mcp.tool()
def render_strut_evidence(
    ct_filepath: str,
    localized_graph_filepath: str,
    profiles_filepath: str,
    output_directory: str,
    strut_id: int,
    threshold: float,
    crop_margin_voxels: int = 8,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render three orthogonal CT crops and an occupancy profile for one strut."""

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
        profiles, _ = _repository_path(
            profiles_filepath, must_exist=True, expected_suffixes={".json"}
        )
        output, _ = _repository_output_directory(output_directory)
        payload = _render_strut_evidence(
            ct,
            graph,
            profiles,
            output,
            strut_id=strut_id,
            threshold=threshold,
            crop_margin_voxels=crop_margin_voxels,
            overwrite=overwrite,
        )
        return _core_response(
            "render_strut_evidence",
            f"Rendered evidence packet for strut {strut_id}",
            payload,
        )

    return _run_structured_tool("render_strut_evidence", operation)


@mcp.tool()
def compute_detection_metrics(
    classifications_filepath: str,
    sealed_labels_filepath: str,
    output_filepath: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Eval-side strict/lenient recall, Wilson intervals, and confusion matrix."""

    def operation() -> dict[str, Any]:
        classifications, _ = _repository_path(
            classifications_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        labels, _ = _repository_path(
            sealed_labels_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output, _ = _repository_path(
            output_filepath, must_exist=False, expected_suffixes={".json"}
        )
        payload = _compute_detection_metrics(
            classifications, labels, output, overwrite=overwrite
        )
        return _core_response(
            "compute_detection_metrics",
            (
                f"Scored {payload['sealed_strut_count']} sealed struts; "
                f"lenient recall {payload['lenient_recall']['value']:.3f}"
            ),
            payload,
        )

    return _run_structured_tool("compute_detection_metrics", operation)


@mcp.tool()
def get_strut_report(
    strut_id: int,
    metrics_filepath: str,
    classifications_filepath: str,
    thresholds_filepath: str,
    evidence_manifest_filepath: str | None = None,
) -> dict[str, Any]:
    """Return one compact artifact-backed strut record without recomputation."""

    def operation() -> dict[str, Any]:
        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        classifications, _ = _repository_path(
            classifications_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        thresholds_path, _ = _repository_path(
            thresholds_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        evidence = (
            _repository_path(
                evidence_manifest_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )[0]
            if evidence_manifest_filepath
            else None
        )
        payload = _get_strut_report(
            strut_id,
            metrics,
            classifications,
            thresholds_path,
            evidence_manifest_path=evidence,
        )
        return _core_response(
            "get_strut_report",
            f"Loaded artifact-backed report for strut {strut_id}",
            payload,
        )

    return _run_structured_tool("get_strut_report", operation)


@mcp.tool()
def segment_ct_dataset(
    input_filepath: str,
    output_filepath: str,
    threshold: float,
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"] = "autonomous_v2",
    retention: Literal["committed", "regenerable"] = "committed",
    chunk_depth: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"] = "autonomous_v2",
    overwrite: bool = False,
) -> dict[str, Any]:
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
    registration_mode: Literal["challenge_aligned_json", "autonomous_v2"] = "autonomous_v2",
    overwrite: bool = False,
) -> dict[str, Any]:
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
            repository_root=REPOSITORY_ROOT,
        )
        stats = dict(payload["result"])
        stats["candidates"] = [dict(item) for item in stats["candidates"]]
        payload["result"] = stats
        envelope = _core_response(
            "compare_segmentation_masks",
            f"Compared {len(masks)} aligned segmentation mask(s)",
            payload,
        )
        # Preserve compact legacy summary fields for direct Python callers.
        return {**envelope, **stats}

    return _run_structured_tool("compare_segmentation_masks", operation)


@mcp.tool()
def summarize_nde_artifacts(
    raw_filepath: str,
    mask_filepath: str,
    skeleton_filepath: str | None = None,
) -> dict[str, Any]:
    """Summarize aligned raw, mask, and optional skeleton NPY artifacts.

    The tool returns report-ready scalar metrics and never returns voxel arrays.
    Skeleton endpoints and branch points use a 26-connected neighborhood.
    """
    return _summarize_nde_artifacts(
        raw_filepath,
        mask_filepath,
        skeleton_filepath,
    )


@mcp.tool()
def render_volume_3d(
    input_filepath: str,
    output_filepath: str,
    surface_level: float = 0.5,
    downsample_factor: int = 2,
    elevation: float = 30.0,
    azimuth: float = 45.0,
    skeleton_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render a volume isosurface and optional skeleton overlay to PNG.

    surface_level is normalized to the downsampled volume range and must be
    strictly between zero and one. The tool writes the image and returns only
    compact render metadata.
    """
    return _render_volume_3d(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        surface_level=surface_level,
        downsample_factor=downsample_factor,
        elevation=elevation,
        azimuth=azimuth,
        skeleton_filepath=skeleton_filepath,
        overwrite=overwrite,
    )


@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.

    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).

    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = _input_npy_path(input_filepath)
        mask = np.load(input_path, mmap_mode="r", allow_pickle=False)
        if mask.ndim != 3:
            raise ValueError(
                f"Expected a 3D mask, but {input_path} has shape {mask.shape}."
            )

        output_path = _output_path(output_filepath, ".npy")

        # skeletonize_mask reports progress with print(). Redirect that output to
        # stderr so it cannot interfere with MCP's JSON-RPC messages on stdout.
        with redirect_stdout(sys.stderr):
            skeleton = skeletonize_mask(str(input_path), str(output_path))

        if skeleton is None or not output_path.is_file():
            raise RuntimeError("Skeletonization did not produce an output file.")

        skeleton_voxels = int(np.count_nonzero(skeleton))
        return (
            f"Skeletonized {input_path}. Saved {skeleton_voxels} skeleton "
            f"voxels to {output_path}."
        )
    except Exception as exc:
        return f"Error skeletonizing segmentation mask: {exc}"


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
