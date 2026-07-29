import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from fastmcp import FastMCP
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field

try:
    from .part2_core import (
        classify_struts as _classify_struts,
        compare_segmentation_masks as _compare_masks_core,
        compute_registration_qa as _compute_registration_qa,
        compute_spatial_stats as _compute_spatial_stats,
        compute_strut_metrics as _compute_strut_metrics,
        error_response as _error_response,
        get_strut_report as _get_strut_report,
        load_volume as _load_volume,
        localize_lattice_nodes as _localize_lattice_nodes,
        normalize_lattice_graph as _normalize_lattice_graph,
        register_lattice_to_ct as _register_lattice_to_ct,
        render_strut_evidence as _render_strut_evidence,
        render_lattice_3d as _render_lattice_3d,
        replay_exact_otsu as _replay_exact_otsu,
        success_response as _success_response,
        segment_ct_dataset as _segment_ct_dataset_core,
        volume_metadata as _volume_metadata,
        visualize_slice as _visualize_slice_core,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from .part2_core.artifacts import sha256_json, write_json_atomic
    from .volume_metadata import inspect_volume_envelope
    from .specimen_manifest import (
        canonical_json_sha256 as _canonical_json_sha256,
        load_json as _load_json,
        validate_manifest as _validate_specimen_manifest,
    )
except ImportError:
    from part2_core import (
        classify_struts as _classify_struts,
        compare_segmentation_masks as _compare_masks_core,
        compute_registration_qa as _compute_registration_qa,
        compute_spatial_stats as _compute_spatial_stats,
        compute_strut_metrics as _compute_strut_metrics,
        error_response as _error_response,
        get_strut_report as _get_strut_report,
        load_volume as _load_volume,
        localize_lattice_nodes as _localize_lattice_nodes,
        normalize_lattice_graph as _normalize_lattice_graph,
        register_lattice_to_ct as _register_lattice_to_ct,
        render_strut_evidence as _render_strut_evidence,
        render_lattice_3d as _render_lattice_3d,
        replay_exact_otsu as _replay_exact_otsu,
        success_response as _success_response,
        segment_ct_dataset as _segment_ct_dataset_core,
        volume_metadata as _volume_metadata,
        visualize_slice as _visualize_slice_core,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from part2_core.artifacts import sha256_json, write_json_atomic
    from volume_metadata import inspect_volume_envelope
    from specimen_manifest import (
        canonical_json_sha256 as _canonical_json_sha256,
        load_json as _load_json,
        validate_manifest as _validate_specimen_manifest,
    )


# Initialize the MCP server
mcp = FastMCP("CT Segmentation")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_VERIFICATION_EVIDENCE_SCHEMA_VERSION = (
    "segmentation-verification-mcp-evidence/1.0.0"
)


class MCPErrorEnvelope(BaseModel):
    """Closed error member for the shared Part 2 MCP response envelope."""

    model_config = ConfigDict(extra="forbid")

    code: str
    type: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class MCPResponseEnvelope(BaseModel):
    """Closed top-level response shape used by production Part 2 tools."""

    model_config = ConfigDict(extra="forbid")

    response_schema_version: Literal["part2-mcp-response/1.0.0"]
    tool: str
    status: Literal["ok", "error"]
    gate: Literal["pass", "halt", "manual_review"]
    summary: str
    result: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    hashes: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: MCPErrorEnvelope | None = None

    def __getitem__(self, key: str) -> Any:
        """Retain compact direct-Python access without opening outputSchema."""

        if key in type(self).model_fields:
            return getattr(self, key)
        return self.result[key]


class EmptyMCPPayload(BaseModel):
    """Closed empty member used by structured error responses."""

    model_config = ConfigDict(extra="forbid")


class VolumeSpacingProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    field: str
    raw_value: str | int | float


class VolumeResolutionSpacingProvenance(VolumeSpacingProvenance):
    resolution_unit: str


class VolumeSpacingAxis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | int | float
    unit: str
    provenance: VolumeResolutionSpacingProvenance | VolumeSpacingProvenance


class VolumeSpacing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    z: VolumeSpacingAxis
    y: VolumeSpacingAxis
    x: VolumeSpacingAxis


class VolumeStatistics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_computed", "computed"]
    minimum: str | int | float | None
    maximum: str | int | float | None
    mean: str | int | float | None
    finite_count: str | int
    nonfinite_count: str | int


class VolumeArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    role: Literal["ct_volume"]
    retention: Literal["committed", "external", "regenerable"]


class VolumeManifestMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["npy", "tiff"]
    shape: list[int]
    dtype: str
    byte_order: Literal["little", "big", "not_applicable"]
    array_axes: Literal["unknown"] | list[str]
    voxel_spacing: VolumeSpacing


class VolumeManifestFragment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ct_volume: VolumeArtifactBinding
    ct_metadata: VolumeManifestMetadata


class VolumeMetadataResult(BaseModel):
    """Exact metadata payload emitted by the trusted volume inspector."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    authoritative: bool
    inspection_mode: Literal["header_only", "streaming_statistics"]
    method: Literal["volume_metadata"]
    method_version: Literal["1.0.0"]
    output_schema_version: Literal["volume-metadata/1.0.0"]
    path: str
    sha256: str
    file_bytes: int
    format: Literal["npy", "tiff"]
    shape: list[int]
    ndim: int
    dtype: str
    dtype_string: str
    byte_order: Literal["little", "big", "not_applicable"]
    axes: str
    voxel_count: int
    array_bytes: int
    voxel_spacing: VolumeSpacing
    statistics: VolumeStatistics
    manifest_fragment: VolumeManifestFragment


class VolumeMetadataRequestBinding(BaseModel):
    """Exact normalized arguments covered by Stage 0 MCP evidence."""

    model_config = ConfigDict(extra="forbid")

    input_filepath: str
    output_filepath: str
    call_receipt_filepath: str
    header_only: bool
    include_sha256: bool
    retention: Literal["committed", "external", "regenerable"]


class VolumeMetadataHeaderFacts(BaseModel):
    """Header facts parsed by the MCP-owned volume inspector."""

    model_config = ConfigDict(extra="forbid")

    file_bytes: int
    format: Literal["npy", "tiff"]
    shape: list[int]
    ndim: int
    dtype: str
    dtype_string: str
    byte_order: Literal["little", "big", "not_applicable"]
    axes: str
    voxel_count: int
    array_bytes: int


class VolumeMetadataEvidence(BaseModel):
    """Closed persisted scientific evidence written before the call receipt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["volume-metadata-mcp-evidence/1.1.0"]
    response_schema_version: Literal["part2-mcp-response/1.0.0"]
    tool: Literal["inspect_volume_metadata"]
    status: Literal["ok"]
    gate: Literal["pass", "manual_review"]
    summary: str
    request: VolumeMetadataRequestBinding
    result: VolumeMetadataResult
    warnings: list[str]
    error: None


class VolumeMetadataCallArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    role: Literal["ct_metadata_mcp_response"]
    retention: Literal["committed"]


class VolumeMetadataCallArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_response: VolumeMetadataCallArtifactBinding


class VolumeMetadataCallHashes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_sha256: str
    request_sha256: str
    result_sha256: str
    header_facts_sha256: str
    metadata_response_sha256: str


class VolumeMetadataMCPCallReceipt(BaseModel):
    """Closed, self-hashed receipt for one actual MCP metadata call."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["volume-metadata-mcp-call-receipt/1.0.0"]
    response_schema_version: Literal["part2-mcp-response/1.0.0"]
    tool: Literal["inspect_volume_metadata"]
    status: Literal["ok"]
    gate: Literal["pass", "manual_review"]
    summary: str
    request: VolumeMetadataRequestBinding
    header_facts: VolumeMetadataHeaderFacts
    artifacts: VolumeMetadataCallArtifacts
    hashes: VolumeMetadataCallHashes
    warnings: list[str]
    error: None
    canonical_call_receipt_sha256: str


class VolumeMetadataResponseArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    changed: bool
    role: Literal["ct_metadata_mcp_response"]
    retention: Literal["committed"]


class VolumeMetadataCallReceiptResponseArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    changed: bool
    role: Literal["ct_metadata_mcp_call_receipt"]
    retention: Literal["committed"]


class VolumeMetadataResponseArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_response: VolumeMetadataResponseArtifact
    call_receipt: VolumeMetadataCallReceiptResponseArtifact


class VolumeMetadataResponseHashes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_sha256: str
    request_sha256: str
    metadata_response_sha256: str
    call_receipt_sha256: str


class VolumeMetadataMCPError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    type: str
    message: str
    details: EmptyMCPPayload


class VolumeMetadataMCPResponse(BaseModel):
    """Fully closed output contract for ``inspect_volume_metadata``."""

    model_config = ConfigDict(extra="forbid")

    response_schema_version: Literal["part2-mcp-response/1.0.0"]
    tool: Literal["inspect_volume_metadata"]
    status: Literal["ok", "error"]
    gate: Literal["pass", "halt", "manual_review"]
    summary: str
    result: VolumeMetadataResult | EmptyMCPPayload
    artifacts: VolumeMetadataResponseArtifacts | EmptyMCPPayload
    hashes: VolumeMetadataResponseHashes | EmptyMCPPayload
    warnings: list[str]
    error: VolumeMetadataMCPError | None

    def __getitem__(self, key: str) -> Any:
        """Retain compact direct-Python access used by contract tests."""

        if key in type(self).model_fields:
            value = getattr(self, key)
        elif isinstance(self.result, VolumeMetadataResult):
            value = getattr(self.result, key)
        else:
            raise KeyError(key)
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


@mcp.tool()
def inspect_volume_metadata(
    input_filepath: str,
    output_filepath: str,
    call_receipt_filepath: str,
    header_only: bool = True,
    include_sha256: bool = True,
    retention: Literal["committed", "external", "regenerable"] = "external",
) -> VolumeMetadataMCPResponse:
    """Inspect one repository CT volume and return manifest-ready metadata.

    Use header-only mode for specimen intake. It reads the NPY/TIFF header and
    streams the file for SHA-256 without decoding voxel intensities. Set
    include_sha256 to false only for a non-authoritative preview. Inputs are
    constrained to this repository and are never modified.
    """
    def operation() -> dict[str, Any]:
        source, source_relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        output, output_relative = _repository_path(
            output_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        call_receipt_output, call_receipt_relative = _repository_path(
            call_receipt_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        if call_receipt_output == output:
            raise ValueError(
                "Metadata evidence and MCP call receipt require distinct paths"
            )
        inspection = inspect_volume_envelope(
            source,
            repository_root=REPOSITORY_ROOT,
            header_only=header_only,
            include_sha256=include_sha256,
            retention=retention,
        )
        authoritative = bool(include_sha256 and header_only)
        inspection["authoritative"] = authoritative
        gate: Literal["pass", "manual_review"] = (
            "pass" if authoritative else "manual_review"
        )
        warnings = (
            []
            if authoritative
            else [
                "Only header_only=true with include_sha256=true is authoritative for intake"
            ]
        )
        request_binding = {
            "input_filepath": source_relative,
            "output_filepath": output_relative,
            "call_receipt_filepath": call_receipt_relative,
            "header_only": bool(header_only),
            "include_sha256": bool(include_sha256),
            "retention": retention,
        }
        summary = (
            "Persisted authoritative header-only CT metadata response"
            if authoritative
            else "Persisted non-authoritative CT metadata preview"
        )
        evidence = VolumeMetadataEvidence.model_validate({
            "schema_version": "volume-metadata-mcp-evidence/1.1.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": gate,
            "summary": summary,
            "request": request_binding,
            "result": inspection,
            "warnings": warnings,
            "error": None,
        }).model_dump(mode="json")
        artifact = write_json_atomic(output, evidence)
        artifact["path"] = output_relative
        header_facts = {
            field: inspection[field]
            for field in (
                "file_bytes",
                "format",
                "shape",
                "ndim",
                "dtype",
                "dtype_string",
                "byte_order",
                "axes",
                "voxel_count",
                "array_bytes",
            )
        }
        call_receipt_base = {
            "schema_version": "volume-metadata-mcp-call-receipt/1.0.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": gate,
            "summary": summary,
            "request": request_binding,
            "header_facts": header_facts,
            "artifacts": {
                "metadata_response": {
                    "path": output_relative,
                    "sha256": artifact["sha256"],
                    "role": "ct_metadata_mcp_response",
                    "retention": "committed",
                }
            },
            "hashes": {
                "input_sha256": inspection["sha256"],
                "request_sha256": sha256_json(request_binding),
                "result_sha256": sha256_json(inspection),
                "header_facts_sha256": sha256_json(header_facts),
                "metadata_response_sha256": artifact["sha256"],
            },
            "warnings": warnings,
            "error": None,
        }
        call_receipt_document = VolumeMetadataMCPCallReceipt.model_validate(
            {
                **call_receipt_base,
                "canonical_call_receipt_sha256": sha256_json(call_receipt_base),
            }
        ).model_dump(mode="json")
        call_receipt_artifact = write_json_atomic(
            call_receipt_output, call_receipt_document
        )
        call_receipt_artifact["path"] = call_receipt_relative
        return _success_response(
            tool="inspect_volume_metadata",
            gate=gate,
            summary=summary,
            result=inspection,
            artifacts={
                "metadata_response": {
                    **artifact,
                    "role": "ct_metadata_mcp_response",
                    "retention": "committed",
                },
                "call_receipt": {
                    **call_receipt_artifact,
                    "role": "ct_metadata_mcp_call_receipt",
                    "retention": "committed",
                },
            },
            hashes={
                "input_sha256": inspection["sha256"],
                "request_sha256": sha256_json(request_binding),
                "metadata_response_sha256": artifact["sha256"],
                "call_receipt_sha256": call_receipt_artifact["sha256"],
            },
            warnings=warnings,
        )

    try:
        return VolumeMetadataMCPResponse.model_validate(operation())
    except Exception as exc:
        return VolumeMetadataMCPResponse.model_validate(
            _structured_failure("inspect_volume_metadata", exc)
        )


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
) -> MCPResponseEnvelope:
    try:
        return MCPResponseEnvelope.model_validate(operation())
    except Exception as exc:
        return MCPResponseEnvelope.model_validate(_structured_failure(tool, exc))


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
def load_lattice_graph(
    input_filepath: str,
    output_filepath: str,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
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


_STAGE2_METRIC_INPUT_ROLES = frozenset(
    {
        "analysis_ready_specimen_manifest",
        "data_prep_completion_receipt",
        "analysis_config",
        "ct_volume",
        "localized_graph",
        "registration_qa",
        "otsu_report",
        "canonical_segmentation_mask",
        "segmentation_mask_comparison",
    }
)
_FORBIDDEN_STAGE2_PATH_TOKENS = frozenset(
    {"label", "labels", "development", "dev_split", "sealed", "ground_truth", "classification"}
)


def _stage2_json(path: Path, role: str) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Stage 2 {role} must be a JSON object")
    return value


def _stage2_artifact_record(
    records: dict[str, dict[str, Any]],
    role: str,
    expected_relative: str,
    expected_path: Path,
) -> dict[str, Any]:
    record = records.get(role)
    if record is None:
        raise ValueError(f"Stage 2 handoff is missing required artifact role {role}")
    if record.get("path") != expected_relative:
        raise ValueError(f"Stage 2 handoff path mismatch for {role}")
    digest = record.get("sha256")
    if not isinstance(digest, str) or digest != _sha256_file(expected_path):
        raise ValueError(f"Stage 2 handoff hash mismatch for {role}")
    return record


def _validate_stage2_metric_bundle(
    *,
    handoff_path: Path,
    handoff_relative: str,
    paths: dict[str, tuple[Path, str]],
) -> tuple[dict[str, Any], dict[str, Any], float, dict[str, Any]]:
    """Fail closed unless the MCP request reproduces one canonical Stage 2 handoff."""

    handoff = _stage2_json(handoff_path, "handoff")
    expected_handoff_fields = {
        "schema_version",
        "specimen_id",
        "stage_number",
        "stage",
        "owner",
        "attempt",
        "run_token",
        "created_at",
        "registration_mode",
        "config_sha256",
        "contract_version",
        "contract_sha256",
        "predecessor_receipt_sha256",
        "input_artifacts",
        "forbidden_operations",
        "canonical_handoff_sha256",
    }
    if set(handoff) != expected_handoff_fields:
        raise ValueError("Stage 2 handoff is open-ended or schema-incompatible")
    handoff_base = {
        key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"
    }
    if (
        handoff.get("schema_version") != "part2-stage-handoff/1.0.0"
        or handoff.get("stage_number") != 2
        or handoff.get("stage") != "strut_metrics"
        or handoff.get("owner") != "strut_metrics"
        or handoff.get("registration_mode") != "autonomous_v2"
        or handoff.get("canonical_handoff_sha256")
        != _canonical_json_sha256(handoff_base)
    ):
        raise ValueError("Stage 2 handoff identity or canonical hash is invalid")
    contract_path = REPOSITORY_ROOT / "analysis" / "contracts" / "strut_metrics.json"
    contract = _stage2_json(contract_path, "contract")
    if (
        handoff.get("contract_version") != contract.get("schema_version")
        or handoff.get("contract_sha256") != _sha256_file(contract_path)
    ):
        raise ValueError("Stage 2 handoff is bound to a stale strut-metrics contract")
    if handoff.get("forbidden_operations") != contract.get("forbidden_operations"):
        raise ValueError("Stage 2 handoff forbidden operations differ from the live contract")
    predecessor = handoff.get("predecessor_receipt_sha256")
    if (
        not isinstance(predecessor, str)
        or len(predecessor) != 64
        or any(character not in "0123456789abcdef" for character in predecessor)
    ):
        raise ValueError("Stage 2 handoff lacks the verified Stage 1 receipt hash")
    artifacts = handoff.get("input_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Stage 2 handoff input_artifacts must be an array")
    records: dict[str, dict[str, Any]] = {}
    for record in artifacts:
        if not isinstance(record, dict) or not {"role", "path", "sha256"} <= set(record):
            raise ValueError("Stage 2 handoff contains a malformed artifact record")
        role = record.get("role")
        artifact_path = record.get("path")
        if not isinstance(role, str) or role in records or not isinstance(artifact_path, str):
            raise ValueError("Stage 2 handoff contains duplicate or invalid artifact roles")
        lowered_parts = {part.lower() for part in Path(artifact_path).parts}
        if (
            role not in _STAGE2_METRIC_INPUT_ROLES
            or lowered_parts & _FORBIDDEN_STAGE2_PATH_TOKENS
            or any(token in role.lower() for token in _FORBIDDEN_STAGE2_PATH_TOKENS)
        ):
            raise ValueError(f"Stage 2 handoff exposes forbidden or unauthorized artifact {role}")
        records[role] = record
    if set(records) != _STAGE2_METRIC_INPUT_ROLES:
        missing = sorted(_STAGE2_METRIC_INPUT_ROLES - set(records))
        extra = sorted(set(records) - _STAGE2_METRIC_INPUT_ROLES)
        raise ValueError(f"Stage 2 handoff artifact roles mismatch; missing={missing}, extra={extra}")
    for role, (path, relative) in paths.items():
        _stage2_artifact_record(records, role, relative, path)
    control_config_hash = handoff.get("config_sha256")
    if (
        not isinstance(control_config_hash, str)
        or len(control_config_hash) != 64
        or any(character not in "0123456789abcdef" for character in control_config_hash)
    ):
        raise ValueError("Stage 2 handoff has an invalid frozen control-config hash")

    specimen_id = handoff.get("specimen_id")
    if not isinstance(specimen_id, str) or not specimen_id:
        raise ValueError("Stage 2 handoff specimen_id is invalid")
    expected_handoff = (
        Path("analysis")
        / specimen_id
        / "handoffs"
        / f"stage_2_strut_metrics_attempt_{handoff['attempt']}.json"
    ).as_posix()
    if handoff_relative != expected_handoff:
        raise ValueError("Stage 2 handoff is outside its canonical specimen path")

    config = _stage2_json(paths["analysis_config"][0], "analysis config")
    schema_path = REPOSITORY_ROOT / "analysis" / "schema" / "strut_metrics_input.schema.json"
    schema = _stage2_json(schema_path, "configuration schema")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        detail = "; ".join(error.message for error in errors[:5])
        raise ValueError(f"Frozen Stage 2 analysis config is invalid: {detail}")
    fragment = config["stage_2_strut_metrics"]

    manifest = _stage2_json(
        paths["analysis_ready_specimen_manifest"][0], "analysis-ready specimen manifest"
    )
    if manifest.get("lifecycle_state") != "analysis_ready" or manifest.get("specimen_id") != specimen_id:
        raise ValueError("Stage 2 requires the matching analysis-ready specimen manifest")
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("Analysis-ready manifest has no closed inputs object")
    ct_binding = manifest_inputs.get("ct")
    mask_binding = manifest_inputs.get("canonical_mask")
    if not isinstance(ct_binding, dict) or not isinstance(mask_binding, dict):
        raise ValueError("Analysis-ready manifest does not bind CT and canonical mask")
    if (
        ct_binding.get("path") != paths["ct_volume"][1]
        or ct_binding.get("sha256") != records["ct_volume"]["sha256"]
        or ct_binding.get("role") != "ct_volume"
        or mask_binding.get("path") != paths["canonical_segmentation_mask"][1]
        or mask_binding.get("sha256") != records["canonical_segmentation_mask"]["sha256"]
        or mask_binding.get("role") != "canonical_segmentation_mask"
        or mask_binding.get("dtype") != "uint8"
        or mask_binding.get("array_axes") != ["z", "y", "x"]
        or mask_binding.get("retention") != "committed"
    ):
        raise ValueError("Manifest CT or canonical-mask binding differs from Stage 2 handoff")
    ct_metadata = manifest_inputs.get("ct_metadata")
    if not isinstance(ct_metadata, dict) or mask_binding.get("shape") != ct_metadata.get("shape"):
        raise ValueError("Canonical mask shape binding differs from CT metadata")
    analysis_parameters = manifest.get("analysis_parameters")
    if not isinstance(analysis_parameters, dict):
        raise ValueError("Analysis-ready manifest has no frozen analysis_parameters")
    requested_scope = analysis_parameters.get("requested_analysis_scope")
    if requested_scope not in {"roi_screening", "direct_metrology"}:
        raise ValueError("Analysis-ready manifest requested_analysis_scope is invalid")
    if analysis_parameters.get("registration", {}).get("mode") != handoff.get(
        "registration_mode"
    ):
        raise ValueError("Analysis-ready manifest registration mode differs from the handoff")
    derived = manifest.get("derived")
    registration_result = (
        derived.get("registration_result") if isinstance(derived, dict) else None
    )
    registration_values = (
        registration_result.get("values")
        if isinstance(registration_result, dict)
        else None
    )
    if not isinstance(registration_values, dict):
        raise ValueError("Analysis-ready manifest has no finalized registration result")

    completion = _stage2_json(
        paths["data_prep_completion_receipt"][0], "data-prep completion receipt"
    )
    completion_base = {
        key: value for key, value in completion.items() if key != "canonical_completion_sha256"
    }
    if (
        completion.get("schema_version") != "data-prep-completion/1.2.0"
        or completion.get("specimen_id") != specimen_id
        or completion.get("design_id") != manifest.get("design_id")
        or completion.get("lifecycle_state") != "analysis_ready"
        or completion.get("canonical_completion_sha256")
        != _canonical_json_sha256(completion_base)
        or completion.get("analysis_ready_manifest_sha256")
        != _canonical_json_sha256(manifest)
        or completion.get("canonical_mask") != mask_binding
        or completion.get("registration_mode") != handoff.get("registration_mode")
        or completion.get("requested_analysis_scope") != requested_scope
    ):
        raise ValueError("Data-prep completion receipt is stale or incompatible")
    authorized_outputs = completion.get("authorized_outputs")
    unauthorized_outputs = completion.get("unauthorized_outputs")
    if (
        not isinstance(authorized_outputs, list)
        or not isinstance(unauthorized_outputs, list)
        or any(not isinstance(value, str) for value in authorized_outputs)
        or any(not isinstance(value, str) for value in unauthorized_outputs)
        or set(authorized_outputs) & set(unauthorized_outputs)
    ):
        raise ValueError("Data-prep output authorization lists are malformed or overlap")
    if (
        registration_values.get("requested_analysis_scope") != requested_scope
        or registration_values.get("authorized_outputs") != authorized_outputs
        or registration_values.get("unauthorized_outputs") != unauthorized_outputs
    ):
        raise ValueError(
            "Finalized manifest registration scope or output authorizations differ "
            "from the data-prep completion receipt"
        )
    metrology_outputs = {"absolute_metrology", "direct_dimensional_measurement"}
    if requested_scope == "roi_screening" and (
        set(authorized_outputs) & metrology_outputs
        or not metrology_outputs <= set(unauthorized_outputs)
    ):
        raise ValueError("ROI-screening Stage 1 receipt incorrectly authorizes metrology")
    if requested_scope == "direct_metrology" and not metrology_outputs <= set(
        authorized_outputs
    ):
        raise ValueError("Direct-metrology Stage 1 receipt lacks metrology authorization")

    qa = _stage2_json(paths["registration_qa"][0], "registration QA")
    if (
        qa.get("specimen_id") != specimen_id
        or qa.get("gate") != "pass"
        or qa.get("overall_pass") is not True
        or qa.get("registration_mode") != handoff.get("registration_mode")
    ):
        raise ValueError("Stage 1 registration QA did not pass for this specimen")

    otsu = _stage2_json(paths["otsu_report"][0], "exact Otsu report")
    threshold = otsu.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or otsu.get("overall_pass") is not True
        or not math.isclose(
            float(fragment["otsu_threshold"]), float(threshold), rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("Frozen Stage 2 threshold does not match the passing Stage 1 Otsu result")

    comparison = _stage2_json(
        paths["segmentation_mask_comparison"][0], "segmentation mask comparison"
    )
    candidates = comparison.get("candidates")
    matching_candidates = [
        candidate
        for candidate in candidates or []
        if isinstance(candidate, dict)
        and candidate.get("path") == paths["canonical_segmentation_mask"][1]
        and candidate.get("sha256") == records["canonical_segmentation_mask"]["sha256"]
        and candidate.get("exact_threshold_match") is True
        and math.isclose(float(candidate.get("threshold", math.nan)), float(threshold), abs_tol=1e-12)
    ]
    if comparison.get("overall_pass") is not True or len(matching_candidates) != 1:
        raise ValueError("Canonical mask lacks an exact passing Stage 1 comparison binding")

    input_provenance = {
        "stage_2_handoff": {
            "path": handoff_relative,
            "sha256": _sha256_file(handoff_path),
            "canonical_sha256": handoff["canonical_handoff_sha256"],
        },
        "predecessor_stage_1_receipt_sha256": predecessor,
        "contract_sha256": handoff["contract_sha256"],
        "config_sha256": handoff["config_sha256"],
        "requested_analysis_scope": requested_scope,
        "authorized_outputs": sorted(authorized_outputs),
        "unauthorized_outputs": sorted(unauthorized_outputs),
        "input_artifacts": {
            role: {"path": record["path"], "sha256": record["sha256"]}
            for role, record in sorted(records.items())
        },
    }
    return handoff, config, float(threshold), input_provenance


@mcp.tool()
def compute_strut_metrics(
    stage_2_handoff_filepath: str,
    analysis_ready_specimen_manifest_filepath: str,
    data_prep_completion_receipt_filepath: str,
    analysis_config_filepath: str,
    ct_filepath: str,
    localized_graph_filepath: str,
    registration_qa_filepath: str,
    otsu_report_filepath: str,
    canonical_segmentation_mask_filepath: str,
    segmentation_mask_comparison_filepath: str,
    output_directory: str,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Compute strict Stage 2 label-blind evidence from one sealed Stage 1 handoff."""

    def operation() -> dict[str, Any]:
        handoff, handoff_relative = _repository_path(
            stage_2_handoff_filepath, must_exist=True, expected_suffixes={".json"}
        )
        specifications = {
            "analysis_ready_specimen_manifest": (
                analysis_ready_specimen_manifest_filepath,
                {".json"},
            ),
            "data_prep_completion_receipt": (
                data_prep_completion_receipt_filepath,
                {".json"},
            ),
            "analysis_config": (analysis_config_filepath, {".json"}),
            "ct_volume": (ct_filepath, {".npy", ".tif", ".tiff"}),
            "localized_graph": (localized_graph_filepath, {".json"}),
            "registration_qa": (registration_qa_filepath, {".json"}),
            "otsu_report": (otsu_report_filepath, {".json"}),
            "canonical_segmentation_mask": (
                canonical_segmentation_mask_filepath,
                {".npy"},
            ),
            "segmentation_mask_comparison": (
                segmentation_mask_comparison_filepath,
                {".json"},
            ),
        }
        paths = {
            role: _repository_path(filepath, must_exist=True, expected_suffixes=suffixes)
            for role, (filepath, suffixes) in specifications.items()
        }
        handoff_document, config, threshold, input_provenance = _validate_stage2_metric_bundle(
            handoff_path=handoff,
            handoff_relative=handoff_relative,
            paths=paths,
        )
        output, output_relative = _repository_output_directory(output_directory)
        expected_output = (
            Path("analysis") / str(handoff_document["specimen_id"]) / "struts"
        ).as_posix()
        if output_relative != expected_output:
            raise ValueError("Stage 2 output_directory is not the canonical specimen struts path")
        if overwrite:
            raise ValueError(
                "Strict Stage 2 artifacts are immutable; use exact idempotent replay"
            )
        payload = _compute_strut_metrics(
            paths["ct_volume"][0],
            paths["localized_graph"][0],
            output / "per_strut_metrics.csv",
            output / "per_strut_profiles.json",
            output / "metrics_report.json",
            threshold=threshold,
            registration_mode=str(handoff_document["registration_mode"]),
            config=config,
            registration_qa_path=paths["registration_qa"][0],
            canonical_mask_path=paths["canonical_segmentation_mask"][0],
            output_calibration_path=output / "corridor_calibration.json",
            input_provenance=input_provenance,
            overwrite=overwrite,
        )
        return _core_response(
            "compute_strut_metrics",
            f"Measured {payload['counts']['metric_rows']} nominal struts with gate {payload['gate']}",
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
) -> MCPResponseEnvelope:
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
) -> MCPResponseEnvelope:
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
def get_strut_report(
    strut_id: int,
    metrics_filepath: str,
    classifications_filepath: str,
    thresholds_filepath: str,
    evidence_manifest_filepath: str | None = None,
) -> MCPResponseEnvelope:
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
def compute_spatial_stats(
    localized_graph_filepath: str,
    classifications_filepath: str,
    metrics_filepath: str,
    output_statistics_filepath: str,
    output_figure_filepath: str,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Write graph-aware spatial statistics and a compact figure for Stage 4."""

    def operation() -> dict[str, Any]:
        graph, _ = _repository_path(
            localized_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        classifications, _ = _repository_path(
            classifications_filepath, must_exist=True, expected_suffixes={".json"}
        )
        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        statistics, _ = _repository_path(
            output_statistics_filepath, must_exist=False, expected_suffixes={".json"}
        )
        figure, _ = _repository_path(
            output_figure_filepath, must_exist=False, expected_suffixes={".png"}
        )
        payload = _compute_spatial_stats(
            graph,
            classifications,
            metrics,
            statistics,
            figure,
            overwrite=overwrite,
        )
        return _core_response(
            "compute_spatial_stats",
            f"Summarized {payload['counts']['struts']} classified struts",
            payload,
        )

    return _run_structured_tool("compute_spatial_stats", operation)


@mcp.tool()
def render_lattice_3d(
    localized_graph_filepath: str,
    classifications_filepath: str,
    output_filepath: str,
    elevation: float = 24.0,
    azimuth: float = 38.0,
    line_width: float = 0.8,
    node_size: float = 1.5,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Render every nominal strut from frozen graph-aware classifications."""

    def operation() -> dict[str, Any]:
        graph, _ = _repository_path(
            localized_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        classifications, _ = _repository_path(
            classifications_filepath, must_exist=True, expected_suffixes={".json"}
        )
        output, _ = _repository_path(
            output_filepath, must_exist=False, expected_suffixes={".png"}
        )
        payload = _render_lattice_3d(
            graph,
            classifications,
            output,
            elevation=elevation,
            azimuth=azimuth,
            line_width=line_width,
            node_size=node_size,
            overwrite=overwrite,
        )
        return _core_response(
            "render_lattice_3d",
            f"Rendered {payload['counts']['total']} classified graph struts",
            payload,
        )

    return _run_structured_tool("render_lattice_3d", operation)


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
            repository_root=REPOSITORY_ROOT,
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
            repository_root=REPOSITORY_ROOT,
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
        specimen_root = (REPOSITORY_ROOT / "analysis" / specimen_id).resolve()
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
            repository_root=REPOSITORY_ROOT,
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


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
