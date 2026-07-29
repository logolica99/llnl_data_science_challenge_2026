"""Stage 0 specimen-ingest MCP adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from part2_core import normalize_lattice_graph as _normalize_lattice_graph
from part2_core import success_response as _success_response
from part2_core.artifacts import sha256_json, write_json_atomic
from volume_metadata import inspect_volume_envelope

from . import common
from .common import (
    MCPResponseEnvelope,
    _config_sha256,
    _repository_path,
    _run_structured_tool,
    _structured_failure,
)
from .registry import mcp


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
            repository_root=common.REPOSITORY_ROOT,
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
