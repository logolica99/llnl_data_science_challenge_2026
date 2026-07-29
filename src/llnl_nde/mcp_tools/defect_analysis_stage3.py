"""Stage 3 defect-classification and evidence MCP adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llnl_nde.core import classify_struts as _classify_struts
from llnl_nde.core import render_strut_evidence as _render_strut_evidence

from .common import (
    MCPResponseEnvelope,
    _core_response,
    _repository_output_directory,
    _repository_path,
    _run_structured_tool,
)
from .registry import mcp


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
