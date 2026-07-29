"""Stage 4 reporting and spatial-visualization MCP adapters."""

from __future__ import annotations

from typing import Any

from part2_core import compute_spatial_stats as _compute_spatial_stats
from part2_core import get_strut_report as _get_strut_report
from part2_core import render_lattice_3d as _render_lattice_3d

from .common import (
    MCPResponseEnvelope,
    _core_response,
    _repository_path,
    _run_structured_tool,
)
from .registry import mcp


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
