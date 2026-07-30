"""Seal-free hackathon MCP adapters for agentic Stage 0–3 demos.

These tools wrap the same ``llnl_nde.core`` science as production, but they do
not require hash-sealed handoffs or stage receipts. Production Stage 2/3 tools
remain the authoritative sealed path.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from llnl_nde.core.defect_analysis import (
    DEFAULT_STAGE3_CONFIG,
    DEFECT_KINDS,
    analyze_strut_specialist as _analyze_strut_specialist,
    export_stage3_validation_csvs as _export_stage3_validation_csvs,
    merge_strut_classifications as _merge_strut_classifications,
    prepare_hackathon_report_classifications as _prepare_hackathon_report_classifications,
)
from llnl_nde.core.localization import localize_lattice_nodes as _localize_lattice_nodes
from llnl_nde.core.strut_metrics import compute_strut_metrics as _compute_strut_metrics

from .common import (
    MCPResponseEnvelope,
    _core_response,
    _repository_output_directory,
    _repository_path,
    _run_structured_tool,
)
from .registry import mcp


HACKATHON_LOCALIZATION_CONFIG: dict[str, Any] = {
    "minimum_primary_or_stable_coarse_fraction": 0.0,
    "maximum_fallback_fraction": 1.0,
    "maximum_ambiguous_fraction": 1.0,
    "maximum_rejected_fraction": 1.0,
    "maximum_boundary_limited_fraction": 1.0,
}


@mcp.tool()
def hackathon_localize_lattice_nodes(
    ct_filepath: str,
    registered_graph_filepath: str,
    output_graph_filepath: str,
    output_report_filepath: str,
    threshold: float,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Localize registered nodes without a hashed specimen-manifest policy."""

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
        merged = {
            **HACKATHON_LOCALIZATION_CONFIG,
            **(config or {}),
        }
        payload = _localize_lattice_nodes(
            ct,
            graph,
            output_graph,
            output_report,
            threshold=threshold,
            registration_mode=registration_mode,
            config=merged,
            overwrite=overwrite,
        )
        compact = dict(payload)
        compact.pop("records", None)
        return _core_response(
            "hackathon_localize_lattice_nodes",
            (
                f"Localized {payload['counts']['accepted_nodes']} of "
                f"{payload['counts']['nodes']} nodes (hackathon)"
            ),
            compact,
        )

    return _run_structured_tool("hackathon_localize_lattice_nodes", operation)


@mcp.tool()
def hackathon_compute_strut_metrics(
    ct_filepath: str,
    localized_graph_filepath: str,
    output_metrics_filepath: str,
    output_profiles_filepath: str,
    output_report_filepath: str,
    threshold: float,
    registration_mode: Literal["autonomous_v2"] = "autonomous_v2",
    config: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Measure every strut without a Stage 2 hash-sealed handoff."""

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
        payload = _compute_strut_metrics(
            ct,
            graph,
            metrics,
            profiles,
            report,
            threshold=threshold,
            registration_mode=registration_mode,
            config=config,
            overwrite=overwrite,
        )
        return _core_response(
            "hackathon_compute_strut_metrics",
            (
                f"Measured {(payload.get('counts') or {}).get('metric_rows', '?')} "
                "struts (hackathon)"
            ),
            payload,
        )

    return _run_structured_tool("hackathon_compute_strut_metrics", operation)


@mcp.tool()
def hackathon_analyze_defect(
    metrics_filepath: str,
    profiles_filepath: str,
    output_findings_filepath: str,
    specimen_id: str,
    defect_kind: Literal["missing", "broken", "thin", "bent"],
    analysis_config_filepath: str | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Run one defect specialist agent tool without Stage 3 receipt binding."""

    def operation() -> dict[str, Any]:
        import json

        if defect_kind not in DEFECT_KINDS:
            raise ValueError(f"Unsupported defect_kind: {defect_kind}")
        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        profiles, _ = _repository_path(
            profiles_filepath, must_exist=True, expected_suffixes={".json"}
        )
        output, _ = _repository_path(
            output_findings_filepath, must_exist=False, expected_suffixes={".json"}
        )
        if analysis_config_filepath is None:
            config: dict[str, Any] = {
                "stage_3_defect_analysis": copy.deepcopy(DEFAULT_STAGE3_CONFIG)
            }
        else:
            config_path, _ = _repository_path(
                analysis_config_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
        payload = _analyze_strut_specialist(
            metrics,
            profiles,
            config,
            output,
            specimen_id=specimen_id,
            defect_kind=defect_kind,
            overwrite=overwrite,
        )
        return _core_response(
            "hackathon_analyze_defect",
            f"Analyzed {defect_kind} findings for {specimen_id}",
            payload,
        )

    return _run_structured_tool("hackathon_analyze_defect", operation)


@mcp.tool()
def hackathon_merge_defect_classifications(
    metrics_filepath: str,
    profiles_filepath: str,
    findings_missing_filepath: str,
    findings_broken_filepath: str,
    findings_thin_filepath: str,
    findings_bent_filepath: str,
    output_classifications_filepath: str,
    output_thresholds_filepath: str,
    output_decision_log_filepath: str,
    specimen_id: str,
    analysis_config_filepath: str | None = None,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Merge specialist findings with fixed precedence (no Stage 3 handoff)."""

    def operation() -> dict[str, Any]:
        import json

        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        profiles, _ = _repository_path(
            profiles_filepath, must_exist=True, expected_suffixes={".json"}
        )
        findings = {
            "missing": _repository_path(
                findings_missing_filepath, must_exist=True, expected_suffixes={".json"}
            )[0],
            "broken": _repository_path(
                findings_broken_filepath, must_exist=True, expected_suffixes={".json"}
            )[0],
            "thin": _repository_path(
                findings_thin_filepath, must_exist=True, expected_suffixes={".json"}
            )[0],
            "bent": _repository_path(
                findings_bent_filepath, must_exist=True, expected_suffixes={".json"}
            )[0],
        }
        classifications, _ = _repository_path(
            output_classifications_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        thresholds, _ = _repository_path(
            output_thresholds_filepath, must_exist=False, expected_suffixes={".json"}
        )
        decision_log, _ = _repository_path(
            output_decision_log_filepath, must_exist=False, expected_suffixes={".md"}
        )
        if analysis_config_filepath is None:
            config: dict[str, Any] = {
                "stage_3_defect_analysis": copy.deepcopy(DEFAULT_STAGE3_CONFIG)
            }
        else:
            config_path, _ = _repository_path(
                analysis_config_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
        payload = _merge_strut_classifications(
            metrics,
            profiles,
            config,
            findings,
            classifications,
            thresholds,
            decision_log,
            specimen_id=specimen_id,
            overwrite=overwrite,
        )
        return _core_response(
            "hackathon_merge_defect_classifications",
            f"Merged defect classifications for {specimen_id}",
            payload,
        )

    return _run_structured_tool("hackathon_merge_defect_classifications", operation)


@mcp.tool()
def hackathon_export_defect_csvs(
    classifications_filepath: str,
    missing_findings_filepath: str,
    broken_findings_filepath: str,
    metrics_filepath: str,
    nominal_graph_filepath: str,
    output_directory: str,
    excluded_nominal_axis: str = "y",
    excluded_nominal_value: float = 18.0,
    coordinate_tolerance: float = 1e-9,
) -> MCPResponseEnvelope:
    """Export missing/broken CSVs under analysis/ (seal-free hackathon path)."""

    def operation() -> dict[str, Any]:
        classifications, _ = _repository_path(
            classifications_filepath, must_exist=True, expected_suffixes={".json"}
        )
        missing, _ = _repository_path(
            missing_findings_filepath, must_exist=True, expected_suffixes={".json"}
        )
        broken, _ = _repository_path(
            broken_findings_filepath, must_exist=True, expected_suffixes={".json"}
        )
        metrics, _ = _repository_path(
            metrics_filepath, must_exist=True, expected_suffixes={".csv"}
        )
        nominal, _ = _repository_path(
            nominal_graph_filepath, must_exist=True, expected_suffixes={".json"}
        )
        output, _ = _repository_output_directory(output_directory)
        payload = _export_stage3_validation_csvs(
            classifications,
            missing,
            broken,
            metrics,
            nominal,
            output,
            excluded_nominal_axis=excluded_nominal_axis,
            excluded_nominal_value=excluded_nominal_value,
            coordinate_tolerance=coordinate_tolerance,
            overwrite=True,
        )
        return _core_response(
            "hackathon_export_defect_csvs",
            "Exported missing/broken defect CSVs (hackathon)",
            payload,
        )

    return _run_structured_tool("hackathon_export_defect_csvs", operation)


@mcp.tool()
def hackathon_prepare_report_classifications(
    classifications_filepath: str,
    output_classifications_filepath: str,
    nominal_graph_filepath: str | None = None,
    metrics_filepath: str | None = None,
    excluded_nominal_axis: str = "y",
    excluded_nominal_value: float = 18.0,
    coordinate_tolerance: float = 1e-9,
    require_disconnected_for_broken: bool = True,
    overwrite: bool = False,
) -> MCPResponseEnvelope:
    """Remap deferred + crop-plane + connected-bite broken → present for report."""

    def operation() -> dict[str, Any]:
        source, _ = _repository_path(
            classifications_filepath, must_exist=True, expected_suffixes={".json"}
        )
        destination, _ = _repository_path(
            output_classifications_filepath,
            must_exist=False,
            expected_suffixes={".json"},
        )
        nominal = None
        if nominal_graph_filepath is not None:
            nominal, _ = _repository_path(
                nominal_graph_filepath,
                must_exist=True,
                expected_suffixes={".json"},
            )
        metrics = None
        if metrics_filepath is not None:
            metrics, _ = _repository_path(
                metrics_filepath,
                must_exist=True,
                expected_suffixes={".csv"},
            )
        payload = _prepare_hackathon_report_classifications(
            source,
            destination,
            nominal_graph_path=nominal,
            metrics_path=metrics,
            excluded_nominal_axis=excluded_nominal_axis,
            excluded_nominal_value=excluded_nominal_value,
            coordinate_tolerance=coordinate_tolerance,
            require_disconnected_for_broken=require_disconnected_for_broken,
            overwrite=overwrite,
        )
        return _core_response(
            "hackathon_prepare_report_classifications",
            (
                "Prepared report classifications "
                f"(missing={payload['counts'].get('missing')}, "
                f"broken={payload['counts'].get('broken')})"
            ),
            payload,
        )

    return _run_structured_tool(
        "hackathon_prepare_report_classifications", operation
    )
