"""Production MCP server assembly and public imports.

Tool implementations are grouped by immutable pipeline stage under
``mcp_tools``. Importing this module registers the complete production surface.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llnl_nde.mcp_tools.common import (
    MCPErrorEnvelope,
    MCPResponseEnvelope,
    REPOSITORY_ROOT,
)
from llnl_nde.mcp_tools.registry import mcp
from llnl_nde.mcp_tools.specimen_ingest_stage0 import (
    inspect_volume_metadata,
    load_lattice_graph,
)
from llnl_nde.mcp_tools.data_prep_stage1 import (
    compare_segmentation_masks,
    compute_registration_qa,
    localize_lattice_nodes,
    register_lattice_to_ct,
    replay_exact_otsu,
    segment_ct_dataset,
    verify_canonical_segmentation,
    visualize_slice,
    volume_info,
)
from llnl_nde.mcp_tools.strut_metrics_stage2 import compute_strut_metrics
from llnl_nde.mcp_tools.defect_analysis_stage3 import (
    classify_struts,
    render_strut_evidence,
)
from llnl_nde.mcp_tools.reporting_stage4 import (
    compute_spatial_stats,
    get_strut_report,
    render_lattice_3d,
)

__all__ = [
    "MCPErrorEnvelope",
    "MCPResponseEnvelope",
    "REPOSITORY_ROOT",
    "classify_struts",
    "compare_segmentation_masks",
    "compute_registration_qa",
    "compute_spatial_stats",
    "compute_strut_metrics",
    "get_strut_report",
    "inspect_volume_metadata",
    "load_lattice_graph",
    "localize_lattice_nodes",
    "mcp",
    "register_lattice_to_ct",
    "render_lattice_3d",
    "render_strut_evidence",
    "replay_exact_otsu",
    "segment_ct_dataset",
    "verify_canonical_segmentation",
    "visualize_slice",
    "volume_info",
]

if __name__ == "__main__":
    mcp.run()
