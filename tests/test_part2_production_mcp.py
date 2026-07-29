"""MCP-client smoke tests for the production Part 2 wrappers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastmcp import Client
from fastmcp.exceptions import ToolError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.server import mcp  # noqa: E402


class ProductionMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        graph = {
            "junctions": [
                {"id": 10, "position": [5, 5, 5]},
                {"id": 20, "position": [10, 5, 5]},
            ],
            "struts": [{"id": 30, "junction0": 10, "junction1": 20}],
            "unit_cells": [{"id": 40, "indices": [0, 0, 0], "struts": [30]}],
        }
        self.nominal = self.root / "nominal.json"
        self.aligned = self.root / "aligned.json"
        self.nominal.write_text(json.dumps(graph), encoding="utf-8")
        self.aligned.write_text(json.dumps(graph), encoding="utf-8")
        self.volume = self.root / "volume.npy"
        np.save(self.volume, np.zeros((16, 16, 16), dtype=np.uint16))

    async def test_tools_registered_with_autonomous_only_schema(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        contract_tools: set[str] = set()
        for contract_path in (REPOSITORY_ROOT / "analysis" / "contracts").glob(
            "*.json"
        ):
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract_tools.update(
                item["name"]
                for item in contract["required_dependencies"]["mcp_tools"]
            )
        self.assertEqual(contract_tools, set(tools))
        expected = {
            "register_lattice_to_ct",
            "localize_lattice_nodes",
            "compute_registration_qa",
            "compute_strut_metrics",
            "classify_struts",
            "render_strut_evidence",
            "get_strut_report",
            "compute_spatial_stats",
            "render_lattice_3d",
        }
        self.assertTrue(expected.issubset(set(tools)))
        for research_only in (
            "compute_detection_metrics",
            "summarize_nde_artifacts",
            "render_volume_3d",
            "skeletonize",
            "explore_ct_thresholds",
            "export_stage3_validation_csvs",
        ):
            self.assertNotIn(research_only, tools)
        registration_mode = tools["register_lattice_to_ct"].parameters[
            "properties"
        ]["registration_mode"]
        self.assertEqual("autonomous_v2", registration_mode["const"])
        async with Client(mcp) as client:
            with self.assertRaisesRegex(ToolError, "autonomous_v2"):
                await client.call_tool(
                    "register_lattice_to_ct",
                    {
                        "nominal_graph_filepath": str(self.nominal),
                        "output_graph_filepath": str(self.root / "rejected.json"),
                        "output_report_filepath": str(self.root / "rejected-report.json"),
                        "registration_mode": "challenge_aligned_json",
                        "ct_filepath": str(self.volume),
                        "aligned_graph_filepath": str(self.aligned),
                    },
                )

    async def test_autonomous_mode_rejects_aligned_json_fail_closed(self) -> None:
        async with Client(mcp) as client:
            call = await client.call_tool(
                "register_lattice_to_ct",
                {
                    "nominal_graph_filepath": str(self.nominal),
                    "output_graph_filepath": str(self.root / "bad.json"),
                    "output_report_filepath": str(self.root / "bad-report.json"),
                    "registration_mode": "autonomous_v2",
                    "ct_filepath": str(self.volume),
                    "aligned_graph_filepath": str(self.aligned),
                    "threshold": 1,
                },
            )
        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("forbids aligned_graph_path", result["error"]["message"])


if __name__ == "__main__":
    unittest.main()
