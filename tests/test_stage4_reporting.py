"""Stage 4 graph-aware reporting tool contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from fastmcp import Client


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import MCPResponseEnvelope, mcp  # noqa: E402
from part2_core.spatial import (  # noqa: E402
    compute_spatial_stats,
    render_lattice_3d,
)
from part2_core.struts import METRIC_FIELDS  # noqa: E402


class Stage4ReportingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph = self.root / "localized_graph.json"
        self.classifications = self.root / "classified_struts.json"
        self.metrics = self.root / "per_strut_metrics.csv"
        self.graph.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": 10, "position": [0.0, 0.0, 0.0]},
                        {"id": 20, "position": [1.0, 0.0, 0.0]},
                        {"id": 30, "position": [2.0, 0.0, 0.0]},
                        {"id": 40, "position": [2.0, 1.0, 0.0]},
                    ],
                    "struts": [
                        {"id": 101, "junction0": 10, "junction1": 20},
                        {"id": 205, "junction0": 20, "junction1": 30},
                        {"id": 999, "junction0": 30, "junction1": 40},
                    ],
                    "unit_cells": [
                        {"id": 7, "indices": [0, 0, 0], "struts": [101, 205, 999]}
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.classifications.write_text(
            json.dumps(
                {
                    "schema_version": "part2-strut-classification/1.0.0",
                    "classifications": [
                        {"strut_id": 999, "class": "present", "bent": False},
                        {"strut_id": 101, "class": "missing", "bent": False},
                        {"strut_id": 205, "class": "broken", "bent": True},
                    ],
                }
            ),
            encoding="utf-8",
        )
        rows = [
            self._metric_row(101, 10, 20, occupancy=0.01, gap=0.9, radius=0.0),
            self._metric_row(205, 20, 30, occupancy=0.4, gap=0.4, radius=1.0),
            self._metric_row(999, 30, 40, occupancy=0.9, gap=0.0, radius=2.0),
        ]
        with self.metrics.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _metric_row(
        strut_id: int,
        junction0: int,
        junction1: int,
        *,
        occupancy: float,
        gap: float,
        radius: float,
    ) -> dict[str, object]:
        return {
            "strut_id": strut_id,
            "junction0_id": junction0,
            "junction1_id": junction1,
            "length_voxels": 1.0,
            "corridor_foreground_fraction": occupancy,
            "maximum_axial_gap_samples": int(round(gap * 10)),
            "maximum_axial_gap_fraction": gap,
            "endpoint0_support_fraction": 1.0,
            "endpoint1_support_fraction": 1.0,
            "interior_component_count": 1,
            "largest_component_fraction": 1.0 - gap,
            "edt_radius_median_voxels": radius,
            "centerline_curvature_rms_voxels": 0.2,
            "roi_in_bounds_fraction": 1.0,
            "roi_valid": "true",
        }

    def test_core_writes_id_complete_spatial_artifacts(self) -> None:
        statistics = self.root / "spatial_statistics.json"
        figure = self.root / "spatial_statistics.png"

        result = compute_spatial_stats(
            self.graph,
            self.classifications,
            self.metrics,
            statistics,
            figure,
        )

        self.assertEqual("pass", result["gate"])
        self.assertEqual(
            {"missing": 1, "broken": 1, "thin": 0, "present": 1},
            result["class_counts"],
        )
        self.assertEqual([[101, 205]], result["defect_clusters"]["cluster_strut_ids"])
        self.assertEqual(1, result["counts"]["bent"])
        self.assertTrue(statistics.is_file())
        self.assertEqual(b"\x89PNG\r\n\x1a\n", figure.read_bytes()[:8])
        replay = compute_spatial_stats(
            self.graph,
            self.classifications,
            self.metrics,
            statistics,
            figure,
        )
        self.assertFalse(replay["artifacts"]["spatial_statistics"]["changed"])
        self.assertFalse(replay["artifacts"]["spatial_statistics_figure"]["changed"])

    def test_core_renders_noncontiguous_strut_ids(self) -> None:
        output = self.root / "lattice.png"

        result = render_lattice_3d(self.graph, self.classifications, output)

        self.assertEqual(3, result["counts"]["total"])
        self.assertEqual(1, result["counts"]["missing"])
        self.assertEqual(b"\x89PNG\r\n\x1a\n", output.read_bytes()[:8])

    async def test_tools_are_registered_and_return_closed_envelopes(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        self.assertIn("compute_spatial_stats", tools)
        self.assertIn("render_lattice_3d", tools)
        for retired in (
            "compute_detection_metrics",
            "summarize_nde_artifacts",
            "render_volume_3d",
            "skeletonize",
        ):
            self.assertNotIn(retired, tools)

        async with Client(mcp) as client:
            statistics = await client.call_tool(
                "compute_spatial_stats",
                {
                    "localized_graph_filepath": str(self.graph),
                    "classifications_filepath": str(self.classifications),
                    "metrics_filepath": str(self.metrics),
                    "output_statistics_filepath": str(self.root / "stats.json"),
                    "output_figure_filepath": str(self.root / "stats.png"),
                },
            )
            render = await client.call_tool(
                "render_lattice_3d",
                {
                    "localized_graph_filepath": str(self.graph),
                    "classifications_filepath": str(self.classifications),
                    "output_filepath": str(self.root / "render.png"),
                },
            )
        for call in (statistics, render):
            self.assertFalse(call.is_error)
            payload = MCPResponseEnvelope.model_validate(call.structured_content)
            self.assertEqual("pass", payload.gate)
            for artifact in payload.artifacts.values():
                self.assertFalse(Path(artifact["path"]).is_absolute())

    async def test_mcp_rejects_incomplete_classification_coverage(self) -> None:
        incomplete = self.root / "incomplete.json"
        incomplete.write_text(
            json.dumps(
                {"classifications": [{"strut_id": 101, "class": "missing"}]}
            ),
            encoding="utf-8",
        )
        async with Client(mcp) as client:
            call = await client.call_tool(
                "render_lattice_3d",
                {
                    "localized_graph_filepath": str(self.graph),
                    "classifications_filepath": str(incomplete),
                    "output_filepath": str(self.root / "bad.png"),
                },
            )
        self.assertFalse(call.is_error)
        self.assertEqual("error", call.structured_content["status"])
        self.assertEqual("halt", call.structured_content["gate"])
        self.assertEqual("invalid_input", call.structured_content["error"]["code"])


if __name__ == "__main__":
    unittest.main()
