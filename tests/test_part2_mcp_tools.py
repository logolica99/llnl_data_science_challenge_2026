"""MCP-client contract tests for the first Part 2 tool slice."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from fastmcp import Client
import numpy as np
import tifffile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import mcp  # noqa: E402
from part2_core.response import RESPONSE_SCHEMA_VERSION  # noqa: E402


class Part2MCPToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        self.volume = self.root / "scan.npy"
        lower = np.linspace(10_000, 15_000, 4_000, dtype=np.uint16)
        upper = np.linspace(45_000, 50_000, 4_000, dtype=np.uint16)
        np.save(self.volume, np.concatenate((lower, upper)).reshape(20, 20, 20))

        self.graph = self.root / "graph.json"
        self.graph.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": 10, "position": [0, 0, 0]},
                        {"id": 30, "position": [1, 1, 1]},
                    ],
                    "struts": [
                        {"id": 90, "junction0": 10, "junction1": 30},
                    ],
                    "unit_cells": [
                        {"id": 5, "indices": [0, 0, 0], "struts": [90]},
                    ],
                }
            ),
            encoding="utf-8",
        )

    async def test_tools_are_registered_with_typed_schemas(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}

        self.assertIn("volume_info", tools)
        self.assertIn("load_lattice_graph", tools)
        self.assertIn("replay_exact_otsu", tools)
        volume_properties = tools["volume_info"].parameters["properties"]
        self.assertEqual("string", volume_properties["input_filepath"]["type"])
        self.assertEqual(True, volume_properties["include_sha256"]["default"])
        graph_properties = tools["load_lattice_graph"].parameters["properties"]
        self.assertEqual(False, graph_properties["overwrite"]["default"])
        otsu_properties = tools["replay_exact_otsu"].parameters["properties"]
        self.assertEqual(
            ["auto", "native_uint16", "full_volume_affine_uint16"],
            otsu_properties["histogram_encoding"]["enum"],
        )

    async def test_volume_info_returns_hash_and_axis_mapping_through_mcp(self) -> None:
        async with Client(mcp) as client:
            call = await client.call_tool(
                "volume_info",
                {"input_filepath": str(self.volume)},
            )

        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual(RESPONSE_SCHEMA_VERSION, result["response_schema_version"])
        self.assertEqual("ok", result["status"])
        self.assertEqual("pass", result["gate"])
        self.assertEqual([20, 20, 20], result["result"]["shape"])
        self.assertTrue(result["result"]["memory_mapped"])
        self.assertEqual(
            ["z", "y", "x"],
            result["result"]["axis_mapping"]["array_axes"],
        )
        self.assertEqual(64, len(result["hashes"]["input_sha256"]))

    async def test_graph_normalization_returns_manual_review_and_artifact(self) -> None:
        output = self.root / "normalized.npz"
        async with Client(mcp) as client:
            call = await client.call_tool(
                "load_lattice_graph",
                {
                    "input_filepath": str(self.graph),
                    "output_filepath": str(output),
                },
            )

        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("ok", result["status"])
        self.assertEqual("manual_review", result["gate"])
        self.assertEqual(
            {"nodes": 2, "edges": 1, "cells": 1},
            result["result"]["counts"],
        )
        self.assertFalse(Path(result["artifacts"]["normalized_graph"]["path"]).is_absolute())
        self.assertTrue(output.is_file())
        with np.load(output, allow_pickle=False) as normalized:
            self.assertEqual([10, 30], normalized["node_id_keys"].tolist())
            self.assertEqual([0, 1], normalized["node_id_rows"].tolist())

    async def test_exact_otsu_persists_artifacts_and_uses_halt_gate(self) -> None:
        output = self.root / "otsu"
        async with Client(mcp) as client:
            call = await client.call_tool(
                "replay_exact_otsu",
                {
                    "input_filepath": str(self.volume),
                    "output_directory": str(output),
                    "chunk_voxels": 257,
                    "minimum_significant_peaks": 3,
                },
            )

        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("ok", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertFalse(result["result"]["overall_pass"])
        self.assertIn("histogram rejection gates failed:", result["warnings"][0])
        self.assertTrue(
            any(not passed for passed in result["result"]["gates"].values())
        )
        for artifact in result["artifacts"].values():
            self.assertFalse(Path(artifact["path"]).is_absolute())
            self.assertTrue((REPOSITORY_ROOT / artifact["path"]).is_file())
            self.assertEqual(64, len(artifact["sha256"]))
        report = json.loads(
            (output / "histogram_report.json").read_text(encoding="utf-8")
        )
        self.assertFalse(Path(report["source_path"]).is_absolute())

    async def test_errors_are_structured_instead_of_json_rpc_failures(self) -> None:
        missing = self.root / "missing.npy"
        async with Client(mcp) as client:
            call = await client.call_tool(
                "volume_info",
                {"input_filepath": str(missing)},
            )

        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertEqual("input_not_found", result["error"]["code"])
        self.assertEqual("FileNotFoundError", result["error"]["type"])

    async def test_legacy_segmentation_uses_shared_tiff_loader(self) -> None:
        tiff = self.root / "scan.tiff"
        mask = self.root / "mask.npy"
        values = np.arange(64, dtype=np.dtype(">u2")).reshape(4, 4, 4)
        tifffile.imwrite(
            tiff,
            values,
            byteorder=">",
            photometric="minisblack",
        )
        async with Client(mcp) as client:
            call = await client.call_tool(
                "segment_ct_dataset",
                {
                    "input_filepath": str(tiff),
                    "output_filepath": str(mask),
                    "threshold": 32,
                },
            )

        self.assertFalse(call.is_error)
        self.assertIn("Saved 32 foreground voxels", call.content[0].text)
        segmented = np.load(mask, allow_pickle=False)
        self.assertEqual(np.uint8, segmented.dtype)
        self.assertEqual(32, int(np.count_nonzero(segmented)))


if __name__ == "__main__":
    unittest.main()
