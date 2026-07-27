"""Production contract tests for the Stage 1 and Stage 2 agent slice."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
import tomllib
import unittest

from fastmcp import Client
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import mcp  # noqa: E402
from part2_core.design_diff import (  # noqa: E402
    label_deleted_edges,
    resolve_cad_graph_orientation,
)


def write_binary_stl(path: Path, centroids: list[np.ndarray]) -> None:
    header = b"deterministic-stage1-test".ljust(80, b"\0")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(centroids)))
        for centroid in centroids:
            stream.write(struct.pack("<3f", 0.0, 0.0, 1.0))
            for _ in range(3):
                stream.write(struct.pack("<3f", *centroid.tolist()))
            stream.write(struct.pack("<H", 0))


class AgentSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def test_agents_are_gpt_5_6_sol_and_skills_are_mcp_only(self) -> None:
        for name in ("design_diff", "data_prep"):
            document = tomllib.loads(
                (REPOSITORY_ROOT / ".codex" / "agents" / f"{name}.toml").read_text()
            )
            self.assertEqual("gpt-5.6-sol", document["model"])
        for name in ("stl-design-diff", "ct-registration", "ct-threshold-optimizer"):
            root = REPOSITORY_ROOT / ".agents" / "skills" / name
            self.assertFalse((root / "scripts").exists())
            self.assertIn("segmentation-tools", (root / "agents" / "openai.yaml").read_text())

    async def test_stage_tools_are_registered_through_actual_mcp_client(self) -> None:
        tools = {tool.name for tool in await mcp.list_tools()}
        self.assertTrue(
            {
                "load_lattice_graph",
                "resolve_cad_graph_orientation",
                "label_deleted_edges",
                "volume_info",
                "replay_exact_otsu",
                "segment_ct_dataset",
                "compare_segmentation_masks",
                "visualize_slice",
                "register_lattice_to_ct",
                "localize_lattice_nodes",
                "compute_registration_qa",
            }.issubset(tools)
        )

    async def test_canonical_mask_replay_is_idempotent_and_config_drift_halts(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary:
            root = Path(temporary)
            volume = root / "volume.npy"
            mask = root / "mask.npy"
            np.save(volume, np.arange(64, dtype=np.uint16).reshape(4, 4, 4))
            async with Client(mcp) as client:
                first = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 32},
                )
                replay = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 32},
                )
                drift = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 31},
                )
            self.assertTrue(first.structured_content["result"]["changed"])
            self.assertFalse(replay.structured_content["result"]["changed"])
            self.assertEqual("error", drift.structured_content["status"])
            self.assertEqual("halt", drift.structured_content["gate"])


class DesignLabelCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph_path = self.root / "graph.json"
        graph = {
            "junctions": [
                {"id": 10, "position": [0, 0, 0]},
                {"id": 20, "position": [1, 0, 0]},
                {"id": 30, "position": [2, 0, 0]},
                {"id": 40, "position": [3, 0, 0]},
                {"id": 50, "position": [4, 0, 0]},
            ],
            "struts": [
                {"id": 101, "junction0": 10, "junction1": 20},
                {"id": 205, "junction0": 20, "junction1": 30},
                {"id": 309, "junction0": 30, "junction1": 40},
                {"id": 413, "junction0": 40, "junction1": 50},
            ],
            "unit_cells": [
                {"id": 7, "indices": [0, 0, 0], "struts": [101, 205, 309, 413]}
            ],
        }
        self.graph_path.write_text(json.dumps(graph), encoding="utf-8")
        # The core subtracts the graph center [2, 0, 0].
        edge_centers = [np.asarray([value, 0.0, 0.0]) for value in (-1.5, -0.5, 0.5, 1.5)]
        baseline = [point for point in edge_centers for _ in range(175)]
        write_binary_stl(self.root / "0.stl", baseline)
        write_binary_stl(self.root / "0p1.stl", baseline[175:])
        write_binary_stl(self.root / "0p5.stl", baseline[350:])
        write_binary_stl(self.root / "1p0.stl", baseline[525:])
        orientation = {
            "gate": "pass",
            "gates": {"orientation_unambiguous": True},
            "transform": {
                "rotation_matrix": np.eye(3).tolist(),
                "scale_mm_per_design_unit": 1.0,
                "translation_mm": [0.0, 0.0, 0.0],
            },
        }
        self.orientation_path = self.root / "orientation.json"
        self.orientation_path.write_text(json.dumps(orientation), encoding="utf-8")

    def test_tube_emptiness_preserves_ids_monotone_sets_and_triangle_evidence(self) -> None:
        report = label_deleted_edges(
            self.graph_path,
            self.root / "0.stl",
            {
                "0p1": self.root / "0p1.stl",
                "0p5": self.root / "0p5.stl",
                "1p0": self.root / "1p0.stl",
            },
            self.orientation_path,
            self.root / "labels",
            development_split_path=self.root / "dev.json",
            sealed_split_path=self.root / "sealed.json",
            label_report_path=self.root / "report.md",
            expected_deletions={"0p1": 1, "0p5": 2, "1p0": 3},
        )
        # This small fixture intentionally fails only the production graph-count gate.
        self.assertEqual("halt", report["gate"])
        self.assertTrue(report["gates"]["deletion_sets_monotone"])
        self.assertTrue(report["gates"]["triangle_deficit_ratio_between_170_and_180"])
        first = json.loads((self.root / "labels" / "intentional_deletions_0p1.json").read_text())
        self.assertEqual([101], first["deleted_strut_ids"])
        dev = set(json.loads((self.root / "dev.json").read_text())["strut_ids"])
        sealed = set(json.loads((self.root / "sealed.json").read_text())["strut_ids"])
        self.assertTrue(dev.isdisjoint(sealed))
        self.assertEqual({101, 205}, dev | sealed)
        self.assertIn("Design-diff label report", (self.root / "report.md").read_text())

        replay = label_deleted_edges(
            self.graph_path,
            self.root / "0.stl",
            {
                "0p1": self.root / "0p1.stl",
                "0p5": self.root / "0p5.stl",
                "1p0": self.root / "1p0.stl",
            },
            self.orientation_path,
            self.root / "labels",
            development_split_path=self.root / "dev.json",
            sealed_split_path=self.root / "sealed.json",
            label_report_path=self.root / "report.md",
            expected_deletions={"0p1": 1, "0p5": 2, "1p0": 3},
        )
        self.assertFalse(replay["artifacts"]["labels_0p1"]["changed"])

    def test_equivalent_orientation_hypotheses_abstain(self) -> None:
        orientation = resolve_cad_graph_orientation(
            self.graph_path,
            self.root / "0.stl",
            self.root / "ambiguous-orientation.json",
            scale_candidates=[1.0],
            expected_counts={"nodes": 5, "edges": 4, "cells": 1},
        )
        self.assertEqual("manual_review", orientation["gate"])
        self.assertGreater(
            orientation["ambiguity"]["equivalent_hypothesis_count"], 1
        )
        self.assertFalse(orientation["provenance"]["ct_accessed"])
        self.assertFalse(orientation["provenance"]["aligned_graph_accessed"])


if __name__ == "__main__":
    unittest.main()
