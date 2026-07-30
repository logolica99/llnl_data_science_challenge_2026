"""MCP-client contract tests for the first Part 2 tool slice."""

from __future__ import annotations

import hashlib
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

from llnl_nde.server import MCPResponseEnvelope, mcp  # noqa: E402
from llnl_nde.core.response import RESPONSE_SCHEMA_VERSION  # noqa: E402
from llnl_nde.orchestration.contracts import (  # noqa: E402
    canonical_json_sha256,
    validate_manifest,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        self.policy = self._write_policy_artifact()

    def _write_policy_artifact(
        self, *, minimum_significant_peaks: int = 2
    ) -> Path:
        source = (
            REPOSITORY_ROOT
            / "analysis"
            / "brian_tran_9x9x9_0point5dash1_production"
            / "config"
            / "specimen_manifest.json"
        )
        manifest = json.loads(source.read_text(encoding="utf-8"))
        ct_relative = self.volume.relative_to(REPOSITORY_ROOT).as_posix()
        graph_relative = self.graph.relative_to(REPOSITORY_ROOT).as_posix()
        manifest["specimen_id"] = "mcp_tool_fixture"
        manifest["design_id"] = "mcp_tool_design"
        manifest["analysis_parameters"]["segmentation"][
            "minimum_significant_peaks"
        ] = minimum_significant_peaks
        manifest["analysis_parameters_sha256"] = canonical_json_sha256(
            manifest["analysis_parameters"]
        )
        manifest["inputs"]["ct"] = {
            "path": ct_relative,
            "sha256": _sha256_file(self.volume),
            "role": "ct_volume",
            "retention": "external",
        }
        manifest["inputs"]["ct_metadata"] = {
            "array_axes": ["z", "y", "x"],
            "byte_order": "little",
            "dtype": "uint16",
            "format": "npy",
            "shape": [20, 20, 20],
            "voxel_spacing": {
                axis: {
                    "provenance": {
                        "field": "unknown",
                        "raw_value": "unknown",
                        "source": "unknown",
                    },
                    "unit": "unknown",
                    "value": "unknown",
                }
                for axis in ("x", "y", "z")
            },
        }
        manifest["inputs"]["design_graph"] = {
            "path": graph_relative,
            "sha256": _sha256_file(self.graph),
            "role": "design_graph",
            "retention": "external",
        }
        manifest["inputs"].pop("normalized_nominal_graph", None)
        if "intake" in manifest:
            manifest["intake"]["graph_inspection"]["path"] = graph_relative
            manifest["intake"]["graph_inspection"]["sha256"] = _sha256_file(
                self.graph
            )
            manifest["intake"]["graph_inspection"].update(
                {
                    "junction_count": 2,
                    "strut_count": 1,
                    "unit_cell_count": 1,
                }
            )
        if "graph_summary" in manifest.get("derived", {}):
            manifest["derived"]["graph_summary"]["values"].update(
                {
                    "junction_count": 2,
                    "strut_count": 1,
                    "unit_cell_count": 1,
                }
            )
            manifest["derived"]["graph_summary"]["provenance"][
                "config_sha256"
            ] = manifest["analysis_parameters_sha256"]
            manifest["derived"]["graph_summary"]["provenance"]["input_sha256"] = [
                _sha256_file(self.graph)
            ]
        policy_path = self.root / "specimen_manifest.json"
        policy_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_manifest(
            policy_path,
            repository_root=REPOSITORY_ROOT,
            verify_files=False,
        )
        return policy_path

    async def test_tools_are_registered_with_typed_schemas(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}

        self.assertIn("volume_info", tools)
        self.assertIn("load_lattice_graph", tools)
        self.assertIn("replay_exact_otsu", tools)
        self.assertNotIn("resolve_cad_graph_orientation", tools)
        self.assertNotIn("label_deleted_edges", tools)
        self.assertIn("verify_canonical_segmentation", tools)
        self.assertIn("localize_lattice_nodes", tools)
        self.assertIn("compute_registration_qa", tools)
        volume_properties = tools["volume_info"].parameters["properties"]
        self.assertEqual("string", volume_properties["input_filepath"]["type"])
        self.assertEqual(True, volume_properties["include_sha256"]["default"])
        graph_properties = tools["load_lattice_graph"].parameters["properties"]
        self.assertEqual(False, graph_properties["overwrite"]["default"])
        otsu_schema = tools["replay_exact_otsu"].parameters
        self.assertIn("analysis_policy_artifact_filepath", otsu_schema["required"])
        self.assertNotIn("chunk_voxels", otsu_schema["properties"])
        self.assertNotIn("enforce_reference_replay", otsu_schema["properties"])
        localization_schema = tools["localize_lattice_nodes"].parameters
        self.assertIn(
            "analysis_policy_artifact_filepath", localization_schema["required"]
        )
        qa_schema = tools["compute_registration_qa"].parameters
        self.assertIn("analysis_scope_artifact_filepath", qa_schema["required"])
        self.assertNotIn("absolute_registration_uncertainty", qa_schema["properties"])
        verification_schema = tools["verify_canonical_segmentation"].parameters
        self.assertEqual(
            {
                "specimen_id",
                "design_id",
                "analysis_policy_artifact_filepath",
                "exact_otsu_report_filepath",
                "canonical_mask_filepath",
                "mask_comparison_report_filepath",
                "output_filepath",
                "registration_mode",
            },
            set(verification_schema["required"]),
        )
        self.assertEqual(False, verification_schema["additionalProperties"])
        self.assertEqual(
            False, verification_schema["properties"]["overwrite"]["default"]
        )
        for name in (
            "volume_info",
            "load_lattice_graph",
            "replay_exact_otsu",
            "verify_canonical_segmentation",
            "register_lattice_to_ct",
            "localize_lattice_nodes",
            "compute_registration_qa",
        ):
            output_schema = tools[name].output_schema
            self.assertFalse(output_schema["additionalProperties"], name)
            self.assertEqual(
                {
                    "response_schema_version",
                    "tool",
                    "status",
                    "gate",
                    "summary",
                    "result",
                    "artifacts",
                    "hashes",
                    "warnings",
                    "error",
                },
                set(output_schema["properties"]),
                name,
            )

    async def test_volume_info_returns_hash_and_axis_mapping_through_mcp(self) -> None:
        async with Client(mcp) as client:
            call = await client.call_tool(
                "volume_info",
                {"input_filepath": str(self.volume)},
            )

        self.assertFalse(call.is_error)
        result = call.structured_content
        MCPResponseEnvelope.model_validate(result)
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
        MCPResponseEnvelope.model_validate(result)
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
        policy = self._write_policy_artifact(minimum_significant_peaks=3)
        output = self.root / "otsu"
        async with Client(mcp) as client:
            call = await client.call_tool(
                "replay_exact_otsu",
                {
                    "input_filepath": str(self.volume),
                    "output_directory": str(output),
                    "analysis_policy_artifact_filepath": str(policy),
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
        self.assertEqual(
            "hashed_analysis_parameters",
            result["result"]["provenance"]["policy_binding"],
        )
        self.assertEqual(
            64,
            len(result["hashes"]["analysis_parameters_sha256"]),
        )
        for artifact in result["artifacts"].values():
            self.assertFalse(Path(artifact["path"]).is_absolute())
            self.assertTrue((REPOSITORY_ROOT / artifact["path"]).is_file())
            self.assertEqual(64, len(artifact["sha256"]))
        report = json.loads(
            (output / "histogram_report.json").read_text(encoding="utf-8")
        )
        self.assertFalse(Path(report["source_path"]).is_absolute())
        self.assertEqual(
            policy.relative_to(REPOSITORY_ROOT).as_posix(),
            report["analysis_policy_artifact"]["path"],
        )

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
        result = call.structured_content
        self.assertEqual("ok", result["status"])
        self.assertEqual("pass", result["gate"])
        self.assertTrue(mask.is_file())


if __name__ == "__main__":
    unittest.main()
