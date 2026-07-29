"""Contract tests for the volume-metadata MCP boundary."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np
from fastmcp import Client
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.server import (  # noqa: E402
    inspect_volume_metadata,
    mcp,
)
from llnl_nde.orchestration.specimen_ingest import ingest_specimen  # noqa: E402
from llnl_nde.orchestration.contracts import DEFAULT_SCHEMA, canonical_json_sha256  # noqa: E402
from llnl_nde.core.volume_inspection import UNKNOWN  # noqa: E402


class VolumeMetadataMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.volume = self.root / "scan.npy"
        np.save(
            self.volume,
            np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
        )

    def test_tool_returns_compact_authoritative_contract(self) -> None:
        result = inspect_volume_metadata(
            str(self.volume),
            str(self.root / "metadata.json"),
            str(self.root / "metadata-call-receipt.json"),
            retention="regenerable",
        )

        self.assertEqual("ok", result["status"])
        self.assertTrue(result["authoritative"])
        self.assertEqual("header_only", result["inspection_mode"])
        self.assertEqual([2, 3, 4], result["shape"])
        self.assertEqual("uint16", result["dtype"])
        self.assertEqual("not_computed", result["statistics"]["status"])
        self.assertEqual(64, len(result["sha256"]))
        self.assertFalse(Path(result["path"]).is_absolute())
        self.assertEqual(
            "regenerable",
            result["manifest_fragment"]["ct_volume"]["retention"],
        )
        artifact = result["artifacts"]["metadata_response"]
        persisted = REPOSITORY_ROOT / artifact["path"]
        self.assertTrue(persisted.is_file())
        self.assertEqual(
            artifact["sha256"],
            hashlib.sha256(persisted.read_bytes()).hexdigest(),
        )
        evidence = json.loads(persisted.read_text(encoding="utf-8"))
        self.assertEqual("inspect_volume_metadata", evidence["tool"])
        self.assertEqual(result["result"], evidence["result"])
        call_artifact = result["artifacts"]["call_receipt"]
        persisted_call = REPOSITORY_ROOT / call_artifact["path"]
        self.assertEqual(
            call_artifact["sha256"],
            hashlib.sha256(persisted_call.read_bytes()).hexdigest(),
        )
        call_receipt = json.loads(persisted_call.read_text(encoding="utf-8"))
        self.assertEqual(
            "volume-metadata-mcp-call-receipt/1.0.0",
            call_receipt["schema_version"],
        )
        self.assertEqual([2, 3, 4], call_receipt["header_facts"]["shape"])
        self.assertEqual(
            {
                key: artifact[key]
                for key in ("path", "sha256", "role", "retention")
            },
            call_receipt["artifacts"]["metadata_response"],
        )
        self.assertEqual(
            call_receipt["canonical_call_receipt_sha256"],
            canonical_json_sha256(
                {
                    key: value
                    for key, value in call_receipt.items()
                    if key != "canonical_call_receipt_sha256"
                }
            ),
        )

    def test_preview_is_explicitly_non_authoritative(self) -> None:
        result = inspect_volume_metadata(
            str(self.volume),
            str(self.root / "preview.json"),
            str(self.root / "preview-call-receipt.json"),
            include_sha256=False,
        )

        self.assertFalse(result["authoritative"])
        self.assertEqual(UNKNOWN, result["sha256"])

    def test_tool_rejects_paths_outside_repository(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npy") as outside:
            result = inspect_volume_metadata(
                outside.name,
                str(self.root / "outside.json"),
                str(self.root / "outside-call-receipt.json"),
            )
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])

    def test_skill_does_not_bundle_a_cli_fallback(self) -> None:
        script = (
            REPOSITORY_ROOT
            / ".agents/skills/volume-metadata/scripts/extract_metadata.py"
        )
        self.assertFalse(script.exists())

    async def test_tool_is_registered_with_typed_schema(self) -> None:
        tools = await mcp.list_tools()
        tool = next(
            item for item in tools if item.name == "inspect_volume_metadata"
        )

        properties = tool.parameters["properties"]
        self.assertEqual("string", properties["input_filepath"]["type"])
        self.assertIn("output_filepath", tool.parameters["required"])
        self.assertIn("call_receipt_filepath", tool.parameters["required"])
        self.assertEqual(True, properties["header_only"]["default"])
        self.assertEqual(True, properties["include_sha256"]["default"])
        self.assertEqual(
            ["committed", "external", "regenerable"],
            properties["retention"]["enum"],
        )
        self.assertFalse(tool.output_schema["additionalProperties"])

        open_objects: list[str] = []

        def find_open_objects(value: object, path: str = "$") -> None:
            if isinstance(value, dict):
                if (
                    value.get("type") == "object"
                    and value.get("additionalProperties") is not False
                ):
                    open_objects.append(path)
                for key, child in value.items():
                    find_open_objects(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    find_open_objects(child, f"{path}[{index}]")

        find_open_objects(tool.output_schema)
        self.assertEqual([], open_objects)

    async def test_tool_returns_structured_content_through_mcp(self) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "inspect_volume_metadata",
                {
                    "input_filepath": str(self.volume),
                    "output_filepath": str(self.root / "mcp-metadata.json"),
                    "call_receipt_filepath": str(
                        self.root / "mcp-metadata-call-receipt.json"
                    ),
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual("ok", result.structured_content["status"])
        self.assertEqual(
            [2, 3, 4], result.structured_content["result"]["shape"]
        )
        self.assertEqual(
            result.structured_content["result"]["sha256"],
            result.structured_content["result"]["manifest_fragment"][
                "ct_volume"
            ]["sha256"],
        )
        artifact = result.structured_content["artifacts"]["metadata_response"]
        self.assertEqual(
            artifact["sha256"],
            result.structured_content["hashes"]["metadata_response_sha256"],
        )
        call_artifact = result.structured_content["artifacts"]["call_receipt"]
        self.assertEqual(
            call_artifact["sha256"],
            result.structured_content["hashes"]["call_receipt_sha256"],
        )

    async def test_fastmcp_response_and_receipt_are_accepted_by_intake(self) -> None:
        analysis_root = REPOSITORY_ROOT / "analysis"
        with tempfile.TemporaryDirectory(
            dir=analysis_root,
            prefix="mcp_stage0_",
        ) as specimen_directory:
            specimen_root = Path(specimen_directory)
            specimen_id = specimen_root.name
            graph = self.root / "design.json"
            graph.write_text(
                json.dumps(
                    {
                        "junctions": [
                            {"id": 0, "position": [0.0, 0.0, 0.0]},
                            {"id": 1, "position": [1.0, 1.0, 1.0]},
                        ],
                        "struts": [
                            {"id": 10, "junction0": 0, "junction1": 1}
                        ],
                        "unit_cells": [{"id": 20, "struts": [10]}],
                    }
                ),
                encoding="utf-8",
            )
            cad = self.root / "design.stl"
            trimesh.creation.box().export(cad)
            metadata_path = specimen_root / "config" / "ct_metadata_response.json"
            call_receipt_path = (
                specimen_root / "config" / "ct_metadata_mcp_call_receipt.json"
            )
            async with Client(mcp) as client:
                call = await client.call_tool(
                    "inspect_volume_metadata",
                    {
                        "input_filepath": str(self.volume),
                        "output_filepath": str(metadata_path),
                        "call_receipt_filepath": str(call_receipt_path),
                        "header_only": True,
                        "include_sha256": True,
                        "retention": "external",
                    },
                )
            response = call.structured_content
            self.assertEqual("pass", response["gate"])
            response_sha256 = response["artifacts"]["metadata_response"][
                "sha256"
            ]
            call_receipt_sha256 = response["artifacts"]["call_receipt"][
                "sha256"
            ]
            ingest = ingest_specimen(
                repository_root=REPOSITORY_ROOT,
                specimen_id=specimen_id,
                design_id="mcp_stage0_design",
                requested_analysis_scope="roi_screening",
                cad_path=cad,
                design_graph_path=graph,
                ct_path=self.volume,
                ct_metadata_response_path=metadata_path,
                ct_metadata_response_sha256=response_sha256,
                ct_metadata_call_receipt_path=call_receipt_path,
                ct_metadata_call_receipt_sha256=call_receipt_sha256,
                registration_mode="autonomous_v2",
                association_confirmed=True,
                allowed_data_roots=[self.root],
                cad_units="millimeter",
                cad_units_provenance="synthetic test declaration",
                graph_axes="xyz",
                array_axes="zyx",
                aligned_graph_units="simulation_voxel",
                retention="external",
                schema_path=DEFAULT_SCHEMA,
            )
            receipt = json.loads(
                Path(ingest["paths"]["ingest_receipt"]).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                receipt["self_verification"][
                    "ct_metadata_mcp_integrity_chain_valid"
                ]
            )
            self.assertEqual(
                response_sha256,
                receipt["input_sha256"]["ct_metadata_response"],
            )


if __name__ == "__main__":
    unittest.main()
