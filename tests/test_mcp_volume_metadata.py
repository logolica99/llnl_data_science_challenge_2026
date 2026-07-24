"""Contract tests for the volume-metadata MCP boundary."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from fastmcp import Client


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import (  # noqa: E402
    inspect_volume_metadata,
    mcp,
)
from volume_metadata import UNKNOWN, VolumeMetadataError  # noqa: E402


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

    def test_preview_is_explicitly_non_authoritative(self) -> None:
        result = inspect_volume_metadata(
            str(self.volume),
            include_sha256=False,
        )

        self.assertFalse(result["authoritative"])
        self.assertEqual(UNKNOWN, result["sha256"])

    def test_tool_rejects_paths_outside_repository(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npy") as outside:
            with self.assertRaisesRegex(
                VolumeMetadataError, "escapes repository"
            ):
                inspect_volume_metadata(outside.name)

    def test_single_file_cli_fallback_matches_authority_contract(self) -> None:
        script = (
            REPOSITORY_ROOT
            / ".agents/skills/volume-metadata/scripts/extract_metadata.py"
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--header-only",
                str(self.volume),
            ],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)

        self.assertEqual("ok", result["status"])
        self.assertTrue(result["authoritative"])
        self.assertEqual("header_only", result["inspection_mode"])
        self.assertEqual([2, 3, 4], result["shape"])
        self.assertEqual(64, len(result["sha256"]))

    async def test_tool_is_registered_with_typed_schema(self) -> None:
        tools = await mcp.list_tools()
        tool = next(
            item for item in tools if item.name == "inspect_volume_metadata"
        )

        properties = tool.parameters["properties"]
        self.assertEqual("string", properties["input_filepath"]["type"])
        self.assertEqual(True, properties["header_only"]["default"])
        self.assertEqual(True, properties["include_sha256"]["default"])
        self.assertEqual(
            ["committed", "external", "regenerable"],
            properties["retention"]["enum"],
        )

    async def test_tool_returns_structured_content_through_mcp(self) -> None:
        async with Client(mcp) as client:
            result = await client.call_tool(
                "inspect_volume_metadata",
                {"input_filepath": str(self.volume)},
            )

        self.assertFalse(result.is_error)
        self.assertEqual("ok", result.structured_content["status"])
        self.assertEqual([2, 3, 4], result.structured_content["shape"])
        self.assertEqual(
            result.structured_content["sha256"],
            result.structured_content["manifest_fragment"]["ct_volume"][
                "sha256"
            ],
        )


if __name__ == "__main__":
    unittest.main()
