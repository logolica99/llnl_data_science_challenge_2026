"""Contract tests for MCP-owned comparison, summary, and 3D rendering."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from fastmcp import Client
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import (  # noqa: E402
    compare_segmentation_masks,
    mcp,
    render_volume_3d,
    summarize_nde_artifacts,
)


class MCPArtifactToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        raw = np.zeros((12, 12, 12), dtype=np.float32)
        raw[3:9, 3:9, 3:9] = 1.0
        self.raw = self.root / "raw.npy"
        np.save(self.raw, raw)

        self.mask = self.root / "mask.npy"
        np.save(self.mask, (raw > 0).astype(np.uint8))
        self.empty_mask = self.root / "empty-mask.npy"
        np.save(self.empty_mask, np.zeros_like(raw, dtype=np.uint8))

        skeleton = np.zeros_like(raw, dtype=np.uint8)
        skeleton[3:9, 6, 6] = 1
        skeleton[6, 3:9, 6] = 1
        self.skeleton = self.root / "skeleton.npy"
        np.save(self.skeleton, skeleton)

    def test_compare_masks_returns_compact_statistics(self) -> None:
        result = compare_segmentation_masks(
            str(self.raw),
            [str(self.mask), str(self.empty_mask)],
            [0.5, 1.5],
        )

        self.assertEqual("ok", result["status"])
        self.assertEqual([12, 12, 12], result["shape"])
        self.assertEqual(216, result["candidates"][0]["foreground_voxels"])
        self.assertEqual(0, result["candidates"][1]["foreground_voxels"])
        self.assertNotIn("array", result)

    def test_summary_reports_aligned_scalar_metrics(self) -> None:
        result = summarize_nde_artifacts(
            str(self.raw),
            str(self.mask),
            str(self.skeleton),
        )

        self.assertEqual("computed", result["mask"]["mean_foreground_intensity_status"])
        self.assertEqual(1.0, result["mask"]["mean_foreground_intensity"])
        self.assertEqual(216, result["mask"]["foreground_voxels"])
        self.assertGreater(result["skeleton"]["skeleton_voxels"], 0)
        self.assertGreater(
            result["skeleton"]["branch_points_26_connected"],
            0,
        )

    def test_skeleton_connectivity_crosses_processing_slabs(self) -> None:
        raw = np.ones((40, 3, 3), dtype=np.float32)
        mask = np.ones_like(raw, dtype=np.uint8)
        skeleton = np.zeros_like(raw, dtype=np.uint8)
        skeleton[:, 1, 1] = 1
        raw_path = self.root / "long-raw.npy"
        mask_path = self.root / "long-mask.npy"
        skeleton_path = self.root / "long-skeleton.npy"
        np.save(raw_path, raw)
        np.save(mask_path, mask)
        np.save(skeleton_path, skeleton)

        result = summarize_nde_artifacts(
            str(raw_path),
            str(mask_path),
            str(skeleton_path),
        )

        self.assertEqual(40, result["skeleton"]["skeleton_voxels"])
        self.assertEqual(2, result["skeleton"]["endpoints_26_connected"])
        self.assertEqual(0, result["skeleton"]["branch_points_26_connected"])

    def test_render_writes_png_and_returns_only_metadata(self) -> None:
        output = self.root / "render.png"
        result = render_volume_3d(
            str(self.raw),
            str(output),
            downsample_factor=1,
            skeleton_filepath=str(self.skeleton),
        )

        self.assertEqual("ok", result["status"])
        self.assertGreater(result["vertices"], 0)
        self.assertGreater(result["faces"], 0)
        self.assertGreater(result["rendered_skeleton_points"], 0)
        self.assertEqual(b"\x89PNG\r\n\x1a\n", output.read_bytes()[:8])

    def test_render_refuses_implicit_overwrite(self) -> None:
        output = self.root / "render.png"
        output.write_bytes(b"existing")

        with self.assertRaisesRegex(FileExistsError, "enable overwrite"):
            render_volume_3d(str(self.raw), str(output))

    async def test_tools_are_registered_with_typed_schemas(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}

        self.assertIn("compare_segmentation_masks", tools)
        self.assertIn("summarize_nde_artifacts", tools)
        self.assertIn("render_volume_3d", tools)
        compare_properties = tools["compare_segmentation_masks"].parameters[
            "properties"
        ]
        self.assertEqual("array", compare_properties["mask_filepaths"]["type"])
        self.assertEqual("number", compare_properties["thresholds"]["items"]["type"])
        render_properties = tools["render_volume_3d"].parameters["properties"]
        self.assertEqual(0.5, render_properties["surface_level"]["default"])
        self.assertEqual(False, render_properties["overwrite"]["default"])

    async def test_tools_return_structured_content_through_mcp(self) -> None:
        output = self.root / "mcp-render.png"
        async with Client(mcp) as client:
            comparison = await client.call_tool(
                "compare_segmentation_masks",
                {
                    "raw_filepath": str(self.raw),
                    "mask_filepaths": [str(self.mask)],
                    "thresholds": [0.5],
                },
            )
            summary = await client.call_tool(
                "summarize_nde_artifacts",
                {
                    "raw_filepath": str(self.raw),
                    "mask_filepath": str(self.mask),
                    "skeleton_filepath": str(self.skeleton),
                },
            )
            render = await client.call_tool(
                "render_volume_3d",
                {
                    "input_filepath": str(self.raw),
                    "output_filepath": str(output),
                    "downsample_factor": 1,
                    "skeleton_filepath": str(self.skeleton),
                },
            )

        self.assertFalse(comparison.is_error)
        self.assertEqual("ok", comparison.structured_content["status"])
        self.assertFalse(summary.is_error)
        self.assertEqual(216, summary.structured_content["mask"]["foreground_voxels"])
        self.assertFalse(render.is_error)
        self.assertEqual(str(output), render.structured_content["output_path"])
        self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
