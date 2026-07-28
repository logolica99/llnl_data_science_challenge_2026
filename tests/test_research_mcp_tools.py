"""Contract tests for the isolated, disabled-by-default research MCP."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from fastmcp import Client
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from research.evaluation import compute_detection_metrics  # noqa: E402
from research.mcp_server import mcp  # noqa: E402
from research.volume_artifacts import (  # noqa: E402
    compare_segmentation_masks,
    render_volume_3d,
    summarize_nde_artifacts,
)


class MCPArtifactToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        runs = REPOSITORY_ROOT / "research" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runs)
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

        self.assertIn("summarize_nde_artifacts", tools)
        self.assertIn("render_volume_3d", tools)
        self.assertIn("skeletonize", tools)
        self.assertIn("compute_detection_metrics", tools)
        self.assertIn("explore_ct_thresholds", tools)
        render_properties = tools["render_volume_3d"].parameters["properties"]
        self.assertEqual(0.5, render_properties["surface_level"]["default"])
        self.assertEqual(False, render_properties["overwrite"]["default"])

    async def test_tools_return_structured_content_through_mcp(self) -> None:
        output = self.root / "mcp-render.png"
        async with Client(mcp) as client:
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

        self.assertFalse(summary.is_error)
        self.assertTrue(summary.structured_content["research_only"])
        self.assertEqual(
            216,
            summary.structured_content["result"]["mask"]["foreground_voxels"],
        )
        self.assertFalse(render.is_error)
        self.assertEqual(
            output.relative_to(REPOSITORY_ROOT).as_posix(),
            render.structured_content["result"]["output_path"],
        )
        self.assertTrue(output.is_file())

    async def test_labeled_evaluation_requires_research_copies(self) -> None:
        classifications = self.root / "classifications.json"
        labels = self.root / "labels.json"
        output = self.root / "detection.json"
        classifications.write_text(
            '{"classifications":[{"strut_id":7,"class":"missing"}]}',
            encoding="utf-8",
        )
        labels.write_text('{"strut_ids":[7]}', encoding="utf-8")
        direct = compute_detection_metrics(classifications, labels, output)
        self.assertEqual(1.0, direct["strict_recall"]["value"])

        output.unlink()
        async with Client(mcp) as client:
            call = await client.call_tool(
                "compute_detection_metrics",
                {
                    "classifications_filepath": str(classifications),
                    "sealed_labels_filepath": str(labels),
                    "output_filepath": str(output),
                },
            )
        self.assertFalse(call.is_error)
        self.assertTrue(call.structured_content["research_only"])
        self.assertEqual(
            1.0,
            call.structured_content["result"]["strict_recall"]["value"],
        )

    async def test_threshold_exploration_writes_only_research_artifacts(self) -> None:
        volume = np.linspace(0, 100, 8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
        source = self.root / "threshold-source.npy"
        np.save(source, volume)
        output = self.root / "threshold-run"
        async with Client(mcp) as client:
            call = await client.call_tool(
                "explore_ct_thresholds",
                {
                    "input_filepath": str(source),
                    "output_directory": str(output),
                    "threshold_offsets": [-1.0, 0.0, 1.0],
                },
            )
        self.assertFalse(call.is_error, call)
        result = call.structured_content
        self.assertTrue(result["research_only"])
        self.assertEqual(3, result["result"]["candidate_count"])
        self.assertTrue((output / "threshold_comparison.json").is_file())


if __name__ == "__main__":
    unittest.main()
