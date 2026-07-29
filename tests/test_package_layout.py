"""Architecture tests for the canonical ``llnl_nde`` package layout."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


class PackageLayoutTests(unittest.TestCase):
    def test_stage_adapters_have_descriptive_stage_suffixes(self) -> None:
        tool_directory = REPOSITORY_ROOT / "src" / "llnl_nde" / "mcp_tools"
        stage_files = {
            path.name for path in tool_directory.glob("*_stage[0-4].py")
        }
        self.assertEqual(
            {
                "specimen_ingest_stage0.py",
                "data_prep_stage1.py",
                "strut_metrics_stage2.py",
                "defect_analysis_stage3.py",
                "reporting_stage4.py",
            },
            stage_files,
        )

    def test_requested_package_boundaries_exist(self) -> None:
        package = REPOSITORY_ROOT / "src" / "llnl_nde"
        expected = {
            "server.py",
            "mcp_tools/specimen_ingest_stage0.py",
            "mcp_tools/data_prep_stage1.py",
            "mcp_tools/strut_metrics_stage2.py",
            "mcp_tools/defect_analysis_stage3.py",
            "mcp_tools/reporting_stage4.py",
            "core/volume.py",
            "core/segmentation.py",
            "core/registration.py",
            "core/strut_metrics.py",
            "core/classification.py",
            "core/evidence.py",
            "core/reporting.py",
            "orchestration/contracts.py",
            "orchestration/receipts.py",
            "orchestration/pipeline.py",
            "cli/specimen_ingest.py",
            "cli/segmentation_replay.py",
            "cli/cloud_smoke_check.py",
        }
        missing = sorted(path for path in expected if not (package / path).is_file())
        self.assertEqual([], missing)

    def test_legacy_source_layout_is_absent(self) -> None:
        source = REPOSITORY_ROOT / "src"
        legacy_paths = {
            "cloud_smoke_check.py",
            "data_prep_handoff.py",
            "mcp_server.py",
            "mcp_tools",
            "part2_core",
            "part2_orchestration.py",
            "segmentation_replay.py",
            "specimen_ingest.py",
            "specimen_manifest.py",
            "volume_metadata.py",
        }
        present = sorted(path for path in legacy_paths if (source / path).exists())
        self.assertEqual([], present)


if __name__ == "__main__":
    unittest.main()
