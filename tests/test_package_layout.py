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
        }
        missing = sorted(path for path in expected if not (package / path).is_file())
        self.assertEqual([], missing)

    def test_legacy_imports_alias_canonical_implementations(self) -> None:
        from llnl_nde.core import compute_strut_metrics as canonical_metrics
        from llnl_nde.server import mcp as canonical_mcp
        from mcp_server import mcp as legacy_mcp
        from part2_core import compute_strut_metrics as legacy_metrics

        self.assertIs(canonical_metrics, legacy_metrics)
        self.assertIs(canonical_mcp, legacy_mcp)


if __name__ == "__main__":
    unittest.main()
