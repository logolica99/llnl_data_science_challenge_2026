"""Bounded reference-data integration checks.

The expensive full-volume and multi-million-triangle replays are opt-in via
``RUN_PART2_REFERENCE_INTEGRATION=1``. Header/contract checks stay in the
ordinary suite.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.server import replay_exact_otsu, volume_info  # noqa: E402


CT_PATH = REPOSITORY_ROOT / (
    "data/missing_struts/tif_stacks/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
POLICY_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "brian_tran_9x9x9_0point5dash1_production"
    / "config"
    / "specimen_manifest.json"
)
RUN_EXPENSIVE = os.environ.get("RUN_PART2_REFERENCE_INTEGRATION") == "1"


class ReferenceIntegrationTests(unittest.TestCase):
    def test_reference_ct_header_contract_is_bounded(self) -> None:
        result = volume_info(str(CT_PATH), include_sha256=False)
        self.assertEqual("ok", result["status"])
        self.assertEqual([761, 815, 837], result["result"]["shape"])
        self.assertEqual("uint16", result["result"]["dtype"])
        self.assertEqual(["z", "y", "x"], result["result"]["axis_mapping"]["array_axes"])

    @unittest.skipUnless(RUN_EXPENSIVE, "set RUN_PART2_REFERENCE_INTEGRATION=1")
    def test_reference_exact_otsu_replay(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary:
            result = replay_exact_otsu(
                str(CT_PATH),
                temporary,
                analysis_policy_artifact_filepath=str(
                    POLICY_PATH.relative_to(REPOSITORY_ROOT)
                ),
                registration_mode="autonomous_v2",
            )
        self.assertEqual("pass", result["gate"])
        self.assertEqual(40054, result["result"]["threshold"])
        self.assertEqual(58_653_410, result["result"]["foreground_voxel_count"])
        self.assertEqual(
            "hashed_analysis_parameters",
            result["result"]["provenance"]["policy_binding"],
        )
        self.assertEqual(
            POLICY_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            result["result"]["analysis_policy_artifact"]["path"],
        )


if __name__ == "__main__":
    unittest.main()
