"""Evidence-only tests for the real fail-closed orchestrator proof."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    REPOSITORY_ROOT
    / "demo"
    / "part2-orchestrator"
    / "server"
    / "real_proof.py"
)


class RealOrchestratorProofTests(unittest.TestCase):
    def test_real_proof_halts_without_fabricating_specialist_work(self) -> None:
        with tempfile.TemporaryDirectory(prefix="part2-real-proof-test-") as temporary:
            output = Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--output-directory",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            report = json.loads(
                (output / "proof-evidence" / "latest" / "proof-report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("halt", report["status"]["pipeline_state"])
            self.assertEqual(0, report["status"]["stages"]["0"]["attempt_count"])
            self.assertTrue(
                all(
                    report["status"]["stages"][str(number)]["state"] == "locked"
                    for number in range(1, 5)
                )
            )
            self.assertFalse(report["synthetic_specialist_artifacts"])
            self.assertFalse(report["fallback_used"])
            self.assertEqual(
                "missing_or_incompatible_dependency", report["halt"]["code"]
            )
            self.assertEqual(
                {"agent", "mcp_server"},
                {failure["kind"] for failure in report["halt"]["failures"]},
            )
            self.assertTrue((output / "real-proof.html").is_file())
            self.assertIn("Real fail-closed run", (output / "real-proof.html").read_text())


if __name__ == "__main__":
    unittest.main()
