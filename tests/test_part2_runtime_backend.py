"""Tests for the real Codex app-server bridge without mock agent output."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND = (
    REPOSITORY_ROOT
    / "demo"
    / "part2-orchestrator"
    / "server"
    / "runtime_backend.py"
)


class RuntimeBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        specification = importlib.util.spec_from_file_location("runtime_backend", BACKEND)
        assert specification is not None and specification.loader is not None
        cls.module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.module)

    def test_association_comes_from_real_frozen_specimen(self) -> None:
        association = self.module.association_projection()
        self.assertEqual("brian_tran_9x9x9_0point5dash1", association["sourceSpecimenId"])
        self.assertTrue(association["ct"].endswith("Slices.tif"))
        self.assertTrue(association["nominalGraph"].endswith("octet_truss_9x9x9.json"))
        self.assertEqual("autonomous_v2", association["registrationMode"])
        self.assertNotIn("cad", association)
        self.assertNotIn("alignedGraph", association)

    def test_installed_app_server_accepts_initialize(self) -> None:
        if shutil.which("codex") is None:
            self.skipTest("Codex CLI is not installed")
        completed = subprocess.run(
            [sys.executable, str(BACKEND), "--check-runtime"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertIn("Codex app-server initialize: pass", completed.stdout)

    def test_segmentation_tools_are_preapproved_without_human_prompt(self) -> None:
        import tomllib

        with (REPOSITORY_ROOT / ".codex/config.toml").open("rb") as stream:
            config = tomllib.load(stream)
        server = config["mcp_servers"]["segmentation-tools"]
        self.assertEqual("approve", server["default_tools_approval_mode"])

    def test_mcp_server_configuration_is_clone_portable_and_frozen(self) -> None:
        import tomllib

        config_path = REPOSITORY_ROOT / ".codex/config.toml"
        with config_path.open("rb") as stream:
            config = tomllib.load(stream)

        expected = {
            "segmentation-tools": "src/llnl_nde/server.py",
            "segmentation-tools-research": "research/mcp_server.py",
        }
        for name, entrypoint in expected.items():
            with self.subTest(server=name):
                server = config["mcp_servers"][name]
                self.assertEqual("uv", server["command"])
                self.assertEqual(
                    ["run", "--frozen", "python", entrypoint],
                    server["args"],
                )
                self.assertEqual("..", server["cwd"])
                self.assertEqual(
                    REPOSITORY_ROOT,
                    (config_path.parent / server["cwd"]).resolve(),
                )
                self.assertFalse(Path(server["command"]).is_absolute())
                self.assertTrue(
                    all(not Path(argument).is_absolute() for argument in server["args"])
                )

        self.assertTrue(config["mcp_servers"]["segmentation-tools"]["enabled"])
        self.assertTrue(config["mcp_servers"]["segmentation-tools"]["required"])
        self.assertFalse(
            config["mcp_servers"]["segmentation-tools-research"]["enabled"]
        )
        self.assertFalse(
            config["mcp_servers"]["segmentation-tools-research"]["required"]
        )

    def test_running_manifest_requires_reconciliation(self) -> None:
        run = self.module.RuntimeRun()
        run.manifest = {"pipeline_state": "running"}
        self.assertTrue(run._manifest_is_running())
        run.manifest = {"pipeline_state": "halt"}
        self.assertFalse(run._manifest_is_running())

    def test_runtime_request_is_a_required_stage_zero_handoff_input(self) -> None:
        contract = json.loads(
            (REPOSITORY_ROOT / "analysis/contracts/specimen_ingest.json").read_text()
        )
        self.assertIn(
            "scientist_intake_request",
            contract["input_artifacts"]["required_roles"],
        )
        rule = next(
            item
            for item in contract["input_artifacts"]["allowed"]
            if item["role"] == "scientist_intake_request"
        )
        self.assertEqual(
            "analysis/<specimen_id>/config/runtime_request.json", rule["path"]
        )


if __name__ == "__main__":
    unittest.main()
