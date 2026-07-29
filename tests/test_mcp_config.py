"""Tests for the checked-in production MCP server configuration."""

from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MCPConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = REPOSITORY_ROOT / ".codex/config.toml"
        with cls.config_path.open("rb") as stream:
            cls.config = tomllib.load(stream)

    def test_production_tools_are_preapproved_without_human_prompt(self) -> None:
        server = self.config["mcp_servers"]["segmentation-tools"]
        self.assertEqual("approve", server["default_tools_approval_mode"])

    def test_server_configuration_is_frozen_and_resolves_entrypoints(self) -> None:
        expected = {
            "segmentation-tools": "src/llnl_nde/server.py",
            "segmentation-tools-research": "research/mcp_server.py",
        }
        for name, entrypoint in expected.items():
            with self.subTest(server=name):
                server = self.config["mcp_servers"][name]
                self.assertEqual("uv", server["command"])
                self.assertEqual(
                    ["run", "--frozen", "python", entrypoint], server["args"]
                )
                expected_cwd = (
                    str(REPOSITORY_ROOT)
                    if name == "segmentation-tools"
                    else "."
                )
                self.assertEqual(expected_cwd, server["cwd"])
                self.assertEqual(
                    REPOSITORY_ROOT,
                    (REPOSITORY_ROOT / server["cwd"]).resolve(),
                )
                self.assertTrue((REPOSITORY_ROOT / entrypoint).is_file())
                self.assertFalse(Path(server["command"]).is_absolute())
                self.assertTrue(
                    all(not Path(argument).is_absolute() for argument in server["args"])
                )

        production = self.config["mcp_servers"]["segmentation-tools"]
        research = self.config["mcp_servers"]["segmentation-tools-research"]
        self.assertTrue(production["enabled"])
        self.assertTrue(production["required"])
        self.assertFalse(research["enabled"])
        self.assertFalse(research["required"])


if __name__ == "__main__":
    unittest.main()
