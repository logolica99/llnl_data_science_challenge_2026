"""Agent contract tests for production intake and data preparation."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class IntakeAndDataPrepAgentTests(unittest.TestCase):
    def test_only_production_stage_agents_are_discoverable(self) -> None:
        agent_names = {path.stem for path in (ROOT / ".codex" / "agents").glob("*.toml")}
        self.assertEqual(
            {
                "orchestrator",
                "specimen_ingest",
                "data_prep",
                "strut_metrics",
                "defect_lead",
                "missing_strut_agent",
                "broken_strut_agent",
                "thin_strut_agent",
                "classifier_verifier",
                "report_agent",
            },
            agent_names,
        )

    def test_agents_use_the_declared_model(self) -> None:
        for path in (ROOT / ".codex" / "agents").glob("*.toml"):
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("gpt-5.6-sol", document["model"])
            self.assertEqual("high", document["model_reasoning_effort"])

    def test_ingest_owns_graph_normalization(self) -> None:
        prompt = tomllib.loads(
            (ROOT / ".codex" / "agents" / "specimen_ingest.toml").read_text(
                encoding="utf-8"
            )
        )["developer_instructions"]
        self.assertIn("load_lattice_graph", prompt)
        self.assertIn("normalized_nominal_graph.npz", prompt)
        self.assertIn("nominal lattice graph JSON", prompt)
        self.assertIn("CT TIFF/NPY", prompt)

    def test_production_agents_do_not_reference_design_diff(self) -> None:
        for path in (ROOT / ".codex" / "agents").glob("*.toml"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("`design_diff`", text)
            self.assertNotIn("stl-design-diff", text)
            self.assertNotIn("sealed_split", text)

    def test_data_prep_is_stage_one_and_autonomous(self) -> None:
        document = tomllib.loads(
            (ROOT / ".codex" / "agents" / "data_prep.toml").read_text(
                encoding="utf-8"
            )
        )
        prompt = document["developer_instructions"]
        self.assertIn("Stage 1", prompt)
        self.assertIn("autonomous", prompt)
        self.assertNotIn("challenge_aligned_json", prompt)

    def test_strut_metrics_is_stage_two_and_mcp_only(self) -> None:
        document = tomllib.loads(
            (ROOT / ".codex" / "agents" / "strut_metrics.toml").read_text(
                encoding="utf-8"
            )
        )
        prompt = document["developer_instructions"]
        self.assertIn("Stage 2", prompt)
        self.assertIn("compute_strut_metrics", prompt)
        self.assertIn("canonical_segmentation_mask", prompt)
        self.assertIn("Never call a", prompt)

    def test_contracts_route_stage_zero_directly_to_data_prep(self) -> None:
        intake = json.loads(
            (ROOT / "analysis" / "contracts" / "specimen_ingest.json").read_text()
        )
        data_prep = json.loads(
            (ROOT / "analysis" / "contracts" / "data_prep.json").read_text()
        )
        self.assertEqual("data_prep", intake["next_stage"])
        self.assertEqual(1, data_prep["stage_number"])
        self.assertIn(
            "normalized_nominal_graph",
            data_prep["input_artifacts"]["required_roles"],
        )


if __name__ == "__main__":
    unittest.main()
