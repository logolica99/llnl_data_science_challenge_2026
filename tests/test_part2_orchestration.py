"""Production topology tests for the nominal-graph + CT control plane."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.pipeline import (  # noqa: E402
    ManifestValidationError,
    REGISTRATION_MODES,
    STAGE_NAMES,
    STAGE_NUMBERS,
    _load_contracts,
    create_pipeline_manifest,
    validate_pipeline_manifest,
)


class ProductionPipelineTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        shutil.copytree(
            REPOSITORY_ROOT / "analysis" / "contracts",
            self.root / "analysis" / "contracts",
        )
        config = self.root / "config" / "frozen.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"frozen":true}\n', encoding="utf-8")
        self.config = config

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_production_stage_order_is_contiguous_zero_through_four(self) -> None:
        self.assertEqual(tuple(range(5)), STAGE_NUMBERS)
        self.assertEqual(
            (
                "specimen_ingest",
                "data_prep",
                "strut_metrics",
                "defect_analysis",
                "nde_report",
            ),
            STAGE_NAMES,
        )
        contracts = _load_contracts(self.root, "analysis/contracts")
        self.assertEqual(
            list(enumerate(STAGE_NAMES)),
            [
                (item["document"]["stage_number"], item["document"]["stage"])
                for item in contracts
            ],
        )

    def test_intake_requires_graph_and_ct_not_cad_or_design_variants(self) -> None:
        contracts = _load_contracts(self.root, "analysis/contracts")
        intake = contracts[0]["document"]
        self.assertEqual(
            {
                "stage_handoff",
                "scientist_intake_request",
                "nominal_graph",
                "ct_volume",
                "specimen_manifest_schema",
                "volume_metadata_skill",
            },
            set(intake["input_artifacts"]["required_roles"]),
        )
        serialized = json.dumps(intake).lower()
        self.assertNotIn("design_diff", serialized)
        self.assertNotIn("0.1.stl", serialized)
        self.assertNotIn("0.5.stl", serialized)
        self.assertIn("load_lattice_graph", serialized)
        self.assertIn("normalized_nominal_graph", serialized)

    def test_only_autonomous_registration_is_production_mode(self) -> None:
        self.assertEqual({"autonomous_v2"}, REGISTRATION_MODES)
        with self.assertRaisesRegex(Exception, "Unsupported registration mode"):
            create_pipeline_manifest(
                repository_root=self.root,
                specimen_id="sample",
                config_path=self.config,
                registration_mode="challenge_aligned_json",
            )

    def test_manifest_contains_five_hash_bound_stages(self) -> None:
        created = create_pipeline_manifest(
            repository_root=self.root,
            specimen_id="sample",
            config_path=self.config,
            registration_mode="autonomous_v2",
            timestamp="2026-07-28T00:00:00Z",
        )
        manifest = created["manifest"]
        self.assertEqual([0, 1, 2, 3, 4], manifest["stage_order"])
        self.assertEqual("ready", manifest["stages"]["0"]["state"])
        self.assertTrue(
            all(manifest["stages"][str(number)]["state"] == "locked" for number in range(1, 5))
        )
        validated = validate_pipeline_manifest(
            created["path"], repository_root=self.root
        )
        self.assertEqual(manifest["manifest_sha256"], validated["manifest_sha256"])

    def test_research_contracts_are_not_loaded_as_production_stages(self) -> None:
        contracts = _load_contracts(self.root, "analysis/contracts")
        loaded_paths = {Path(item["path"]).name for item in contracts}
        self.assertNotIn("design_diff.json", loaded_paths)
        self.assertNotIn("sealed_evaluation.json", loaded_paths)


if __name__ == "__main__":
    unittest.main()
