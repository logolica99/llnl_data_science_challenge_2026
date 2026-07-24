"""End-to-end tests for orchestrator → ingest → data_prep boundaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest

import numpy as np
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data_prep_handoff import (  # noqa: E402
    DataPrepHandoffError,
    apply_data_prep_result,
    create_data_prep_handoff,
)
from specimen_ingest import ingest_specimen, inspect_lattice_graph  # noqa: E402
from specimen_manifest import (  # noqa: E402
    DEFAULT_SCHEMA,
    canonical_json_sha256,
    load_json,
    require_analysis_ready,
)


class DataPrepHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.design = self.data / "design.json"
        self.aligned = self.data / "derived-aligned.json"
        graph = {
            "junctions": [
                {"id": 0, "position": [0.0, 0.0, 0.0]},
                {"id": 1, "position": [1.0, 1.0, 1.0]},
            ],
            "struts": [{"id": 10, "junction0": 0, "junction1": 1}],
            "unit_cells": [{"id": 20, "struts": [10]}],
        }
        self.design.write_text(json.dumps(graph), encoding="utf-8")
        self.aligned.write_text(json.dumps(graph, indent=2), encoding="utf-8")
        self.cad = self.data / "design.stl"
        trimesh.creation.box().export(self.cad)
        self.ct = self.data / "scan.npy"
        np.save(self.ct, np.arange(24, dtype=np.float32).reshape(2, 3, 4))

    def _ingest(self, *, provisional: bool = False) -> dict[str, object]:
        return ingest_specimen(
            repository_root=self.root,
            specimen_id="handoff_specimen",
            cad_path=self.cad,
            design_graph_path=self.design,
            ct_path=self.ct,
            registration_mode="autonomous_v2",
            association_confirmed=True,
            cad_units="unknown" if provisional else "millimeter",
            cad_units_provenance="unknown" if provisional else "scientist declaration",
            graph_axes="unknown" if provisional else "xyz",
            array_axes="unknown" if provisional else "zyx",
            aligned_graph_units="unknown" if provisional else "simulation_voxel",
            retention="external",
            schema_path=DEFAULT_SCHEMA,
        )

    def _data_prep_result(self, manifest_path: Path) -> dict[str, object]:
        manifest = load_json(manifest_path)
        aligned = inspect_lattice_graph(
            self.aligned,
            repository_root=self.root,
            allowed_roots=[self.data],
        )
        config_hash = manifest["analysis_parameters_sha256"]
        design_hash = manifest["inputs"]["design_graph"]["sha256"]
        ct_hash = manifest["inputs"]["ct"]["sha256"]
        aligned_artifact = {
            "path": aligned["path"],
            "sha256": aligned["sha256"],
            "role": "derived_aligned_graph",
            "retention": "external",
        }
        graph_values = manifest["derived"]["graph_summary"]["values"]
        derived = {
            "graph_summary": {
                "method": "canonical_lattice_topology",
                "method_version": "1.0.0",
                "provenance": {
                    "source": "nominal and data-prep aligned graph inspection",
                    "input_sha256": sorted({design_hash, aligned["sha256"]}),
                    "config_sha256": config_hash,
                },
                "values": graph_values,
                "aligned_values": graph_values,
            },
            "voxel_spacing": {
                "method": "simulation_grid_index",
                "method_version": "1.0.0",
                "provenance": {
                    "source": "declared simulation grid",
                    "input_sha256": [ct_hash],
                    "config_sha256": config_hash,
                },
                "values": {
                    "spacing": [1.0, 1.0, 1.0],
                    "axes": ["z", "y", "x"],
                    "unit": "simulation_voxel",
                },
            },
            "segmentation_result": {
                "method": "exact_histogram_otsu",
                "method_version": "2.0.0",
                "provenance": {
                    "source": "synthetic integration-test data-prep result",
                    "input_sha256": [ct_hash],
                    "config_sha256": config_hash,
                },
                "values": {
                    "threshold": 11.0,
                    "voxel_count": 24,
                    "foreground_voxel_count": 12,
                    "foreground_fraction": 0.5,
                    "otsu_separability": 0.9,
                    "background_mean": 5.5,
                    "foreground_mean": 17.5,
                    "class_mean_separation_sigma": 2.0,
                    "significant_modes": [5.0, 18.0],
                    "histogram_sha256": "1" * 64,
                    "overall_pass": True,
                },
            },
            "registration_result": {
                "method": "autonomous_v2",
                "method_version": "1.0.0",
                "provenance": {
                    "source": "synthetic integration-test registration result",
                    "input_sha256": sorted({ct_hash, aligned["sha256"]}),
                    "config_sha256": config_hash,
                },
                "values": {
                    "aligned_graph_state": "derived",
                    "overall_pass": True,
                    "local_recenter_complete": True,
                    "roi_gate_pass": True,
                    "metrology_gate_pass": True,
                },
            },
        }
        return {
            "schema_version": "data-prep-result/1.0.0",
            "specimen_id": manifest["specimen_id"],
            "input_manifest_sha256": canonical_json_sha256(manifest),
            "analysis_parameters_sha256": config_hash,
            "aligned_graph": aligned_artifact,
            "derived": derived,
            "self_verification": {
                "exact_otsu_complete": True,
                "registration_complete": True,
                "local_recenter_complete": True,
                "roi_gate_pass": True,
                "metrology_gate_pass": True,
                "defect_labels_not_accessed": True,
            },
        }

    def test_ready_intake_emits_idempotent_data_prep_handoff(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        receipt_path = Path(intake["paths"]["ingest_receipt"])

        first = create_data_prep_handoff(
            manifest_path, receipt_path, repository_root=self.root
        )
        second = create_data_prep_handoff(
            manifest_path, receipt_path, repository_root=self.root
        )

        self.assertEqual("ready", first["handoff"]["status"])
        self.assertEqual("run_data_prep", first["handoff"]["action"])
        self.assertFalse(second["changed"])
        self.assertEqual(
            first["handoff"]["canonical_handoff_sha256"],
            second["handoff"]["canonical_handoff_sha256"],
        )

    def test_provisional_intake_halts_with_unresolved_fields(self) -> None:
        intake = self._ingest(provisional=True)
        result = create_data_prep_handoff(
            Path(intake["paths"]["specimen_manifest"]),
            Path(intake["paths"]["ingest_receipt"]),
            repository_root=self.root,
        )

        self.assertEqual("halt", result["handoff"]["status"])
        self.assertTrue(result["handoff"]["unresolved_fields"])

    def test_tampered_intake_receipt_cannot_unlock_data_prep(self) -> None:
        intake = self._ingest()
        receipt_path = Path(intake["paths"]["ingest_receipt"])
        receipt = load_json(receipt_path)
        receipt["manifest_sha256"] = "0" * 64
        receipt["canonical_receipt_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "canonical_receipt_sha256"
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "manifest_sha256"):
            create_data_prep_handoff(
                Path(intake["paths"]["specimen_manifest"]),
                receipt_path,
                repository_root=self.root,
            )

    def test_invalid_receipt_self_hash_cannot_unlock_data_prep(self) -> None:
        intake = self._ingest()
        receipt_path = Path(intake["paths"]["ingest_receipt"])
        receipt = load_json(receipt_path)
        receipt["canonical_receipt_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "canonical hash"):
            create_data_prep_handoff(
                Path(intake["paths"]["specimen_manifest"]),
                receipt_path,
                repository_root=self.root,
            )

    def test_data_prep_result_advances_manifest_to_analysis_ready(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        result_path = manifest_path.parent / "data_prep_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        completion = apply_data_prep_result(
            manifest_path,
            result_path,
            repository_root=self.root,
        )

        finalized = require_analysis_ready(
            manifest_path,
            consumer="roi_metrics",
            schema_path=DEFAULT_SCHEMA,
            repository_root=self.root,
        )
        self.assertEqual("analysis_ready", finalized["lifecycle_state"])
        self.assertEqual(
            "derived_aligned_graph",
            finalized["inputs"]["aligned_graph"]["role"],
        )
        self.assertTrue(Path(completion["completion_receipt_path"]).is_file())

    def test_failed_data_prep_self_verification_cannot_advance(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        result["self_verification"]["metrology_gate_pass"] = False
        result_path = manifest_path.parent / "failed_data_prep_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "metrology_gate_pass"):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )
        self.assertEqual(
            "ready_for_data_prep", load_json(manifest_path)["lifecycle_state"]
        )

    def test_agent_and_stage_contracts_are_bounded(self) -> None:
        agent = tomllib.loads(
            (
                REPOSITORY_ROOT / ".codex/agents/specimen_ingest.toml"
            ).read_text(encoding="utf-8")
        )
        contract = json.loads(
            (
                REPOSITORY_ROOT / "analysis/contracts/specimen_ingest.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual("specimen_ingest", agent["name"])
        self.assertIn("at most 2 correction attempts", agent["developer_instructions"])
        self.assertIn("does not compute or choose Otsu", agent["developer_instructions"])
        self.assertEqual("orchestrator", contract["invoked_by"])
        self.assertEqual(2, contract["maximum_attempts"])
        self.assertEqual("data_prep", contract["next_stage"])


if __name__ == "__main__":
    unittest.main()
