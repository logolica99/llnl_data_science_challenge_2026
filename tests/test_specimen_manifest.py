"""Tests for the Part 2 manifest schema and provenance gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.contracts import (  # noqa: E402
    ManifestValidationError,
    canonical_json_sha256,
    manifest_paths,
    require_analysis_ready,
    topology_summary,
    validate_manifest,
)
from llnl_nde.cli.segmentation_replay import histogram_sha256, otsu_from_histogram  # noqa: E402

import numpy as np


class SpecimenManifestTests(unittest.TestCase):
    @staticmethod
    def _promote_analysis_ready(
        manifest: dict[str, object],
    ) -> dict[str, object]:
        manifest["lifecycle_state"] = "analysis_ready"
        inputs = manifest["inputs"]
        inputs["canonical_mask"] = {
            "path": (
                f"analysis/{manifest['specimen_id']}/segmentation/"
                "canonical_mask.npy"
            ),
            "sha256": "a" * 64,
            "role": "canonical_segmentation_mask",
            "retention": "committed",
            "dtype": "uint8",
            "shape": inputs["ct_metadata"]["shape"],
            "array_axes": ["z", "y", "x"],
        }
        return manifest

    def test_all_example_manifests_validate(self) -> None:
        paths = manifest_paths()
        self.assertGreaterEqual(len(paths), 2)
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual([], validate_manifest(path))

    def test_manifest_rejects_stale_config_hash(self) -> None:
        source = manifest_paths()[0]
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["analysis_parameters"]["budgets"]["maximum_agent_retries"] += 1
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)
        with self.assertRaisesRegex(
            ManifestValidationError, "analysis_parameters_sha256"
        ):
            validate_manifest(temporary)

    def test_derived_records_reject_arbitrary_extra_fields(self) -> None:
        source = manifest_paths()[0]
        for section in (
            "graph_summary",
            "voxel_spacing",
            "segmentation_result",
            "registration_result",
        ):
            with self.subTest(section=section):
                manifest = json.loads(source.read_text(encoding="utf-8"))
                manifest["derived"][section]["unrecognized_field"] = True
                temporary = self._write_temporary(manifest)
                self.addCleanup(
                    lambda path=temporary: path.unlink(missing_ok=True)
                )
                with self.assertRaisesRegex(
                    ManifestValidationError, "unrecognized_field"
                ):
                    validate_manifest(temporary)

    def test_analysis_ready_requires_canonical_mask(self) -> None:
        source = next(
            path
            for path in manifest_paths()
            if "brian_tran_9x9x9_0point5dash1" in path.as_posix()
        )
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["lifecycle_state"] = "analysis_ready"
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "canonical_mask"):
            validate_manifest(temporary)

    def test_analysis_ready_canonical_mask_is_specimen_scoped_and_shape_bound(
        self,
    ) -> None:
        source = next(
            path
            for path in manifest_paths()
            if "brian_tran_9x9x9_0point5dash1" in path.as_posix()
        )
        for name, mutate, pattern in (
            (
                "path",
                lambda value: value["inputs"]["canonical_mask"].__setitem__(
                    "path", "analysis/other_specimen/segmentation/canonical_mask.npy"
                ),
                "specimen-scoped canonical path",
            ),
            (
                "shape",
                lambda value: value["inputs"]["canonical_mask"].__setitem__(
                    "shape", [1, 1, 1]
                ),
                "shape differs from CT shape",
            ),
        ):
            with self.subTest(case=name):
                manifest = self._promote_analysis_ready(
                    json.loads(source.read_text(encoding="utf-8"))
                )
                mutate(manifest)
                temporary = self._write_temporary(manifest)
                self.addCleanup(lambda path=temporary: path.unlink(missing_ok=True))
                with self.assertRaisesRegex(ManifestValidationError, pattern):
                    validate_manifest(temporary)

    def test_registration_result_rejects_open_or_unknown_control_fields(self) -> None:
        source = next(
            path
            for path in manifest_paths()
            if "brian_tran_9x9x9_0point5dash1" in path.as_posix()
        )
        cases = (
            (
                "roi-gate-extra",
                lambda values: values["roi_gate_results"].__setitem__(
                    "unrecognized_gate", True
                ),
                "roi_gate_results",
            ),
            (
                "localization-count-extra",
                lambda values: values["localization_quality_counts"].__setitem__(
                    "unrecognized_count", 0
                ),
                "localization_quality_counts",
            ),
            (
                "authorization-enum",
                lambda values: values["authorized_outputs"].append(
                    "unrecognized_output"
                ),
                "authorized_outputs",
            ),
            (
                "reason-enum",
                lambda values: values["reason_codes"].append(
                    "UNRECOGNIZED_REASON"
                ),
                "reason_codes",
            ),
        )
        for name, mutate, pattern in cases:
            with self.subTest(case=name):
                manifest = self._promote_analysis_ready(
                    json.loads(source.read_text(encoding="utf-8"))
                )
                mutate(manifest["derived"]["registration_result"]["values"])
                temporary = self._write_temporary(manifest)
                self.addCleanup(lambda path=temporary: path.unlink(missing_ok=True))
                with self.assertRaisesRegex(ManifestValidationError, pattern):
                    validate_manifest(temporary)

    def test_registration_result_requires_exact_scope_authorization_and_reasons(
        self,
    ) -> None:
        source = next(
            path
            for path in manifest_paths()
            if "brian_tran_9x9x9_0point5dash1" in path.as_posix()
        )
        for name, mutate, pattern in (
            (
                "authorization",
                lambda values: values["authorized_outputs"].append(
                    "absolute_metrology"
                ),
                "exact roi_screening allowlist",
            ),
            (
                "reasons",
                lambda values: values.__setitem__(
                    "reason_codes", ["ROI_GATES_PASS", "METROLOGY_GATES_PASS"]
                ),
                "exact roi_screening result",
            ),
        ):
            with self.subTest(case=name):
                manifest = self._promote_analysis_ready(
                    json.loads(source.read_text(encoding="utf-8"))
                )
                mutate(manifest["derived"]["registration_result"]["values"])
                temporary = self._write_temporary(manifest)
                self.addCleanup(lambda path=temporary: path.unlink(missing_ok=True))
                with self.assertRaisesRegex(ManifestValidationError, pattern):
                    validate_manifest(temporary)

    def test_intake_voxel_spacing_provenance_is_closed(self) -> None:
        source = manifest_paths()[0]
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["inputs"]["ct_metadata"]["voxel_spacing"] = {
            axis: {
                "value": "unknown",
                "unit": "unknown",
                "provenance": {
                    "source": "unknown",
                    "field": "unknown",
                    "raw_value": "unknown",
                    "unexpected": True,
                },
            }
            for axis in ("z", "y", "x")
        }
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "unexpected"):
            validate_manifest(temporary)

    def test_manifest_rejects_conflicting_frozen_stage_2_policy(self) -> None:
        source = manifest_paths()[0]
        for name, mutate, pattern in (
            (
                "search-budget",
                lambda value: value["analysis_parameters"]["localization_policy"].__setitem__(
                    "search_radius_voxels", 7.5
                ),
                "search_radius_voxels differs",
            ),
            (
                "padding-budget",
                lambda value: value["analysis_parameters"]["qa_policy"].__setitem__(
                    "roi_padding_fraction", 0.1
                ),
                "roi_padding_fraction differs",
            ),
            (
                "support-weights",
                lambda value: value["analysis_parameters"]["localization_policy"].__setitem__(
                    "incident_support_weight", 0.4
                ),
                "support weights must sum to 1",
            ),
        ):
            with self.subTest(case=name):
                manifest = json.loads(source.read_text(encoding="utf-8"))
                mutate(manifest)
                new_hash = canonical_json_sha256(manifest["analysis_parameters"])
                manifest["analysis_parameters_sha256"] = new_hash
                for record in manifest["derived"].values():
                    record["provenance"]["config_sha256"] = new_hash
                temporary = self._write_temporary(manifest)
                self.addCleanup(lambda path=temporary: path.unlink(missing_ok=True))
                with self.assertRaisesRegex(ManifestValidationError, pattern):
                    validate_manifest(temporary)

    def test_nominal_and_registered_9x9_topologies_match(self) -> None:
        nominal = topology_summary(
            REPOSITORY_ROOT / "data/missing_struts/octet_truss_9x9x9.json"
        )
        aligned = topology_summary(
            REPOSITORY_ROOT
            / "data/missing_struts/registered_jsons"
            / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
        )
        self.assertEqual(nominal, aligned)
        self.assertEqual(10_206, nominal["junction_count"])
        self.assertEqual(18_468, nominal["strut_count"])
        self.assertEqual(729, nominal["unit_cell_count"])

    def test_config_hash_is_key_order_independent(self) -> None:
        left = {"b": 2, "a": {"d": 4, "c": 3}}
        right = {"a": {"c": 3, "d": 4}, "b": 2}
        self.assertEqual(canonical_json_sha256(left), canonical_json_sha256(right))

    def test_otsu_replay_uses_manifest_threshold_convention(self) -> None:
        histogram = np.zeros(65_536, dtype=np.int64)
        histogram[10] = 5
        histogram[20] = 5
        threshold, separability = otsu_from_histogram(histogram)
        self.assertEqual(10, threshold)
        self.assertEqual(1.0, separability)
        self.assertEqual(64, len(histogram_sha256(histogram)))

    def test_autonomous_provisional_manifest_allows_pending_aligned_graph(self) -> None:
        source = next(
            path
            for path in manifest_paths()
            if "brian_tran_9x9x9_0point5dash1_production" in path.as_posix()
            and path.as_posix().endswith(
                "brian_tran_9x9x9_0point5dash1_production/config/specimen_manifest.json"
            )
        )
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["lifecycle_state"] = "provisional"
        manifest["unresolved_fields"] = ["analysis_parameters.coordinates.array_axes"]
        manifest["analysis_parameters"]["coordinates"]["array_axes"] = "unknown"
        manifest["inputs"].pop("aligned_graph", None)
        graph_summary = manifest["derived"]["graph_summary"]
        graph_summary.pop("aligned_values", None)
        graph_summary["provenance"]["input_sha256"] = [
            manifest["inputs"]["design_graph"]["sha256"]
        ]
        manifest["derived"] = {"graph_summary": graph_summary}
        manifest["analysis_parameters_sha256"] = canonical_json_sha256(
            manifest["analysis_parameters"]
        )
        graph_summary["provenance"]["config_sha256"] = manifest[
            "analysis_parameters_sha256"
        ]
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        self.assertEqual([], validate_manifest(temporary))

    def test_challenge_provisional_manifest_requires_supplied_aligned_graph(self) -> None:
        source = next(
            path
            for path in manifest_paths()
            if path.as_posix().endswith(
                "brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json"
            )
        )
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["lifecycle_state"] = "provisional"
        del manifest["inputs"]["aligned_graph"]
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "aligned_graph"):
            validate_manifest(temporary)

    def test_ready_for_data_prep_rejects_unresolved_fields(self) -> None:
        source = manifest_paths()[0]
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["lifecycle_state"] = "ready_for_data_prep"
        manifest["unresolved_fields"] = ["inputs.ct_metadata.array_axes"]
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "unresolved_fields"):
            validate_manifest(temporary)

    def test_downstream_consumer_rejects_provisional_manifest(self) -> None:
        source = manifest_paths()[0]
        manifest = json.loads(source.read_text(encoding="utf-8"))
        manifest["lifecycle_state"] = "provisional"
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(
            ManifestValidationError, "roi_metrics rejected manifest"
        ):
            require_analysis_ready(temporary, consumer="roi_metrics")

    def test_roi_analysis_ready_rejects_failed_roi_gate(self) -> None:
        source = manifest_paths()[0]
        manifest = self._promote_analysis_ready(
            json.loads(source.read_text(encoding="utf-8"))
        )
        manifest["derived"]["registration_result"]["values"][
            "roi_gate_results"
        ]["padded_roi_in_bounds"] = False
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "failed ROI gate"):
            validate_manifest(temporary)

    def test_direct_metrology_analysis_ready_requires_passing_uncertainty(self) -> None:
        source = next(
            path
            for path in manifest_paths()
            if path.as_posix().endswith(
                "brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json"
            )
        )
        manifest = self._promote_analysis_ready(
            json.loads(source.read_text(encoding="utf-8"))
        )
        manifest["analysis_parameters"]["requested_analysis_scope"] = "direct_metrology"
        manifest["analysis_parameters_sha256"] = canonical_json_sha256(
            manifest["analysis_parameters"]
        )
        values = manifest["derived"]["registration_result"]["values"]
        values["requested_analysis_scope"] = "direct_metrology"
        values["metrology_gate_status"] = "insufficient_evidence"
        values["reason_codes"] = ["ROI_GATES_PASS", "METROLOGY_GATES_PASS"]
        values["authorized_outputs"] = [
            "segmentation",
            "registration",
            "node_localization",
            "coarse_region_screening",
            "padded_roi_definition",
            "absolute_metrology",
            "direct_dimensional_measurement",
        ]
        values["unauthorized_outputs"] = []
        for section in manifest["derived"].values():
            if isinstance(section, dict) and isinstance(section.get("provenance"), dict):
                section["provenance"]["config_sha256"] = manifest[
                    "analysis_parameters_sha256"
                ]
        temporary = self._write_temporary(manifest)
        self.addCleanup(temporary.unlink)

        with self.assertRaisesRegex(ManifestValidationError, "metrology_gate_status"):
            validate_manifest(temporary)

    def _write_temporary(self, manifest: dict[str, object]) -> Path:
        path = REPOSITORY_ROOT / "analysis" / "schema" / ".invalid-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
