"""End-to-end tests for orchestrator → ingest → data_prep boundaries."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from fastmcp import Client
import numpy as np
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.receipts import (  # noqa: E402
    DataPrepHandoffError,
    apply_data_prep_result,
    create_data_prep_handoff,
)
from llnl_nde import server as mcp_server  # noqa: E402
from llnl_nde.mcp_tools import common as mcp_common  # noqa: E402
from llnl_nde.core.otsu import replay_exact_otsu  # noqa: E402
from llnl_nde.core.segmentation import compare_segmentation_masks  # noqa: E402
from llnl_nde.orchestration.specimen_ingest import (  # noqa: E402
    ingest_specimen,
    inspect_lattice_graph,
)
from llnl_nde.orchestration.contracts import (  # noqa: E402
    DEFAULT_SCHEMA,
    canonical_json_sha256,
    load_json,
    require_analysis_ready,
    sha256_file,
)
from tests.stage0_metadata_fixture import (  # noqa: E402
    write_ct_metadata_response_fixture,
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
        zz, yy, xx = np.indices((10, 10, 10))
        background = (180 + ((17 * zz + 11 * yy + 5 * xx) % 41)).astype(
            np.uint16
        )
        foreground = xx < 2
        self.ct_values = np.where(
            foreground,
            980 + ((13 * zz + 7 * yy + 3 * xx) % 41),
            background,
        ).astype(np.uint16)
        np.save(self.ct, self.ct_values)
        self.segmentation = (
            self.root / "analysis" / "handoff_specimen" / "segmentation"
        )
        self.segmentation.mkdir(parents=True)
        self.mask = self.segmentation / "canonical_mask.npy"
        self.otsu_report = self.segmentation / "histogram_report.json"
        self.mask_comparison = self.segmentation / "mask_comparison.json"
        self.segmentation_verification = (
            self.segmentation / "segmentation_verification_mcp_response.json"
        )
        self.localization_report = (
            self.root
            / "analysis"
            / "handoff_specimen"
            / "registration"
            / "localization_report.json"
        )
        self.localization_report.parent.mkdir(parents=True)
        self.localized_graph = self.localization_report.parent / "localized_graph.json"
        self.localized_graph.write_text(
            json.dumps(graph, sort_keys=True), encoding="utf-8"
        )
        self.registration_report = (
            self.localization_report.parent / "registration_report.json"
        )
        self.registration_report.write_text(
            json.dumps(
                {
                    "schema_version": "part2-registration/1.0.0",
                    "mode": "autonomous_v2",
                    "gate": "pass",
                    "hashes": {
                        "ct_sha256": sha256_file(self.ct),
                        "registered_graph_sha256": sha256_file(self.aligned),
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.registration_qa = (
            self.root
            / "analysis"
            / "handoff_specimen"
            / "qa"
            / "registration_qa.json"
        )
        self.registration_qa.parent.mkdir(parents=True)

    def _exact_otsu_values(
        self, manifest: dict[str, object]
    ) -> dict[str, object]:
        policy = manifest["analysis_parameters"]["segmentation"]
        shape = manifest["inputs"]["ct_metadata"]["shape"]
        recipe = {
            "histogram_encoding": policy["histogram_encoding"],
            "edge_slices_excluded": policy["edge_slices_excluded"],
            "chunk_voxels": int(policy["chunk_depth"])
            * int(shape[1])
            * int(shape[2]),
            "coarse_bins": policy["coarse_bins"],
            "peak_smoothing_sigma_bins": policy["peak_smoothing_sigma_bins"],
            "peak_prominence_fraction": policy["peak_prominence_fraction"],
            "minimum_significant_peaks": policy["minimum_significant_peaks"],
            "minimum_foreground_fraction": policy["minimum_foreground_fraction"],
            "maximum_foreground_fraction": policy["maximum_foreground_fraction"],
            "minimum_otsu_separability": policy["minimum_otsu_separability"],
            "minimum_class_mean_separation_sigma": policy[
                "minimum_class_mean_separation_sigma"
            ],
        }
        replay, _ = replay_exact_otsu(self.ct, recipe=recipe)
        self.assertTrue(replay["overall_pass"])
        return replay

    def _ingest(self, *, provisional: bool = False) -> dict[str, object]:
        metadata_path, metadata_sha256 = write_ct_metadata_response_fixture(
            repository_root=self.root,
            specimen_id="handoff_specimen",
            ct_path=self.ct,
            shape=self.ct_values.shape,
            dtype="uint16",
            dtype_string=np.dtype(np.uint16).str,
            byte_order="little",
            volume_format="npy",
            retention="external",
        )
        return ingest_specimen(
            repository_root=self.root,
            specimen_id="handoff_specimen",
            design_id="handoff_design",
            requested_analysis_scope="roi_screening",
            cad_path=self.cad,
            design_graph_path=self.design,
            ct_path=self.ct,
            ct_metadata_response_path=metadata_path,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=metadata_path.with_name(
                "ct_metadata_mcp_call_receipt.json"
            ),
            ct_metadata_call_receipt_sha256=sha256_file(
                metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
            ),
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

    def _write_stage2_reports(
        self,
        manifest_path: Path,
        manifest: dict[str, object],
        *,
        threshold: float,
    ) -> None:
        parameters = manifest["analysis_parameters"]
        analysis_parameters_sha256 = manifest["analysis_parameters_sha256"]
        scope_artifact_sha256 = sha256_file(manifest_path)
        ct_sha256 = manifest["inputs"]["ct"]["sha256"]
        localized_graph_sha256 = sha256_file(self.localized_graph)
        localization_policy = parameters["localization_policy"]
        localization_policy_sha256 = canonical_json_sha256(localization_policy)
        localization_config_sha256 = canonical_json_sha256(
            {
                key: value
                for key, value in localization_policy.items()
                if key != "schema_version"
            }
        )
        qa_policy_sha256 = canonical_json_sha256(parameters["qa_policy"])
        qa_config_sha256 = canonical_json_sha256(
            {
                key: value
                for key, value in parameters["qa_policy"].items()
                if key != "schema_version"
            }
        )
        segmentation_policy_sha256 = canonical_json_sha256(
            parameters["segmentation"]
        )
        quantitative_policy = {
            key: localization_policy[key]
            for key in (
                "minimum_primary_or_stable_coarse_fraction",
                "maximum_fallback_fraction",
                "maximum_ambiguous_fraction",
                "maximum_rejected_fraction",
                "maximum_boundary_limited_fraction",
            )
        }
        segmentation_binding = {
            "method": "exact_histogram_otsu",
            "method_version": "2.0.0",
            "threshold": threshold,
            "threshold_comparison": "value >= threshold",
            "ct_sha256": ct_sha256,
            "segmentation_policy_sha256": segmentation_policy_sha256,
            "overall_pass": True,
        }
        localization_document = {
            "schema_version": "part2-node-localization/1.2.0",
            "specimen_id": manifest["specimen_id"],
            "design_id": manifest["design_id"],
            "requested_analysis_scope": parameters["requested_analysis_scope"],
            "registration_mode": parameters["registration"]["mode"],
            "threshold": threshold,
            "gate": "pass",
            "overall_pass": True,
            "quantitative_policy": quantitative_policy,
            "segmentation_binding": segmentation_binding,
            "analysis_policy_source": {
                "source_artifact_path": str(manifest_path.resolve()),
                "source_artifact_sha256": scope_artifact_sha256,
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "localization_policy_sha256": localization_policy_sha256,
                "requested_analysis_scope": parameters[
                    "requested_analysis_scope"
                ],
                "registration_mode": parameters["registration"]["mode"],
                "specimen_id": manifest["specimen_id"],
                "design_id": manifest["design_id"],
                "declared_ct_sha256": ct_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
            },
            "hashes": {
                "ct_sha256": ct_sha256,
                "input_registered_graph_sha256": sha256_file(self.aligned),
                "localized_graph_sha256": localized_graph_sha256,
                "registration_report_sha256": sha256_file(
                    self.registration_report
                ),
                "analysis_policy_artifact_sha256": scope_artifact_sha256,
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "localization_policy_sha256": localization_policy_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
            },
            "provenance": {
                "registration_mode": parameters["registration"]["mode"],
                "config_sha256": localization_config_sha256,
                "policy_binding": "hashed_analysis_parameters",
                "sealed_labels_read": False,
            },
            "artifacts": {
                "localized_graph": {
                    "path": str(self.localized_graph.resolve()),
                    "sha256": localized_graph_sha256,
                    "role": "independently_localized_lattice_graph",
                    "retention": "regenerable",
                },
                "registration_report": {
                    "path": str(self.registration_report.resolve()),
                    "sha256": sha256_file(self.registration_report),
                    "role": "registration_report",
                    "retention": "committed",
                },
            },
        }
        self.localization_report.write_text(
            json.dumps(localization_document, sort_keys=True), encoding="utf-8"
        )
        localization_report_sha256 = sha256_file(self.localization_report)
        binding_values = {
            "specimen_id": manifest["specimen_id"],
            "design_id": manifest["design_id"],
            "requested_analysis_scope": parameters["requested_analysis_scope"],
            "registration_mode": parameters["registration"]["mode"],
            "ct_sha256": ct_sha256,
            "localized_graph_sha256": localized_graph_sha256,
            "analysis_parameters_sha256": analysis_parameters_sha256,
            "localization_policy_sha256": localization_policy_sha256,
            "analysis_policy_artifact_sha256": scope_artifact_sha256,
        }
        qa_document = {
            "schema_version": "part2-registration-qa/1.2.0",
            "specimen_id": manifest["specimen_id"],
            "design_id": manifest["design_id"],
            "requested_analysis_scope": parameters["requested_analysis_scope"],
            "registration_mode": parameters["registration"]["mode"],
            "threshold": threshold,
            "gate": "pass",
            "overall_pass": True,
            "authorized_outputs": [
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
            ],
            "unauthorized_outputs": [
                "absolute_metrology",
                "direct_dimensional_measurement",
            ],
            "roi_gate_results": {
                "image_support": True,
                "localization_quality": True,
                "coarse_region_support": True,
                "padded_roi_in_bounds": True,
            },
            "metrology": {"status": "not_authorized"},
            "localization_quality_counts": {
                "primary": 2,
                "stable_coarse": 0,
                "fallback": 0,
                "ambiguous": 0,
                "rejected": 0,
                "boundary_limited": 0,
            },
            "reason_codes": ["ROI_GATES_PASS", "METROLOGY_NOT_AUTHORIZED"],
            "segmentation_binding": {
                **segmentation_binding,
                "gates": {
                    "exact_otsu_replay_passed": True,
                    "threshold_matches_exact_otsu": True,
                },
            },
            "localization_binding": {
                "artifact": {
                    "path": str(self.localization_report.resolve()),
                    "sha256": localization_report_sha256,
                    "role": "localization_report",
                },
                "expected": binding_values,
                "observed": binding_values,
                "gates": {"all_bindings_match": True},
                "overall_pass": True,
            },
            "scope_source": {
                "requested_analysis_scope": parameters[
                    "requested_analysis_scope"
                ],
                "source_artifact_kind": "specimen_manifest",
                "source_artifact_path": str(manifest_path.resolve()),
                "source_artifact_sha256": scope_artifact_sha256,
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "localization_policy_sha256": localization_policy_sha256,
                "qa_policy_sha256": qa_policy_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
                "specimen_id": manifest["specimen_id"],
                "design_id": manifest["design_id"],
                "declared_ct_sha256": ct_sha256,
                "registration_mode": parameters["registration"]["mode"],
            },
            "hashes": {
                "ct_sha256": ct_sha256,
                "localized_graph_sha256": localized_graph_sha256,
                "registration_report_sha256": sha256_file(
                    self.registration_report
                ),
                "analysis_scope_artifact_sha256": scope_artifact_sha256,
                "analysis_parameters_sha256": analysis_parameters_sha256,
                "localization_policy_sha256": localization_policy_sha256,
                "qa_policy_sha256": qa_policy_sha256,
                "segmentation_policy_sha256": segmentation_policy_sha256,
                "localization_report_sha256": localization_report_sha256,
            },
            "provenance": {
                "registration_mode": parameters["registration"]["mode"],
                "config_sha256": qa_config_sha256,
                "policy_binding": "hashed_analysis_parameters",
            },
        }
        self.registration_qa.write_text(
            json.dumps(qa_document, sort_keys=True), encoding="utf-8"
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
        exact_otsu = self._exact_otsu_values(manifest)
        threshold = exact_otsu["threshold"]
        replay_config_sha256 = canonical_json_sha256(
            {
                "recipe": exact_otsu["recipe"],
                "registration_mode": "autonomous_v2",
                "enforce_reference_replay": False,
            }
        )
        np.save(
            self.mask,
            np.asarray(self.ct_values >= threshold, dtype=np.uint8),
        )
        exact_otsu.update(
            {
                "source_path": self.ct.relative_to(self.root).as_posix(),
                "registration_mode": "autonomous_v2",
                "analysis_policy_artifact": {
                    "path": manifest_path.relative_to(self.root).as_posix(),
                    "sha256": sha256_file(manifest_path),
                    "role": "specimen_manifest",
                },
                "hashes": {
                    "input_sha256": ct_hash,
                    "config_sha256": replay_config_sha256,
                    "analysis_parameters_sha256": config_hash,
                    "segmentation_policy_sha256": canonical_json_sha256(
                        manifest["analysis_parameters"]["segmentation"]
                    ),
                    "analysis_policy_artifact_sha256": sha256_file(manifest_path),
                },
                "provenance": {
                    "registration_mode": "autonomous_v2",
                    "threshold_selected_per_scan": True,
                    "target_foreground_fraction_used": False,
                    "defect_labels_read": False,
                    "policy_binding": "hashed_analysis_parameters",
                },
            }
        )
        self.otsu_report.write_text(
            json.dumps(exact_otsu, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison = compare_segmentation_masks(
            self.ct,
            [self.mask],
            [threshold],
            registration_mode="autonomous_v2",
            output_report_path=self.mask_comparison,
            repository_root=self.root,
        )
        self.assertEqual("pass", comparison["gate"])
        with patch.object(mcp_common, "REPOSITORY_ROOT", self.root):
            verification = mcp_server.verify_canonical_segmentation(
                specimen_id=manifest["specimen_id"],
                design_id=manifest["design_id"],
                analysis_policy_artifact_filepath=manifest_path.relative_to(
                    self.root
                ).as_posix(),
                exact_otsu_report_filepath=self.otsu_report.relative_to(
                    self.root
                ).as_posix(),
                canonical_mask_filepath=self.mask.relative_to(self.root).as_posix(),
                mask_comparison_report_filepath=self.mask_comparison.relative_to(
                    self.root
                ).as_posix(),
                output_filepath=self.segmentation_verification.relative_to(
                    self.root
                ).as_posix(),
                registration_mode="autonomous_v2",
            )
        self.assertEqual("ok", verification.status)
        self.assertEqual("pass", verification.gate)
        self._write_stage2_reports(manifest_path, manifest, threshold=threshold)
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
                    field: exact_otsu[field]
                    for field in (
                        "threshold",
                        "voxel_count",
                        "foreground_voxel_count",
                        "foreground_fraction",
                        "otsu_separability",
                        "background_mean",
                        "foreground_mean",
                        "class_mean_separation_sigma",
                        "significant_modes",
                        "histogram_sha256",
                        "overall_pass",
                    )
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
                    "specimen_id": "handoff_specimen",
                    "design_id": "handoff_design",
                    "aligned_graph_state": "derived",
                    "requested_analysis_scope": "roi_screening",
                    "overall_pass": True,
                    "local_recenter_complete": True,
                    "roi_gate_pass": True,
                    "metrology_gate_status": "not_authorized",
                    "authorized_outputs": [
                        "segmentation",
                        "registration",
                        "node_localization",
                        "coarse_region_screening",
                        "padded_roi_definition",
                    ],
                    "unauthorized_outputs": [
                        "absolute_metrology",
                        "direct_dimensional_measurement",
                    ],
                    "roi_gate_results": {
                        "image_support": True,
                        "localization_quality": True,
                        "coarse_region_support": True,
                        "padded_roi_in_bounds": True,
                    },
                    "localization_quality_counts": {
                        "primary": 2,
                        "stable_coarse": 0,
                        "fallback": 0,
                        "ambiguous": 0,
                        "rejected": 0,
                        "boundary_limited": 0,
                    },
                    "reason_codes": [
                        "ROI_GATES_PASS",
                        "METROLOGY_NOT_AUTHORIZED",
                    ],
                },
            },
        }

        return {
            "schema_version": "data-prep-result/1.2.0",
            "specimen_id": manifest["specimen_id"],
            "design_id": manifest["design_id"],
            "requested_analysis_scope": "roi_screening",
            "registration_mode": "autonomous_v2",
            "input_manifest_sha256": canonical_json_sha256(manifest),
            "input_manifest_artifact_sha256": sha256_file(manifest_path),
            "analysis_parameters_sha256": config_hash,
            "authorized_outputs": [
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
            ],
            "unauthorized_outputs": [
                "absolute_metrology",
                "direct_dimensional_measurement",
            ],
            "roi_gate_results": {
                "image_support": True,
                "localization_quality": True,
                "coarse_region_support": True,
                "padded_roi_in_bounds": True,
            },
            "metrology_gate_status": "not_authorized",
            "localization_quality_counts": {
                "primary": 2,
                "stable_coarse": 0,
                "fallback": 0,
                "ambiguous": 0,
                "rejected": 0,
                "boundary_limited": 0,
            },
            "reason_codes": [
                "ROI_GATES_PASS",
                "METROLOGY_NOT_AUTHORIZED",
            ],
            "artifact_bindings": {
                "localization_report": {
                    "path": self.localization_report.relative_to(self.root).as_posix(),
                    "sha256": sha256_file(self.localization_report),
                    "role": "localization_report",
                },
                "registration_qa": {
                    "path": self.registration_qa.relative_to(self.root).as_posix(),
                    "sha256": sha256_file(self.registration_qa),
                    "role": "registration_qa",
                },
                "segmentation_verification_mcp_response": {
                    "path": self.segmentation_verification.relative_to(
                        self.root
                    ).as_posix(),
                    "sha256": sha256_file(self.segmentation_verification),
                    "role": "segmentation_verification_mcp_response",
                },
            },
            "aligned_graph": aligned_artifact,
            "canonical_mask": {
                "path": self.mask.relative_to(self.root).as_posix(),
                "sha256": sha256_file(self.mask),
                "role": "canonical_segmentation_mask",
                "retention": "committed",
                "dtype": "uint8",
                "shape": list(manifest["inputs"]["ct_metadata"]["shape"]),
                "array_axes": ["z", "y", "x"],
            },
            "derived": derived,
            "self_verification": {
                "exact_otsu_complete": True,
                "registration_complete": True,
                "local_recenter_complete": True,
                "roi_gate_pass": True,
                "scope_bound_to_hashed_intake": True,
                "localization_quality_propagated": True,
                "defect_labels_not_accessed": True,
            },
        }

    def _rewrite_bound_stage2_reports(
        self,
        result: dict[str, object],
        *,
        mutate_localization=None,
        mutate_qa=None,
    ) -> None:
        localization = load_json(self.localization_report)
        if mutate_localization is not None:
            mutate_localization(localization)
        self.localization_report.write_text(
            json.dumps(localization, sort_keys=True), encoding="utf-8"
        )
        localization_sha256 = sha256_file(self.localization_report)

        qa = load_json(self.registration_qa)
        qa["hashes"]["localization_report_sha256"] = localization_sha256
        qa["localization_binding"]["artifact"]["sha256"] = localization_sha256
        if mutate_qa is not None:
            mutate_qa(qa)
        self.registration_qa.write_text(
            json.dumps(qa, sort_keys=True), encoding="utf-8"
        )
        result["artifact_bindings"]["localization_report"][
            "sha256"
        ] = localization_sha256
        result["artifact_bindings"]["registration_qa"]["sha256"] = sha256_file(
            self.registration_qa
        )

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

    def test_handoff_rehashes_persisted_metadata_evidence(self) -> None:
        intake = self._ingest()
        metadata_path = Path(intake["paths"]["ct_metadata_response"])
        metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")

        with self.assertRaisesRegex(
            DataPrepHandoffError, "metadata.*SHA-256|artifact bundle"
        ):
            create_data_prep_handoff(
                Path(intake["paths"]["specimen_manifest"]),
                Path(intake["paths"]["ingest_receipt"]),
                repository_root=self.root,
                schema_path=DEFAULT_SCHEMA,
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
        self.assertEqual(
            "canonical_segmentation_mask",
            finalized["inputs"]["canonical_mask"]["role"],
        )
        self.assertTrue(Path(completion["completion_receipt_path"]).is_file())

    def test_data_prep_result_replay_is_idempotent_and_divergence_is_rejected(
        self,
    ) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        result_path = manifest_path.parent / "idempotent_data_prep_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        first = apply_data_prep_result(
            manifest_path,
            result_path,
            repository_root=self.root,
        )
        second = apply_data_prep_result(
            manifest_path,
            result_path,
            repository_root=self.root,
        )
        self.assertEqual(
            first["canonical_completion_sha256"],
            second["canonical_completion_sha256"],
        )
        self.assertEqual(
            {"manifest": False, "completion_receipt": False},
            second["changed"],
        )

        np.save(
            self.mask,
            np.asarray(np.load(self.ct, allow_pickle=False) >= 123.0, dtype=np.uint8),
        )
        with self.assertRaisesRegex(DataPrepHandoffError, "Canonical mask"):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )

        divergent = json.loads(json.dumps(result))
        divergent["reason_codes"].append("DIVERGENT_REPLAY")
        divergent_path = manifest_path.parent / "divergent_data_prep_result.json"
        divergent_path.write_text(json.dumps(divergent), encoding="utf-8")
        with self.assertRaisesRegex(DataPrepHandoffError, "sealed completion"):
            apply_data_prep_result(
                manifest_path,
                divergent_path,
                repository_root=self.root,
            )

    def test_self_consistent_threshold_123_mask_cannot_finalize(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        fabricated_threshold = 123.0
        fabricated_mask = np.asarray(
            self.ct_values >= fabricated_threshold,
            dtype=np.uint8,
        )
        np.save(
            self.mask,
            fabricated_mask,
        )
        result["canonical_mask"]["sha256"] = sha256_file(self.mask)
        segmentation_values = result["derived"]["segmentation_result"]["values"]
        segmentation_values["threshold"] = fabricated_threshold
        segmentation_values["voxel_count"] = int(fabricated_mask.size)
        segmentation_values["foreground_voxel_count"] = int(
            np.count_nonzero(fabricated_mask)
        )
        segmentation_values["foreground_fraction"] = float(
            np.count_nonzero(fabricated_mask) / fabricated_mask.size
        )

        def mutate_localization(document: dict[str, object]) -> None:
            document["threshold"] = fabricated_threshold
            document["segmentation_binding"]["threshold"] = fabricated_threshold

        def mutate_qa(document: dict[str, object]) -> None:
            document["threshold"] = fabricated_threshold
            document["segmentation_binding"]["threshold"] = fabricated_threshold

        self._rewrite_bound_stage2_reports(
            result,
            mutate_localization=mutate_localization,
            mutate_qa=mutate_qa,
        )
        fabricated_otsu = load_json(self.otsu_report)
        fabricated_otsu["threshold"] = fabricated_threshold
        fabricated_otsu["threshold_histogram_bin"] = int(fabricated_threshold)
        fabricated_otsu["foreground_voxel_count"] = int(
            np.count_nonzero(fabricated_mask)
        )
        fabricated_otsu["foreground_fraction"] = float(
            np.count_nonzero(fabricated_mask) / fabricated_mask.size
        )
        self.otsu_report.write_text(
            json.dumps(fabricated_otsu, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.mask_comparison.unlink()
        fabricated_comparison = compare_segmentation_masks(
            self.ct,
            [self.mask],
            [fabricated_threshold],
            registration_mode="autonomous_v2",
            output_report_path=self.mask_comparison,
            repository_root=self.root,
        )
        self.assertEqual("pass", fabricated_comparison["gate"])
        self.segmentation_verification.unlink()
        with patch.object(mcp_common, "REPOSITORY_ROOT", self.root):
            verification = mcp_server.verify_canonical_segmentation(
                specimen_id="handoff_specimen",
                design_id="handoff_design",
                analysis_policy_artifact_filepath=manifest_path.relative_to(
                    self.root
                ).as_posix(),
                exact_otsu_report_filepath=self.otsu_report.relative_to(
                    self.root
                ).as_posix(),
                canonical_mask_filepath=self.mask.relative_to(self.root).as_posix(),
                mask_comparison_report_filepath=self.mask_comparison.relative_to(
                    self.root
                ).as_posix(),
                output_filepath=self.segmentation_verification.relative_to(
                    self.root
                ).as_posix(),
                registration_mode="autonomous_v2",
            )
        self.assertEqual("error", verification.status)
        self.assertEqual("halt", verification.gate)
        self.assertFalse(self.segmentation_verification.exists())
        result_path = manifest_path.parent / "threshold-123-mask-result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(
            DataPrepHandoffError,
            "Segmentation verification evidence is missing",
        ):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )
        self.assertEqual(
            "ready_for_data_prep", load_json(manifest_path)["lifecycle_state"]
        )

    def test_segmentation_verification_evidence_is_closed_scoped_and_hash_bound(
        self,
    ) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        original_evidence = load_json(self.segmentation_verification)
        original_hash = result["artifact_bindings"][
            "segmentation_verification_mcp_response"
        ]["sha256"]

        def attempt(name: str, evidence: dict[str, object], *, rebind: bool) -> str:
            self.segmentation_verification.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            candidate = json.loads(json.dumps(result))
            candidate_binding = candidate["artifact_bindings"][
                "segmentation_verification_mcp_response"
            ]
            candidate_binding["sha256"] = (
                sha256_file(self.segmentation_verification) if rebind else original_hash
            )
            candidate_path = manifest_path.parent / f"{name}.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.assertRaises(DataPrepHandoffError) as raised:
                apply_data_prep_result(
                    manifest_path,
                    candidate_path,
                    repository_root=self.root,
                )
            return str(raised.exception)

        open_evidence = json.loads(json.dumps(original_evidence))
        open_evidence["unexpected"] = True
        self.assertIn("schema is open", attempt("open-evidence", open_evidence, rebind=True))

        cross_specimen = json.loads(json.dumps(original_evidence))
        cross_specimen["specimen_id"] = "other_specimen"
        self.assertIn(
            "identity or terminal gate",
            attempt("cross-specimen-evidence", cross_specimen, rebind=True),
        )

        stale_manifest_binding = json.loads(json.dumps(original_evidence))
        stale_manifest_binding["bindings"]["analysis_policy_artifact"][
            "sha256"
        ] = "0" * 64
        stale_manifest_binding["hashes"][
            "analysis_policy_artifact_sha256"
        ] = "0" * 64
        self.assertIn(
            "artifact bindings are stale",
            attempt(
                "stale-manifest-binding",
                stale_manifest_binding,
                rebind=True,
            ),
        )

        stale = json.loads(json.dumps(original_evidence))
        stale["summary"] = "tampered"
        self.assertIn("SHA-256 mismatch", attempt("stale-evidence", stale, rebind=False))

        self.segmentation_verification.unlink()
        missing_path = manifest_path.parent / "missing-evidence.json"
        missing_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(
            DataPrepHandoffError,
            "Segmentation verification evidence is missing",
        ):
            apply_data_prep_result(
                manifest_path,
                missing_path,
                repository_root=self.root,
            )

    def test_data_prep_result_control_objects_are_closed(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        mutations = (
            ("root", lambda value: value.__setitem__("unexpected_probe", True)),
            (
                "artifact-bindings",
                lambda value: value["artifact_bindings"].__setitem__(
                    "unexpected_probe",
                    {
                        "path": "analysis/unexpected.json",
                        "sha256": "0" * 64,
                        "role": "unexpected_probe",
                    },
                ),
            ),
            (
                "artifact-binding",
                lambda value: value["artifact_bindings"][
                    "localization_report"
                ].__setitem__("unexpected_probe", True),
            ),
            (
                "self-verification",
                lambda value: value["self_verification"].__setitem__(
                    "unexpected_probe", True
                ),
            ),
            (
                "aligned-graph",
                lambda value: value["aligned_graph"].__setitem__(
                    "unexpected_probe", True
                ),
            ),
            (
                "canonical-mask",
                lambda value: value["canonical_mask"].__setitem__(
                    "unexpected_probe", True
                ),
            ),
            (
                "roi-gates",
                lambda value: value["roi_gate_results"].__setitem__(
                    "unexpected_probe", True
                ),
            ),
            (
                "localization-counts",
                lambda value: value["localization_quality_counts"].__setitem__(
                    "unexpected_probe", 0
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(layer=name):
                result = self._data_prep_result(manifest_path)
                mutate(result)
                candidate = manifest_path.parent / f"open-{name}.json"
                candidate.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(DataPrepHandoffError) as raised:
                    apply_data_prep_result(
                        manifest_path,
                        candidate,
                        repository_root=self.root,
                    )
                self.assertRegex(
                    str(raised.exception),
                    r"(?:schema is open|schema is open or incompatible)",
                )

    def test_integer_otsu_threshold_verifies_through_real_mcp_client(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        threshold = result["derived"]["segmentation_result"]["values"]["threshold"]
        self.assertIs(type(threshold), int)
        self.segmentation_verification.unlink()

        async def invoke() -> dict[str, object]:
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool(
                    "verify_canonical_segmentation",
                    {
                        "specimen_id": "handoff_specimen",
                        "design_id": "handoff_design",
                        "analysis_policy_artifact_filepath": manifest_path.relative_to(
                            self.root
                        ).as_posix(),
                        "exact_otsu_report_filepath": self.otsu_report.relative_to(
                            self.root
                        ).as_posix(),
                        "canonical_mask_filepath": self.mask.relative_to(
                            self.root
                        ).as_posix(),
                        "mask_comparison_report_filepath": self.mask_comparison.relative_to(
                            self.root
                        ).as_posix(),
                        "output_filepath": self.segmentation_verification.relative_to(
                            self.root
                        ).as_posix(),
                        "registration_mode": "autonomous_v2",
                    },
                )
            self.assertFalse(call.is_error)
            return call.structured_content

        with patch.object(mcp_common, "REPOSITORY_ROOT", self.root):
            response = asyncio.run(invoke())
            replay = asyncio.run(invoke())
        self.assertEqual("ok", response["status"])
        self.assertEqual("pass", response["gate"])
        self.assertEqual(threshold, response["result"]["threshold"])
        artifact = response["artifacts"]["segmentation_verification"]
        self.assertEqual(
            "segmentation_verification_mcp_response", artifact["role"]
        )
        self.assertEqual(
            self.segmentation_verification.relative_to(self.root).as_posix(),
            artifact["path"],
        )
        self.assertTrue(artifact["changed"])
        self.assertEqual(sha256_file(self.segmentation_verification), artifact["sha256"])
        replay_artifact = replay["artifacts"]["segmentation_verification"]
        self.assertFalse(replay_artifact["changed"])
        self.assertEqual(artifact["sha256"], replay_artifact["sha256"])

    def test_exact_otsu_policy_binding_is_required_by_verifier(
        self,
    ) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        self._data_prep_result(manifest_path)
        self.segmentation_verification.unlink()
        accepted_report = load_json(self.otsu_report)

        def verify() -> object:
            with patch.object(mcp_common, "REPOSITORY_ROOT", self.root):
                return mcp_server.verify_canonical_segmentation(
                    specimen_id="handoff_specimen",
                    design_id="handoff_design",
                    analysis_policy_artifact_filepath=manifest_path.relative_to(
                        self.root
                    ).as_posix(),
                    exact_otsu_report_filepath=self.otsu_report.relative_to(
                        self.root
                    ).as_posix(),
                    canonical_mask_filepath=self.mask.relative_to(self.root).as_posix(),
                    mask_comparison_report_filepath=self.mask_comparison.relative_to(
                        self.root
                    ).as_posix(),
                    output_filepath=self.segmentation_verification.relative_to(
                        self.root
                    ).as_posix(),
                    registration_mode="autonomous_v2",
                )

        accepted = verify()
        self.assertEqual("ok", accepted.status)
        self.assertEqual("pass", accepted.gate)

        def missing_policy_artifact(value: dict[str, object]) -> None:
            value.pop("analysis_policy_artifact")

        def stale_policy_hash(value: dict[str, object]) -> None:
            value["hashes"]["analysis_parameters_sha256"] = "0" * 64

        def missing_policy_binding(value: dict[str, object]) -> None:
            value["provenance"].pop("policy_binding")

        def legacy_reference_replay(value: dict[str, object]) -> None:
            value["reference_replay"] = {
                "enforced": True,
                "expected_threshold": value["threshold"],
                "expected_foreground_voxels": value["foreground_voxel_count"],
                "gates": {
                    "reference_threshold_matches": True,
                    "reference_foreground_count_matches": True,
                },
            }

        for name, mutate in (
            ("missing-policy-artifact", missing_policy_artifact),
            ("stale-policy-hash", stale_policy_hash),
            ("missing-policy-binding", missing_policy_binding),
            ("legacy-reference-replay", legacy_reference_replay),
        ):
            with self.subTest(case=name):
                self.segmentation_verification.unlink(missing_ok=True)
                mutated = json.loads(json.dumps(accepted_report))
                mutate(mutated)
                self.otsu_report.write_text(
                    json.dumps(mutated, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                rejected = verify()
                self.assertEqual("error", rejected.status)
                self.assertEqual("halt", rejected.gate)
                self.assertFalse(self.segmentation_verification.exists())

    def test_segmentation_verifier_rejects_json_scalar_type_coercions(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        self._data_prep_result(manifest_path)
        self.segmentation_verification.unlink()
        otsu = load_json(self.otsu_report)
        comparison = load_json(self.mask_comparison)

        def verify() -> object:
            with patch.object(mcp_common, "REPOSITORY_ROOT", self.root):
                return mcp_server.verify_canonical_segmentation(
                    specimen_id="handoff_specimen",
                    design_id="handoff_design",
                    analysis_policy_artifact_filepath=manifest_path.relative_to(
                        self.root
                    ).as_posix(),
                    exact_otsu_report_filepath=self.otsu_report.relative_to(
                        self.root
                    ).as_posix(),
                    canonical_mask_filepath=self.mask.relative_to(self.root).as_posix(),
                    mask_comparison_report_filepath=self.mask_comparison.relative_to(
                        self.root
                    ).as_posix(),
                    output_filepath=self.segmentation_verification.relative_to(
                        self.root
                    ).as_posix(),
                    registration_mode="autonomous_v2",
                )

        def float_otsu_count(
            report: dict[str, object], _comparison: dict[str, object]
        ) -> None:
            report["foreground_voxel_count"] = float(
                report["foreground_voxel_count"]
            )

        def integer_otsu_gate(
            report: dict[str, object], _comparison: dict[str, object]
        ) -> None:
            report["gates"]["foreground_fraction_plausible"] = 1

        def float_comparison_count(
            _report: dict[str, object], candidate_report: dict[str, object]
        ) -> None:
            candidate_report["candidates"][0]["mismatched_voxels"] = 0.0

        def integer_comparison_gate(
            _report: dict[str, object], candidate_report: dict[str, object]
        ) -> None:
            candidate_report["candidates"][0]["exact_threshold_match"] = 1

        for name, mutate in (
            ("otsu-count-as-float", float_otsu_count),
            ("otsu-gate-as-integer", integer_otsu_gate),
            ("comparison-count-as-float", float_comparison_count),
            ("comparison-gate-as-integer", integer_comparison_gate),
        ):
            with self.subTest(case=name):
                mutated_otsu = json.loads(json.dumps(otsu))
                mutated_comparison = json.loads(json.dumps(comparison))
                mutate(mutated_otsu, mutated_comparison)
                self.otsu_report.write_text(
                    json.dumps(mutated_otsu, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.mask_comparison.write_text(
                    json.dumps(mutated_comparison, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                self.segmentation_verification.unlink(missing_ok=True)
                rejected = verify()
                self.assertEqual("error", rejected.status)
                self.assertEqual("halt", rejected.gate)
                self.assertFalse(self.segmentation_verification.exists())

    def test_stale_looser_and_cross_input_stage2_reports_are_rejected(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        mutations = (
            (
                "stale-analysis-parameters",
                lambda value: value["hashes"].__setitem__(
                    "analysis_parameters_sha256", "0" * 64
                ),
            ),
            (
                "looser-localization-policy",
                lambda value: value["quantitative_policy"].__setitem__(
                    "maximum_fallback_fraction", 1.0
                ),
            ),
            (
                "wrong-ct",
                lambda value: value["hashes"].__setitem__(
                    "ct_sha256", "0" * 64
                ),
            ),
            (
                "wrong-localized-graph",
                lambda value: value["hashes"].__setitem__(
                    "localized_graph_sha256", "0" * 64
                ),
            ),
            (
                "wrong-registration-mode",
                lambda value: value.__setitem__(
                    "registration_mode", "challenge_aligned_json"
                ),
            ),
            (
                "wrong-scope",
                lambda value: value.__setitem__(
                    "requested_analysis_scope", "direct_metrology"
                ),
            ),
            (
                "wrong-otsu-threshold",
                lambda value: value.__setitem__("threshold", 12.0),
            ),
            (
                "open-segmentation-binding",
                lambda value: value["segmentation_binding"].__setitem__(
                    "unrecognized_field", True
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(case=name):
                result = self._data_prep_result(manifest_path)
                self._rewrite_bound_stage2_reports(
                    result,
                    mutate_localization=mutate,
                )
                result_path = manifest_path.parent / f"{name}-result.json"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaises(DataPrepHandoffError):
                    apply_data_prep_result(
                        manifest_path,
                        result_path,
                        repository_root=self.root,
                    )

        result = self._data_prep_result(manifest_path)
        self._rewrite_bound_stage2_reports(
            result,
            mutate_qa=lambda value: value["hashes"].__setitem__(
                "localization_report_sha256", "0" * 64
            ),
        )
        qa_mismatch_path = manifest_path.parent / "qa-localization-sha-mismatch.json"
        qa_mismatch_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(
            DataPrepHandoffError, "localization_report_sha256"
        ):
            apply_data_prep_result(
                manifest_path,
                qa_mismatch_path,
                repository_root=self.root,
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.root,
            prefix="escaped-registration-",
            suffix=".json",
            delete=False,
        ) as stream:
            escaped_registration = Path(stream.name)
            stream.write(self.registration_report.read_bytes())
        self.addCleanup(escaped_registration.unlink, missing_ok=True)
        result = self._data_prep_result(manifest_path)
        self._rewrite_bound_stage2_reports(
            result,
            mutate_localization=lambda value: value["artifacts"][
                "registration_report"
            ].__setitem__("path", str(escaped_registration)),
        )
        escaped_path = manifest_path.parent / "escaped-registration-result.json"
        escaped_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(DataPrepHandoffError, "outside the run"):
            apply_data_prep_result(
                manifest_path,
                escaped_path,
                repository_root=self.root,
            )

    def test_data_prep_result_without_canonical_mask_cannot_advance(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        del result["canonical_mask"]
        result_path = manifest_path.parent / "missing_mask_data_prep_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "canonical_mask"):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )

    def test_failed_data_prep_self_verification_cannot_advance(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        result["self_verification"]["scope_bound_to_hashed_intake"] = False
        result_path = manifest_path.parent / "failed_data_prep_result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "scope_bound_to_hashed_intake"):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )
        self.assertEqual(
            "ready_for_data_prep", load_json(manifest_path)["lifecycle_state"]
        )

    def test_scope_tampering_and_conflicting_authorizations_are_rejected(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        for name, mutate, pattern in (
            (
                "scope-tampered",
                lambda value: value.__setitem__(
                    "requested_analysis_scope", "direct_metrology"
                ),
                "requested_analysis_scope mismatch",
            ),
            (
                "duplicate-authorization",
                lambda value: value["authorized_outputs"].append("segmentation"),
                "authorization lists",
            ),
        ):
            with self.subTest(case=name):
                result = self._data_prep_result(manifest_path)
                mutate(result)
                result_path = manifest_path.parent / f"{name}.json"
                result_path.write_text(json.dumps(result), encoding="utf-8")
                with self.assertRaisesRegex(DataPrepHandoffError, pattern):
                    apply_data_prep_result(
                        manifest_path,
                        result_path,
                        repository_root=self.root,
                    )

    def test_cross_specimen_stage_2_report_binding_is_rejected(self) -> None:
        intake = self._ingest()
        manifest_path = Path(intake["paths"]["specimen_manifest"])
        result = self._data_prep_result(manifest_path)
        result["artifact_bindings"]["registration_qa"]["path"] = (
            "analysis/other_specimen/qa/registration_qa.json"
        )
        result_path = manifest_path.parent / "cross-specimen-result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")

        with self.assertRaisesRegex(DataPrepHandoffError, "specimen-scoped"):
            apply_data_prep_result(
                manifest_path,
                result_path,
                repository_root=self.root,
            )

    def test_handoff_consumes_evidence_without_scientific_recomputation(self) -> None:
        tree = ast.parse(
            (
                REPOSITORY_ROOT
                / "src/llnl_nde/orchestration/receipts.py"
            ).read_text(
                encoding="utf-8"
            )
        )
        imported_modules: set[str] = set()
        referenced_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
            elif isinstance(node, ast.Name):
                referenced_names.add(node.id)
        forbidden_imports = {
            "numpy",
            "scipy",
            "skimage",
            "llnl_nde.core",
            "llnl_nde.server",
        }
        self.assertFalse(
            {
                imported
                for imported in imported_modules
                if any(
                    imported == forbidden or imported.startswith(f"{forbidden}.")
                    for forbidden in forbidden_imports
                )
            }
        )
        self.assertTrue(
            referenced_names.isdisjoint({"load_volume", "replay_exact_otsu"})
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
