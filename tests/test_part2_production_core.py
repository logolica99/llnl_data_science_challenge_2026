"""Production Part 2 core tests: synthetic, gates, axes, sizes, determinism."""

from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from part2_core.artifacts import sha256_file, sha256_json  # noqa: E402
from part2_core.evidence import render_strut_evidence  # noqa: E402
from part2_core.lattice import load_lattice_json  # noqa: E402
from part2_core.localization import localize_lattice_nodes  # noqa: E402
from part2_core.otsu import replay_exact_otsu  # noqa: E402
from part2_core.qa import compute_registration_qa  # noqa: E402
from part2_core.registration import (  # noqa: E402
    SimilarityTransform,
    register_lattice_to_ct,
    run_synthetic_suite,
    solve_similarity,
)
from part2_core.reports import get_strut_report  # noqa: E402
from part2_core.sampling import sample_corridor  # noqa: E402
from part2_core.struts import classify_struts, compute_strut_metrics  # noqa: E402


class SyntheticFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.nominal = self.root / "nominal.json"
        self.aligned = self.root / "aligned.json"
        topology = {
            "struts": [
                {"id": 101, "junction0": 10, "junction1": 20},
                {"id": 303, "junction0": 20, "junction1": 30},
                {"id": 707, "junction0": 30, "junction1": 40},
            ],
            "unit_cells": [{"id": 5, "indices": [0, 0, 0], "struts": [101, 303, 707]}],
        }
        self.nominal.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": 10, "position": [0, 0, 0]},
                        {"id": 20, "position": [1, 0, 0]},
                        {"id": 30, "position": [1, 1, 0]},
                        {"id": 40, "position": [1, 1, 1]},
                    ],
                    **topology,
                }
            ),
            encoding="utf-8",
        )
        positions = np.asarray(
            [[8, 8, 8], [28, 8, 8], [28, 28, 8], [28, 28, 28]],
            dtype=float,
        )
        self.aligned.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": identifier, "position": position.tolist()}
                        for identifier, position in zip([10, 20, 30, 40], positions)
                    ],
                    **topology,
                }
            ),
            encoding="utf-8",
        )
        zz, yy, xx = np.indices((40, 40, 40))
        points = np.stack((xx, yy, zz), axis=-1).astype(float)
        foreground = np.zeros((40, 40, 40), dtype=bool)
        for position in positions:
            foreground |= np.sum((points - position) ** 2, axis=-1) <= 3.5**2
        endpoints = [
            (positions[0], positions[1]),
            (positions[1], positions[2]),
            (positions[2], positions[3]),
        ]
        for edge_index, (start, end) in enumerate(endpoints):
            direction = end - start
            length_squared = float(np.dot(direction, direction))
            t = np.clip(
                np.sum((points - start) * direction, axis=-1) / length_squared,
                0.0,
                1.0,
            )
            closest = start + t[..., None] * direction
            tube = np.sum((points - closest) ** 2, axis=-1) <= 2.25**2
            if edge_index == 1:
                tube &= ~((t > 0.40) & (t < 0.65))
            foreground |= tube
        self.volume = self.root / "ct.npy"
        background = (
            100 + ((17 * zz + 11 * yy + 5 * xx) % 201)
        ).astype(np.uint16)
        np.save(
            self.volume,
            np.where(foreground, 1_000, background).astype(np.uint16),
        )
        self.otsu_result, _ = replay_exact_otsu(
            self.volume,
            recipe={"histogram_encoding": "native_uint16"},
        )
        self.threshold = float(self.otsu_result["threshold"])
        self.roi_scope = self._write_scope_artifact(
            "roi_screening", "challenge_aligned_json"
        )
        self.direct_scope = self._write_scope_artifact(
            "direct_metrology", "challenge_aligned_json"
        )
        self.direct_autonomous_scope = self._write_scope_artifact(
            "direct_metrology", "autonomous_v2"
        )

    def _write_scope_artifact(self, scope: str, registration_mode: str) -> Path:
        parameters = {
            "requested_analysis_scope": scope,
            "registration": {
                "mode": registration_mode,
                "local_recenter_required": True,
            },
            "segmentation": {
                "method": "exact_histogram_otsu",
                "method_version": "2.0.0",
                "comparison": "value >= threshold",
                "histogram_bins": 65536,
                "histogram_encoding": "native_uint16",
                "edge_slices_excluded": 0,
                "chunk_depth": 8,
                "coarse_bins": 1024,
                "peak_smoothing_sigma_bins": 2.0,
                "peak_prominence_fraction": 0.003,
                "minimum_significant_peaks": 2,
                "minimum_foreground_fraction": 0.01,
                "maximum_foreground_fraction": 0.35,
                "minimum_otsu_separability": 0.45,
                "minimum_class_mean_separation_sigma": 0.75,
            },
            "localization_policy": {
                "schema_version": "stage2-localization-policy/1.1.0",
                "patch_radius_voxels": 10,
                "search_radius_voxels": 8.0,
                "maximum_shift_voxels": 8.0,
                "smoothing_sigma_voxels": 1.25,
                "mean_shift_bandwidth_voxels": 4.0,
                "mean_shift_max_iterations": 12,
                "mean_shift_tolerance_voxels": 0.05,
                "seed_perturbation_voxels": 2.0,
                "seed_cluster_radius_voxels": 2.0,
                "minimum_seed_consensus_fraction": 0.7,
                "minimum_candidate_support": 0.05,
                "minimum_relative_support_improvement": 0.01,
                "incident_sample_distances_voxels": [3.0, 5.0, 7.0],
                "core_support_weight": 0.7,
                "incident_support_weight": 0.3,
                "minimum_primary_or_stable_coarse_fraction": 0.0,
                "maximum_fallback_fraction": 1.0,
                "maximum_ambiguous_fraction": 1.0,
                "maximum_rejected_fraction": 1.0,
                "maximum_boundary_limited_fraction": 1.0,
            },
            "qa_policy": {
                "schema_version": "stage2-qa-policy/1.1.0",
                "junction_patch_radius_voxels": 2,
                "corridor_axial_samples": 9,
                "corridor_radius_voxels": 6.0,
                "corridor_angular_samples": 8,
                "roi_padding_fraction": 0.2,
                "spatial_bins_per_axis": 2,
                "radial_foreground_probability": 0.5,
                "minimum_mean_junction_foreground_fraction": 0.1,
                "minimum_median_corridor_foreground_fraction": 0.01,
                "maximum_spatial_bin_median_range": 1.0,
                "minimum_roi_in_bounds_fraction": 0.5,
                "maximum_uncertainty_to_radius_ratio": 1.0,
            },
            "artifact_schema_versions": {
                "specimen_manifest": "2.1.0",
                "node_localization": "1.2.0",
                "registration_qa": "1.2.0",
                "per_strut_metrics": "1.0.0",
                "classified_struts": "1.0.0",
                "nde_report": "1.0.0",
            },
        }
        path = self.root / f"{scope}-{registration_mode}-specimen-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "2.1.0",
                    "specimen_id": "synthetic_fixture",
                    "design_id": "synthetic_design",
                    "requested_analysis_scope": scope,
                    "analysis_parameters": parameters,
                    "analysis_parameters_sha256": sha256_json(parameters),
                    "inputs": {
                        "ct": {
                            "path": str(self.volume),
                            "sha256": sha256_file(self.volume),
                            "role": "ct_volume",
                            "retention": "external",
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def _register_and_localize(
        self, scope: str = "roi_screening"
    ) -> tuple[Path, Path]:
        scope_artifact = (
            self.roi_scope if scope == "roi_screening" else self.direct_scope
        )
        registered = self.root / "registered.json"
        registration_report = self.root / "registration.json"
        registration = register_lattice_to_ct(
            self.nominal,
            registered,
            registration_report,
            mode="challenge_aligned_json",
            ct_path=self.volume,
            aligned_graph_path=self.aligned,
        )
        self.assertEqual("pass", registration["gate"])
        localized = self.root / "localized.json"
        localization_report = self.root / "localization.json"
        localization = localize_lattice_nodes(
            self.volume,
            registered,
            localized,
            localization_report,
            threshold=self.threshold,
            registration_mode="challenge_aligned_json",
            analysis_policy_artifact_path=scope_artifact,
            registration_report_path=registration_report,
            config={
                "minimum_primary_or_stable_coarse_fraction": 0.0,
                "maximum_fallback_fraction": 1.0,
                "maximum_ambiguous_fraction": 1.0,
                "maximum_rejected_fraction": 1.0,
                "maximum_boundary_limited_fraction": 1.0,
            },
        )
        self.assertNotEqual("halt", localization["gate"])
        self.assertTrue(localization["localization"]["independent_positions_retained"])
        self.assertFalse(localization["localization"]["global_refit_performed"])
        return localized, localization_report


class ProductionPipelineTests(SyntheticFixture):
    def test_localization_quantitative_fallback_gates_and_quality_propagation(
        self,
    ) -> None:
        node_ids = [1000 + index for index in range(21)]
        edge_ids = [2000 + index for index in range(20)]
        graph_path = self.root / "quality-registered.json"
        graph_path.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": node_id, "position": [10 + index, 20, 20]}
                        for index, node_id in enumerate(node_ids)
                    ],
                    "struts": [
                        {
                            "id": edge_id,
                            "junction0": node_ids[index],
                            "junction1": node_ids[index + 1],
                        }
                        for index, edge_id in enumerate(edge_ids)
                    ],
                    "unit_cells": [
                        {"id": 3000, "indices": [0, 0, 0], "struts": edge_ids}
                    ],
                }
            ),
            encoding="utf-8",
        )

        def localization_result(
            prediction: np.ndarray,
            match_class: str,
            *,
            boundary_limited: bool = False,
        ) -> tuple[np.ndarray, dict[str, object]]:
            accepted = match_class in {"localized", "stable_coarse"}
            location = (
                prediction + np.asarray([0.1, 0.0, 0.0])
                if match_class == "localized"
                else prediction.copy()
            )
            reason = {
                "localized": "accepted",
                "stable_coarse": "accepted",
                "fallback": "insufficient_ct_support",
                "ambiguous": "unstable_multistart",
            }[match_class]
            return location, {
                "accepted": accepted,
                "reason": reason,
                "localization_status": match_class,
                "seed_consensus_fraction": 1.0,
                "stability_uncertainty_voxels": 0.01,
                "coarse_support": 0.5,
                "candidate_support": 0.6,
                "selected_support": 0.6,
                "relative_support_improvement": 0.2,
                "shift_voxels": 0.1 if match_class == "localized" else 0.0,
                "boundary_truncated": boundary_limited,
            }

        def bounded_fallback(
            _volume: np.ndarray,
            prediction: np.ndarray,
            _directions: np.ndarray,
            _threshold: float,
            _config: dict[str, object],
        ) -> tuple[np.ndarray, dict[str, object]]:
            if prediction[0] == 30:
                return localization_result(prediction, "fallback")
            if prediction[0] == 29:
                return localization_result(prediction, "stable_coarse")
            return localization_result(prediction, "localized")

        with mock.patch(
            "part2_core.localization._localize_one", side_effect=bounded_fallback
        ):
            first = localize_lattice_nodes(
                self.volume,
                graph_path,
                self.root / "quality-localized.json",
                self.root / "quality-localization.json",
                threshold=500,
                registration_mode="challenge_aligned_json",
            )
            replay = localize_lattice_nodes(
                self.volume,
                graph_path,
                self.root / "quality-localized-replay.json",
                self.root / "quality-localization-replay.json",
                threshold=500,
                registration_mode="challenge_aligned_json",
            )

        self.assertEqual("pass", first["gate"], first)
        self.assertEqual(19, first["counts"]["primary_nodes"])
        self.assertEqual(1, first["counts"]["stable_coarse_nodes"])
        self.assertEqual(1, first["counts"]["fallback_nodes"])
        self.assertEqual(0, first["counts"]["ambiguous_nodes"])
        self.assertTrue(first["gates"]["fallback_fraction_within_limit"])
        fallback = next(
            record for record in first["records"] if record["match_class"] == "fallback"
        )
        self.assertFalse(fallback["primary_match"])
        self.assertTrue(fallback["fallback_provenance"]["used"])
        self.assertEqual(
            "registered_coarse",
            fallback["fallback_provenance"]["coordinate_source"],
        )
        self.assertTrue(first["edge_quality_records"])
        localized_document = json.loads(
            (self.root / "quality-localized.json").read_text(encoding="utf-8")
        )
        self.assertIn(
            "part2_localization_quality", localized_document["junctions"][-1]
        )
        self.assertIn("part2_localization_quality", localized_document["struts"][-1])
        self.assertEqual(
            first["hashes"]["localized_graph_sha256"],
            replay["hashes"]["localized_graph_sha256"],
        )

        def excessive_fallback(
            _volume: np.ndarray,
            prediction: np.ndarray,
            _directions: np.ndarray,
            _threshold: float,
            _config: dict[str, object],
        ) -> tuple[np.ndarray, dict[str, object]]:
            return localization_result(
                prediction,
                "fallback" if prediction[0] >= 29 else "localized",
            )

        with mock.patch(
            "part2_core.localization._localize_one", side_effect=excessive_fallback
        ):
            excessive = localize_lattice_nodes(
                self.volume,
                graph_path,
                self.root / "excessive-fallback-graph.json",
                self.root / "excessive-fallback-report.json",
                threshold=500,
                registration_mode="challenge_aligned_json",
            )
        self.assertEqual("manual_review", excessive["gate"])
        self.assertFalse(excessive["gates"]["fallback_fraction_within_limit"])
        self.assertTrue(excessive["gates"]["rejected_fraction_within_limit"])

        def boundary_limited(
            _volume: np.ndarray,
            prediction: np.ndarray,
            _directions: np.ndarray,
            _threshold: float,
            _config: dict[str, object],
        ) -> tuple[np.ndarray, dict[str, object]]:
            return localization_result(
                prediction,
                "localized",
                boundary_limited=prediction[0] == 30,
            )

        with mock.patch(
            "part2_core.localization._localize_one", side_effect=boundary_limited
        ):
            boundary = localize_lattice_nodes(
                self.volume,
                graph_path,
                self.root / "boundary-graph.json",
                self.root / "boundary-report.json",
                threshold=500,
                registration_mode="challenge_aligned_json",
            )
        self.assertEqual("manual_review", boundary["gate"])
        self.assertFalse(
            boundary["gates"]["boundary_limited_fraction_within_limit"]
        )
        boundary_record = next(
            record for record in boundary["records"] if record["boundary_limited"]
        )
        self.assertFalse(boundary_record["primary_match"])
        self.assertEqual("fallback", boundary_record["match_class"])

        def excessive_ambiguity(
            _volume: np.ndarray,
            prediction: np.ndarray,
            _directions: np.ndarray,
            _threshold: float,
            _config: dict[str, object],
        ) -> tuple[np.ndarray, dict[str, object]]:
            return localization_result(
                prediction,
                "ambiguous" if prediction[0] >= 29 else "localized",
            )

        with mock.patch(
            "part2_core.localization._localize_one", side_effect=excessive_ambiguity
        ):
            ambiguous = localize_lattice_nodes(
                self.volume,
                graph_path,
                self.root / "ambiguous-graph.json",
                self.root / "ambiguous-report.json",
                threshold=500,
                registration_mode="challenge_aligned_json",
            )
        self.assertEqual("manual_review", ambiguous["gate"])
        self.assertEqual(2, ambiguous["counts"]["ambiguous_nodes"])
        self.assertFalse(ambiguous["gates"]["ambiguity_fraction_within_limit"])

    def test_multibranch_junction_uses_stable_ct_center_not_competing_edt_peaks(
        self,
    ) -> None:
        true_center = np.asarray([24.0, 24.0, 24.0])
        offsets = 12.0 * np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        true_positions = true_center + offsets
        coarse_positions = true_positions + np.asarray([1.5, -1.0, 1.0])
        node_ids = [100 + index for index in range(len(true_positions))]
        star_graph = self.root / "star-registered.json"
        star_graph.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": identifier, "position": position.tolist()}
                        for identifier, position in zip(node_ids, coarse_positions)
                    ],
                    "struts": [
                        {"id": 200 + index, "junction0": node_ids[0], "junction1": node_ids[index]}
                        for index in range(1, len(node_ids))
                    ],
                    "unit_cells": [
                        {
                            "id": 300,
                            "indices": [0, 0, 0],
                            "struts": [200 + index for index in range(1, len(node_ids))],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        zz, yy, xx = np.indices((49, 49, 49))
        points = np.stack((xx, yy, zz), axis=-1).astype(float)
        foreground = np.sum((points - true_center) ** 2, axis=-1) <= 4.0**2
        for endpoint in true_positions[1:]:
            direction = endpoint - true_center
            length_squared = float(np.dot(direction, direction))
            t = np.clip(
                np.sum((points - true_center) * direction, axis=-1)
                / length_squared,
                0.0,
                1.0,
            )
            closest = true_center + t[..., None] * direction
            foreground |= np.sum((points - closest) ** 2, axis=-1) <= 2.5**2
        star_ct = self.root / "star-ct.npy"
        np.save(star_ct, np.where(foreground, 1000, 0).astype(np.uint16))

        report = localize_lattice_nodes(
            star_ct,
            star_graph,
            self.root / "star-localized.json",
            self.root / "star-localization.json",
            threshold=self.threshold,
            registration_mode="challenge_aligned_json",
            config={
                "minimum_primary_or_stable_coarse_fraction": 0.5,
                "maximum_ambiguous_fraction": 0.5,
            },
        )
        center_record = next(
            record for record in report["records"] if record["node_id"] == node_ids[0]
        )
        self.assertTrue(center_record["accepted"], center_record)
        self.assertNotIn("unstable_multistart", center_record["reason"])
        self.assertGreaterEqual(center_record["seed_consensus_fraction"], 0.7)
        self.assertLess(
            np.linalg.norm(
                np.asarray(center_record["localized_xyz"]) - true_center
            ),
            np.linalg.norm(coarse_positions[0] - true_center),
        )
        self.assertFalse(
            report["localization"]["stability_uncertainty_voxels"][
                "absolute_registration_accuracy_claimed"
            ]
        )

    def test_challenge_metrics_classification_evidence_lookup_and_scoring(self) -> None:
        localized, localization_report = self._register_and_localize(
            "direct_metrology"
        )
        qa_path = self.root / "qa.json"
        qa = compute_registration_qa(
            self.volume,
            localized,
            qa_path,
            threshold=self.threshold,
            registration_mode="challenge_aligned_json",
            analysis_scope_artifact_path=self.direct_scope,
            localization_report_path=localization_report,
            config={
                "minimum_mean_junction_foreground_fraction": 0.1,
                "minimum_median_corridor_foreground_fraction": 0.01,
                "maximum_spatial_bin_median_range": 1.0,
                "minimum_roi_in_bounds_fraction": 0.5,
                "spatial_bins_per_axis": 2,
            },
        )
        self.assertEqual("manual_review", qa["gate"])
        self.assertEqual("direct_metrology", qa["requested_analysis_scope"])
        self.assertTrue(qa["coarse_capture"]["overall_pass"])
        self.assertFalse(qa["metrology"]["overall_pass"])
        self.assertEqual("fail", qa["metrology"]["status"])
        self.assertIn("METROLOGY_EVIDENCE_MISSING", qa["reason_codes"])
        self.assertIsNone(
            qa["metrology"]["absolute_registration_uncertainty_voxels"]
        )
        self.assertFalse(
            qa["metrology"]["gates"][
                "absolute_registration_uncertainty_available"
            ]
        )

        metrics_path = self.root / "metrics.csv"
        profiles_path = self.root / "profiles.json"
        metrics_report = self.root / "metrics-report.json"
        metrics = compute_strut_metrics(
            self.volume,
            localized,
            metrics_path,
            profiles_path,
            metrics_report,
            threshold=500,
            registration_mode="challenge_aligned_json",
            registration_qa_path=qa_path,
            config={
                "corridor_radius_voxels": 4.0,
                "axial_samples": 31,
                "minimum_valid_roi_fraction": 1.0,
            },
        )
        self.assertEqual(3, metrics["counts"]["metric_rows"])
        self.assertEqual("pass", metrics["gate"])

        classifications_path = self.root / "classifications.json"
        thresholds_path = self.root / "thresholds.json"
        classified = classify_struts(
            metrics_path,
            {
                "missing_occupancy_max": 0.8,
                "missing_gap_fraction_min": 0.1,
                "broken_gap_fraction_min": 0.9,
                "broken_largest_component_fraction_max": 0.05,
                "thin_radius_max": 0.1,
                "bent_curvature_min": 10.0,
            },
            classifications_path,
            thresholds_path,
        )
        labels = {
            row["strut_id"]: row["class"] for row in classified["classifications"]
        }
        self.assertEqual("missing", labels[303])
        self.assertEqual(3, classified["counts"]["total"])

        report = get_strut_report(
            303,
            metrics_path,
            classifications_path,
            thresholds_path,
            evidence_manifest_path=render_strut_evidence(
                self.volume,
                localized,
                profiles_path,
                self.root / "evidence",
                strut_id=303,
                threshold=self.threshold,
            )["artifacts"]["manifest"]["path"],
        )
        self.assertEqual("missing", report["class"])
        self.assertFalse(report["provenance"]["metrics_recomputed"])
        self.assertIn("axial", report["evidence"])

    def test_registration_qa_is_scope_aware_and_rejects_scope_tampering(self) -> None:
        localized, localization_report = self._register_and_localize()
        localization_document = json.loads(
            localization_report.read_text(encoding="utf-8")
        )
        scope_document = json.loads(self.roi_scope.read_text(encoding="utf-8"))
        self.assertEqual(
            scope_document["analysis_parameters_sha256"],
            localization_document["hashes"]["analysis_parameters_sha256"],
        )
        self.assertEqual(
            "hashed_analysis_parameters",
            localization_document["provenance"]["policy_binding"],
        )
        self.assertEqual("synthetic_fixture", localization_document["specimen_id"])
        self.assertEqual("synthetic_design", localization_document["design_id"])
        with self.assertRaisesRegex(ValueError, "conflicts with hashed policy"):
            localize_lattice_nodes(
                self.volume,
                localized,
                self.root / "conflicting-policy-graph.json",
                self.root / "conflicting-policy-localization.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_policy_artifact_path=self.roi_scope,
                registration_report_path=self.root / "registration.json",
                config={"maximum_fallback_fraction": 0.5},
            )
        qa_config = {
            "minimum_mean_junction_foreground_fraction": 0.1,
            "minimum_median_corridor_foreground_fraction": 0.01,
            "maximum_spatial_bin_median_range": 1.0,
            "minimum_roi_in_bounds_fraction": 0.5,
            "spatial_bins_per_axis": 2,
        }
        roi = compute_registration_qa(
            self.volume,
            localized,
            self.root / "roi-qa.json",
            threshold=self.threshold,
            registration_mode="challenge_aligned_json",
            analysis_scope_artifact_path=self.roi_scope,
            localization_report_path=localization_report,
            config=qa_config,
        )
        self.assertEqual("pass", roi["gate"], roi)
        self.assertEqual("synthetic_fixture", roi["specimen_id"])
        self.assertEqual("synthetic_design", roi["design_id"])
        self.assertEqual("roi_screening", roi["requested_analysis_scope"])
        self.assertEqual("not_authorized", roi["metrology"]["status"])
        self.assertIsNone(roi["metrology"]["overall_pass"])
        self.assertTrue(roi["roi_gate_results"]["overall_pass"])
        self.assertIn("coarse_region_screening", roi["authorized_outputs"])
        self.assertIn("absolute_metrology", roi["unauthorized_outputs"])
        self.assertIn("METROLOGY_NOT_AUTHORIZED", roi["reason_codes"])
        self.assertEqual(
            3,
            roi["localization_quality_counts"]["fallback_nodes"],
        )
        self.assertEqual(
            scope_document["analysis_parameters_sha256"],
            roi["hashes"]["analysis_parameters_sha256"],
        )
        self.assertEqual(
            sha256_json(scope_document["analysis_parameters"]["qa_policy"]),
            roi["hashes"]["qa_policy_sha256"],
        )
        with self.assertRaisesRegex(ValueError, "conflicts with hashed policy"):
            compute_registration_qa(
                self.volume,
                localized,
                self.root / "conflicting-policy-qa.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_scope_artifact_path=self.roi_scope,
                localization_report_path=localization_report,
                config={
                    **qa_config,
                    "maximum_uncertainty_to_radius_ratio": 0.5,
                },
            )

        binding_mutations = {
            "wrong-scope": lambda value: value.__setitem__(
                "requested_analysis_scope", "direct_metrology"
            ),
            "wrong-mode": lambda value: value.__setitem__(
                "registration_mode", "autonomous_v2"
            ),
            "wrong-ct": lambda value: value["hashes"].__setitem__(
                "ct_sha256", "0" * 64
            ),
            "wrong-graph": lambda value: value["hashes"].__setitem__(
                "localized_graph_sha256", "0" * 64
            ),
            "stale-analysis": lambda value: value["hashes"].__setitem__(
                "analysis_parameters_sha256", "0" * 64
            ),
            "wrong-policy-hash": lambda value: value["hashes"].__setitem__(
                "localization_policy_sha256", "0" * 64
            ),
            "looser-policy": lambda value: value["quantitative_policy"].__setitem__(
                "maximum_fallback_fraction", 0.5
            ),
        }
        for case, mutate in binding_mutations.items():
            with self.subTest(localization_binding=case):
                tampered_localization = json.loads(
                    localization_report.read_text(encoding="utf-8")
                )
                mutate(tampered_localization)
                tampered_localization_path = self.root / f"{case}-localization.json"
                tampered_localization_path.write_text(
                    json.dumps(tampered_localization, sort_keys=True),
                    encoding="utf-8",
                )
                rejected = compute_registration_qa(
                    self.volume,
                    localized,
                    self.root / f"{case}-qa.json",
                    threshold=self.threshold,
                    registration_mode="challenge_aligned_json",
                    analysis_scope_artifact_path=self.roi_scope,
                    localization_report_path=tampered_localization_path,
                    config=qa_config,
                )
                self.assertEqual("halt", rejected["gate"], rejected)
                self.assertFalse(rejected["localization_binding"]["overall_pass"])
                self.assertTrue(
                    any(
                        code.startswith("LOCALIZATION_BINDING_FAILED_")
                        for code in rejected["reason_codes"]
                    )
                )

        direct_registration_path = self.root / "direct-registration.json"
        direct_registration_path.write_text(
            json.dumps(
                {
                    "schema_version": "part2-registration/1.0.0",
                    "mode": "autonomous_v2",
                    "gate": "pass",
                    "hashes": {
                        "ct_sha256": sha256_file(self.volume),
                        "registered_graph_sha256": sha256_file(localized),
                    },
                    "mode_details": {
                        "bounded_robustness": {
                            "p95_prediction_spread_voxels": 0.1,
                            "overall_pass": True,
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        direct_localized_path = self.root / "direct-localized.json"
        direct_localization_path = self.root / "direct-localization.json"
        direct_localization = localize_lattice_nodes(
            self.volume,
            localized,
            direct_localized_path,
            direct_localization_path,
            threshold=self.threshold,
            registration_mode="autonomous_v2",
            analysis_policy_artifact_path=self.direct_autonomous_scope,
            registration_report_path=direct_registration_path,
            config={
                "minimum_primary_or_stable_coarse_fraction": 0.0,
                "maximum_fallback_fraction": 1.0,
                "maximum_ambiguous_fraction": 1.0,
                "maximum_rejected_fraction": 1.0,
                "maximum_boundary_limited_fraction": 1.0,
            },
        )
        self.assertEqual("pass", direct_localization["gate"])
        direct = compute_registration_qa(
            self.volume,
            direct_localized_path,
            self.root / "direct-qa.json",
            threshold=self.threshold,
            registration_mode="autonomous_v2",
            analysis_scope_artifact_path=self.direct_autonomous_scope,
            localization_report_path=direct_localization_path,
            config=qa_config,
        )
        self.assertEqual("pass", direct["gate"], direct)
        self.assertEqual("pass", direct["metrology"]["status"])
        self.assertIn(
            "direct_dimensional_measurement", direct["authorized_outputs"]
        )

        tampered_registration = json.loads(
            direct_registration_path.read_text(encoding="utf-8")
        )
        tampered_registration["mode_details"]["bounded_robustness"][
            "p95_prediction_spread_voxels"
        ] = 0.2
        direct_registration_path.write_text(
            json.dumps(tampered_registration, sort_keys=True), encoding="utf-8"
        )
        tampered_direct = compute_registration_qa(
            self.volume,
            direct_localized_path,
            self.root / "tampered-direct-qa.json",
            threshold=self.threshold,
            registration_mode="autonomous_v2",
            analysis_scope_artifact_path=self.direct_autonomous_scope,
            localization_report_path=direct_localization_path,
            config=qa_config,
        )
        self.assertEqual("halt", tampered_direct["gate"])
        self.assertFalse(
            tampered_direct["metrology"]["gates"][
                "absolute_registration_uncertainty_artifact_backed"
            ]
        )

        tampered_scope = self.root / "tampered-scope.json"
        tampered_scope_document = json.loads(
            self.roi_scope.read_text(encoding="utf-8")
        )
        tampered_scope_document["requested_analysis_scope"] = "direct_metrology"
        tampered_scope_document["analysis_parameters"][
            "requested_analysis_scope"
        ] = "direct_metrology"
        tampered_scope.write_text(
            json.dumps(tampered_scope_document),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "analysis_parameters_sha256"):
            compute_registration_qa(
                self.volume,
                localized,
                self.root / "tampered-qa.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_scope_artifact_path=tampered_scope,
                localization_report_path=localization_report,
                config=qa_config,
            )

    def test_hashed_stage2_policy_and_exact_otsu_bindings_are_closed(self) -> None:
        localized, localization_report = self._register_and_localize()

        def mutated_scope(
            name: str,
            mutate: object,
        ) -> Path:
            document = json.loads(self.roi_scope.read_text(encoding="utf-8"))
            mutate(document["analysis_parameters"])
            document["analysis_parameters_sha256"] = sha256_json(
                document["analysis_parameters"]
            )
            path = self.root / f"{name}-scope.json"
            path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            return path

        invalid_policies = (
            (
                "wrong-policy-version",
                lambda parameters: parameters["localization_policy"].__setitem__(
                    "schema_version", "stage2-localization-policy/9.9.9"
                ),
                "schema_version",
            ),
            (
                "extra-localization-field",
                lambda parameters: parameters["localization_policy"].__setitem__(
                    "unhashed_safety_override", 1
                ),
                "closed schema",
            ),
            (
                "weights-not-normalized",
                lambda parameters: parameters["localization_policy"].__setitem__(
                    "core_support_weight", 0.8
                ),
                "sum to 1",
            ),
            (
                "duplicate-incident-distance",
                lambda parameters: parameters["localization_policy"].__setitem__(
                    "incident_sample_distances_voxels", [3.0, 3.0, 7.0]
                ),
                "strictly increasing",
            ),
            (
                "extra-segmentation-field",
                lambda parameters: parameters["segmentation"].__setitem__(
                    "unhashed_threshold_adjustment", 1
                ),
                "closed schema",
            ),
            (
                "extra-artifact-version",
                lambda parameters: parameters["artifact_schema_versions"].__setitem__(
                    "unrecognized_report", "1.0.0"
                ),
                "artifact_schema_versions",
            ),
        )
        for name, mutate, pattern in invalid_policies:
            with self.subTest(policy=name):
                scope_path = mutated_scope(name, mutate)
                with self.assertRaisesRegex(ValueError, pattern):
                    localize_lattice_nodes(
                        self.volume,
                        self.aligned,
                        self.root / f"{name}-graph.json",
                        self.root / f"{name}-report.json",
                        threshold=self.threshold,
                        registration_mode="challenge_aligned_json",
                        analysis_policy_artifact_path=scope_path,
                    )

        qa_extra = mutated_scope(
            "extra-qa-field",
            lambda parameters: parameters["qa_policy"].__setitem__(
                "unhashed_safety_override", 1
            ),
        )
        with self.assertRaisesRegex(ValueError, "closed schema"):
            compute_registration_qa(
                self.volume,
                localized,
                self.root / "extra-qa-field-report.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_scope_artifact_path=qa_extra,
                localization_report_path=localization_report,
            )

        with self.assertRaisesRegex(ValueError, "outside the closed hashed policy"):
            localize_lattice_nodes(
                self.volume,
                self.aligned,
                self.root / "legacy-override-graph.json",
                self.root / "legacy-override-report.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_policy_artifact_path=self.roi_scope,
                registration_report_path=self.root / "registration.json",
                config={"minimum_accepted_fraction": 0.0},
            )
        with self.assertRaisesRegex(ValueError, "outside the closed hashed policy"):
            compute_registration_qa(
                self.volume,
                localized,
                self.root / "unknown-qa-config.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_scope_artifact_path=self.roi_scope,
                localization_report_path=localization_report,
                config={"unhashed_safety_override": 1},
            )
        with self.assertRaisesRegex(ValueError, "exact Otsu replay"):
            localize_lattice_nodes(
                self.volume,
                self.aligned,
                self.root / "wrong-threshold-graph.json",
                self.root / "wrong-threshold-localization.json",
                threshold=self.threshold + 1.0,
                registration_mode="challenge_aligned_json",
                analysis_policy_artifact_path=self.roi_scope,
                registration_report_path=self.root / "registration.json",
            )
        wrong_threshold_qa = compute_registration_qa(
            self.volume,
            localized,
            self.root / "wrong-threshold-qa.json",
            threshold=self.threshold + 1.0,
            registration_mode="challenge_aligned_json",
            analysis_scope_artifact_path=self.roi_scope,
            localization_report_path=localization_report,
        )
        self.assertEqual("halt", wrong_threshold_qa["gate"])
        self.assertFalse(
            wrong_threshold_qa["segmentation_binding"]["overall_pass"]
        )
        with self.assertRaisesRegex(ValueError, "registration_mode"):
            localize_lattice_nodes(
                self.volume,
                self.aligned,
                self.root / "wrong-mode-graph.json",
                self.root / "wrong-mode-localization.json",
                threshold=self.threshold,
                registration_mode="autonomous_v2",
                analysis_policy_artifact_path=self.roi_scope,
                registration_report_path=self.root / "registration.json",
            )

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=REPOSITORY_ROOT,
            prefix=".escaped-registration-",
            suffix=".json",
            delete=False,
        ) as stream:
            escaped_registration = Path(stream.name)
            stream.write((self.root / "registration.json").read_bytes())
        self.addCleanup(escaped_registration.unlink, missing_ok=True)
        with self.assertRaisesRegex(ValueError, "must share the bounded"):
            localize_lattice_nodes(
                self.volume,
                self.aligned,
                self.root / "escaped-registration-graph.json",
                self.root / "escaped-registration-localization.json",
                threshold=self.threshold,
                registration_mode="challenge_aligned_json",
                analysis_policy_artifact_path=self.roi_scope,
                registration_report_path=escaped_registration,
            )
        escaped_localization = json.loads(
            localization_report.read_text(encoding="utf-8")
        )
        escaped_localization["artifacts"]["registration_report"]["path"] = str(
            escaped_registration
        )
        escaped_localization_path = self.root / "escaped-embedded-localization.json"
        escaped_localization_path.write_text(
            json.dumps(escaped_localization, sort_keys=True), encoding="utf-8"
        )
        escaped_qa = compute_registration_qa(
            self.volume,
            localized,
            self.root / "escaped-embedded-qa.json",
            threshold=self.threshold,
            registration_mode="challenge_aligned_json",
            analysis_scope_artifact_path=self.roi_scope,
            localization_report_path=escaped_localization_path,
        )
        self.assertEqual("halt", escaped_qa["gate"])
        self.assertFalse(
            escaped_qa["localization_binding"]["gates"][
                "registration_report_path_within_run_root"
            ]
        )

    def test_empty_ct_localization_halts_with_structured_gate(self) -> None:
        empty = self.root / "empty.npy"
        np.save(empty, np.zeros((40, 40, 40), dtype=np.uint16))
        report = localize_lattice_nodes(
            empty,
            self.aligned,
            self.root / "fallback.json",
            self.root / "fallback-report.json",
            threshold=500,
            registration_mode="challenge_aligned_json",
        )
        self.assertEqual("halt", report["gate"])
        self.assertEqual(0, report["counts"]["accepted_nodes"])
        self.assertEqual(4, report["counts"]["fallback_nodes"])


class AxisAndDeterminismTests(SyntheticFixture):
    def test_corridor_sampler_pins_xyz_to_zyx(self) -> None:
        sentinel = np.zeros((5, 6, 7), dtype=np.uint16)
        sentinel[4, 2, 6] = 999
        sampled = sample_corridor(
            sentinel,
            np.asarray([6.0, 2.0, 4.0]),
            np.asarray([5.0, 2.0, 4.0]),
            threshold=900,
            axial_samples=3,
            radius_voxels=0.1,
            angular_samples=4,
        )
        self.assertTrue(sampled["foreground"][0, 0])

    def test_registration_and_synthetic_suite_are_deterministic(self) -> None:
        first = register_lattice_to_ct(
            self.nominal,
            self.root / "first.json",
            self.root / "first-report.json",
            mode="challenge_aligned_json",
            ct_path=self.volume,
            aligned_graph_path=self.aligned,
        )
        second = register_lattice_to_ct(
            self.nominal,
            self.root / "second.json",
            self.root / "second-report.json",
            mode="challenge_aligned_json",
            ct_path=self.volume,
            aligned_graph_path=self.aligned,
        )
        self.assertEqual(
            first["hashes"]["registered_graph_sha256"],
            second["hashes"]["registered_graph_sha256"],
        )
        x, y, z = np.meshgrid(
            np.linspace(0.0, 18.0, 5),
            np.linspace(0.0, 18.0, 5),
            np.linspace(0.0, 18.0, 5),
            indexing="ij",
        )
        source = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        config = {
            "synthetic": {
                "case_count": 2,
                "minimum_pass_fraction": 0.5,
            }
        }
        self.assertEqual(
            run_synthetic_suite(source, config),
            run_synthetic_suite(source, config),
        )

    def test_exact_similarity_synthetic_recovery(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(100, 3))
        expected = SimilarityTransform(
            12.5,
            Rotation.from_euler("z", 1.0, degrees=True).as_matrix(),
            np.asarray([4.0, 5.0, 6.0]),
        )
        fitted = solve_similarity(source, expected.apply(source))
        self.assertAlmostEqual(12.5, fitted.scale, places=8)
        self.assertLess(np.linalg.norm(fitted.translation - expected.translation), 1e-8)

    def test_autonomous_v2_recovers_seeded_ct_node_grid(self) -> None:
        design_points = np.asarray(
            [[x, y, z] for x in range(4) for y in range(4) for z in range(4)],
            dtype=float,
        )
        node_ids = [100 + index * 3 for index in range(len(design_points))]
        autonomous_graph = self.root / "autonomous-nominal.json"
        autonomous_graph.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": identifier, "position": point.tolist()}
                        for identifier, point in zip(node_ids, design_points)
                    ],
                    "struts": [
                        {
                            "id": 9001,
                            "junction0": node_ids[0],
                            "junction1": node_ids[1],
                        }
                    ],
                    "unit_cells": [
                        {
                            "id": 77,
                            "indices": [0, 0, 0],
                            "struts": [9001],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        target_points = 6.0 * design_points + 4.0
        zz, yy, xx = np.indices((25, 25, 25))
        points = np.stack((xx, yy, zz), axis=-1)
        mask = np.zeros((25, 25, 25), dtype=bool)
        for point in target_points:
            mask |= np.sum((points - point) ** 2, axis=-1) <= 2.0**2
        autonomous_ct = self.root / "autonomous.npy"
        np.save(autonomous_ct, np.where(mask, 1000, 0).astype(np.uint16))
        result = register_lattice_to_ct(
            autonomous_graph,
            self.root / "autonomous-registered.json",
            self.root / "autonomous-report.json",
            mode="autonomous_v2",
            ct_path=autonomous_ct,
            threshold=500,
            config={
                "detection": {
                    "downsample_factor": 1,
                    "central_z_margin_fraction": 0.0,
                    "edt_peak_threshold_downsampled_voxels": 1.5,
                    "minimum_component_voxels": 1,
                    "maximum_component_voxels": 999,
                    "candidate_holdout_fraction": 0.2,
                },
                "gates": {
                    "maximum_holdout_median_distance_voxels": 1.0,
                    "minimum_candidate_to_unique_node_ratio": 0.9,
                },
            },
        )
        self.assertEqual("pass", result["gate"], result)
        self.assertFalse(result["provenance"]["aligned_graph_used_for_fit"])
        self.assertAlmostEqual(6.0, result["transform"]["scale"], places=6)


class GraphSizeAndWrapperBoundaryTests(unittest.TestCase):
    def test_8x8x8_and_9x9x9_graphs_are_input_derived(self) -> None:
        graph8 = load_lattice_json(
            REPOSITORY_ROOT / "data/octet_truss_8x8x8/octet_truss_8x8x8.json"
        )
        graph9 = load_lattice_json(
            REPOSITORY_ROOT / "data/missing_struts/octet_truss_9x9x9.json"
        )
        self.assertEqual({"nodes": 7_168, "edges": 13_056, "cells": 512}, graph8.counts)
        self.assertEqual(
            {"nodes": 10_206, "edges": 18_468, "cells": 729}, graph9.counts
        )

    def test_new_mcp_wrappers_contain_no_numpy_numerical_calls(self) -> None:
        modules = [
            ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted((REPOSITORY_ROOT / "src/mcp_tools").glob("stage*.py"))
        ]
        names = {
            "register_lattice_to_ct",
            "localize_lattice_nodes",
            "compute_registration_qa",
            "compute_strut_metrics",
            "classify_struts",
            "render_strut_evidence",
            "get_strut_report",
            "compute_spatial_stats",
            "render_lattice_3d",
        }
        wrappers = [
            node
            for module in modules
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        self.assertEqual(names, {node.name for node in wrappers})
        for wrapper in wrappers:
            numpy_calls = [
                node
                for node in ast.walk(wrapper)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "np"
            ]
            self.assertEqual([], numpy_calls, wrapper.name)


if __name__ == "__main__":
    unittest.main()
