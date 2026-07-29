from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from registration_core import (  # noqa: E402
    SimilarityTransform,
    deterministic_65536_histogram,
    downstream_tolerance_gate,
    histogram_diagnostics,
    otsu_from_histogram,
    rotation_difference_deg,
    run_synthetic_suite,
    sha256_file,
    solve_similarity,
    split_candidates,
)
from validate_against_ground_truth import verify_completion  # noqa: E402


class SimilarityTests(unittest.TestCase):
    def test_exact_similarity_recovery(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(200, 3))
        rotation = Rotation.from_euler(
            "xyz", [2.0, -1.0, 0.5], degrees=True
        ).as_matrix()
        expected = SimilarityTransform(
            scale=39.4,
            rotation=rotation,
            translation=np.array([59.0, 52.0, 26.0]),
        )
        target = expected.apply(source)
        fitted = solve_similarity(source, target)
        self.assertAlmostEqual(fitted.scale, expected.scale, places=10)
        self.assertLess(
            rotation_difference_deg(fitted.rotation, expected.rotation), 1e-5
        )
        self.assertLess(
            np.linalg.norm(fitted.translation - expected.translation), 1e-10
        )

    def test_degenerate_similarity_rejected(self) -> None:
        source = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            solve_similarity(source, source)


class HistogramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "config.default.json").read_text())

    def test_bimodal_histogram_passes(self) -> None:
        levels = np.arange(65536)
        histogram = (
            np.exp(-0.5 * ((levels - 12000) / 1200) ** 2) * 9_000
            + np.exp(-0.5 * ((levels - 50000) / 1800) ** 2) * 1_500
        ).astype(np.int64)
        threshold, separability = otsu_from_histogram(histogram)
        report = histogram_diagnostics(
            histogram, threshold, separability, self.config
        )
        self.assertTrue(report["overall_pass"], report)
        self.assertGreater(threshold, 12000)
        self.assertLess(threshold, 50000)

    def test_unimodal_histogram_rejected(self) -> None:
        levels = np.arange(65536)
        histogram = (
            np.exp(-0.5 * ((levels - 30000) / 3000) ** 2) * 10_000
        ).astype(np.int64)
        threshold, separability = otsu_from_histogram(histogram)
        report = histogram_diagnostics(
            histogram, threshold, separability, self.config
        )
        self.assertFalse(report["gates"]["histogram_not_unimodal"])
        self.assertFalse(report["overall_pass"])

    def test_float_histogram_counts_every_voxel_and_records_mapping(self) -> None:
        volume = np.array(
            [
                [[-0.5, -0.25], [0.0, 0.25]],
                [[0.5, 0.75], [1.0, 1.25]],
            ],
            dtype=np.float32,
        )
        histogram, count, encoding = deterministic_65536_histogram(
            volume, chunk_depth=1
        )
        self.assertEqual(count, volume.size)
        self.assertEqual(int(histogram.sum()), volume.size)
        self.assertEqual(encoding["encoding"], "full_volume_affine_uint16")
        self.assertEqual(encoding["native_min"], -0.5)
        self.assertEqual(encoding["native_max"], 1.25)


class IsolationAndGateTests(unittest.TestCase):
    def test_fitting_cli_has_no_ground_truth_input(self) -> None:
        fit_source = (ROOT / "fit_registration.py").read_text()
        self.assertNotIn("--ground-truth", fit_source)
        self.assertNotIn("registered_jsons", fit_source)

    def test_candidate_split_has_no_overlap(self) -> None:
        candidates = np.arange(300, dtype=float).reshape(100, 3)
        fit, holdout, fit_indices, holdout_indices = split_candidates(
            candidates, 0.2, 42
        )
        self.assertEqual(len(fit), 80)
        self.assertEqual(len(holdout), 20)
        self.assertEqual(
            len(set(fit_indices.tolist()) & set(holdout_indices.tolist())), 0
        )

    def test_downstream_gate_blocks_large_uncertainty(self) -> None:
        config = json.loads((ROOT / "config.default.json").read_text())
        multistart = {
            "near_optimal_p95_prediction_spread_voxels": 0.1
        }
        robustness = {"maximum_case_p95_prediction_shift_voxels": 0.2}
        image = {
            "candidate_holdout": {
                "median_distance_to_predicted_unique_node_voxels": 5.0
            },
            "corridors": {"measured_strut_radius_voxels": 3.0},
        }
        result = downstream_tolerance_gate(
            multistart, robustness, image, config
        )
        self.assertFalse(result["classification_allowed"])

    def test_validator_rejects_missing_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "FIT_COMPLETE"):
                verify_completion(Path(directory))

    def test_validator_rejects_tampered_frozen_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "fit_manifest.json"
            transform = root / "fitted_transform.json"
            registered = root / "our_registered.json"
            manifest.write_text(
                json.dumps({"ground_truth_used_for_fit": False})
            )
            transform.write_text("{}")
            registered.write_text("{}")
            records = {}
            for key, path in (
                ("fit_manifest", manifest),
                ("fitted_transform", transform),
                ("our_registered", registered),
            ):
                records[key] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            marker = {
                **records,
                "ground_truth_used_for_fit": False,
                "fit_completed_before_ground_truth_access": True,
            }
            (root / "FIT_COMPLETE.json").write_text(json.dumps(marker))
            registered.write_text('{"tampered": true}')
            with self.assertRaisesRegex(RuntimeError, "integrity"):
                verify_completion(root)


class SyntheticRecoveryTests(unittest.TestCase):
    def test_small_synthetic_suite(self) -> None:
        config = json.loads((ROOT / "config.default.json").read_text())
        config = copy.deepcopy(config)
        config["synthetic"]["case_count"] = 2
        config["synthetic"]["minimum_pass_fraction"] = 0.5
        x, y, z = np.meshgrid(
            np.linspace(0.0, 18.0, 8),
            np.linspace(0.0, 18.0, 8),
            np.linspace(0.0, 18.0, 8),
            indexing="ij",
        )
        source = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        result = run_synthetic_suite(source, config)
        self.assertTrue(result["overall_pass"], result)


if __name__ == "__main__":
    unittest.main()
