"""Tests for the Part 2 manifest schema and provenance gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from specimen_manifest import (  # noqa: E402
    ManifestValidationError,
    canonical_json_sha256,
    manifest_paths,
    topology_summary,
    validate_manifest,
)
from segmentation_replay import histogram_sha256, otsu_from_histogram  # noqa: E402

import numpy as np


class SpecimenManifestTests(unittest.TestCase):
    def test_all_example_manifests_validate(self) -> None:
        paths = manifest_paths()
        self.assertEqual(2, len(paths))
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

    def _write_temporary(self, manifest: dict[str, object]) -> Path:
        path = REPOSITORY_ROOT / "analysis" / "schema" / ".invalid-manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
