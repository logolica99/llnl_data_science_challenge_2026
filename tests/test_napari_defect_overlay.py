from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from napari_defect_overlay import (  # noqa: E402
    CT_VALIDATED_AXIS_MAP,
    IDENTITY_AXIS_MAP,
    remap_missing_strut_ids,
)


NOMINAL_DESIGN = PROJECT_ROOT / "data/missing_struts/octet_truss_9x9x9.json"
MISSING_STRUTS = (
    PROJECT_ROOT
    / "data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv"
)


class MissingStrutOrientationTests(unittest.TestCase):
    def test_ct_validated_map_is_a_cube_rotation(self) -> None:
        np.testing.assert_allclose(
            CT_VALIDATED_AXIS_MAP @ CT_VALIDATED_AXIS_MAP.T,
            np.eye(3),
        )
        self.assertAlmostEqual(np.linalg.det(CT_VALIDATED_AXIS_MAP), 1.0)

        center = np.array([9.0, 9.0, 9.0])
        point = np.array([16.0, 1.0, 1.0])
        transformed = (point - center) @ CT_VALIDATED_AXIS_MAP.T + center
        np.testing.assert_allclose(transformed, [2.0, 17.0, 17.0])

    def test_identity_orientation_preserves_ids(self) -> None:
        source, target, mapping = remap_missing_strut_ids(
            MISSING_STRUTS,
            NOMINAL_DESIGN,
            IDENTITY_AXIS_MAP,
        )
        self.assertEqual(len(source), 93)
        self.assertEqual(source, target)
        self.assertTrue(all(source_id == target_id for source_id, target_id in mapping.items()))

    def test_ct_orientation_remaps_known_deleted_edges(self) -> None:
        source, target, mapping = remap_missing_strut_ids(
            MISSING_STRUTS,
            NOMINAL_DESIGN,
            CT_VALIDATED_AXIS_MAP,
        )
        self.assertEqual(len(source), 93)
        self.assertEqual(len(target), 93)
        self.assertEqual(source & target, set())
        self.assertEqual(mapping[210], 18200)
        self.assertEqual(mapping[638], 13888)
        self.assertEqual(mapping[18438], 10)


if __name__ == "__main__":
    unittest.main()
