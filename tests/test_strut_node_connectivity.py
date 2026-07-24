"""Focused regression tests for the A-to-B connectivity primitive."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_strut_node_connectivity.py"
SPEC = importlib.util.spec_from_file_location("strut_node_connectivity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONNECTIVITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONNECTIVITY
SPEC.loader.exec_module(CONNECTIVITY)


class SharedComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.local_z = np.arange(41, dtype=np.float64)
        self.disk = np.zeros((3, 3), dtype=bool)
        self.disk[1, 1] = True
        self.structure = ndimage.generate_binary_structure(3, 3)

    def measure(self, binary: np.ndarray) -> dict[str, object]:
        labels, _ = ndimage.label(binary, structure=self.structure)
        return CONNECTIVITY.shared_component_between_windows(
            labels,
            self.local_z,
            self.disk,
            first_z=0.0,
            second_z=40.0,
            half_length=2.0,
        )

    def test_one_intact_component_connects_a_to_b(self) -> None:
        binary = np.zeros((41, 3, 3), dtype=bool)
        binary[:, 1, 1] = True
        result = self.measure(binary)
        self.assertTrue(result["same_component_observed"])
        self.assertEqual(result["shared_component_count"], 1)

    def test_disconnected_halves_do_not_connect_a_to_b(self) -> None:
        binary = np.zeros((41, 3, 3), dtype=bool)
        binary[:11, 1, 1] = True
        binary[30:, 1, 1] = True
        labels, _ = ndimage.label(binary, structure=self.structure)

        endpoint_a = CONNECTIVITY.endpoint_connection_measurement(
            binary, labels, self.local_z, self.disk, endpoint_z=0.0, collar_z=8.0
        )
        endpoint_b = CONNECTIVITY.endpoint_connection_measurement(
            binary, labels, self.local_z, self.disk, endpoint_z=40.0, collar_z=32.0
        )
        direct = CONNECTIVITY.shared_component_between_windows(
            labels,
            self.local_z,
            self.disk,
            first_z=0.0,
            second_z=40.0,
            half_length=2.0,
        )

        self.assertTrue(endpoint_a["node_to_collar_component_observed"])
        self.assertTrue(endpoint_b["node_to_collar_component_observed"])
        self.assertFalse(direct["same_component_observed"])
        self.assertEqual(direct["shared_component_count"], 0)


class NormalizedCuboidTests(unittest.TestCase):
    def test_frame_is_right_handed_for_diagonal_and_reversal(self) -> None:
        node_a = np.array([1.0, 2.0, 3.0])
        node_b = np.array([5.0, 7.0, 11.0])
        frame = CONNECTIVITY.stable_frame(node_a, node_b)
        reversed_frame = CONNECTIVITY.stable_frame(node_b, node_a)
        basis = np.asarray(frame[:3])

        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=12)
        np.testing.assert_allclose(frame[2], -reversed_frame[2], atol=1e-12)
        self.assertAlmostEqual(frame[3], reversed_frame[3], places=12)

    def test_a_and_b_are_exact_first_and_last_slices(self) -> None:
        zz, yy, xx = np.indices((9, 9, 9))
        volume = (100 * xx + 10 * yy + zz).astype(np.float32)
        nodes = {0: np.array([2.0, 3.0, 1.0]), 1: np.array([6.0, 3.0, 1.0])}
        strut = {"id": 7, "junction0": 0, "junction1": 1}
        sampled, geometry, local_x, local_y, local_z = CONNECTIVITY.build_cuboid(
            volume, strut, nodes, half_width=1.0, corridor_radius=0.5
        )
        x_zero = int(np.flatnonzero(local_x == 0.0)[0])
        y_zero = int(np.flatnonzero(local_y == 0.0)[0])

        self.assertEqual(local_z[0], 0.0)
        self.assertEqual(local_z[-1], geometry.length_voxels)
        np.testing.assert_allclose(geometry.center_xyz, [4.0, 3.0, 1.0])
        self.assertEqual(sampled[0, y_zero, x_zero], volume[1, 3, 2])
        self.assertEqual(sampled[-1, y_zero, x_zero], volume[1, 3, 6])

    def test_batched_interpolation_matches_reference_and_uses_one_scipy_call(self) -> None:
        zz, yy, xx = np.indices((20, 20, 20))
        volume = (100.0 * xx + 10.0 * yy + zz).astype(np.float32)
        nodes = {
            0: np.array([4.0, 5.0, 6.0]),
            1: np.array([13.0, 5.0, 6.0]),
            2: np.array([5.0, 4.0, 5.0]),
            3: np.array([12.0, 13.0, 14.0]),
        }
        struts = [
            {"id": 7, "junction0": 0, "junction1": 1},
            {"id": 8, "junction0": 2, "junction1": 3},
        ]

        original_map_coordinates = CONNECTIVITY.ndimage.map_coordinates
        with mock.patch.object(
            CONNECTIVITY.ndimage,
            "map_coordinates",
            wraps=original_map_coordinates,
        ) as interpolation:
            cuboids = CONNECTIVITY.build_cuboids_batch(
                volume, struts, nodes, half_width=2.0, corridor_radius=1.5
            )
        self.assertEqual(interpolation.call_count, 1)

        for strut, (sampled, geometry, local_x, local_y, local_z) in zip(
            struts, cuboids, strict=True
        ):
            basis_x, basis_y, basis_z, length = CONNECTIVITY.stable_frame(
                nodes[strut["junction0"]], nodes[strut["junction1"]]
            )
            center = (nodes[strut["junction0"]] + nodes[strut["junction1"]]) / 2.0
            grid_z, grid_y, grid_x = np.meshgrid(
                local_z, local_y, local_x, indexing="ij"
            )
            world = (
                center[None, None, None, :]
                + grid_x[..., None] * basis_x
                + grid_y[..., None] * basis_y
                + (grid_z[..., None] - 0.5 * length) * basis_z
            )
            reference = original_map_coordinates(
                volume,
                np.vstack(
                    [world[..., 2].ravel(), world[..., 1].ravel(), world[..., 0].ravel()]
                ),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
                output=np.float32,
            ).reshape(grid_z.shape)
            np.testing.assert_array_equal(sampled, reference)
            self.assertEqual(list(sampled.shape), geometry.local_shape_zyx)


class OutputCleanupTests(unittest.TestCase):
    def test_cleanup_removes_only_script_owned_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            owned = [output_dir / name for name in CONNECTIVITY.GENERATED_RUN_FILENAMES]
            owned.append(output_dir / "strut_42_cuboid.npz")
            preserved = [output_dir / "analysis_config.json", output_dir / "review_notes.md"]
            for path in owned + preserved:
                path.write_text("test", encoding="utf-8")

            CONNECTIVITY.clear_previous_run_artifacts(output_dir)

            self.assertTrue(all(not path.exists() for path in owned))
            self.assertTrue(all(path.exists() for path in preserved))


class ExplicitSelectionTests(unittest.TestCase):
    def test_explicit_ids_preserve_requested_order(self) -> None:
        struts = [{"id": 10}, {"id": 20}, {"id": 30}]
        selected = CONNECTIVITY.select_struts_by_ids(struts, [30, 10])
        self.assertEqual([strut["id"] for strut in selected], [30, 10])


if __name__ == "__main__":
    unittest.main()
