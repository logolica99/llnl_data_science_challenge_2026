import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strut_cross_section_viewer import (
    add_registration_metrics,
    classify_strut,
    consensus_ct_centerline,
    detect_dense_boundary_limits,
    enumerate_component_candidates,
    extract_cross_sections,
    recover_bounded_path_gaps,
)


def cylinder_volume(broken=False, junction_flare=False):
    volume = np.zeros((81, 48, 48), dtype=np.uint16)
    yy, xx = np.indices((48, 48))
    disk = (xx - 24) ** 2 + (yy - 22) ** 2 <= 3 ** 2
    flare = (xx - 24) ** 2 + (yy - 22) ** 2 <= 9 ** 2
    for z in range(5, 76):
        if broken and 32 <= z <= 50:
            continue
        volume[z][flare if junction_flare and (z <= 18 or z >= 62) else disk] = 1000
    return volume


def registration_metrics(sections):
    return {
        key: sections[0][key] for key in (
            "tracking_coverage",
            "median_registration_offset_voxels",
            "registration_offset_variation_voxels",
        )
    }


class RegistrationTolerantScreeningTests(unittest.TestCase):
    def test_tracking_confidence_is_conditional_on_supported_planes(self):
        sections = [
            {
                "distance_voxels": float(index),
                "centroid_u_voxels": (
                    float(index) if index < 3 else float("nan")
                ),
                "centroid_v_voxels": (
                    0.0 if index < 3 else float("nan")
                ),
                "measurement_eligible": True,
                "tracking_confidence": 0.9 if index < 3 else 0.0,
            }
            for index in range(4)
        ]
        metrics = add_registration_metrics(sections)
        self.assertAlmostEqual(metrics["tracking_coverage"], 0.75)
        self.assertAlmostEqual(metrics["mean_tracking_confidence"], 0.9)

    def test_consensus_ignores_ambiguous_junction_planes(self):
        def candidate(u, v, area=120):
            return {
                "centroid_u": float(u),
                "centroid_v": float(v),
                "registered_distance": float(np.hypot(u, v)),
                "area_pixels": int(area),
            }

        planes = [
            [candidate(0.5, 5.2), candidate(-5.0, 0.0)],
            *[
                [candidate(u, v)]
                for u, v in (
                    (0.1, 5.3), (-0.2, 5.5), (-0.1, 5.4),
                    (-1.3, 5.2), (-2.1, 5.5), (-0.8, 4.8),
                    (-0.8, 4.8), (-0.5, 4.5), (-0.7, 5.1),
                )
            ],
            [candidate(-0.2, 4.0), candidate(0.5, -6.3)],
        ]
        centerline = consensus_ct_centerline(
            planes, ignore_edge_sections=1
        )
        self.assertIsNotNone(centerline)
        self.assertEqual(centerline.shape, (11, 2))
        self.assertGreater(float(np.median(centerline[:, 1])), 4.0)

    def test_adjacent_centerline_deviation_is_classified_as_bent(self):
        residuals = [0.1, 0.2, 0.8, 1.6, 0.8, 0.2, 0.1]
        sections = [
            {
                "equivalent_radius_voxels": 3.0,
                "plane_material_fraction": 0.08,
                "cross_section_circularity": 0.9,
                "excess_area_fraction": 0.0,
                "contour_radius_cv": 0.05,
                "registration_residual_voxels": residual,
                "measurement_eligible": True,
                "dense_boundary_interference": False,
            }
            for residual in residuals
        ]
        result, reasons, _ = classify_strut(
            sections,
            rms_curvature=0.01,
            curvature_threshold=0.15,
            registration_metrics={
                "tracking_coverage": 1.0,
                "median_registration_offset_voxels": 5.0,
                "registration_offset_variation_voxels": 0.7,
            },
        )
        self.assertEqual(result, "potentially_bent")
        self.assertTrue(any("best-fit straight CT centerline" in reason
                            for reason in reasons))

    def test_shifted_continuous_strut_is_not_called_missing(self):
        sections, _, curvature = extract_cross_sections(
            cylinder_volume(),
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=10,
            grid_size=41,
            tracking_radius_voxels=6,
        )
        metrics = registration_metrics(sections)
        result, reasons, radius = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            min_broken_sections=3, registration_metrics=metrics,
        )
        self.assertEqual(result, "normal")
        self.assertGreater(metrics["tracking_coverage"], 0.95)
        self.assertGreater(metrics["median_registration_offset_voxels"], 3.0)
        self.assertGreater(radius, 2.0)
        self.assertTrue(any("registered centerline" in reason for reason in reasons))

    def test_coherent_large_registration_shift_is_measured_not_missing(self):
        volume = np.zeros((81, 56, 56), dtype=np.uint16)
        yy, xx = np.indices((56, 56))
        shifted = (xx - 28) ** 2 + (yy - 20) ** 2 <= 3 ** 2
        for z in range(5, 76):
            volume[z][shifted] = 1000
        sections, _, curvature = extract_cross_sections(
            volume,
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=12,
            grid_size=49,
            tracking_radius_voxels=6,
        )
        result, _, radius = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            min_broken_sections=3,
            registration_metrics=registration_metrics(sections),
        )
        self.assertNotIn("missing", result)
        self.assertGreater(registration_metrics(sections)["tracking_coverage"], 0.8)
        self.assertGreater(radius, 2.0)
        self.assertTrue(np.isfinite(sections[len(sections) // 2][
            "equivalent_radius_voxels"
        ]))

    def test_visible_material_in_isolated_path_gap_is_recovered(self):
        volume = np.zeros((81, 56, 56), dtype=np.uint16)
        yy, xx = np.indices((56, 56))
        for z in range(5, 76):
            center_x = 30 if 38 <= z <= 42 else 28
            disk = (xx - center_x) ** 2 + (yy - 20) ** 2 <= 3 ** 2
            volume[z][disk] = 1000
        sections, _, _ = extract_cross_sections(
            volume,
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=12,
            grid_size=49,
            tracking_radius_voxels=6,
        )
        middle = sections[len(sections) // 2]
        self.assertTrue(middle["tracking_recovered"])
        self.assertGreaterEqual(middle["tracking_confidence"], 0.45)
        self.assertTrue(np.isfinite(middle["equivalent_radius_voxels"]))
        self.assertGreater(sections[0]["tracking_coverage"], 0.95)

    def test_ambiguous_gap_recovery_remains_untracked(self):
        axis = np.linspace(-5.0, 5.0, 21)
        spacing = float(axis[1] - axis[0])
        yy, xx = np.indices((21, 21))
        left_disk = (xx - 6) ** 2 + (yy - 10) ** 2 <= 2 ** 2
        right_disk = (xx - 14) ** 2 + (yy - 10) ** 2 <= 2 ** 2
        masks = [
            np.zeros((21, 21), dtype=bool),
            left_disk | right_disk,
            np.zeros((21, 21), dtype=bool),
        ]
        anchor = {
            "centroid_u": 0.0,
            "centroid_v": 0.0,
            "area_pixels": 13,
            "circularity": 0.9,
        }
        path = [
            {"candidate": anchor, "confidence": 0.9},
            {"candidate": None, "confidence": 0.0},
            {"candidate": anchor, "confidence": 0.9},
        ]
        recovered = recover_bounded_path_gaps(
            path, masks, axis, spacing, expected_area_pixels=13,
            search_radius=6,
        )
        self.assertIsNone(recovered[1]["candidate"])

    def test_two_plane_gap_recovers_local_core_from_merged_material(self):
        axis = np.linspace(-5.0, 5.0, 21)
        spacing = float(axis[1] - axis[0])
        yy, xx = np.indices((21, 21))
        target = (xx - 10) ** 2 + (yy - 10) ** 2 <= 2 ** 2
        neighbor = (xx - 17) ** 2 + (yy - 10) ** 2 <= 2 ** 2
        bridge = (yy == 10) & (xx >= 12) & (xx <= 15)
        merged = target | bridge | neighbor
        masks = [
            np.zeros((21, 21), dtype=bool),
            merged,
            merged,
            np.zeros((21, 21), dtype=bool),
        ]
        anchor = {
            "centroid_u": 0.0,
            "centroid_v": 0.0,
            "area_pixels": 13,
            "circularity": 0.9,
        }
        path = [
            {"candidate": anchor, "confidence": 0.9},
            {"candidate": None, "confidence": 0.0},
            {"candidate": None, "confidence": 0.0},
            {"candidate": anchor, "confidence": 0.9},
        ]
        recovered = recover_bounded_path_gaps(
            path, masks, axis, spacing, expected_area_pixels=13,
            search_radius=6,
        )
        for item in recovered[1:3]:
            self.assertIsNotNone(item["candidate"])
            self.assertTrue(item["recovered"])
            self.assertGreaterEqual(item["confidence"], 0.45)
            self.assertLess(item["candidate"]["area_pixels"], merged.sum())

    def test_recovery_can_inspect_component_with_distant_merged_centroid(self):
        axis = np.linspace(-15.0, 15.0, 61)
        spacing = float(axis[1] - axis[0])
        yy, xx = np.indices((61, 61))
        target = (xx - 30) ** 2 + (yy - 30) ** 2 <= 2 ** 2
        neighbor = (xx - 52) ** 2 + (yy - 30) ** 2 <= 8 ** 2
        bridge = (yy == 30) & (xx >= 32) & (xx <= 44)
        merged = target | bridge | neighbor
        ordinary = enumerate_component_candidates(
            merged, axis, 4.5, spacing, search_uv=(0.0, 0.0)
        )
        recovery = enumerate_component_candidates(
            merged,
            axis,
            4.5,
            spacing,
            search_uv=(0.0, 0.0),
            allow_distant_centroid=True,
        )
        self.assertFalse(ordinary)
        self.assertEqual(len(recovery), 1)

    def test_consecutive_gap_is_called_possible_break(self):
        sections, _, curvature = extract_cross_sections(
            cylinder_volume(broken=True),
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=10,
            grid_size=41,
            tracking_radius_voxels=6,
        )
        result, reasons, _ = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            min_broken_sections=3,
            registration_metrics=registration_metrics(sections),
        )
        self.assertEqual(result, "potentially_broken_or_missing")
        self.assertTrue(any("consecutive" in reason for reason in reasons))
        self.assertTrue(any(
            not np.isfinite(section["equivalent_radius_voxels"])
            and not section["tracking_recovered"]
            for section in sections
        ))

    def test_empty_search_tube_is_called_possible_missing(self):
        sections, _, curvature = extract_cross_sections(
            np.zeros((81, 48, 48), dtype=np.uint16),
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=10,
            grid_size=41,
            tracking_radius_voxels=6,
        )
        result, _, _ = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            min_broken_sections=3,
            registration_metrics=registration_metrics(sections),
        )
        self.assertEqual(result, "potentially_missing")

    def test_junction_radius_flare_is_excluded_from_dross_decision(self):
        sections, _, curvature = extract_cross_sections(
            cylinder_volume(junction_flare=True),
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=12,
            grid_size=49,
            tracking_radius_voxels=6,
        )
        result, reasons, _ = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            registration_metrics=registration_metrics(sections),
        )
        self.assertEqual(result, "normal")
        self.assertTrue(sections[0]["junction_excluded"])
        self.assertTrue(sections[-1]["junction_excluded"])
        self.assertFalse(any("high-radius outliers" in reason for reason in reasons))

    def test_global_tracker_does_not_jump_to_neighboring_strut(self):
        volume = np.zeros((81, 56, 56), dtype=np.uint16)
        yy, xx = np.indices((56, 56))
        target = (xx - 21) ** 2 + (yy - 20) ** 2 <= 3 ** 2
        neighbor = (xx - 29) ** 2 + (yy - 20) ** 2 <= 3 ** 2
        for z in range(5, 76):
            volume[z][target | neighbor] = 1000
        sections, _, _ = extract_cross_sections(
            volume,
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=12,
            grid_size=49,
            tracking_radius_voxels=8,
        )
        confident = [
            section for section in sections
            if section["tracking_confidence"] >= 0.55
            and not section["junction_excluded"]
        ]
        self.assertGreaterEqual(len(confident), 7)
        self.assertLess(
            max(abs(section["centroid_v_voxels"]) for section in confident),
            3.0,
        )

    def test_coherent_neighbor_outside_registration_limit_is_not_adopted(self):
        volume = np.zeros((81, 56, 56), dtype=np.uint16)
        yy, xx = np.indices((56, 56))
        neighbor = (xx - 29) ** 2 + (yy - 20) ** 2 <= 3 ** 2
        for z in range(5, 76):
            volume[z][neighbor] = 1000
        sections, _, _ = extract_cross_sections(
            volume,
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.1, 0.9, 11),
            extent=12,
            grid_size=49,
            tracking_radius_voxels=6,
        )
        self.assertTrue(all(
            not np.isfinite(section["equivalent_radius_voxels"])
            for section in sections
        ))

    def test_dense_boundary_slab_is_excluded_from_tracking_and_radius(self):
        volume = cylinder_volume()
        volume[:22] = 1000
        safe_min, safe_max, _ = detect_dense_boundary_limits(
            volume, threshold=500, sample_stride=2,
            density_threshold=0.12, consecutive=3, padding_slices=1,
        )
        self.assertGreaterEqual(safe_min, 22)
        sections, _, curvature = extract_cross_sections(
            volume,
            np.array([20.0, 20.0, 5.0]),
            np.array([20.0, 20.0, 75.0]),
            threshold=500,
            positions=np.linspace(0.0, 0.9, 11),
            extent=10,
            grid_size=41,
            tracking_radius_voxels=6,
            valid_z_range=(safe_min, safe_max),
        )
        affected = [
            section for section in sections
            if section["dense_boundary_interference"]
        ]
        measured = [
            section for section in sections
            if section["measurement_eligible"]
        ]
        self.assertTrue(affected)
        self.assertTrue(all(not section["measurement_eligible"] for section in affected))
        self.assertTrue(all(not np.isfinite(section["equivalent_radius_voxels"])
                            for section in affected))
        self.assertTrue(measured)
        self.assertLess(max(section["equivalent_radius_voxels"]
                            for section in measured), 5.0)
        result, reasons, radius = classify_strut(
            sections, curvature, curvature_threshold=0.15,
            registration_metrics=registration_metrics(sections),
        )
        self.assertNotIn("thick", result)
        self.assertLess(radius, 5.0)

if __name__ == "__main__":
    unittest.main()
