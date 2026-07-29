import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strut_defect_pipeline import (  # noqa: E402
    SECTION_FIELDS,
    SUMMARY_FIELDS,
    classify_struts,
    compute_strut_metrics,
    render_strut_evidence,
    run_pipeline,
)
from strut_cross_section_viewer import (  # noqa: E402
    make_basis,
    recover_refined_tangent_gaps_from_3d_component,
    sample_nearest,
)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_row(strut_id, radius, *, bent=False):
    return {
        "strut_id": strut_id,
        "unit_cell_edge_idx": 1,
        "peer_group_id": "unit_cell_edge_1",
        "junction0": strut_id * 2,
        "junction1": strut_id * 2 + 1,
        "length_voxels": 60,
        "length_mm": "",
        "midpoint_x_voxels": 10 + strut_id,
        "midpoint_y_voxels": 20,
        "midpoint_z_voxels": 40,
        "orientation_x": 0,
        "orientation_y": 0,
        "orientation_z": 1,
        "valid_sample_count": 9,
        "median_radius_voxels": radius,
        "min_radius_voxels": radius * 0.95,
        "max_radius_voxels": radius * 1.05,
        "median_radius_mm": "",
        "interior_radius_cv": 0.04,
        "tracking_coverage": 1.0,
        "mean_tracking_confidence": 0.95,
        "junction_contamination_fraction": 0.0,
        "dense_boundary_interference_fraction": 0.0,
        "median_registration_offset_voxels": 0.5,
        "centerline_deviation_rms_voxels": 1.0 if bent else 0.1,
        "centerline_deviation_max_voxels": 2.0 if bent else 0.2,
        "curvature_rms_inverse_voxels": 0.2 if bent else 0.01,
        "max_turn_angle_degrees": 8.0 if bent else 1.0,
        "continuity_status": "continuous",
        "same_component_connects_collar_a_to_b": True,
        "endpoint0_support_fraction": 0.20,
        "endpoint1_support_fraction": 0.20,
        "maximum_axial_gap_samples": 0,
        "maximum_axial_gap_fraction": 0.0,
        "continuity_corridor_radius_voxels": 5.0,
        "continuity_sample_count": 49,
        "measurement_quality": "usable",
    }


def section_rows(strut_id, radius, *, bent=False):
    rows = []
    deviations = [0.1, 0.2, 0.8, 1.7, 2.0, 1.7, 0.8, 0.2, 0.1]
    for index, fraction in enumerate(np.linspace(0.1, 0.9, 9)):
        deviation = deviations[index] if bent else 0.1
        rows.append({
            "strut_id": strut_id,
            "sample_index": index,
            "axis_fraction": fraction,
            "distance_voxels": fraction * 60,
            "radius_voxels": radius,
            "radius_mm": "",
            "area_voxels_squared": np.pi * radius * radius,
            "center_x_voxels": 10,
            "center_y_voxels": 20,
            "center_z_voxels": fraction * 60,
            "tracked_center_u_voxels": deviation,
            "tracked_center_v_voxels": 0,
            "centerline_deviation_voxels": deviation,
            "centerline_deviation_mm": "",
            "curvature_inverse_voxels": 0.2 if bent else 0.01,
            "tracking_confidence": 0.95,
            "valid": True,
            "exclusion_reason": "",
            "junction_excluded": False,
            "junction_contaminated": False,
            "dense_boundary_interference": False,
        })
    return rows


class ThinThickBentPipelineTests(unittest.TestCase):
    def test_3d_component_recovery_requires_connected_flanks(self):
        volume = np.zeros((31, 31, 31), dtype=np.uint16)
        yy_volume, xx_volume = np.indices((31, 31))
        disk = (xx_volume - 15) ** 2 + (yy_volume - 15) ** 2 <= 2 ** 2
        for z_index in range(5, 26):
            volume[z_index][disk] = 1000

        start = np.asarray([15.0, 15.0, 5.0])
        end = np.asarray([15.0, 15.0, 25.0])
        direction, basis_u, basis_v, _ = make_basis(start, end)
        axis = np.linspace(-5.0, 5.0, 21)
        uu, vv = np.meshgrid(axis, axis, indexing="xy")
        centers = [
            np.asarray([15.0, 15.0, z_value])
            for z_value in (10.0, 15.0, 20.0)
        ]
        planes = []
        records = []
        for index, center in enumerate(centers):
            xyz = (
                center
                + uu[..., None] * basis_u
                + vv[..., None] * basis_v
            )
            intensities = sample_nearest(
                volume, xyz.reshape(-1, 3)
            ).reshape(len(axis), len(axis))
            mask = intensities >= 500
            selected = (
                np.zeros_like(mask, dtype=bool) if index == 1 else mask
            )
            records.append({
                "selected": selected,
                "sampling_plane_center": center,
                "local_tangent": direction,
                "local_u": basis_u,
                "local_v": basis_v,
                "tracked_center": (
                    np.full(3, np.nan) if index == 1 else center
                ),
                "tracking_confidence": 0.9 if index != 1 else 0.0,
                "dense_boundary_interference": False,
            })
            planes.append((intensities, mask))

        recovered, _ = recover_refined_tangent_gaps_from_3d_component(
            volume,
            500,
            centers,
            basis_u,
            basis_v,
            records,
            planes,
            axis,
            uu,
            vv,
            float(axis[1] - axis[0]),
            int(np.count_nonzero(planes[0][1])),
            6.0,
        )
        self.assertTrue(np.any(recovered[1]["selected"]))
        self.assertTrue(recovered[1]["tracking_recovered"])
        self.assertEqual(
            recovered[1]["tracking_method"],
            "3d_centerline_local_tangent_component_recovery",
        )

        broken = volume.copy()
        broken[14:17] = 0
        not_recovered, _ = recover_refined_tangent_gaps_from_3d_component(
            broken,
            500,
            centers,
            basis_u,
            basis_v,
            records,
            planes,
            axis,
            uu,
            vv,
            float(axis[1] - axis[0]),
            int(np.count_nonzero(planes[0][1])),
            6.0,
        )
        self.assertFalse(np.any(not_recovered[1]["selected"]))

    def test_measurement_and_complete_handoff_run_without_agents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = np.zeros((81, 48, 48), dtype=np.uint16)
            yy, xx = np.indices((48, 48))
            disk = (xx - 24) ** 2 + (yy - 22) ** 2 <= 3 ** 2
            for z in range(5, 76):
                volume[z][disk] = 1000
            tiff_path = root / "fixture.tif"
            json_path = root / "registered.json"
            tifffile.imwrite(tiff_path, volume)
            json_path.write_text(json.dumps({
                "junctions": [
                    {"id": 0, "position": [20.0, 20.0, 5.0]},
                    {"id": 1, "position": [20.0, 20.0, 75.0]},
                ],
                "struts": [{
                    "id": 0,
                    "unit_cell_edge_idx": 1,
                    "junction0": 0,
                    "junction1": 1,
                    "thickness": 0.1,
                }],
                "unit_cells": [],
            }), encoding="utf-8")

            measurement = compute_strut_metrics(
                tiff_path, json_path, root / "metrics_only", 500,
            )
            self.assertEqual(measurement["strut_count"], 1)
            self.assertEqual(measurement["section_count"], 21)
            with (root / "metrics_only" / "strut_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                measured = next(csv.DictReader(handle))
            self.assertGreater(float(measured["median_radius_voxels"]), 2.0)
            self.assertEqual(measured["measurement_quality"], "usable")
            self.assertEqual(measured["continuity_status"], "continuous")
            self.assertEqual(
                measured["same_component_connects_collar_a_to_b"], "True"
            )
            with (root / "metrics_only" / "strut_section_measurements.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                measured_section = next(csv.DictReader(handle))
            self.assertEqual(
                measured_section["tracking_method"],
                "3d_centerline_local_tangent",
            )
            self.assertNotEqual(measured_section["local_tangent_z"], "")
            measurement_manifest = json.loads(
                (root / "metrics_only" / "measurement_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                measurement_manifest["config"]["tracking_method"],
                "3d_centerline_local_tangent",
            )
            self.assertEqual(measurement_manifest["config"]["positions"], 21)

            result = run_pipeline(
                tiff_path, json_path, root / "complete", 500,
            )
            self.assertEqual(result["status"], "ready")
            handoff = json.loads(
                (root / "complete" / "classification_handoff.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                handoff["self_verification"]["all_required_artifacts_exist"]
            )
            self.assertEqual(handoff["scope"], ["thin", "thick", "bent"])

    def test_measurement_marks_broken_cylinder_noncontinuous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = np.zeros((81, 48, 48), dtype=np.uint16)
            yy, xx = np.indices((48, 48))
            disk = (xx - 24) ** 2 + (yy - 22) ** 2 <= 3 ** 2
            for z in range(5, 76):
                if not 36 <= z <= 44:
                    volume[z][disk] = 1000
            tiff_path = root / "broken_fixture.tif"
            json_path = root / "registered.json"
            tifffile.imwrite(tiff_path, volume)
            json_path.write_text(json.dumps({
                "junctions": [
                    {"id": 0, "position": [20.0, 20.0, 5.0]},
                    {"id": 1, "position": [20.0, 20.0, 75.0]},
                ],
                "struts": [{
                    "id": 0,
                    "unit_cell_edge_idx": 1,
                    "junction0": 0,
                    "junction1": 1,
                }],
                "unit_cells": [],
            }), encoding="utf-8")
            compute_strut_metrics(
                tiff_path, json_path, root / "metrics", 500,
            )
            with (root / "metrics" / "strut_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                measured = next(csv.DictReader(handle))
            self.assertEqual(
                measured["continuity_status"], "noncontinuous"
            )
            self.assertEqual(
                measured["same_component_connects_collar_a_to_b"], "False"
            )
            self.assertGreaterEqual(
                int(measured["maximum_axial_gap_samples"]), 3
            )

    def test_classification_and_class_specific_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "strut_summary.csv"
            sections_path = root / "strut_section_measurements.csv"
            summary = [summary_row(index, 3.0) for index in range(5)]
            summary.extend([
                summary_row(5, 1.5),
                summary_row(6, 5.0),
                summary_row(7, 3.0, bent=True),
            ])
            sections = []
            for index in range(5):
                sections.extend(section_rows(index, 3.0))
            sections.extend(section_rows(5, 1.5))
            sections.extend(section_rows(6, 5.0))
            sections.extend(section_rows(7, 3.0, bent=True))
            write_csv(summary_path, SUMMARY_FIELDS, summary)
            write_csv(sections_path, SECTION_FIELDS, sections)
            (root / "measurement_manifest.json").write_text(json.dumps({
                "config": {
                    "threshold": 500,
                    "positions": 9,
                    "start_fraction": 0.1,
                    "end_fraction": 0.9,
                    "tracking_radius_voxels": 6,
                },
                "artifacts": {
                    "section_measurements": {
                        "sha256": "fixture-section-hash",
                    },
                },
            }), encoding="utf-8")
            thresholds_path = root / "test_thresholds.json"
            thresholds_path.write_text(json.dumps({
                "minimum_peer_group_size": 4,
                "minimum_valid_samples": 6,
                "bent_minimum_adjacent_samples": 2,
            }), encoding="utf-8")

            classification_dir = root / "classification"
            classify_struts(
                summary_path,
                sections_path,
                classification_dir,
                thresholds_json=thresholds_path,
            )
            with (classification_dir / "classified_struts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                classified = {
                    int(row["strut_id"]): row for row in csv.DictReader(handle)
                }
            self.assertEqual(classified[5]["classification"], "thin")
            self.assertEqual(classified[6]["classification"], "thick")
            self.assertEqual(classified[7]["classification"], "bent")
            self.assertEqual(
                classified[7]["continuous_for_shape_classification"], "True"
            )

            evidence_dir = root / "evidence"
            rendered = render_strut_evidence(
                classification_dir / "classified_struts.csv",
                sections_path,
                evidence_dir,
                thresholds_json=classification_dir / "thresholds.json",
            )
            self.assertEqual(rendered["plot_count"], 3)
            self.assertTrue((evidence_dir / "thin" / "strut_5_radius.png").is_file())
            self.assertTrue((evidence_dir / "thick" / "strut_6_radius.png").is_file())
            self.assertTrue(
                (evidence_dir / "bent" / "strut_7_centerline.png").is_file()
            )
            with (classification_dir / "classified_struts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rendered_rows = {
                    int(row["strut_id"]): row for row in csv.DictReader(handle)
                }
            self.assertEqual(
                rendered_rows[7]["evidence_png"],
                "../evidence/bent/strut_7_centerline.png",
            )
            bent_findings = json.loads(
                (classification_dir / "findings_bent.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                bent_findings["findings"][0]["evidence_png"],
                "../evidence/bent/strut_7_centerline.png",
            )
            embedded = bent_findings["findings"][0]["measurement_profile"]
            self.assertEqual(embedded["source"], "thin_thick_bent_pipeline")
            self.assertEqual(embedded["ct_threshold"], 500.0)
            self.assertEqual(
                embedded["section_measurements_sha256"],
                "fixture-section-hash",
            )
            self.assertEqual(len(embedded["samples"]), 9)
            self.assertEqual(embedded["samples"][0]["fraction"], 0.1)
            self.assertIn("deviation_voxels", embedded["samples"][0])

    def test_bent_primary_label_wins_close_radius_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "strut_summary.csv"
            sections_path = root / "strut_section_measurements.csv"
            summary = [summary_row(index, 3.0) for index in range(5)]
            competing = summary_row(5, 1.94, bent=True)
            competing["centerline_deviation_rms_voxels"] = 0.74
            competing["centerline_deviation_max_voxels"] = 1.40
            competing["curvature_rms_inverse_voxels"] = 0.14
            summary.append(competing)
            sections = [
                row
                for index in range(5)
                for row in section_rows(index, 3.0)
            ]
            sections.extend(section_rows(5, 1.94, bent=True))
            write_csv(summary_path, SUMMARY_FIELDS, summary)
            write_csv(sections_path, SECTION_FIELDS, sections)
            thresholds_path = root / "thresholds.json"
            thresholds_path.write_text(json.dumps({
                "minimum_peer_group_size": 4,
                "minimum_valid_samples": 6,
                "bent_minimum_adjacent_samples": 2,
            }), encoding="utf-8")
            output = root / "classification"
            classify_struts(
                summary_path,
                sections_path,
                output,
                thresholds_json=thresholds_path,
            )
            with (output / "classified_struts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(
                    row for row in csv.DictReader(handle)
                    if row["strut_id"] == "5"
                )
            self.assertEqual(row["classification"], "bent")
            self.assertEqual(row["is_thin"], "True")
            self.assertEqual(row["is_bent"], "True")
            self.assertIn(
                "near-threshold evidence strength",
                row["classification_priority_reason"],
            )

    def test_bent_can_be_classified_when_only_radius_cv_gate_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "strut_summary.csv"
            sections_path = root / "strut_section_measurements.csv"
            summary = [summary_row(index, 3.0) for index in range(5)]
            bent = summary_row(5, 3.0, bent=True)
            bent["interior_radius_cv"] = 0.40
            summary.append(bent)
            sections = [
                row
                for index in range(5)
                for row in section_rows(index, 3.0)
            ]
            sections.extend(section_rows(5, 3.0, bent=True))
            write_csv(summary_path, SUMMARY_FIELDS, summary)
            write_csv(sections_path, SECTION_FIELDS, sections)
            thresholds_path = root / "thresholds.json"
            thresholds_path.write_text(json.dumps({
                "minimum_peer_group_size": 4,
                "minimum_valid_samples": 6,
                "bent_minimum_adjacent_samples": 2,
            }), encoding="utf-8")
            output = root / "classification"
            classify_struts(
                summary_path,
                sections_path,
                output,
                thresholds_json=thresholds_path,
            )
            with (output / "classified_struts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(
                    row for row in csv.DictReader(handle)
                    if row["strut_id"] == "5"
                )
            self.assertEqual(row["classification"], "bent")
            self.assertEqual(row["is_bent"], "True")

    def test_noncontinuous_strut_is_not_labeled_thin_thick_or_bent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "strut_summary.csv"
            sections_path = root / "strut_section_measurements.csv"
            summary = [summary_row(index, 3.0) for index in range(5)]
            disconnected = summary_row(5, 1.5, bent=True)
            disconnected.update({
                "continuity_status": "noncontinuous",
                "same_component_connects_collar_a_to_b": False,
                "endpoint0_support_fraction": 0.20,
                "endpoint1_support_fraction": 0.20,
                "maximum_axial_gap_samples": 3,
                "maximum_axial_gap_fraction": 0.10,
            })
            summary.append(disconnected)
            sections = [
                row
                for index in range(5)
                for row in section_rows(index, 3.0)
            ]
            sections.extend(section_rows(5, 1.5, bent=True))
            write_csv(summary_path, SUMMARY_FIELDS, summary)
            write_csv(sections_path, SECTION_FIELDS, sections)
            thresholds_path = root / "thresholds.json"
            thresholds_path.write_text(json.dumps({
                "minimum_peer_group_size": 4,
                "minimum_valid_samples": 6,
                "bent_minimum_adjacent_samples": 2,
            }), encoding="utf-8")
            output = root / "classification"
            classify_struts(
                summary_path,
                sections_path,
                output,
                thresholds_json=thresholds_path,
            )
            with (output / "classified_struts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(
                    item for item in csv.DictReader(handle)
                    if item["strut_id"] == "5"
                )
            self.assertEqual(row["classification"], "uncertain")
            self.assertEqual(row["decision_status"], "excluded_noncontinuous")
            self.assertEqual(row["is_thin"], "False")
            self.assertEqual(row["is_thick"], "False")
            self.assertEqual(row["is_bent"], "False")
            self.assertEqual(
                row["continuous_for_shape_classification"], "False"
            )
            self.assertIn(
                "classification suppressed", row["reason"]
            )


if __name__ == "__main__":
    unittest.main()
