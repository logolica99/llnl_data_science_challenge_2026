"""Stage 3 missing/broken development slice and MCP-boundary tests."""

from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np
from fastmcp import Client
from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.server import mcp  # noqa: E402
from research.mcp_server import mcp as research_mcp  # noqa: E402
from llnl_nde.core.artifacts import sha256_file, sha256_json  # noqa: E402
from llnl_nde.core.defect_analysis import (  # noqa: E402
    DEFAULT_STAGE3_CONFIG,
    analyze_strut_specialist,
    merge_strut_classifications,
    verify_strut_classifications,
)
from llnl_nde.core.evidence import render_strut_evidence  # noqa: E402
from llnl_nde.core.strut_metrics import METRIC_FIELDS  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Stage3DefectAnalysisTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.specimen_id = f"stage3_test_{uuid.uuid4().hex}"
        self.analysis = REPOSITORY_ROOT / "analysis" / self.specimen_id
        self.struts = self.analysis / "struts"
        self.struts.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(self.analysis, ignore_errors=True))
        self.validation_output = (
            REPOSITORY_ROOT / "research" / "runs" / "stage3_validation" / self.specimen_id
        )
        self.addCleanup(
            lambda: shutil.rmtree(self.validation_output, ignore_errors=True)
        )

        self.config = self.analysis / "config" / "analysis_config.json"
        _write_json(
            self.config,
            {
                "schema_version": "part2-analysis-config/1.0.0",
                "registration_mode": "autonomous_v2",
                "threshold": 1.0,
                "stage_3_defect_analysis": copy.deepcopy(DEFAULT_STAGE3_CONFIG),
                "label_inputs_accessed": False,
            },
        )
        self.metrics = self.struts / "per_strut_metrics.csv"
        rows = []
        for identifier, connected in ((1, False), (2, False), (3, True), (4, True)):
            row = {field: 0 for field in METRIC_FIELDS}
            row.update(
                {
                    "strut_id": identifier,
                    "junction0_id": identifier * 10,
                    "junction1_id": identifier * 10 + 1,
                    "length_voxels": 10.0,
                    "corridor_radius_voxels": 4.0,
                    "cuboid_half_width_voxels": 6.0,
                    "axial_padding_fraction_total": 0.2,
                    "corridor_foreground_fraction": 0.5,
                    "minimum_foreground_fraction": 0.0,
                    "median_foreground_fraction": 0.8,
                    "maximum_foreground_fraction": 0.9,
                    "maximum_axial_gap_samples": 0,
                    "maximum_axial_gap_fraction": 0.0,
                    "endpoint0_support_fraction": 0.8,
                    "endpoint1_support_fraction": 0.8,
                    "a_collar_foreground_fraction": 0.2,
                    "b_collar_foreground_fraction": 0.2,
                    "interior_component_count": 1,
                    "largest_component_fraction": 0.8,
                    "same_material_component_connects_a_to_b": connected,
                    "same_component_connects_collar_a_to_b": connected,
                    "shared_component_voxel_count_in_corridor": (
                        1000 if connected else 0
                    ),
                    "endpoint0_to_collar_component_voxel_count_in_corridor": 1000,
                    "endpoint1_to_collar_component_voxel_count_in_corridor": 1000,
                    "both_endpoint_segments_observed": True,
                    "junction_masked_collar_shared_component_voxel_count_in_corridor": 800,
                    "edt_radius_median_voxels": 2.0,
                    "centerline_curvature_rms_voxels": 0.1,
                    "roi_in_bounds_fraction": 1.0,
                    "roi_valid": True,
                }
            )
            rows.append(row)
        with self.metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        axial_t = np.linspace(-0.1, 1.1, 13).tolist()
        profiles = {
            1: [0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            2: [0.8, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            3: [0.8, 0.8, 0.8, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            4: [0.8] * 13,
        }
        self.profiles = self.struts / "per_strut_profiles.json"
        _write_json(
            self.profiles,
            {
                "schema_version": "part2-strut-metrics/2.0.0",
                "measurement_only": True,
                "classification_performed": False,
                "profiles": [
                    {
                        "strut_id": identifier,
                        "axial_t": axial_t,
                        "foreground_fraction": values,
                        "occupancy_profile": values,
                    }
                    for identifier, values in profiles.items()
                ],
            },
        )
        self.calibration = self.struts / "corridor_calibration.json"
        _write_json(
            self.calibration,
            {
                "schema_version": "part2-corridor-calibration/1.0.0",
                "label_blind": True,
                "corridor_radius_voxels": 4.0,
            },
        )
        self.ct = self.analysis / "inputs" / "ct.npy"
        self.ct.parent.mkdir(parents=True)
        np.save(self.ct, np.ones((32, 32, 32), dtype=np.uint16))
        self.graph = self.analysis / "registration" / "localized_graph.json"
        junctions = []
        struts = []
        for identifier in range(1, 5):
            junctions.extend(
                [
                    {"id": identifier * 10, "position": [5, 5 + 5 * identifier, 10]},
                    {"id": identifier * 10 + 1, "position": [15, 5 + 5 * identifier, 10]},
                ]
            )
            struts.append(
                {
                    "id": identifier,
                    "junction0": identifier * 10,
                    "junction1": identifier * 10 + 1,
                }
            )
        _write_json(
            self.graph,
            {
                "junctions": junctions,
                "struts": struts,
                "unit_cells": [
                    {"id": 1, "indices": [0, 0, 0], "struts": [1, 2, 3, 4]}
                ],
            },
        )
        self.nominal_graph = self.analysis / "inputs" / "nominal_graph.json"
        nominal_junctions = []
        for identifier in range(1, 5):
            y0 = 18.0 if identifier == 1 else float(identifier * 2)
            nominal_junctions.extend(
                [
                    {
                        "id": identifier * 10,
                        "position": [float(identifier * 2), y0, 0.0],
                    },
                    {
                        "id": identifier * 10 + 1,
                        "position": [float(identifier * 2 + 1), float(identifier * 2 + 1), 1.0],
                    },
                ]
            )
        _write_json(
            self.nominal_graph,
            {
                "junctions": nominal_junctions,
                "struts": struts,
                "unit_cells": [
                    {"id": 1, "indices": [0, 0, 0], "struts": [1, 2, 3, 4]}
                ],
            },
        )
        self.handoff = (
            self.analysis / "handoffs" / "stage_3_defect_analysis_attempt_1.json"
        )
        contract = REPOSITORY_ROOT / "analysis" / "contracts" / "defect_analysis.json"
        contract_document = json.loads(contract.read_text(encoding="utf-8"))
        artifacts = []
        for role, path in (
            ("analysis_config", self.config),
            ("corridor_calibration", self.calibration),
            ("per_strut_metrics", self.metrics),
            ("per_strut_profiles", self.profiles),
            ("localized_graph", self.graph),
            ("ct_volume", self.ct),
        ):
            artifacts.append(
                {
                    "role": role,
                    "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        handoff = {
            "schema_version": "part2-stage-handoff/1.0.0",
            "specimen_id": self.specimen_id,
            "stage_number": 3,
            "stage": "defect_analysis",
            "owner": "defect_lead",
            "attempt": 1,
            "run_token": "stage3-test-token",
            "created_at": "2026-07-29T00:00:00Z",
            "registration_mode": "autonomous_v2",
            "config_sha256": sha256_file(self.config),
            "contract_version": contract_document["schema_version"],
            "contract_sha256": sha256_file(contract),
            "predecessor_receipt_sha256": "b" * 64,
            "input_artifacts": artifacts,
            "forbidden_operations": contract_document["forbidden_operations"],
        }
        handoff["canonical_handoff_sha256"] = sha256_json(handoff)
        _write_json(self.handoff, handoff)

    def test_common_schema_and_prior_missing_broken_math(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        finding_paths = {
            kind: self.struts / f"unit_findings_{kind}.json"
            for kind in ("missing", "broken", "thin", "bent")
        }
        for kind in finding_paths:
            analyze_strut_specialist(
                self.metrics,
                self.profiles,
                config,
                finding_paths[kind],
                specimen_id=self.specimen_id,
                defect_kind=kind,
            )
        missing = json.loads(finding_paths["missing"].read_text(encoding="utf-8"))
        broken = json.loads(finding_paths["broken"].read_text(encoding="utf-8"))
        self.assertEqual("positive", missing["findings"][0]["disposition"])
        broken_by_id = {row["strut_id"]: row for row in broken["findings"]}
        self.assertEqual("negative", broken_by_id[1]["disposition"])
        self.assertEqual("positive", broken_by_id[2]["disposition"])
        self.assertEqual("positive", broken_by_id[3]["disposition"])
        self.assertIn("connected_bite_case", broken_by_id[3]["reasons"])

        result = merge_strut_classifications(
            self.metrics,
            self.profiles,
            config,
            finding_paths,
            self.struts / "unit_classified.json",
            self.struts / "unit_thresholds.json",
            self.struts / "unit_decision_log.md",
            specimen_id=self.specimen_id,
        )
        by_id = {row["strut_id"]: row for row in result["classifications"]}
        self.assertEqual("manual_review", result["gate"])
        self.assertEqual("missing", by_id[1]["class"])
        self.assertEqual("broken", by_id[2]["class"])
        self.assertEqual("broken", by_id[3]["class"])
        self.assertEqual("deferred", by_id[4]["class"])
        self.assertIsNone(by_id[1]["bent"])

        schema = json.loads(
            (REPOSITORY_ROOT / "analysis/schema/classified_struts.schema.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema).validate(
            json.loads((self.struts / "unit_classified.json").read_text(encoding="utf-8"))
        )

    def test_unresolved_disconnection_requires_review_evidence(self) -> None:
        review_metrics = self.struts / "review_metrics.csv"
        with self.metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if int(row["strut_id"]) == 2:
                row["endpoint0_to_collar_component_voxel_count_in_corridor"] = "0"
                row["endpoint1_to_collar_component_voxel_count_in_corridor"] = "0"
                row["both_endpoint_segments_observed"] = "False"
        with review_metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        config = json.loads(self.config.read_text(encoding="utf-8"))
        finding_paths = {
            kind: self.struts / f"review_findings_{kind}.json"
            for kind in ("missing", "broken", "thin", "bent")
        }
        for kind in finding_paths:
            analyze_strut_specialist(
                review_metrics,
                self.profiles,
                config,
                finding_paths[kind],
                specimen_id=self.specimen_id,
                defect_kind=kind,
            )
        result = merge_strut_classifications(
            review_metrics,
            self.profiles,
            config,
            finding_paths,
            self.struts / "review_classified.json",
            self.struts / "review_thresholds.json",
            self.struts / "review_decision_log.md",
            specimen_id=self.specimen_id,
        )
        row = next(item for item in result["classifications"] if item["strut_id"] == 2)
        self.assertEqual("deferred", row["class"])
        self.assertTrue(row["evidence_required"])
        self.assertIn("specialist_review_required", row["reasons"])

    async def test_mcp_partial_team_is_manual_review_and_renders_local_evidence(self) -> None:
        base_arguments = {
            "stage_3_handoff_filepath": self.handoff.relative_to(REPOSITORY_ROOT).as_posix(),
            "analysis_config_filepath": self.config.relative_to(REPOSITORY_ROOT).as_posix(),
            "corridor_calibration_filepath": self.calibration.relative_to(REPOSITORY_ROOT).as_posix(),
            "metrics_filepath": self.metrics.relative_to(REPOSITORY_ROOT).as_posix(),
            "profiles_filepath": self.profiles.relative_to(REPOSITORY_ROOT).as_posix(),
            "localized_graph_filepath": self.graph.relative_to(REPOSITORY_ROOT).as_posix(),
            "ct_filepath": self.ct.relative_to(REPOSITORY_ROOT).as_posix(),
            "output_directory": self.struts.relative_to(REPOSITORY_ROOT).as_posix(),
        }
        async with Client(mcp) as client:
            for kind in ("missing", "broken", "thin", "bent"):
                call = await client.call_tool(
                    "classify_struts",
                    {**base_arguments, "operation": f"analyze_{kind}"},
                )
                result = call.structured_content
                self.assertEqual("ok", result["status"], result)
                expected_gate = "pass" if kind in {"missing", "broken"} else "manual_review"
                self.assertEqual(expected_gate, result["gate"])

            merged_call = await client.call_tool(
                "classify_struts", {**base_arguments, "operation": "merge"}
            )
            merged = merged_call.structured_content
            self.assertEqual("ok", merged["status"], merged)
            self.assertEqual("manual_review", merged["gate"])

            evidence_call = await client.call_tool(
                "render_strut_evidence",
                {
                    "stage_3_handoff_filepath": base_arguments["stage_3_handoff_filepath"],
                    "analysis_config_filepath": base_arguments["analysis_config_filepath"],
                    "corridor_calibration_filepath": base_arguments[
                        "corridor_calibration_filepath"
                    ],
                    "ct_filepath": base_arguments["ct_filepath"],
                    "localized_graph_filepath": base_arguments[
                        "localized_graph_filepath"
                    ],
                    "metrics_filepath": base_arguments["metrics_filepath"],
                    "profiles_filepath": base_arguments["profiles_filepath"],
                    "classifications_filepath": (
                        self.struts / "classified_struts.json"
                    ).relative_to(REPOSITORY_ROOT).as_posix(),
                    "thresholds_filepath": (self.struts / "thresholds.json")
                    .relative_to(REPOSITORY_ROOT)
                    .as_posix(),
                    "output_directory": (self.analysis / "evidence")
                    .relative_to(REPOSITORY_ROOT)
                    .as_posix(),
                    "strut_id": 1,
                },
            )
        evidence = evidence_call.structured_content
        self.assertEqual("ok", evidence["status"], evidence)
        self.assertEqual("pass", evidence["gate"])
        manifest = json.loads(
            (self.analysis / "evidence" / "strut_1" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(manifest["local_frame"]["local_z_is_a_to_b"])
        self.assertFalse(manifest["provenance"]["metrics_recomputed"])
        self.assertFalse(manifest["provenance"]["classification_recomputed"])
        self.assertIn("aligned_xz", manifest["artifacts"])

    async def test_mcp_exports_non_authoritative_missing_broken_validation_csvs(self) -> None:
        base_arguments = {
            "stage_3_handoff_filepath": self.handoff.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "analysis_config_filepath": self.config.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "corridor_calibration_filepath": self.calibration.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "metrics_filepath": self.metrics.relative_to(REPOSITORY_ROOT).as_posix(),
            "profiles_filepath": self.profiles.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "localized_graph_filepath": self.graph.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "ct_filepath": self.ct.relative_to(REPOSITORY_ROOT).as_posix(),
            "output_directory": self.struts.relative_to(REPOSITORY_ROOT).as_posix(),
        }
        async with Client(mcp) as client:
            for kind in ("missing", "broken", "thin", "bent"):
                call = await client.call_tool(
                    "classify_struts",
                    {**base_arguments, "operation": f"analyze_{kind}"},
                )
                self.assertEqual("ok", call.structured_content["status"])
            merged_call = await client.call_tool(
                "classify_struts",
                {**base_arguments, "operation": "merge"},
            )
            self.assertEqual("manual_review", merged_call.structured_content["gate"])
        research_inputs = self.validation_output / "inputs"
        research_inputs.mkdir(parents=True)
        copied = {}
        for name, source in {
            "classifications": self.struts / "classified_struts.json",
            "missing": self.struts / "findings_missing.json",
            "broken": self.struts / "findings_broken.json",
            "metrics": self.metrics,
            "nominal_graph": self.nominal_graph,
        }.items():
            destination = research_inputs / source.name
            shutil.copy2(source, destination)
            copied[name] = destination
        export_directory = self.validation_output / "exports"
        async with Client(research_mcp) as client:
            export_call = await client.call_tool(
                "export_stage3_validation_csvs",
                {
                    "classifications_filepath": str(copied["classifications"]),
                    "missing_findings_filepath": str(copied["missing"]),
                    "broken_findings_filepath": str(copied["broken"]),
                    "metrics_filepath": str(copied["metrics"]),
                    "nominal_graph_filepath": str(copied["nominal_graph"]),
                    "output_directory": str(export_directory),
                    "excluded_nominal_axis": "y",
                    "excluded_nominal_value": 18.0,
                },
            )
        result = export_call.structured_content
        self.assertEqual("ok", result["status"], result)
        self.assertEqual("pass", result["result"]["gate"])
        self.assertTrue(result["result"]["non_authoritative"])
        self.assertEqual(
            {
                "missing_all": 1,
                "broken_all": 2,
                "missing_touching_excluded_plane": 1,
                "missing_viewer_filtered": 0,
            },
            result["result"]["counts"],
        )
        with (export_directory / "missing_struts.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            missing = list(csv.DictReader(stream))
        with (export_directory / "broken_struts.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            broken = list(csv.DictReader(stream))
        with (export_directory / "missing_struts_viewer_filtered.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            filtered = list(csv.DictReader(stream))
        self.assertEqual(["1"], [row["strut_id"] for row in missing])
        self.assertEqual(["2", "3"], [row["strut_id"] for row in broken])
        self.assertEqual([], filtered)
        manifest = json.loads(
            (export_directory / "stage3_validation_export_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(manifest["non_authoritative"])
        self.assertFalse(manifest["production_receipt_artifact"])
        self.assertEqual(
            "either nominal endpoint touches the plane",
            manifest["filter"]["exclusion_rule"],
        )

    async def test_mcp_rejects_stale_stage3_handoff(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["predecessor_receipt_sha256"] = "c" * 64
        _write_json(self.handoff, handoff)
        async with Client(mcp) as client:
            call = await client.call_tool(
                "classify_struts",
                {
                    "stage_3_handoff_filepath": self.handoff.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "analysis_config_filepath": self.config.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "corridor_calibration_filepath": self.calibration.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "metrics_filepath": self.metrics.relative_to(REPOSITORY_ROOT).as_posix(),
                    "profiles_filepath": self.profiles.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "localized_graph_filepath": self.graph.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "ct_filepath": self.ct.relative_to(REPOSITORY_ROOT).as_posix(),
                    "output_directory": self.struts.relative_to(
                        REPOSITORY_ROOT
                    ).as_posix(),
                    "operation": "analyze_missing",
                },
            )
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("canonical hash", result["error"]["message"])

    def test_independent_verifier_requires_complete_team_and_exact_evidence(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        policy = config["stage_3_defect_analysis"]
        policy["development_mode"] = False
        for kind in ("thin", "bent"):
            policy[kind]["implementation_status"] = "complete"
            policy[kind]["policy"] = {"teammate_policy_fixture": True}
        _write_json(self.config, config)
        policy_sha256 = sha256_json(policy)
        finding_paths = {
            kind: self.struts / f"production_findings_{kind}.json"
            for kind in ("missing", "broken", "thin", "bent")
        }
        for kind in ("missing", "broken"):
            analyze_strut_specialist(
                self.metrics,
                self.profiles,
                config,
                finding_paths[kind],
                specimen_id=self.specimen_id,
                defect_kind=kind,
            )
        for kind in ("thin", "bent"):
            rows = [
                {
                    "strut_id": identifier,
                    "disposition": "negative",
                    "reasons": ["teammate_rule_not_satisfied"],
                    "features": {},
                    "evidence_required": False,
                    "evidence_refs": ["per_strut_metrics", "per_strut_profiles"],
                }
                for identifier in range(1, 5)
            ]
            _write_json(
                finding_paths[kind],
                {
                    "schema_version": "part2-specialist-findings/1.0.0",
                    "specimen_id": self.specimen_id,
                    "stage_number": 3,
                    "specialist": "thin_strut_agent",
                    "defect_kind": kind,
                    "status": "complete",
                    "policy_sha256": policy_sha256,
                    "input_hashes": {
                        "per_strut_metrics_sha256": sha256_file(self.metrics),
                        "per_strut_profiles_sha256": sha256_file(self.profiles),
                    },
                    "coverage": {
                        "nominal_strut_count": 4,
                        "evaluated_count": 4,
                        "positive_count": 0,
                        "negative_count": 4,
                        "review_count": 0,
                        "deferred_count": 0,
                    },
                    "findings": rows,
                    "provenance": {
                        "rule_version": "teammate-fixture/1.0.0",
                        "metrics_recomputed": False,
                        "registration_recomputed": False,
                        "training_labels_read": False,
                        "evaluation_labels_read": False,
                        "intentional_deletion_labels_read": False,
                    },
                },
            )
        classifications = self.struts / "production_classified.json"
        thresholds = self.struts / "production_thresholds.json"
        decision_log = self.struts / "production_decision_log.md"
        merged = merge_strut_classifications(
            self.metrics,
            self.profiles,
            config,
            finding_paths,
            classifications,
            thresholds,
            decision_log,
            specimen_id=self.specimen_id,
        )
        self.assertEqual("pass", merged["gate"])
        evidence_paths = []
        for strut_id in (1, 2, 3):
            rendered = render_strut_evidence(
                self.ct,
                self.graph,
                self.profiles,
                self.analysis / "evidence",
                strut_id=strut_id,
                threshold=1.0,
                metrics_path=self.metrics,
                classifications_path=classifications,
                thresholds_path=thresholds,
                specimen_id=self.specimen_id,
            )
            evidence_paths.append(Path(rendered["artifacts"]["manifest"]["path"]))
        verifier = verify_strut_classifications(
            self.metrics,
            self.profiles,
            finding_paths,
            classifications,
            thresholds,
            decision_log,
            evidence_paths,
            self.struts / "production_verifier.json",
            analysis_config=self.config,
            localized_graph_path=self.graph,
            ct_path=self.ct,
            specimen_id=self.specimen_id,
            attempt=1,
            run_token="stage3-test-token",
            config_sha256=sha256_file(self.config),
            contract_sha256=sha256_file(
                REPOSITORY_ROOT / "analysis/contracts/defect_analysis.json"
            ),
            predecessor_receipt_sha256="b" * 64,
            input_handoff_sha256="c" * 64,
        )
        self.assertEqual("pass", verifier["gate"])
        self.assertFalse(verifier["participated_in_classification"])
        self.assertTrue(verifier["self_verification"]["every_strut_labeled_once"])

        tampered = json.loads(classifications.read_text(encoding="utf-8"))
        next(
            row for row in tampered["classifications"] if row["strut_id"] == 3
        )["class"] = "present"
        _write_json(classifications, tampered)
        with self.assertRaisesRegex(ValueError, "precedence"):
            verify_strut_classifications(
                self.metrics,
                self.profiles,
                finding_paths,
                classifications,
                thresholds,
                decision_log,
                evidence_paths,
                self.struts / "tampered_verifier.json",
                analysis_config=self.config,
                localized_graph_path=self.graph,
                ct_path=self.ct,
                specimen_id=self.specimen_id,
                attempt=1,
                run_token="stage3-test-token",
                config_sha256=sha256_file(self.config),
                contract_sha256=sha256_file(
                    REPOSITORY_ROOT / "analysis/contracts/defect_analysis.json"
                ),
                predecessor_receipt_sha256="b" * 64,
                input_handoff_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
