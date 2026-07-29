"""MCP-client tests for the strict deterministic Stage 2 strut-metrics tool."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from fastmcp import Client
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import mcp_server  # noqa: E402
from part2_core import strut_metrics  # noqa: E402
from part2_core.registration import register_lattice_to_ct  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class StrutMetricsMCPTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.specimen = "stage2_fixture"
        self.analysis = self.root / "analysis" / self.specimen

        contract = self.root / "analysis" / "contracts" / "strut_metrics.json"
        schema = self.root / "analysis" / "schema" / "strut_metrics_input.schema.json"
        contract.parent.mkdir(parents=True)
        schema.parent.mkdir(parents=True)
        shutil.copyfile(REPOSITORY_ROOT / "analysis/contracts/strut_metrics.json", contract)
        shutil.copyfile(
            REPOSITORY_ROOT / "analysis/schema/strut_metrics_input.schema.json", schema
        )

        volume = np.zeros((72, 72, 72), dtype=np.uint16)
        edge_locations = [(18, 18), (18, 36), (36, 18), (36, 36), (54, 54)]
        for y_center, z_center in edge_locations[:4]:
            for z in range(72):
                for y in range(72):
                    if (z - z_center) ** 2 + (y - y_center) ** 2 <= 9:
                        volume[z, y, 20:53] = 100
        self.ct = self.root / "data" / "scan.npy"
        self.ct.parent.mkdir()
        np.save(self.ct, volume)
        self.mask = self.analysis / "segmentation" / "canonical_mask.npy"
        self.mask.parent.mkdir(parents=True)
        np.save(self.mask, np.asarray(volume >= 50, dtype=np.uint8))

        junctions = []
        struts = []
        for row, (y, z) in enumerate(edge_locations):
            first = 2 * row + 1
            second = first + 1
            junctions.extend(
                [
                    {"id": first, "position": [20.0, float(y), float(z)]},
                    {"id": second, "position": [52.0, float(y), float(z)]},
                ]
            )
            struts.append(
                {"id": 100 + row, "junction0": first, "junction1": second}
            )
        self.graph = self.analysis / "registration" / "localized_graph.json"
        _write_json(
            self.graph,
            {
                "junctions": junctions,
                "struts": struts,
                "unit_cells": [{"id": 1, "struts": [item["id"] for item in struts]}],
            },
        )

        self.config = self.analysis / "config" / "analysis_config.json"
        _write_json(
            self.config,
            {
                "stage_2_strut_metrics": {
                    "schema_version": "part2-strut-metrics-config/1.0.0",
                    "otsu_threshold": 50,
                    "axial_padding_fraction_total": 0.2,
                    "interpolation_batch_size": 32,
                    "junction_mask_radius_voxels": 2.0,
                    "transverse_margin_fraction": 0.2,
                    "minimum_transverse_margin_voxels": 2.0,
                    "collar_fraction": 0.2,
                    "endpoint_seed_half_length_voxels": 1.0,
                    "collar_half_length_voxels": 1.0,
                    "minimum_axial_foreground_fraction": 0.1,
                    "centerline_smoothing_passes": 2,
                    "minimum_valid_roi_fraction": 0.99,
                    "corridor_bootstrap": {
                        "sample_count": 5,
                        "minimum_valid_samples": 4,
                        "minimum_valid_slice_fraction": 0.75,
                        "central_fraction": 0.6,
                        "calibration_half_width_voxels": 6.0,
                        "maximum_axis_distance_voxels": 4.0,
                        "radial_extent_quantile": 0.9,
                        "radius_multiplier": 1.0,
                        "radius_safety_voxels": 0.5,
                        "minimum_radius_voxels": 2.0,
                        "maximum_radius_voxels": 5.0,
                    },
                }
            },
        )
        self.qa = self.analysis / "qa" / "registration_qa.json"
        _write_json(
            self.qa,
            {
                "specimen_id": self.specimen,
                "registration_mode": "autonomous_v2",
                "gate": "pass",
                "overall_pass": True,
            },
        )
        self.otsu = self.analysis / "segmentation" / "histogram_report.json"
        _write_json(self.otsu, {"overall_pass": True, "threshold": 50})
        self.comparison = self.analysis / "segmentation" / "mask_comparison.json"
        _write_json(
            self.comparison,
            {
                "overall_pass": True,
                "candidates": [
                    {
                        "path": self.mask.relative_to(self.root).as_posix(),
                        "sha256": _sha(self.mask),
                        "threshold": 50,
                        "exact_threshold_match": True,
                    }
                ],
            },
        )
        mask_binding = {
            "path": self.mask.relative_to(self.root).as_posix(),
            "sha256": _sha(self.mask),
            "role": "canonical_segmentation_mask",
            "retention": "committed",
            "dtype": "uint8",
            "shape": list(volume.shape),
            "array_axes": ["z", "y", "x"],
        }
        self.manifest = self.analysis / "config" / "specimen_manifest.json"
        authorized_outputs = [
            "segmentation",
            "registration",
            "node_localization",
            "coarse_region_screening",
            "padded_roi_definition",
        ]
        unauthorized_outputs = [
            "absolute_metrology",
            "direct_dimensional_measurement",
        ]
        manifest = {
            "specimen_id": self.specimen,
            "design_id": "fixture_design",
            "lifecycle_state": "analysis_ready",
            "analysis_parameters": {
                "requested_analysis_scope": "roi_screening",
                "registration": {"mode": "autonomous_v2"},
            },
            "inputs": {
                "ct": {
                    "path": self.ct.relative_to(self.root).as_posix(),
                    "sha256": _sha(self.ct),
                    "role": "ct_volume",
                },
                "ct_metadata": {"shape": list(volume.shape)},
                "canonical_mask": mask_binding,
            },
            "derived": {
                "registration_result": {
                    "values": {
                        "requested_analysis_scope": "roi_screening",
                        "authorized_outputs": authorized_outputs,
                        "unauthorized_outputs": unauthorized_outputs,
                    }
                }
            },
        }
        _write_json(self.manifest, manifest)
        self.completion = self.analysis / "config" / "data_prep_completion_receipt.json"
        completion_base = {
            "schema_version": "data-prep-completion/1.2.0",
            "specimen_id": self.specimen,
            "design_id": "fixture_design",
            "lifecycle_state": "analysis_ready",
            "requested_analysis_scope": "roi_screening",
            "registration_mode": "autonomous_v2",
            "authorized_outputs": authorized_outputs,
            "unauthorized_outputs": unauthorized_outputs,
            "analysis_ready_manifest_sha256": _canonical(manifest),
            "canonical_mask": mask_binding,
        }
        _write_json(
            self.completion,
            {
                **completion_base,
                "canonical_completion_sha256": _canonical(completion_base),
            },
        )

        role_paths = {
            "analysis_ready_specimen_manifest": self.manifest,
            "data_prep_completion_receipt": self.completion,
            "analysis_config": self.config,
            "ct_volume": self.ct,
            "localized_graph": self.graph,
            "registration_qa": self.qa,
            "otsu_report": self.otsu,
            "canonical_segmentation_mask": self.mask,
            "segmentation_mask_comparison": self.comparison,
        }
        artifacts = [
            {
                "role": role,
                "path": path.relative_to(self.root).as_posix(),
                "sha256": _sha(path),
            }
            for role, path in sorted(role_paths.items())
        ]
        contract_document = json.loads(contract.read_text(encoding="utf-8"))
        handoff_base = {
            "schema_version": "part2-stage-handoff/1.0.0",
            "specimen_id": self.specimen,
            "stage_number": 2,
            "stage": "strut_metrics",
            "owner": "strut_metrics",
            "attempt": 1,
            "run_token": "b" * 64,
            "created_at": "2026-01-01T00:00:00Z",
            "registration_mode": "autonomous_v2",
            # The orchestrator's frozen control-config digest is distinct from
            # the Stage 1 analysis-config artifact digest below.
            "config_sha256": "c" * 64,
            "contract_version": contract_document["schema_version"],
            "contract_sha256": _sha(contract),
            "predecessor_receipt_sha256": "a" * 64,
            "input_artifacts": artifacts,
            "forbidden_operations": contract_document["forbidden_operations"],
        }
        self.handoff = self.analysis / "handoffs" / "stage_2_strut_metrics_attempt_1.json"
        _write_json(
            self.handoff,
            {
                **handoff_base,
                "canonical_handoff_sha256": _canonical(handoff_base),
            },
        )

    def _arguments(self) -> dict[str, object]:
        relative = lambda path: path.relative_to(self.root).as_posix()
        return {
            "stage_2_handoff_filepath": relative(self.handoff),
            "analysis_ready_specimen_manifest_filepath": relative(self.manifest),
            "data_prep_completion_receipt_filepath": relative(self.completion),
            "analysis_config_filepath": relative(self.config),
            "ct_filepath": relative(self.ct),
            "localized_graph_filepath": relative(self.graph),
            "registration_qa_filepath": relative(self.qa),
            "otsu_report_filepath": relative(self.otsu),
            "canonical_segmentation_mask_filepath": relative(self.mask),
            "segmentation_mask_comparison_filepath": relative(self.comparison),
            "output_directory": f"analysis/{self.specimen}/struts",
        }

    def test_stage1_analysis_config_freezes_stage2_policy_and_otsu_threshold(self) -> None:
        generated = self.root / "generated_analysis_config.json"
        register_lattice_to_ct(
            self.graph,
            self.root / "registered.json",
            self.root / "registration_report.json",
            mode="challenge_aligned_json",
            aligned_graph_path=self.graph,
            threshold=50,
            analysis_config_path=generated,
        )
        document = json.loads(generated.read_text(encoding="utf-8"))
        fragment = document["stage_2_strut_metrics"]
        self.assertEqual("part2-strut-metrics-config/1.0.0", fragment["schema_version"])
        self.assertEqual(50.0, fragment["otsu_threshold"])
        self.assertEqual(0.2, fragment["axial_padding_fraction_total"])
        self.assertEqual(64, fragment["interpolation_batch_size"])

    async def test_tool_is_registered_and_writes_measurements_through_mcp(self) -> None:
        tools = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
        self.assertIn("compute_strut_metrics", tools)
        original = strut_metrics.ndimage.map_coordinates
        interpolation_calls = 0

        def count_interpolation(*args: object, **kwargs: object) -> np.ndarray:
            nonlocal interpolation_calls
            interpolation_calls += 1
            return original(*args, **kwargs)

        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root), mock.patch.object(
            strut_metrics.ndimage, "map_coordinates", side_effect=count_interpolation
        ):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("ok", result["status"], result)
        self.assertEqual("pass", result["gate"], result)
        self.assertTrue(result["result"]["measurement_only"])
        self.assertFalse(result["result"]["classification_performed"])
        self.assertEqual(5, result["result"]["counts"]["metric_rows"])
        self.assertEqual(1, result["result"]["method"]["strut_interpolation_batch_count"])
        self.assertEqual(2, interpolation_calls, "one calibration batch plus one strut batch")
        output = self.analysis / "struts"
        self.assertEqual(
            {
                "corridor_calibration.json",
                "per_strut_metrics.csv",
                "per_strut_profiles.json",
                "metrics_report.json",
            },
            {path.name for path in output.iterdir()},
        )
        with (output / "per_strut_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = {int(row["strut_id"]): row for row in csv.DictReader(stream)}
        self.assertEqual("true", rows[100]["same_material_component_connects_a_to_b"])
        self.assertEqual("false", rows[104]["same_material_component_connects_a_to_b"])
        self.assertEqual(0.0, float(rows[104]["minimum_foreground_fraction"]))

    async def test_tool_halts_when_handoff_exposes_labels(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["input_artifacts"].append(
            {"role": "development_labels", "path": "labels/dev.json", "sha256": "0" * 64}
        )
        base = {key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"}
        handoff["canonical_handoff_sha256"] = _canonical(base)
        _write_json(self.handoff, handoff)
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        self.assertFalse(call.is_error)
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("forbidden", result["summary"].lower())

    async def test_primary_endpoint_result_is_not_overridden_by_junction_mask(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["stage_2_strut_metrics"]["junction_mask_radius_voxels"] = 12.0
        _write_json(self.config, config)
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        for record in handoff["input_artifacts"]:
            if record["role"] == "analysis_config":
                record["sha256"] = _sha(self.config)
        base = {
            key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"
        }
        handoff["canonical_handoff_sha256"] = _canonical(base)
        _write_json(self.handoff, handoff)
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        result = call.structured_content
        self.assertEqual("pass", result["gate"], result)
        with (self.analysis / "struts" / "per_strut_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            rows = {int(row["strut_id"]): row for row in csv.DictReader(stream)}
        self.assertEqual("true", rows[100]["same_material_component_connects_a_to_b"])
        self.assertEqual("false", rows[100]["same_component_connects_collar_a_to_b"])

    async def test_tool_rejects_tampered_forbidden_operations(self) -> None:
        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        handoff["forbidden_operations"] = handoff["forbidden_operations"][:-1]
        base = {
            key: value for key, value in handoff.items() if key != "canonical_handoff_sha256"
        }
        handoff["canonical_handoff_sha256"] = _canonical(base)
        _write_json(self.handoff, handoff)
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("forbidden operations", result["summary"].lower())

    async def test_tool_rejects_analysis_config_hash_mismatch(self) -> None:
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["stage_2_strut_metrics"]["centerline_smoothing_passes"] = 3
        _write_json(self.config, config)
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("hash mismatch", result["summary"].lower())

    async def test_tool_rejects_manifest_receipt_authorization_disagreement(self) -> None:
        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["derived"]["registration_result"]["values"][
            "authorized_outputs"
        ] = ["segmentation"]
        _write_json(self.manifest, manifest)

        completion = json.loads(self.completion.read_text(encoding="utf-8"))
        completion["analysis_ready_manifest_sha256"] = _canonical(manifest)
        completion_base = {
            key: value
            for key, value in completion.items()
            if key != "canonical_completion_sha256"
        }
        completion["canonical_completion_sha256"] = _canonical(completion_base)
        _write_json(self.completion, completion)

        handoff = json.loads(self.handoff.read_text(encoding="utf-8"))
        updated_hashes = {
            "analysis_ready_specimen_manifest": _sha(self.manifest),
            "data_prep_completion_receipt": _sha(self.completion),
        }
        for record in handoff["input_artifacts"]:
            if record["role"] in updated_hashes:
                record["sha256"] = updated_hashes[record["role"]]
        handoff_base = {
            key: value
            for key, value in handoff.items()
            if key != "canonical_handoff_sha256"
        }
        handoff["canonical_handoff_sha256"] = _canonical(handoff_base)
        _write_json(self.handoff, handoff)

        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("authorizations differ", result["summary"].lower())

    async def test_exact_replay_is_idempotent_and_overwrite_is_forbidden(self) -> None:
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                first = await client.call_tool("compute_strut_metrics", self._arguments())
                second = await client.call_tool("compute_strut_metrics", self._arguments())
                overwrite_arguments = {**self._arguments(), "overwrite": True}
                overwritten = await client.call_tool(
                    "compute_strut_metrics", overwrite_arguments
                )
        first_result = first.structured_content
        second_result = second.structured_content
        overwrite_result = overwritten.structured_content
        self.assertEqual("pass", first_result["gate"], first_result)
        self.assertEqual("pass", second_result["gate"], second_result)
        self.assertTrue(
            all(
                artifact["changed"]
                for artifact in first_result["artifacts"].values()
            )
        )
        self.assertTrue(
            all(
                not artifact["changed"]
                for artifact in second_result["artifacts"].values()
            )
        )
        self.assertEqual("error", overwrite_result["status"])
        self.assertEqual("halt", overwrite_result["gate"])
        self.assertIn("immutable", overwrite_result["summary"].lower())

    async def test_partial_output_bundle_is_rejected_without_repair(self) -> None:
        output = self.analysis / "struts"
        output.mkdir(parents=True)
        sentinel = output / "per_strut_metrics.csv"
        sentinel.write_text("do not replace\n", encoding="utf-8")
        with mock.patch.object(mcp_server, "REPOSITORY_ROOT", self.root):
            async with Client(mcp_server.mcp) as client:
                call = await client.call_tool("compute_strut_metrics", self._arguments())
        result = call.structured_content
        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertIn("partial", result["summary"].lower())
        self.assertEqual("do not replace\n", sentinel.read_text(encoding="utf-8"))

    def test_padded_sampling_preserves_exact_nominal_endpoint_planes(self) -> None:
        _, geometry, local_x, local_y, local_z = strut_metrics._prepare_sampling(
            strut_id=1,
            endpoint_ids=np.asarray([10, 11]),
            node_a_xyz=np.asarray([10.0, 20.0, 30.0]),
            node_b_xyz=np.asarray([42.0, 20.0, 30.0]),
            half_width=5.0,
            corridor_radius=3.0,
            axial_padding_fraction_total=0.2,
        )
        self.assertTrue(np.any(np.isclose(local_z, 0.0)))
        self.assertTrue(np.any(np.isclose(local_z, geometry.length_voxels)))
        self.assertAlmostEqual(-3.2, float(local_z[0]))
        self.assertAlmostEqual(35.2, float(local_z[-1]))

        shape = tuple(geometry.local_shape_zyx)
        valid = np.ones(shape, dtype=bool)
        valid.flat[0] = False
        cuboid = strut_metrics.SampledCuboid(
            foreground_probability_zyx=np.ones(shape, dtype=np.float32),
            valid_zyx=valid,
            geometry=geometry,
            local_x=local_x,
            local_y=local_y,
            local_z=local_z,
        )
        permissive = strut_metrics.normalize_stage2_config(
            {"minimum_valid_roi_fraction": 0.99}
        )
        strict = strut_metrics.normalize_stage2_config(
            {"minimum_valid_roi_fraction": 1.0}
        )
        permissive_row, _ = strut_metrics._analyze_cuboid(cuboid, permissive)
        strict_row, _ = strut_metrics._analyze_cuboid(cuboid, strict)
        self.assertTrue(permissive_row["roi_valid"])
        self.assertFalse(strict_row["roi_valid"])


if __name__ == "__main__":
    unittest.main()
