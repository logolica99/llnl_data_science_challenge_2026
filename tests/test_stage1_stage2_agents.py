"""Production contract tests for the Stage 1 and Stage 2 agent slice."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import struct
import sys
import tempfile
import tomllib
import unittest

from fastmcp import Client
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from mcp_server import mcp  # noqa: E402
from part2_core.artifacts import sha256_file, sha256_json  # noqa: E402
from part2_core.design_diff import (  # noqa: E402
    AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
    DECLARED_TRANSFORM_TOLERANCES,
    ORIENTATION_SCHEMA_VERSION,
    STAGE1_POLICY_SCHEMA_VERSION,
    TRANSFORM_CONVENTION,
    TRANSFORM_DECLARATION_SCHEMA_VERSION,
    label_deleted_edges,
    load_stage1_policy,
    resolve_cad_graph_orientation,
)


def write_binary_stl(path: Path, centroids: list[np.ndarray]) -> None:
    header = b"deterministic-stage1-test".ljust(80, b"\0")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(centroids)))
        for centroid in centroids:
            stream.write(struct.pack("<3f", 0.0, 0.0, 1.0))
            for _ in range(3):
                stream.write(struct.pack("<3f", *centroid.tolist()))
            stream.write(struct.pack("<H", 0))


def write_stage1_policy(
    path: Path,
    *,
    counts: dict[str, int],
    expected_deletions: dict[str, int],
    reflection_authorized: bool = False,
) -> str:
    document = {
        "schema_version": STAGE1_POLICY_SCHEMA_VERSION,
        "policy_id": f"synthetic-stage1-policy-reflection-{reflection_authorized}",
        "intended_use": "test_fixture",
        "orientation_verification": {
            "sample_count": 9,
            "sample_start": 0.4,
            "sample_end": 0.6,
            "maximum_ranked_edges": 2048,
            "scale_candidates_mm_per_design_unit": [
                AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
            ],
            "ambiguity_absolute_mm": 0.0001,
            "ambiguity_relative_fraction": 0.001,
            "reflection_authorized": reflection_authorized,
            "expected_counts": counts,
            "verification_tolerances": dict(DECLARED_TRANSFORM_TOLERANCES),
        },
        "deletion_labeling": {
            "sample_count": 9,
            "sample_start": 0.4,
            "sample_end": 0.6,
            "radius_margin_mm": 0.03,
            "radius_rounding_mm": 0.01,
            "split_seed": 20260723,
            "development_fraction": 0.3,
            "x_bins": 5,
            "z_shells": 3,
            "expected_deletions": expected_deletions,
        },
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


class AgentSurfaceTests(unittest.IsolatedAsyncioTestCase):
    def test_agents_are_gpt_5_6_sol_and_skills_are_mcp_only(self) -> None:
        for name in ("design_diff", "data_prep"):
            document = tomllib.loads(
                (REPOSITORY_ROOT / ".codex" / "agents" / f"{name}.toml").read_text()
            )
            self.assertEqual("gpt-5.6-sol", document["model"])
        for name in ("stl-design-diff", "ct-registration", "ct-threshold-optimizer"):
            root = REPOSITORY_ROOT / ".agents" / "skills" / name
            self.assertFalse((root / "scripts").exists())
            self.assertIn("segmentation-tools", (root / "agents" / "openai.yaml").read_text())

    async def test_stage_tools_are_registered_through_actual_mcp_client(self) -> None:
        registered = {tool.name: tool for tool in await mcp.list_tools()}
        tools = set(registered)
        self.assertTrue(
            {
                "load_lattice_graph",
                "resolve_cad_graph_orientation",
                "label_deleted_edges",
                "volume_info",
                "replay_exact_otsu",
                "segment_ct_dataset",
                "compare_segmentation_masks",
                "visualize_slice",
                "register_lattice_to_ct",
                "localize_lattice_nodes",
                "compute_registration_qa",
            }.issubset(tools)
        )
        qa_schema = registered["compute_registration_qa"].parameters
        self.assertIn("localization_report_filepath", qa_schema["required"])
        self.assertNotIn(
            "registration_uncertainty_voxels", qa_schema["properties"]
        )
        self.assertNotIn("local_search_radius_voxels", qa_schema["properties"])

    async def test_canonical_mask_replay_is_idempotent_and_config_drift_halts(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary:
            root = Path(temporary)
            volume = root / "volume.npy"
            mask = root / "mask.npy"
            np.save(volume, np.arange(64, dtype=np.uint16).reshape(4, 4, 4))
            async with Client(mcp) as client:
                first = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 32},
                )
                replay = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 32},
                )
                drift = await client.call_tool(
                    "segment_ct_dataset",
                    {"input_filepath": str(volume), "output_filepath": str(mask), "threshold": 31},
                )
            self.assertTrue(first.structured_content["result"]["changed"])
            self.assertFalse(replay.structured_content["result"]["changed"])
            self.assertEqual("error", drift.structured_content["status"])
            self.assertEqual("halt", drift.structured_content["gate"])

    async def test_mask_comparison_halts_on_threshold_content_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary:
            root = Path(temporary)
            volume = root / "volume.npy"
            mask = root / "wrong-mask.npy"
            np.save(volume, np.arange(64, dtype=np.uint16).reshape(4, 4, 4))
            np.save(mask, np.zeros((4, 4, 4), dtype=np.uint8))
            async with Client(mcp) as client:
                call = await client.call_tool(
                    "compare_segmentation_masks",
                    {
                        "raw_filepath": str(volume),
                        "mask_filepaths": [str(mask)],
                        "thresholds": [32],
                    },
                )
            result = call.structured_content
            self.assertEqual("halt", result["gate"])
            self.assertFalse(result["result"]["overall_pass"])
            self.assertEqual(
                32,
                result["result"]["candidates"][0]["mismatched_voxels"],
            )


class Stage1MCPProductionShapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_declared_transform_and_labels_pass_real_mcp_flow(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT) as temporary:
            root = Path(temporary)
            graph_path = root / "production-shaped-graph.json"
            baseline_path = root / "0.stl"
            node_count = 10_206
            edge_count = 18_468
            cell_count = 729
            center = (node_count - 1) / 2.0
            nodes = [
                {"id": index, "position": [float(index), 0.0, 0.0]}
                for index in range(node_count)
            ]
            endpoint_rows = [
                (index, index + 1) for index in range(node_count - 1)
            ] + [
                (index, index + 2)
                for index in range(edge_count - (node_count - 1))
            ]
            edge_ids = [100_000 + index for index in range(edge_count)]
            edges = [
                {
                    "id": edge_ids[index],
                    "junction0": endpoints[0],
                    "junction1": endpoints[1],
                }
                for index, endpoints in enumerate(endpoint_rows)
            ]
            cells = [
                {
                    "id": 200_000 + index,
                    "indices": [index, 0, 0],
                    "struts": [edge_ids[index]],
                }
                for index in range(cell_count)
            ]
            graph_path.write_text(
                json.dumps(
                    {
                        "junctions": nodes,
                        "struts": edges,
                        "unit_cells": cells,
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )

            fractions = np.linspace(0.4, 0.6, 9)
            edge_clouds: list[list[np.ndarray]] = []
            for edge_index, (first, second) in enumerate(endpoint_rows):
                samples = [
                    np.asarray(
                        [
                            (
                                first * (1.0 - float(fraction))
                                + second * float(fraction)
                                - center
                            )
                            * AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
                            0.0,
                            0.0,
                        ]
                    )
                    for fraction in fractions
                ]
                # The first 186 edge clouds carry exactly 175 triangles so
                # each independently deleted production variant has the
                # source-backed 170--180 triangle-deficit ratio.
                edge_clouds.append(
                    samples + ([samples[4]] * 166 if edge_index < 186 else [])
                )
            write_binary_stl(
                baseline_path,
                [point for cloud in edge_clouds for point in cloud],
            )
            deletion_counts = {"0p1": 18, "0p5": 93, "1p0": 186}
            variant_paths: dict[str, Path] = {}
            for name, deletion_count in deletion_counts.items():
                variant = root / f"{name}.stl"
                write_binary_stl(
                    variant,
                    [
                        point
                        for edge_index, cloud in enumerate(edge_clouds)
                        if edge_index >= deletion_count
                        for point in cloud
                    ],
                )
                variant_paths[name] = variant

            declaration_path = root / "transform.json"
            declaration = {
                "schema_version": "part2-graph-to-stl-transform-declaration/1.0.0",
                "declaration_id": "production-shaped-transform-001",
                "source_id": "synthetic-scientist-declaration",
                "provenance_id": "stage1-mcp-production-shaped-test",
                "specimen_id": "production-shaped-specimen",
                "design_id": "production-shaped-design",
                "nominal_graph_sha256": sha256_file(graph_path),
                "full_design_stl_sha256": sha256_file(baseline_path),
                "transform": {
                    "convention": TRANSFORM_CONVENTION,
                    "design_center": [center, 0.0, 0.0],
                    "scale_mm_per_design_unit": (
                        AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
                    ),
                    "rotation_matrix": np.eye(3).tolist(),
                    "translation_mm": [0.0, 0.0, 0.0],
                    "reflection_permitted": False,
                },
            }
            declaration["canonical_declaration_sha256"] = sha256_json(
                declaration
            )
            declaration_path.write_text(
                json.dumps(declaration, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            orientation_path = root / "orientation.json"
            labels_directory = root / "labels"
            async with Client(mcp) as client:
                orientation_call = await client.call_tool(
                    "resolve_cad_graph_orientation",
                    {
                        "nominal_graph_filepath": str(graph_path),
                        "full_design_stl_filepath": str(baseline_path),
                        "output_filepath": str(orientation_path),
                        "specimen_id": "production-shaped-specimen",
                        "design_id": "production-shaped-design",
                        "declared_transform_filepath": str(declaration_path),
                        "declared_transform_sha256": sha256_file(
                            declaration_path
                        ),
                    },
                )
                label_call = await client.call_tool(
                    "label_deleted_edges",
                    {
                        "nominal_graph_filepath": str(graph_path),
                        "baseline_stl_filepath": str(baseline_path),
                        "variant_stl_filepaths": {
                            name: str(path)
                            for name, path in variant_paths.items()
                        },
                        "orientation_filepath": str(orientation_path),
                        "output_directory": str(labels_directory),
                        "specimen_id": "production-shaped-specimen",
                        "design_id": "production-shaped-design",
                        "development_split_filepath": str(root / "dev.json"),
                        "sealed_split_filepath": str(root / "sealed.json"),
                        "label_report_filepath": str(root / "label-report.json"),
                    },
                )

            orientation = orientation_call.structured_content
            labels = label_call.structured_content
            self.assertFalse(orientation_call.is_error)
            self.assertFalse(label_call.is_error)
            self.assertEqual("pass", orientation["gate"])
            self.assertEqual("pass", labels["gate"])
            self.assertEqual(
                edge_count * 9,
                orientation["result"]["support"]["sample_support_count"],
            )
            self.assertEqual(
                orientation["hashes"]["declared_transform_artifact_sha256"],
                orientation["hashes"][
                    "intake_declared_transform_artifact_sha256"
                ],
            )
            self.assertEqual(
                deletion_counts,
                {
                    name: labels["result"]["variants"][name]["deleted_count"]
                    for name in deletion_counts
                },
            )
            self.assertTrue(
                labels["result"]["provenance"][
                    "stage1_policy_revalidated_before_variant_access"
                ]
            )


class DesignLabelCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.graph_path = self.root / "graph.json"
        graph = {
            "junctions": [
                {"id": 10, "position": [0, 0, 0]},
                {"id": 20, "position": [1, 0, 0]},
                {"id": 30, "position": [2, 0, 0]},
                {"id": 40, "position": [3, 0, 0]},
                {"id": 50, "position": [4, 0, 0]},
            ],
            "struts": [
                {"id": 101, "junction0": 10, "junction1": 20},
                {"id": 205, "junction0": 20, "junction1": 30},
                {"id": 309, "junction0": 30, "junction1": 40},
                {"id": 413, "junction0": 40, "junction1": 50},
            ],
            "unit_cells": [
                {"id": 7, "indices": [0, 0, 0], "struts": [101, 205, 309, 413]}
            ],
        }
        self.graph_path.write_text(json.dumps(graph), encoding="utf-8")
        # The core subtracts the graph center [2, 0, 0].
        fractions = np.linspace(0.4, 0.6, 9)
        edge_clouds: list[list[np.ndarray]] = []
        for start in (-2.0, -1.0, 0.0, 1.0):
            support = [
                np.asarray(
                    [
                        (start + float(fraction))
                        * AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
                        0.0,
                        0.0,
                    ]
                )
                for fraction in fractions
            ]
            edge_clouds.append([support[index % len(support)] for index in range(175)])
        baseline = [point for cloud in edge_clouds for point in cloud]
        write_binary_stl(self.root / "0.stl", baseline)
        # Each percentage is an independent specimen. These deletion sets are
        # deliberately non-nested to prevent cross-variant subset assumptions.
        write_binary_stl(
            self.root / "0p1.stl",
            [point for cloud in edge_clouds[1:] for point in cloud],
        )
        write_binary_stl(
            self.root / "0p5.stl",
            [point for index in (0, 3) for point in edge_clouds[index]],
        )
        write_binary_stl(self.root / "1p0.stl", edge_clouds[1])
        self.policy_path = self.root / "stage1-policy.json"
        self.policy_sha256 = write_stage1_policy(
            self.policy_path,
            counts={"nodes": 5, "edges": 4, "cells": 1},
            expected_deletions={"0p1": 1, "0p5": 2, "1p0": 3},
        )
        policy_document = json.loads(self.policy_path.read_text(encoding="utf-8"))
        orientation = {
            "schema_version": ORIENTATION_SCHEMA_VERSION,
            "specimen_id": "specimen-test",
            "design_id": "design-test",
            "gate": "pass",
            "overall_pass": True,
            "resolution_source": "geometry_search",
            "gates": {
                "orientation_unambiguous": True,
                "orientation_resolved": True,
                "scale_preserving_transform": True,
                "all_edge_support_finite": True,
                "maximum_edge_support_distance_within_tolerance": True,
                "edge_support_spread_within_tolerance": True,
                "geometry_search_consistent": True,
                "authoritative_transform_verified": False,
            },
            "transform": {
                "convention": TRANSFORM_CONVENTION,
                "design_center": [2.0, 0.0, 0.0],
                "rotation_matrix": np.eye(3).tolist(),
                "scale_mm_per_design_unit": (
                    AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
                ),
                "translation_mm": [0.0, 0.0, 0.0],
                "reflection_permitted": False,
            },
            "declared_transform": {
                "present": False,
                "verification": {
                    "overall_pass": False,
                    "reason_codes": ["declaration_absent"],
                },
            },
            "stage1_policy": {
                "artifact_path": str(self.policy_path.resolve()),
                "artifact_sha256": self.policy_sha256,
                "config_sha256": sha256_json(policy_document),
                "schema_version": STAGE1_POLICY_SCHEMA_VERSION,
                "policy_id": policy_document["policy_id"],
                "intended_use": "test_fixture",
            },
            "verification_tolerances": dict(DECLARED_TRANSFORM_TOLERANCES),
            "hashes": {
                "nominal_graph_sha256": sha256_file(self.graph_path),
                "full_design_stl_sha256": sha256_file(self.root / "0.stl"),
                "verification_tolerances_sha256": sha256_json(
                    DECLARED_TRANSFORM_TOLERANCES
                ),
                "stage1_policy_artifact_sha256": self.policy_sha256,
                "config_sha256": sha256_json(policy_document),
            },
        }
        self.orientation_path = self.root / "orientation.json"
        self.orientation_path.write_text(json.dumps(orientation), encoding="utf-8")
        self._declaration_counter = 0

    def _write_declaration(
        self,
        *,
        transform_overrides: dict[str, object] | None = None,
        document_overrides: dict[str, object] | None = None,
        valid_self_hash: bool = True,
    ) -> tuple[Path, str]:
        self._declaration_counter += 1
        transform: dict[str, object] = {
            "convention": TRANSFORM_CONVENTION,
            "design_center": [2.0, 0.0, 0.0],
            "scale_mm_per_design_unit": (
                AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
            ),
            "rotation_matrix": np.eye(3).tolist(),
            "translation_mm": [0.0, 0.0, 0.0],
            "reflection_permitted": False,
        }
        transform.update(transform_overrides or {})
        document: dict[str, object] = {
            "schema_version": TRANSFORM_DECLARATION_SCHEMA_VERSION,
            "declaration_id": "declared-transform-test-001",
            "source_id": "scientist-approved-cad-frame",
            "provenance_id": "intake-record-test-001",
            "specimen_id": "specimen-test",
            "design_id": "design-test",
            "nominal_graph_sha256": sha256_file(self.graph_path),
            "full_design_stl_sha256": sha256_file(self.root / "0.stl"),
            "transform": transform,
        }
        document.update(document_overrides or {})
        if valid_self_hash:
            document["canonical_declaration_sha256"] = sha256_json(document)
        else:
            document["canonical_declaration_sha256"] = "0" * 64
        path = self.root / f"declared-transform-{self._declaration_counter}.json"
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        return path, sha256_file(path)

    def _resolve_with_declaration(
        self,
        declaration_path: Path,
        declaration_sha256: str | None,
        *,
        output_name: str | None = None,
        specimen_id: str = "specimen-test",
        design_id: str = "design-test",
        allow_reflection: bool = False,
        baseline_stl_path: Path | None = None,
    ) -> dict[str, object]:
        name = output_name or f"orientation-{self._declaration_counter}.json"
        policy_path = self.policy_path
        policy_sha256 = self.policy_sha256
        if allow_reflection:
            policy_path = self.root / "stage1-policy-reflection.json"
            policy_sha256 = write_stage1_policy(
                policy_path,
                counts={"nodes": 5, "edges": 4, "cells": 1},
                expected_deletions={"0p1": 1, "0p5": 2, "1p0": 3},
                reflection_authorized=True,
            )
        return resolve_cad_graph_orientation(
            self.graph_path,
            baseline_stl_path or self.root / "0.stl",
            self.root / name,
            stage1_policy_path=policy_path,
            stage1_policy_sha256=policy_sha256,
            declared_transform_path=declaration_path,
            declared_transform_sha256=declaration_sha256,
            specimen_id=specimen_id,
            design_id=design_id,
        )

    def test_tube_emptiness_validates_non_nested_specimens_independently(self) -> None:
        report = label_deleted_edges(
            self.graph_path,
            self.root / "0.stl",
            {
                "0p1": self.root / "0p1.stl",
                "0p5": self.root / "0p5.stl",
                "1p0": self.root / "1p0.stl",
            },
            self.orientation_path,
            self.root / "labels",
            development_split_path=self.root / "dev.json",
            sealed_split_path=self.root / "sealed.json",
            label_report_path=self.root / "report.md",
            specimen_id="specimen-test",
            design_id="design-test",
        )
        self.assertEqual("pass", report["gate"])
        self.assertEqual("specimen-test", report["specimen_id"])
        self.assertEqual("design-test", report["design_id"])
        self.assertNotIn("deletion_sets_monotone", report["gates"])
        self.assertTrue(all(report["gates"].values()))
        self.assertTrue(report["gates"]["triangle_deficit_ratio_between_170_and_180"])
        first = json.loads((self.root / "labels" / "intentional_deletions_0p1.json").read_text())
        self.assertEqual([101], first["deleted_strut_ids"])
        self.assertEqual("specimen-test", first["specimen_id"])
        self.assertEqual("design-test", first["design_id"])
        second = json.loads((self.root / "labels" / "intentional_deletions_0p5.json").read_text())
        third = json.loads((self.root / "labels" / "intentional_deletions_1p0.json").read_text())
        self.assertEqual([205, 309], second["deleted_strut_ids"])
        self.assertEqual([101, 309, 413], third["deleted_strut_ids"])
        dev = set(json.loads((self.root / "dev.json").read_text())["strut_ids"])
        sealed = set(json.loads((self.root / "sealed.json").read_text())["strut_ids"])
        self.assertTrue(dev.isdisjoint(sealed))
        self.assertEqual({205, 309}, dev | sealed)
        self.assertIn("Design-diff label report", (self.root / "report.md").read_text())

        replay = label_deleted_edges(
            self.graph_path,
            self.root / "0.stl",
            {
                "0p1": self.root / "0p1.stl",
                "0p5": self.root / "0p5.stl",
                "1p0": self.root / "1p0.stl",
            },
            self.orientation_path,
            self.root / "labels",
            development_split_path=self.root / "dev.json",
            sealed_split_path=self.root / "sealed.json",
            label_report_path=self.root / "report.md",
            specimen_id="specimen-test",
            design_id="design-test",
        )
        self.assertFalse(replay["artifacts"]["labels_0p1"]["changed"])

    def test_equivalent_orientation_hypotheses_abstain(self) -> None:
        orientation = resolve_cad_graph_orientation(
            self.graph_path,
            self.root / "0.stl",
            self.root / "ambiguous-orientation.json",
            stage1_policy_path=self.policy_path,
            stage1_policy_sha256=self.policy_sha256,
            specimen_id="specimen-test",
            design_id="design-test",
        )
        self.assertEqual("manual_review", orientation["gate"])
        self.assertEqual("specimen-test", orientation["specimen_id"])
        self.assertEqual("design-test", orientation["design_id"])
        self.assertGreater(
            orientation["ambiguity"]["equivalent_hypothesis_count"], 1
        )
        self.assertEqual(
            ["declaration_absent", "orientation_symmetry_unresolved"],
            orientation["reason_codes"],
        )
        self.assertFalse(orientation["provenance"]["ct_accessed"])
        self.assertFalse(orientation["provenance"]["aligned_graph_accessed"])

    def test_valid_declared_transform_resolves_symmetry_and_replays(self) -> None:
        declaration, digest = self._write_declaration()
        output_name = "declared-pass.json"
        first = self._resolve_with_declaration(
            declaration,
            digest,
            output_name=output_name,
        )
        artifact_bytes = (self.root / output_name).read_bytes()
        replay = self._resolve_with_declaration(
            declaration,
            digest,
            output_name=output_name,
        )

        self.assertEqual("pass", first["gate"])
        self.assertEqual("specimen-test", first["specimen_id"])
        self.assertEqual("design-test", first["design_id"])
        self.assertEqual("declared_transform", first["resolution_source"])
        self.assertGreater(
            first["ambiguity"]["equivalent_hypothesis_count"], 1
        )
        self.assertTrue(first["ambiguity"]["resolved_by_declaration"])
        self.assertTrue(first["gates"]["authoritative_transform_verified"])
        self.assertEqual(
            AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
            first["transform"]["scale_mm_per_design_unit"],
        )
        self.assertEqual(
            "declared-transform-test-001",
            first["declared_transform"]["declaration_id"],
        )
        self.assertEqual(
            ["declared_transform_verified"], first["reason_codes"]
        )
        self.assertEqual(36, first["support"]["sample_support_count"])
        self.assertEqual(9, first["support"]["samples_per_edge"])
        self.assertTrue(first["provenance"]["all_sampled_edge_support_used"])
        self.assertFalse(replay["artifact"]["changed"])
        self.assertEqual(artifact_bytes, (self.root / output_name).read_bytes())

    def test_declared_transform_requires_support_at_every_sampled_edge_point(
        self,
    ) -> None:
        midpoint_only = self.root / "midpoint-only.stl"
        centers = [
            np.asarray([value * AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT, 0.0, 0.0])
            for value in (-1.5, -0.5, 0.5, 1.5)
        ]
        write_binary_stl(
            midpoint_only,
            [point for center in centers for point in [center] * 175],
        )
        declaration, digest = self._write_declaration(
            document_overrides={
                "full_design_stl_sha256": sha256_file(midpoint_only)
            }
        )
        result = self._resolve_with_declaration(
            declaration,
            digest,
            output_name="midpoint-only-support.json",
            baseline_stl_path=midpoint_only,
        )
        self.assertEqual("halt", result["gate"])
        self.assertIn(
            "declared_transform_geometry_unsupported",
            result["reason_codes"],
        )
        self.assertGreater(
            result["support"][
                "p99_minus_p01_nearest_triangle_centroid_mm"
            ],
            DECLARED_TRANSFORM_TOLERANCES[
                "maximum_edge_support_p99_minus_p01_mm"
            ],
        )

    def test_declared_transform_rejects_malformed_and_non_finite_values(self) -> None:
        cases = [
            (
                {"rotation_matrix": [[1.0, 0.0], [0.0, 1.0]]},
                True,
                "transform_non_finite_or_malformed",
            ),
            (
                {"translation_mm": [0.0, float("nan"), 0.0]},
                False,
                "transform_non_finite_or_malformed",
            ),
        ]
        for index, (override, valid_self_hash, reason) in enumerate(cases):
            with self.subTest(index=index):
                declaration, digest = self._write_declaration(
                    transform_overrides=override,
                    valid_self_hash=valid_self_hash,
                )
                result = self._resolve_with_declaration(
                    declaration,
                    digest,
                    output_name=f"malformed-{index}.json",
                )
                self.assertEqual("halt", result["gate"])
                self.assertEqual([reason], result["reason_codes"])

    def test_declared_transform_enforces_handedness_policy(self) -> None:
        reflection = np.diag([-1.0, 1.0, 1.0]).tolist()
        undeclared, undeclared_digest = self._write_declaration(
            transform_overrides={"rotation_matrix": reflection}
        )
        rejected = self._resolve_with_declaration(
            undeclared,
            undeclared_digest,
            output_name="left-handed-rejected.json",
        )
        self.assertEqual("halt", rejected["gate"])
        self.assertEqual(["wrong_handedness"], rejected["reason_codes"])

        declared, declared_digest = self._write_declaration(
            transform_overrides={
                "rotation_matrix": reflection,
                "reflection_permitted": True,
            }
        )
        contract_rejected = self._resolve_with_declaration(
            declared,
            declared_digest,
            output_name="reflection-contract-rejected.json",
        )
        self.assertEqual("halt", contract_rejected["gate"])
        self.assertEqual(
            ["reflection_not_authorized"], contract_rejected["reason_codes"]
        )
        accepted = self._resolve_with_declaration(
            declared,
            declared_digest,
            output_name="reflection-accepted.json",
            allow_reflection=True,
        )
        self.assertEqual("pass", accepted["gate"])
        self.assertLess(accepted["transform"]["determinant"], 0.0)

    def test_declared_transform_rejects_wrong_scale_and_unsupported_geometry(
        self,
    ) -> None:
        wrong_scale, wrong_scale_digest = self._write_declaration(
            transform_overrides={"scale_mm_per_design_unit": 2.3052}
        )
        scale_result = self._resolve_with_declaration(
            wrong_scale,
            wrong_scale_digest,
            output_name="wrong-scale.json",
        )
        self.assertEqual("halt", scale_result["gate"])
        self.assertEqual(
            ["authoritative_scale_mismatch"], scale_result["reason_codes"]
        )

        wrong_translation, wrong_translation_digest = self._write_declaration(
            transform_overrides={"translation_mm": [10.0, 0.0, 0.0]}
        )
        translation_result = self._resolve_with_declaration(
            wrong_translation,
            wrong_translation_digest,
            output_name="wrong-translation.json",
        )
        self.assertEqual("halt", translation_result["gate"])
        self.assertIn(
            "declared_transform_geometry_unsupported",
            translation_result["reason_codes"],
        )

        angle = np.deg2rad(45.0)
        unsupported_rotation = [
            [float(np.cos(angle)), float(-np.sin(angle)), 0.0],
            [float(np.sin(angle)), float(np.cos(angle)), 0.0],
            [0.0, 0.0, 1.0],
        ]
        unsupported, unsupported_digest = self._write_declaration(
            transform_overrides={"rotation_matrix": unsupported_rotation}
        )
        unsupported_result = self._resolve_with_declaration(
            unsupported,
            unsupported_digest,
            output_name="unsupported-geometry.json",
        )
        self.assertEqual("halt", unsupported_result["gate"])
        self.assertIn(
            "declared_transform_contradicts_geometry_search",
            unsupported_result["reason_codes"],
        )

    def test_declared_transform_rejects_identity_and_hash_mismatches(self) -> None:
        cases = [
            ({}, {"specimen_id": "wrong-specimen"}, "specimen_id_mismatch"),
            ({}, {"design_id": "wrong-design"}, "design_id_mismatch"),
            (
                {"nominal_graph_sha256": "0" * 64},
                {},
                "nominal_graph_hash_mismatch",
            ),
            (
                {"full_design_stl_sha256": "0" * 64},
                {},
                "full_design_stl_hash_mismatch",
            ),
        ]
        for index, (document_override, request_override, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                declaration, digest = self._write_declaration(
                    document_overrides=document_override
                )
                result = self._resolve_with_declaration(
                    declaration,
                    digest,
                    output_name=f"identity-mismatch-{index}.json",
                    **request_override,
                )
                self.assertEqual("halt", result["gate"])
                self.assertEqual([reason], result["reason_codes"])

        declaration, digest = self._write_declaration()
        artifact_result = self._resolve_with_declaration(
            declaration,
            "0" * 64,
            output_name="artifact-hash-mismatch.json",
        )
        self.assertEqual("halt", artifact_result["gate"])
        self.assertEqual(
            ["declaration_artifact_hash_mismatch"],
            artifact_result["reason_codes"],
        )

        self_hash_declaration, _ = self._write_declaration(valid_self_hash=False)
        self_hash_result = self._resolve_with_declaration(
            self_hash_declaration,
            sha256_file(self_hash_declaration),
            output_name="self-hash-mismatch.json",
        )
        self.assertEqual("halt", self_hash_result["gate"])
        self.assertEqual(
            ["declaration_self_hash_mismatch"], self_hash_result["reason_codes"]
        )

    def test_labeling_revalidates_declared_transform_before_variant_access(
        self,
    ) -> None:
        declaration, digest = self._write_declaration()
        orientation_name = "declared-for-labeling.json"
        orientation = self._resolve_with_declaration(
            declaration,
            digest,
            output_name=orientation_name,
        )
        self.assertEqual("pass", orientation["gate"])
        report = label_deleted_edges(
            self.graph_path,
            self.root / "0.stl",
            {
                "0p1": self.root / "0p1.stl",
                "0p5": self.root / "0p5.stl",
                "1p0": self.root / "1p0.stl",
            },
            self.root / orientation_name,
            self.root / "declared-labels",
            specimen_id="specimen-test",
            design_id="design-test",
        )
        self.assertTrue(
            report["provenance"]["orientation_revalidated_before_variant_access"]
        )
        self.assertEqual(
            "declared_transform",
            report["provenance"]["orientation_resolution_source"],
        )
        self.assertEqual("specimen-test", report["specimen_id"])
        self.assertEqual("design-test", report["design_id"])

        mismatched_output = self.root / "identifier-mismatch-labels"
        with self.assertRaisesRegex(ValueError, "specimen_id mismatch"):
            label_deleted_edges(
                self.graph_path,
                self.root / "0.stl",
                {"0p1": self.root / "0p1.stl"},
                self.root / orientation_name,
                mismatched_output,
                specimen_id="wrong-specimen",
                design_id="design-test",
            )
        self.assertFalse(mismatched_output.exists())

        tampered = json.loads((self.root / orientation_name).read_text())
        tampered["transform"]["translation_mm"] = [0.25, 0.0, 0.0]
        tampered_path = self.root / "tampered-orientation.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs from its declaration"):
            label_deleted_edges(
                self.graph_path,
                self.root / "0.stl",
                {"0p1": self.root / "0p1.stl"},
                tampered_path,
                self.root / "tampered-labels",
                specimen_id="specimen-test",
                design_id="design-test",
            )
        self.assertFalse((self.root / "tampered-labels").exists())

    def test_orientation_interface_cannot_consume_answer_bearing_inputs(self) -> None:
        parameters = set(inspect.signature(resolve_cad_graph_orientation).parameters)
        self.assertFalse(
            parameters
            & {
                "variant_stl_paths",
                "deleted_edge_ids",
                "deleted_strut_ids",
                "expected_deletions",
            }
        )
        self.assertTrue(
            {
                "allow_reflection",
                "sample_count",
                "sample_start",
                "sample_end",
                "scale_candidates",
                "ambiguity_absolute_mm",
                "ambiguity_relative_fraction",
                "expected_counts",
                "config_sha256",
            }.isdisjoint(parameters)
        )
        label_parameters = set(inspect.signature(label_deleted_edges).parameters)
        self.assertTrue(
            {
                "expected_deletions",
                "radius_margin_mm",
                "radius_rounding_mm",
                "sample_count",
                "split_seed",
                "config_sha256",
            }.isdisjoint(label_parameters)
        )

    def test_stage1_policy_digest_and_authoritative_scale_are_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_stage1_policy(
                self.policy_path,
                expected_artifact_sha256="0" * 64,
            )
        altered = json.loads(self.policy_path.read_text(encoding="utf-8"))
        altered["orientation_verification"][
            "scale_candidates_mm_per_design_unit"
        ] = [2.3052]
        altered_path = self.root / "altered-stage1-policy.json"
        altered_path.write_text(
            json.dumps(altered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "2.28"):
            load_stage1_policy(
                altered_path,
                expected_artifact_sha256=sha256_file(altered_path),
            )


if __name__ == "__main__":
    unittest.main()
