"""Tests for deterministic specimen input inspection and artifacts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from specimen_ingest import (  # noqa: E402
    SpecimenIngestError,
    ingest_specimen,
    inspect_lattice_graph,
    validate_ingest_artifact_bundle,
)
from specimen_manifest import (  # noqa: E402
    DEFAULT_SCHEMA,
    canonical_json_sha256,
    sha256_file,
    validate_manifest,
)
from tests.stage0_metadata_fixture import (  # noqa: E402
    write_ct_metadata_response_fixture,
)


class SpecimenIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.graph = self.data / "design.json"
        self.aligned = self.data / "aligned.json"
        self.cad = self.data / "design.stl"
        self.ct = self.data / "scan.npy"
        self._write_graph(self.graph)
        self._write_graph(self.aligned)
        trimesh.creation.box(extents=(1.0, 2.0, 3.0)).export(self.cad)
        np.save(self.ct, np.arange(24, dtype=np.float32).reshape(2, 3, 4))

    def _write_graph(self, path: Path, *, bad_reference: bool = False) -> None:
        graph = {
            "junctions": [
                {"id": 0, "position": [0.0, 0.0, 0.0]},
                {"id": 1, "position": [1.0, 1.0, 1.0]},
            ],
            "struts": [
                {"id": 10, "junction0": 0, "junction1": 99 if bad_reference else 1}
            ],
            "unit_cells": [{"id": 20, "struts": [10]}],
        }
        path.write_text(json.dumps(graph), encoding="utf-8")

    def _write_declaration(
        self,
        path: Path,
        *,
        specimen_id: str = "test_specimen",
        canonical_hash: str | None = None,
    ) -> None:
        document = {
            "schema_version": "part2-graph-to-stl-transform-declaration/1.0.0",
            "declaration_id": "test-transform-001",
            "source_id": "scientist-source",
            "provenance_id": "test-intake-001",
            "specimen_id": specimen_id,
            "design_id": "test_design",
            "nominal_graph_sha256": sha256_file(self.graph),
            "full_design_stl_sha256": sha256_file(self.cad),
            "transform": {
                "convention": "stl_mm = scale * ((design_xyz - design_center) @ rotation.T) + translation_mm",
                "design_center": [0.5, 0.5, 0.5],
                "scale_mm_per_design_unit": 2.28,
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_mm": [0.0, 0.0, 0.0],
                "reflection_permitted": False,
            },
        }
        document["canonical_declaration_sha256"] = (
            canonical_hash or canonical_json_sha256(document)
        )
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_ct_metadata_response(
        self,
        *,
        specimen_id: str,
        ct_path: Path | None = None,
        shape: tuple[int, ...] = (2, 3, 4),
        dtype: str = "float32",
        dtype_string: str = "<f4",
        byte_order: str = "little",
        volume_format: str = "npy",
        axes: str = "unknown",
        array_axes: list[str] | str = "unknown",
    ) -> tuple[Path, str]:
        return write_ct_metadata_response_fixture(
            repository_root=self.root,
            specimen_id=specimen_id,
            ct_path=ct_path or self.ct,
            shape=shape,
            dtype=dtype,
            dtype_string=dtype_string,
            byte_order=byte_order,
            volume_format=volume_format,
            retention="external",
            axes=axes,
            array_axes=array_axes,
        )

    def _ingest(
        self,
        *,
        specimen_id: str = "test_specimen",
        registration_mode: str = "autonomous_v2",
        aligned_graph_path: Path | None = None,
        ct_path: Path | None = None,
        ct_shape: tuple[int, ...] = (2, 3, 4),
        ct_dtype: str = "float32",
        ct_dtype_string: str = "<f4",
        ct_byte_order: str = "little",
        ct_format: str = "npy",
        ct_axes: str = "unknown",
        ct_array_axes: list[str] | str = "unknown",
        metadata_response: tuple[Path, str] | None = None,
        metadata_call_receipt: tuple[Path, str] | None = None,
    ) -> dict[str, object]:
        selected_ct = ct_path or self.ct
        metadata_path, metadata_sha256 = metadata_response or (
            self._write_ct_metadata_response(
                specimen_id=specimen_id,
                ct_path=selected_ct,
                shape=ct_shape,
                dtype=ct_dtype,
                dtype_string=ct_dtype_string,
                byte_order=ct_byte_order,
                volume_format=ct_format,
                axes=ct_axes,
                array_axes=ct_array_axes,
            )
        )
        call_receipt_path, call_receipt_sha256 = metadata_call_receipt or (
            metadata_path.with_name("ct_metadata_mcp_call_receipt.json"),
            sha256_file(
                metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
            ),
        )
        return ingest_specimen(
            repository_root=self.root,
            specimen_id=specimen_id,
            design_id="test_design",
            requested_analysis_scope="roi_screening",
            cad_path=self.cad,
            design_graph_path=self.graph,
            ct_path=selected_ct,
            ct_metadata_response_path=metadata_path,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=call_receipt_path,
            ct_metadata_call_receipt_sha256=call_receipt_sha256,
            aligned_graph_path=aligned_graph_path,
            registration_mode=registration_mode,
            association_confirmed=True,
            cad_units="millimeter",
            cad_units_provenance="scientist declaration",
            graph_axes="xyz",
            array_axes="zyx",
            aligned_graph_units=(
                "voxel"
                if registration_mode == "challenge_aligned_json"
                else "simulation_voxel"
            ),
            retention="external",
            schema_path=DEFAULT_SCHEMA,
        )

    def _validate_bundle(self, result: dict[str, object]) -> dict[str, object]:
        paths = result["paths"]
        assert isinstance(paths, dict)
        return validate_ingest_artifact_bundle(
            repository_root=self.root,
            manifest_path=Path(paths["specimen_manifest"]),
            request_path=Path(paths["ingest_request"]),
            receipt_path=Path(paths["ingest_receipt"]),
            ct_metadata_response_path=Path(paths["ct_metadata_response"]),
            ct_metadata_call_receipt_path=Path(
                paths["ct_metadata_mcp_call_receipt"]
            ),
            schema_path=DEFAULT_SCHEMA,
            expected_specimen_id=str(result["specimen_id"]),
            expected_design_id=str(result["design_id"]),
            expected_analysis_scope=str(result["requested_analysis_scope"]),
            expected_registration_mode="autonomous_v2",
            require_ready=True,
        )

    def test_core_requires_bound_artifact_and_has_no_inspector_fallback(self) -> None:
        signature = inspect.signature(ingest_specimen)
        self.assertIs(
            inspect.Parameter.empty,
            signature.parameters["ct_metadata_response_path"].default,
        )
        self.assertIs(
            inspect.Parameter.empty,
            signature.parameters["ct_metadata_response_sha256"].default,
        )
        self.assertIs(
            inspect.Parameter.empty,
            signature.parameters["ct_metadata_call_receipt_path"].default,
        )
        self.assertIs(
            inspect.Parameter.empty,
            signature.parameters["ct_metadata_call_receipt_sha256"].default,
        )
        self.assertNotIn("ct_metadata", signature.parameters)

        source = inspect.getsource(sys.modules[ingest_specimen.__module__])
        self.assertNotIn("from volume_metadata import", source)
        self.assertNotIn("inspect_volume(", source)

    def test_production_intake_accepts_nominal_graph_and_ct_without_cad(self) -> None:
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="production_specimen",
            array_axes=["z", "y", "x"],
        )
        call_path = metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
        normalized = (
            self.root
            / "analysis"
            / "production_specimen"
            / "design"
            / "normalized_nominal_graph.npz"
        )
        normalized.parent.mkdir(parents=True)
        np.savez(normalized, junction_ids=np.array([0, 1]), strut_ids=np.array([10]))
        result = ingest_specimen(
            repository_root=self.root,
            specimen_id="production_specimen",
            design_id="test_design",
            requested_analysis_scope="roi_screening",
            design_graph_path=self.graph,
            ct_path=self.ct,
            ct_metadata_response_path=metadata_path,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=call_path,
            ct_metadata_call_receipt_sha256=sha256_file(call_path),
            registration_mode="autonomous_v2",
            association_confirmed=True,
            graph_axes="xyz",
            array_axes="zyx",
            retention="external",
            schema_path=DEFAULT_SCHEMA,
            normalized_graph_path=normalized,
            normalized_graph_sha256=sha256_file(normalized),
        )
        manifest = json.loads(
            Path(result["paths"]["specimen_manifest"]).read_text(encoding="utf-8")
        )
        self.assertNotIn("cad", manifest["inputs"])
        self.assertNotIn("cad_inspection", manifest["intake"])
        self.assertEqual(
            "normalized_nominal_graph",
            manifest["inputs"]["normalized_nominal_graph"]["role"],
        )
        self._validate_bundle({**result, "requested_analysis_scope": "roi_screening"})

    def test_autonomous_intake_writes_ready_idempotent_artifacts(self) -> None:
        first = self._ingest()
        second = self._ingest()

        self.assertEqual("ready_for_data_prep", first["lifecycle_state"])
        self.assertEqual(first["canonical_hashes"], second["canonical_hashes"])
        self.assertEqual(
            {
                "ingest_request": False,
                "specimen_manifest": False,
                "ingest_receipt": False,
            },
            second["changed"],
        )
        manifest_path = Path(first["paths"]["specimen_manifest"])
        self.assertEqual(
            [],
            validate_manifest(
                manifest_path,
                schema_path=DEFAULT_SCHEMA,
                repository_root=self.root,
            ),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual("test_design", manifest["design_id"])
        self.assertEqual(
            "roi_screening",
            manifest["analysis_parameters"]["requested_analysis_scope"],
        )
        self.assertNotIn("aligned_graph", manifest["inputs"])
        self.assertEqual(
            {"graph_summary"}, set(manifest["derived"])
        )

    def test_challenge_mode_is_not_a_production_intake_mode(self) -> None:
        with self.assertRaisesRegex(SpecimenIngestError, "Unsupported registration mode"):
            self._ingest(
                specimen_id="challenge_specimen",
                registration_mode="challenge_aligned_json",
                aligned_graph_path=self.aligned,
            )

    def test_challenge_mode_rejects_missing_aligned_graph(self) -> None:
        with self.assertRaisesRegex(SpecimenIngestError, "Unsupported registration mode"):
            self._ingest(registration_mode="challenge_aligned_json")

    def test_malformed_graph_reference_is_rejected(self) -> None:
        self._write_graph(self.graph, bad_reference=True)
        with self.assertRaisesRegex(SpecimenIngestError, "unknown junctions"):
            self._ingest()

    def test_unreadable_stl_is_rejected(self) -> None:
        self.cad.write_bytes(b"not an STL")
        with self.assertRaisesRegex(SpecimenIngestError, "Unreadable STL|non-empty"):
            self._ingest()

    def test_non_3d_ct_is_rejected(self) -> None:
        non_3d = self.data / "flat.npy"
        np.save(non_3d, np.zeros((4, 5), dtype=np.uint16))
        with self.assertRaisesRegex(SpecimenIngestError, "must be 3D"):
            self._ingest(
                ct_path=non_3d,
                ct_shape=(4, 5),
                ct_dtype="uint16",
                ct_dtype_string="<u2",
            )

    def test_tiff_intake_uses_header_metadata_without_segmentation(self) -> None:
        tiff = self.data / "scan.tiff"
        tifffile.imwrite(
            tiff,
            np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        result = self._ingest(
            ct_path=tiff,
            ct_dtype="uint16",
            ct_dtype_string="<u2",
            ct_format="tiff",
            ct_axes="ZYX",
            ct_array_axes=["z", "y", "x"],
        )

        manifest = json.loads(
            Path(result["paths"]["specimen_manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual("tiff", manifest["inputs"]["ct_metadata"]["format"])
        receipt = json.loads(
            Path(result["paths"]["ingest_receipt"]).read_text(encoding="utf-8")
        )
        self.assertTrue(receipt["self_verification"]["segmentation_not_run"])

    def test_ct_metadata_response_is_closed_and_hash_bound(self) -> None:
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        with self.assertRaisesRegex(SpecimenIngestError, "SHA-256"):
            self._ingest(
                metadata_response=(metadata_path, "0" * 64)
            )

        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        document["unexpected_authority_channel"] = True
        metadata_path.write_text(json.dumps(document), encoding="utf-8")
        open_hash = sha256_file(metadata_path)
        with self.assertRaisesRegex(SpecimenIngestError, "unexpected keys"):
            self._ingest(metadata_response=(metadata_path, open_hash))

    def test_ct_metadata_response_rejects_preview_and_stale_ct(self) -> None:
        metadata_path, _ = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        document = json.loads(metadata_path.read_text(encoding="utf-8"))
        document["request"]["header_only"] = False
        document["result"]["inspection_mode"] = "streaming_statistics"
        metadata_path.write_text(json.dumps(document), encoding="utf-8")
        preview_hash = sha256_file(metadata_path)
        with self.assertRaisesRegex(SpecimenIngestError, "request binding"):
            self._ingest(metadata_response=(metadata_path, preview_hash))

        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        np.save(self.ct, np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1)
        with self.assertRaisesRegex(SpecimenIngestError, "CT file"):
            self._ingest(
                metadata_response=(metadata_path, metadata_sha256)
            )

    def test_ct_metadata_call_receipt_is_required_closed_and_current(self) -> None:
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        call_path = metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
        original_call = call_path.read_bytes()
        call_path.unlink()
        with self.assertRaisesRegex(SpecimenIngestError, "call receipt.*exist"):
            self._ingest(
                metadata_response=(metadata_path, metadata_sha256),
                metadata_call_receipt=(call_path, "0" * 64),
            )

        call_path.write_bytes(original_call)
        call = json.loads(call_path.read_text(encoding="utf-8"))
        call["undeclared"] = True
        call_path.write_text(json.dumps(call), encoding="utf-8")
        with self.assertRaisesRegex(SpecimenIngestError, "unexpected keys"):
            self._ingest(
                metadata_response=(metadata_path, metadata_sha256),
                metadata_call_receipt=(call_path, sha256_file(call_path)),
            )

        metadata_path, _ = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(
            SpecimenIngestError, "exact evidence artifact|hash bindings"
        ):
            self._ingest(
                metadata_response=(metadata_path, sha256_file(metadata_path))
            )

    def test_ct_metadata_header_facts_reject_false_shape(self) -> None:
        with self.assertRaisesRegex(
            SpecimenIngestError, "header facts differ from persisted evidence"
        ):
            self._ingest(ct_shape=(4, 3, 2))

    def test_ct_metadata_artifacts_cannot_be_reused_across_specimens(self) -> None:
        metadata = self._write_ct_metadata_response(specimen_id="first_specimen")
        with self.assertRaisesRegex(
            SpecimenIngestError, "fixed specimen config path"
        ):
            self._ingest(
                specimen_id="second_specimen",
                metadata_response=metadata,
            )

    def test_bundle_reinspection_rejects_hash_consistent_malformed_stl(self) -> None:
        result = self._ingest()
        paths = result["paths"]
        assert isinstance(paths, dict)
        manifest_path = Path(paths["specimen_manifest"])
        receipt_path = Path(paths["ingest_receipt"])
        self.cad.write_bytes(b"not an STL")
        cad_sha256 = sha256_file(self.cad)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["inputs"]["cad"]["sha256"] = cad_sha256
        manifest["intake"]["cad_inspection"]["sha256"] = cad_sha256
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["input_sha256"]["cad"] = cad_sha256
        receipt["manifest_sha256"] = canonical_json_sha256(manifest)
        receipt["canonical_receipt_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "canonical_receipt_sha256"
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        with self.assertRaisesRegex(
            SpecimenIngestError, "semantic reinspection|Unreadable STL|non-empty"
        ):
            self._validate_bundle(result)

    def test_bundle_rejects_rehashed_declared_value_disagreement(self) -> None:
        result = self._ingest()
        paths = result["paths"]
        assert isinstance(paths, dict)
        request_path = Path(paths["ingest_request"])
        receipt_path = Path(paths["ingest_receipt"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["declared"]["cad_units"] = "inch"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["request_sha256"] = canonical_json_sha256(request)
        receipt["canonical_receipt_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "canonical_receipt_sha256"
            }
        )
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

        with self.assertRaisesRegex(
            SpecimenIngestError, "declared intake values differ"
        ):
            self._validate_bundle(result)

    def test_path_outside_configured_data_root_is_rejected(self) -> None:
        outside = self.root / "outside.json"
        self._write_graph(outside)
        with self.assertRaisesRegex(SpecimenIngestError, "outside configured data roots"):
            inspect_lattice_graph(
                outside,
                repository_root=self.root,
                allowed_roots=[self.data],
            )

    def test_missing_or_malformed_scope_is_rejected(self) -> None:
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        with self.assertRaisesRegex(
            SpecimenIngestError, "requested_analysis_scope"
        ):
            ingest_specimen(
                repository_root=self.root,
                specimen_id="test_specimen",
                design_id="test_design",
                requested_analysis_scope="agent_selected_roi",
                cad_path=self.cad,
                design_graph_path=self.graph,
                ct_path=self.ct,
                ct_metadata_response_path=metadata_path,
                ct_metadata_response_sha256=metadata_sha256,
                ct_metadata_call_receipt_path=metadata_path.with_name(
                    "ct_metadata_mcp_call_receipt.json"
                ),
                ct_metadata_call_receipt_sha256=sha256_file(
                    metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
                ),
                registration_mode="autonomous_v2",
                association_confirmed=True,
                cad_units="millimeter",
                cad_units_provenance="scientist declaration",
                graph_axes="xyz",
                array_axes="zyx",
                aligned_graph_units="simulation_voxel",
                retention="external",
                schema_path=DEFAULT_SCHEMA,
            )

    def test_transform_declaration_is_rejected_by_production_intake(self) -> None:
        declaration = self.data / "transform.json"
        self._write_declaration(declaration)
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        with self.assertRaisesRegex(SpecimenIngestError, "not supported"):
            ingest_specimen(
                repository_root=self.root,
                specimen_id="test_specimen",
                design_id="test_design",
                requested_analysis_scope="roi_screening",
                cad_path=self.cad,
                design_graph_path=self.graph,
                ct_path=self.ct,
                ct_metadata_response_path=metadata_path,
                ct_metadata_response_sha256=metadata_sha256,
                ct_metadata_call_receipt_path=metadata_path.with_name(
                    "ct_metadata_mcp_call_receipt.json"
                ),
                ct_metadata_call_receipt_sha256=sha256_file(
                    metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
                ),
                design_transform_declaration_path=declaration,
                registration_mode="autonomous_v2",
                association_confirmed=True,
                cad_units="millimeter",
                cad_units_provenance="scientist declaration",
                graph_axes="xyz",
                array_axes="zyx",
                aligned_graph_units="simulation_voxel",
                retention="external",
                schema_path=DEFAULT_SCHEMA,
            )

    def test_transform_declaration_rejects_bad_self_hash_before_ready(self) -> None:
        declaration = self.data / "transform-bad-self-hash.json"
        self._write_declaration(declaration, canonical_hash="0" * 64)
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="test_specimen"
        )
        with self.assertRaisesRegex(
            SpecimenIngestError, "not supported"
        ):
            ingest_specimen(
                repository_root=self.root,
                specimen_id="test_specimen",
                design_id="test_design",
                requested_analysis_scope="roi_screening",
                cad_path=self.cad,
                design_graph_path=self.graph,
                ct_path=self.ct,
                ct_metadata_response_path=metadata_path,
                ct_metadata_response_sha256=metadata_sha256,
                ct_metadata_call_receipt_path=metadata_path.with_name(
                    "ct_metadata_mcp_call_receipt.json"
                ),
                ct_metadata_call_receipt_sha256=sha256_file(
                    metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
                ),
                design_transform_declaration_path=declaration,
                registration_mode="autonomous_v2",
                association_confirmed=True,
                cad_units="millimeter",
                cad_units_provenance="scientist declaration",
                graph_axes="xyz",
                array_axes="zyx",
                aligned_graph_units="simulation_voxel",
                retention="external",
                schema_path=DEFAULT_SCHEMA,
            )

    def test_input_file_change_invalidates_prior_receipt(self) -> None:
        first = self._ingest()
        graph = json.loads(self.graph.read_text(encoding="utf-8"))
        self.graph.write_text(json.dumps(graph, indent=4), encoding="utf-8")
        second = self._ingest()

        self.assertNotEqual(
            first["canonical_hashes"]["receipt"],
            second["canonical_hashes"]["receipt"],
        )
        self.assertTrue(second["changed"]["specimen_manifest"])
        self.assertTrue(second["changed"]["ingest_receipt"])

    def test_manifest_rejects_tampered_intake_inspection_hash(self) -> None:
        result = self._ingest()
        manifest_path = Path(result["paths"]["specimen_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["intake"]["cad_inspection"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError, "cad_inspection.sha256 differs"
        ):
            validate_manifest(
                manifest_path,
                schema_path=DEFAULT_SCHEMA,
                repository_root=self.root,
            )

    def test_unknown_declarations_remain_provisional(self) -> None:
        metadata_path, metadata_sha256 = self._write_ct_metadata_response(
            specimen_id="ambiguous_specimen"
        )
        result = ingest_specimen(
            repository_root=self.root,
            specimen_id="ambiguous_specimen",
            design_id="test_design",
            requested_analysis_scope="roi_screening",
            cad_path=self.cad,
            design_graph_path=self.graph,
            ct_path=self.ct,
            ct_metadata_response_path=metadata_path,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=metadata_path.with_name(
                "ct_metadata_mcp_call_receipt.json"
            ),
            ct_metadata_call_receipt_sha256=sha256_file(
                metadata_path.with_name("ct_metadata_mcp_call_receipt.json")
            ),
            registration_mode="autonomous_v2",
            association_confirmed=True,
            cad_units="unknown",
            graph_axes="unknown",
            array_axes="unknown",
            aligned_graph_units="unknown",
            retention="external",
            schema_path=DEFAULT_SCHEMA,
        )

        self.assertEqual("provisional", result["lifecycle_state"])
        self.assertTrue(result["unresolved_fields"])


if __name__ == "__main__":
    unittest.main()
