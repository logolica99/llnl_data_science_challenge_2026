"""Tests for deterministic specimen input inspection and artifacts."""

from __future__ import annotations

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
)
from specimen_manifest import DEFAULT_SCHEMA, validate_manifest  # noqa: E402


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

    def _ingest(
        self,
        *,
        specimen_id: str = "test_specimen",
        registration_mode: str = "autonomous_v2",
        aligned_graph_path: Path | None = None,
        ct_path: Path | None = None,
    ) -> dict[str, object]:
        return ingest_specimen(
            repository_root=self.root,
            specimen_id=specimen_id,
            cad_path=self.cad,
            design_graph_path=self.graph,
            ct_path=ct_path or self.ct,
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
        self.assertNotIn("aligned_graph", manifest["inputs"])
        self.assertEqual(
            {"graph_summary"}, set(manifest["derived"])
        )

    def test_challenge_mode_requires_and_hashes_aligned_graph(self) -> None:
        result = self._ingest(
            specimen_id="challenge_specimen",
            registration_mode="challenge_aligned_json",
            aligned_graph_path=self.aligned,
        )

        manifest = json.loads(
            Path(result["paths"]["specimen_manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual("aligned_graph", manifest["inputs"]["aligned_graph"]["role"])
        self.assertEqual(
            manifest["derived"]["graph_summary"]["values"],
            manifest["derived"]["graph_summary"]["aligned_values"],
        )

    def test_challenge_mode_rejects_missing_aligned_graph(self) -> None:
        with self.assertRaisesRegex(SpecimenIngestError, "requires.*aligned graph"):
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
            self._ingest(ct_path=non_3d)

    def test_tiff_intake_uses_header_metadata_without_segmentation(self) -> None:
        tiff = self.data / "scan.tiff"
        tifffile.imwrite(
            tiff,
            np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        result = self._ingest(ct_path=tiff)

        manifest = json.loads(
            Path(result["paths"]["specimen_manifest"]).read_text(encoding="utf-8")
        )
        self.assertEqual("tiff", manifest["inputs"]["ct_metadata"]["format"])
        receipt = json.loads(
            Path(result["paths"]["ingest_receipt"]).read_text(encoding="utf-8")
        )
        self.assertTrue(receipt["self_verification"]["segmentation_not_run"])

    def test_path_outside_configured_data_root_is_rejected(self) -> None:
        outside = self.root / "outside.json"
        self._write_graph(outside)
        with self.assertRaisesRegex(SpecimenIngestError, "outside configured data roots"):
            inspect_lattice_graph(
                outside,
                repository_root=self.root,
                allowed_roots=[self.data],
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
        result = ingest_specimen(
            repository_root=self.root,
            specimen_id="ambiguous_specimen",
            cad_path=self.cad,
            design_graph_path=self.graph,
            ct_path=self.ct,
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
