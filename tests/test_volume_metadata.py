"""Tests for memory-aware, manifest-ready volume metadata extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from volume_metadata import (  # noqa: E402
    OUTPUT_SCHEMA_VERSION,
    UNKNOWN,
    VolumeMetadataError,
    _array_chunks,
    inspect_volume,
)


class VolumeMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_npy_metadata_is_repository_relative_and_counts_nonfinite(self) -> None:
        path = self.root / "float-volume.npy"
        values = np.array(
            [[[1.0, np.nan], [np.inf, -2.0]], [[3.0, 4.0], [5.0, 6.0]]],
            dtype=">f4",
        )
        np.save(path, values)

        result = inspect_volume(path, repository_root=REPOSITORY_ROOT, chunk_voxels=3)

        self.assertEqual(OUTPUT_SCHEMA_VERSION, result["output_schema_version"])
        self.assertFalse(Path(result["path"]).is_absolute())
        self.assertEqual([2, 2, 2], result["shape"])
        self.assertEqual("float32", result["dtype"])
        self.assertEqual("big", result["byte_order"])
        self.assertEqual(6, result["statistics"]["finite_count"])
        self.assertEqual(2, result["statistics"]["nonfinite_count"])
        self.assertEqual(-2.0, result["statistics"]["minimum"])
        self.assertEqual(UNKNOWN, result["axes"])
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), result["sha256"]
        )
        self.assertEqual(
            "external", result["manifest_fragment"]["ct_volume"]["retention"]
        )

    def test_fortran_order_memmap_chunks_without_full_array_copy(self) -> None:
        path = self.root / "fortran.npy"
        np.save(path, np.asfortranarray(np.arange(24).reshape(2, 3, 4)))
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)

        chunks = list(_array_chunks(mapped, 5))

        self.assertEqual(24, sum(chunk.size for chunk in chunks))
        self.assertTrue(np.shares_memory(mapped, chunks[0]))

    def test_skip_hash_and_invalid_chunk_size(self) -> None:
        path = self.root / "preview.npy"
        np.save(path, np.arange(8, dtype=np.uint16).reshape(2, 2, 2))

        preview = inspect_volume(
            path,
            repository_root=REPOSITORY_ROOT,
            header_only=True,
            include_sha256=False,
            retention="regenerable",
        )
        self.assertEqual(UNKNOWN, preview["sha256"])
        self.assertEqual(
            "regenerable",
            preview["manifest_fragment"]["ct_volume"]["retention"],
        )
        with self.assertRaisesRegex(VolumeMetadataError, "chunk_voxels"):
            inspect_volume(
                path,
                repository_root=REPOSITORY_ROOT,
                chunk_voxels=0,
            )

    def test_header_only_skips_voxel_statistics_but_keeps_hash(self) -> None:
        path = self.root / "header.npy"
        np.save(path, np.arange(24, dtype=np.uint16).reshape(2, 3, 4))

        result = inspect_volume(
            path, repository_root=REPOSITORY_ROOT, header_only=True
        )

        self.assertEqual("not_computed", result["statistics"]["status"])
        self.assertEqual(UNKNOWN, result["statistics"]["finite_count"])
        self.assertEqual(64, len(result["sha256"]))

    def test_ome_tiff_spacing_records_exact_provenance(self) -> None:
        path = self.root / "ome-volume.tiff"
        values = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        tifffile.imwrite(
            path,
            values,
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "ZYX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.75,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": 1.25,
                "PhysicalSizeZUnit": "µm",
            },
        )

        result = inspect_volume(
            path, repository_root=REPOSITORY_ROOT, header_only=True
        )

        self.assertEqual("ZYX", result["axes"])
        self.assertEqual(["z", "y", "x"], result["manifest_fragment"]["ct_metadata"]["array_axes"])
        self.assertEqual(0.5, result["voxel_spacing"]["x"]["value"])
        self.assertEqual(
            "Pixels.PhysicalSizeX",
            result["voxel_spacing"]["x"]["provenance"]["field"],
        )

    def test_missing_spacing_stays_unknown(self) -> None:
        path = self.root / "plain.tif"
        tifffile.imwrite(
            path,
            np.zeros((2, 3, 4), dtype=np.dtype(">u2")),
            byteorder=">",
            photometric="minisblack",
        )

        result = inspect_volume(
            path, repository_root=REPOSITORY_ROOT, header_only=True
        )

        self.assertEqual(UNKNOWN, result["voxel_spacing"]["z"]["value"])
        self.assertEqual("big", result["byte_order"])

    def test_path_escape_is_rejected(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npy") as outside:
            with self.assertRaisesRegex(VolumeMetadataError, "escapes repository"):
                inspect_volume(Path(outside.name), repository_root=self.root)

    def test_non_numeric_npy_is_rejected(self) -> None:
        path = self.root / "strings.npy"
        np.save(path, np.array(["not", "numeric"]))
        with self.assertRaisesRegex(VolumeMetadataError, "real numeric"):
            inspect_volume(path, repository_root=REPOSITORY_ROOT)


if __name__ == "__main__":
    unittest.main()
