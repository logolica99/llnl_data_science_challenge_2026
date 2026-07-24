"""Deterministic core tests for the first Part 2 tooling slice."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from part2_core.graph import normalize_lattice_graph  # noqa: E402
from part2_core.otsu import (  # noqa: E402
    deterministic_histogram,
    otsu_from_histogram,
    replay_exact_otsu,
    write_otsu_artifacts,
)
from part2_core.response import error_response, success_response  # noqa: E402
from part2_core.volume import (  # noqa: E402
    AXIS_MAPPING,
    load_volume,
    sample_xyz,
    xyz_to_zyx_indices,
)


class Part2VolumeCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_npy_float_is_memory_mapped_without_full_conversion(self) -> None:
        path = self.root / "float.npy"
        values = np.arange(24, dtype=np.dtype(">f4")).reshape(2, 3, 4)
        np.save(path, values)

        volume = load_volume(path)

        self.assertIsInstance(volume.array, np.memmap)
        self.assertEqual("float32", volume.dtype.name)
        self.assertEqual("big", volume.byte_order)
        self.assertEqual((2, 3, 4), volume.shape)
        self.assertEqual(float(values[1, 2, 3]), sample_xyz(volume, [3, 2, 1]))

    def test_big_endian_uint16_tiff_is_memory_mapped(self) -> None:
        path = self.root / "big-endian.tiff"
        values = np.arange(60, dtype=np.dtype(">u2")).reshape(3, 4, 5)
        tifffile.imwrite(
            path,
            values,
            byteorder=">",
            photometric="minisblack",
        )

        volume = load_volume(path)

        self.assertIsInstance(volume.array, np.memmap)
        self.assertEqual("uint16", volume.dtype.name)
        self.assertEqual("big", volume.byte_order)
        self.assertEqual(int(values[2, 3, 4]), sample_xyz(volume, [4, 3, 2]))

    def test_axis_sentinel_pins_xyz_to_volume_zyx(self) -> None:
        sentinel = np.zeros((3, 4, 5), dtype=np.uint16)
        sentinel[2, 1, 4] = 65_000

        self.assertEqual(
            {
                "coordinate_order": ["x", "y", "z"],
                "array_axes": ["z", "y", "x"],
                "numpy_index_expression": "volume[round(z), round(y), round(x)]",
            },
            AXIS_MAPPING,
        )
        self.assertEqual((2, 1, 4), xyz_to_zyx_indices([4, 1, 2], sentinel.shape))
        self.assertEqual(65_000, sample_xyz(sentinel, [4, 1, 2]))


class Part2GraphCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _assert_lattice(
        self,
        source: Path,
        *,
        size: int,
        nodes: int,
        edges: int,
        cells: int,
    ) -> None:
        output = self.root / f"normalized-{size}.npz"
        result = normalize_lattice_graph(source, output)

        self.assertEqual(
            {"nodes": nodes, "edges": edges, "cells": cells},
            result["counts"],
        )
        self.assertEqual([size, size, size], result["cell_grid_shape_xyz"])
        self.assertEqual([], result["warnings"])
        self.assertEqual(64, len(result["source_sha256"]))
        self.assertEqual(64, len(result["artifact_sha256"]))
        with np.load(output, allow_pickle=False) as graph:
            self.assertEqual(nodes, graph["node_ids"].size)
            self.assertEqual(edges, graph["edge_ids"].size)
            self.assertEqual(cells + 1, graph["cell_edge_offsets"].size)
            self.assertTrue(
                np.array_equal(
                    graph["node_id_rows"],
                    np.arange(nodes, dtype=np.int64),
                )
            )
            first_edge_node_ids = graph["edge_node_ids"][0]
            first_edge_node_rows = graph["edge_node_rows"][0]
            self.assertTrue(
                np.array_equal(
                    graph["node_ids"][first_edge_node_rows],
                    first_edge_node_ids,
                )
            )

    def test_8x8x8_graph_counts_are_derived_from_input(self) -> None:
        self._assert_lattice(
            REPOSITORY_ROOT
            / "data/octet_truss_8x8x8/octet_truss_8x8x8.json",
            size=8,
            nodes=7_168,
            edges=13_056,
            cells=512,
        )

    def test_9x9x9_graph_counts_are_derived_from_input(self) -> None:
        self._assert_lattice(
            REPOSITORY_ROOT / "data/missing_struts/octet_truss_9x9x9.json",
            size=9,
            nodes=10_206,
            edges=18_468,
            cells=729,
        )

    def test_noncontiguous_ids_use_explicit_maps(self) -> None:
        source = self.root / "noncontiguous.json"
        source.write_text(
            json.dumps(
                {
                    "junctions": [
                        {"id": 10, "position": [0, 0, 0]},
                        {"id": 30, "position": [1, 0, 0]},
                    ],
                    "struts": [
                        {"id": 99, "junction0": 30, "junction1": 10},
                    ],
                    "unit_cells": [
                        {"id": 7, "indices": [0, 0, 0], "struts": [99]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "noncontiguous.npz"

        result = normalize_lattice_graph(source, output)

        self.assertFalse(result["ids_contiguous"]["nodes"])
        self.assertIn("explicit ID maps", result["warnings"][-1])
        with np.load(output, allow_pickle=False) as graph:
            self.assertEqual([30, 10], graph["edge_node_ids"][0].tolist())
            self.assertEqual([1, 0], graph["edge_node_rows"][0].tolist())


class Part2OtsuCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_native_uint16_histogram_is_exact_for_big_endian_input(self) -> None:
        values = np.array([10, 10, 20, 20, 20], dtype=np.dtype(">u2")).reshape(
            1,
            1,
            5,
        )

        histogram, encoding = deterministic_histogram(
            values,
            chunk_voxels=2,
            encoding="native_uint16",
        )
        threshold, separability = otsu_from_histogram(histogram)

        self.assertEqual(2, histogram[10])
        self.assertEqual(3, histogram[20])
        self.assertEqual(5, int(histogram.sum()))
        self.assertEqual("native_uint16", encoding["encoding"])
        self.assertEqual(10, threshold)
        self.assertEqual(1.0, separability)

    def test_float_replay_uses_chunked_affine_encoding_and_writes_artifacts(self) -> None:
        path = self.root / "float-scan.npy"
        lower = np.linspace(1.0, 2.0, 4_000, dtype=np.float32)
        upper = np.linspace(8.0, 9.0, 4_000, dtype=np.float32)
        np.save(path, np.concatenate((lower, upper)).reshape(20, 20, 20))

        result, histogram = replay_exact_otsu(
            path,
            recipe={
                "chunk_voxels": 127,
                "minimum_significant_peaks": 1,
                "maximum_foreground_fraction": 0.75,
            },
        )
        artifacts = write_otsu_artifacts(self.root / "otsu", result, histogram)

        self.assertEqual(
            "full_volume_affine_uint16",
            result["histogram_encoding"]["encoding"],
        )
        self.assertEqual(8_000, result["voxel_count"])
        self.assertEqual(64, len(result["histogram_sha256"]))
        self.assertTrue(Path(artifacts["histogram"]["path"]).is_file())
        self.assertTrue(Path(artifacts["report"]["path"]).is_file())


class Part2ResponseSchemaTests(unittest.TestCase):
    def test_success_gates_include_pass_and_manual_review(self) -> None:
        passed = success_response(
            tool="example",
            gate="pass",
            summary="done",
            result={"count": 1},
        )
        review = success_response(
            tool="example",
            gate="manual_review",
            summary="review",
            result={},
        )

        self.assertEqual("pass", passed["gate"])
        self.assertEqual("manual_review", review["gate"])
        self.assertEqual("ok", review["status"])
        self.assertIsNone(review["error"])

    def test_error_is_structured_and_halts(self) -> None:
        result = error_response(
            tool="example",
            code="invalid_input",
            error_type="ValueError",
            message="bad input",
            details={"field": "path"},
        )

        self.assertEqual("error", result["status"])
        self.assertEqual("halt", result["gate"])
        self.assertEqual("invalid_input", result["error"]["code"])
        self.assertEqual({"field": "path"}, result["error"]["details"])


if __name__ == "__main__":
    unittest.main()
