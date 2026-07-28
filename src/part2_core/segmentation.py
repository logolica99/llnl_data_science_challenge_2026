"""Memory-aware canonical segmentation, comparison, and slice rendering."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import sha256_file, sha256_json, write_json_atomic
from .volume import AXIS_MAPPING, load_volume


RegistrationMode = Literal["challenge_aligned_json", "autonomous_v2"]


def segment_ct_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    threshold: float,
    registration_mode: RegistrationMode,
    retention: Literal["committed", "regenerable"] = "committed",
    chunk_depth: int = 16,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a canonical uint8 mask in bounded Z slabs."""

    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be positive")
    volume = load_volume(input_path)
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".npy":
        raise ValueError("Canonical mask output must use .npy")
    total = int(np.prod(volume.shape, dtype=np.int64))

    def scan(existing: np.ndarray | None = None) -> tuple[int, bool]:
        count = 0
        exact = True
        for start in range(0, volume.shape[0], chunk_depth):
            end = min(start + chunk_depth, volume.shape[0])
            expected = np.asarray(volume.array[start:end] >= threshold, dtype=np.uint8)
            count += int(np.count_nonzero(expected))
            if existing is not None and not np.array_equal(existing[start:end], expected):
                exact = False
        return count, exact

    changed = True
    if destination.exists():
        existing = np.load(destination, mmap_mode="r", allow_pickle=False)
        if existing.shape != volume.shape or existing.dtype != np.uint8:
            raise FileExistsError("Existing canonical mask violates dtype/shape contract")
        foreground, exact = scan(existing)
        if not exact:
            raise FileExistsError("Existing canonical mask has different content")
        changed = False
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npy",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            target = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.uint8, shape=volume.shape
            )
            foreground = 0
            for start in range(0, volume.shape[0], chunk_depth):
                end = min(start + chunk_depth, volume.shape[0])
                slab = np.asarray(volume.array[start:end] >= threshold, dtype=np.uint8)
                target[start:end] = slab
                foreground += int(np.count_nonzero(slab))
            target.flush()
            del target
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    config_hash = sha256_json(
        {
            "threshold": float(threshold),
            "comparison": "value >= threshold",
            "registration_mode": registration_mode,
            "dtype": "uint8",
            "shape_zyx": list(volume.shape),
            "retention": retention,
            "chunk_depth": chunk_depth,
        }
    )
    return {
        "gate": "pass",
        "result": {
            "threshold": float(threshold),
            "threshold_comparison": "value >= threshold",
            "shape": list(volume.shape),
            "dtype": "uint8",
            "foreground_voxels": foreground,
            "total_voxels": total,
            "foreground_fraction": foreground / total if total else 0.0,
            "registration_mode": registration_mode,
            "axis_mapping": AXIS_MAPPING,
            "changed": changed,
        },
        "artifacts": {
            "canonical_mask": {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "role": "canonical_segmentation_mask",
                "dtype": "uint8",
                "shape": list(volume.shape),
                "array_axes": ["z", "y", "x"],
                "retention": retention,
            }
        },
        "hashes": {
            "input_sha256": sha256_file(volume.path),
            "canonical_mask_sha256": sha256_file(destination),
            "config_sha256": config_hash,
        },
        "warnings": [],
    }


def compare_segmentation_masks(
    raw_path: str | Path,
    mask_paths: list[str | Path],
    thresholds: list[float],
    *,
    registration_mode: RegistrationMode,
    output_report_path: str | Path | None = None,
    overwrite: bool = False,
    chunk_depth: int = 16,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate aligned mask contracts and return scalar statistics only."""

    if not mask_paths or len(mask_paths) != len(thresholds):
        raise ValueError("mask_paths and thresholds must be non-empty positional pairs")
    raw = load_volume(raw_path)
    root = Path(repository_root).expanduser().resolve() if repository_root else None

    def portable(path: Path) -> str:
        return path.relative_to(root).as_posix() if root is not None else str(path)
    candidates: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for index, (path, threshold) in enumerate(zip(mask_paths, thresholds, strict=True)):
        if not np.isfinite(threshold):
            raise ValueError("Every threshold must be finite")
        mask = load_volume(path)
        if mask.shape != raw.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match raw shape {raw.shape}")
        if mask.dtype.kind not in "bui":
            raise TypeError(f"Mask dtype must be boolean/integer, found {mask.dtype}")
        foreground = 0
        expected_foreground = 0
        mismatched = 0
        false_positive = 0
        false_negative = 0
        for start in range(0, mask.shape[0], chunk_depth):
            end = min(start + chunk_depth, mask.shape[0])
            actual = np.asarray(mask.array[start:end] != 0, dtype=bool)
            expected = np.asarray(raw.array[start:end] >= threshold, dtype=bool)
            foreground += int(np.count_nonzero(actual))
            expected_foreground += int(np.count_nonzero(expected))
            mismatch = actual != expected
            mismatched += int(np.count_nonzero(mismatch))
            false_positive += int(np.count_nonzero(actual & ~expected))
            false_negative += int(np.count_nonzero(~actual & expected))
        item = {
            "threshold": float(threshold),
            "path": portable(mask.path),
            "dtype": str(mask.dtype),
            "foreground_voxels": foreground,
            "expected_foreground_voxels": expected_foreground,
            "total_voxels": int(mask.array.size),
            "foreground_percent": 100.0 * foreground / int(mask.array.size),
            "mismatched_voxels": mismatched,
            "false_positive_voxels": false_positive,
            "false_negative_voxels": false_negative,
            "exact_threshold_match": mismatched == 0,
            "sha256": sha256_file(mask.path),
        }
        candidates.append(item)
        artifacts[f"mask_{index}"] = {
            "path": portable(mask.path),
            "sha256": item["sha256"],
            "role": "segmentation_mask_candidate",
            "dtype": item["dtype"],
            "shape": list(mask.shape),
            "retention": "regenerable",
        }
    config_hash = sha256_json(
        {"thresholds": thresholds, "registration_mode": registration_mode, "chunk_depth": chunk_depth}
    )
    all_exact = all(item["exact_threshold_match"] for item in candidates)
    result = {
        "status": "ok",
        "raw_path": portable(raw.path),
        "shape": list(raw.shape),
        "candidates": candidates,
        "registration_mode": registration_mode,
        "config_sha256": config_hash,
        "overall_pass": all_exact,
    }
    if output_report_path is not None:
        artifact = write_json_atomic(output_report_path, result, overwrite=overwrite)
        artifacts["comparison_report"] = {
            **artifact,
            "role": "segmentation_mask_comparison",
            "retention": "committed",
        }
    return {
        "gate": "pass" if all_exact else "halt",
        "result": result,
        "artifacts": artifacts,
        "hashes": {"raw_sha256": sha256_file(raw.path), "config_sha256": config_hash},
        "warnings": (
            []
            if all_exact
            else ["one or more masks do not exactly match raw >= threshold"]
        ),
    }


def visualize_slice(
    input_path: str | Path,
    output_path: str | Path,
    *,
    slice_index: int,
    axis: int,
    registration_mode: RegistrationMode,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render one bounded slice to PNG without returning pixel data."""

    volume = load_volume(input_path)
    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0, 1, or 2")
    if not 0 <= slice_index < volume.shape[axis]:
        raise IndexError(f"Slice {slice_index} is outside axis {axis}")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".png":
        raise ValueError("Slice output must use .png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = np.take(volume.array, slice_index, axis=axis)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".png",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    figure, axes = plt.subplots(figsize=(8, 8))
    try:
        axes.imshow(image, cmap="gray")
        axes.set_title(f"{volume.path.name}: axis {axis}, slice {slice_index}")
        axes.axis("off")
        figure.tight_layout()
        figure.savefig(
            temporary,
            dpi=150,
            bbox_inches="tight",
            metadata={"Software": "part2-core"},
        )
    finally:
        plt.close(figure)
    changed = True
    try:
        if destination.exists():
            if destination.read_bytes() != temporary.read_bytes():
                raise FileExistsError(
                    f"Slice artifact already exists with different bytes: {destination}"
                )
            changed = False
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    config_hash = sha256_json(
        {"axis": axis, "slice_index": slice_index, "registration_mode": registration_mode}
    )
    return {
        "gate": "pass",
        "result": {
            "axis": axis,
            "slice_index": slice_index,
            "source_shape": list(volume.shape),
            "registration_mode": registration_mode,
            "changed": changed,
        },
        "artifacts": {
            "slice": {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "role": "qa_slice",
                "retention": "committed",
            }
        },
        "hashes": {
            "input_sha256": sha256_file(volume.path),
            "slice_sha256": sha256_file(destination),
            "config_sha256": config_hash,
        },
        "warnings": [],
    }
