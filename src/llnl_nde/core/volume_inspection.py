"""Memory-aware metadata extraction for CT volumes.

The output is intentionally deterministic and shaped for direct use by the
specimen-ingest manifest builder.  Paths are always repository-relative and
physical spacing is reported only when the file itself supplies provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator
from xml.etree import ElementTree

import numpy as np
import tifffile


METHOD_NAME = "volume_metadata"
METHOD_VERSION = "1.0.0"
OUTPUT_SCHEMA_VERSION = "volume-metadata/1.0.0"
SUPPORTED_SUFFIXES = {".npy": "npy", ".tif": "tiff", ".tiff": "tiff"}
UNKNOWN = "unknown"


class VolumeMetadataError(ValueError):
    """Raised when a volume cannot be inspected safely."""


def _resolve_repository_path(path: Path, repository_root: Path) -> tuple[Path, str]:
    root = repository_root.expanduser().resolve()
    candidate = path.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise VolumeMetadataError(
            f"Input path escapes repository root {root}: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise VolumeMetadataError(f"Input file does not exist: {resolved}")
    return resolved, relative.as_posix()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_dtype(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    if dtype.fields or dtype.subdtype or dtype.kind not in "buif":
        raise VolumeMetadataError(f"Expected a real numeric array, found dtype {dtype}")
    return dtype.name


def _byte_order(dtype: np.dtype[Any]) -> str:
    byteorder = np.dtype(dtype).byteorder
    if byteorder == "|":
        return "not_applicable"
    if byteorder == "=":
        return sys.byteorder
    return "little" if byteorder == "<" else "big"


def _tiff_byte_order(dtype: np.dtype[Any], file_byteorder: str) -> str:
    if np.dtype(dtype).itemsize == 1:
        return "not_applicable"
    if file_byteorder == "<":
        return "little"
    if file_byteorder == ">":
        return "big"
    raise VolumeMetadataError(f"Unsupported TIFF byte order: {file_byteorder}")


def _json_number(value: np.generic[Any] | float | int) -> int | float:
    converted = value.item() if isinstance(value, np.generic) else value
    return int(converted) if isinstance(converted, (bool, int)) else float(converted)


def _statistics(chunks: Iterable[np.ndarray[Any, Any]]) -> dict[str, Any]:
    minimum: int | float | None = None
    maximum: int | float | None = None
    total = 0.0
    finite_count = 0
    nonfinite_count = 0
    for chunk in chunks:
        array = np.asarray(chunk)
        if not array.size:
            continue
        if array.dtype.kind == "f":
            finite_mask = np.isfinite(array)
            chunk_finite = int(np.count_nonzero(finite_mask))
            nonfinite_count += int(array.size) - chunk_finite
            if not chunk_finite:
                continue
            values = array[finite_mask]
        else:
            values = array
            chunk_finite = int(array.size)
        chunk_min = _json_number(np.min(values))
        chunk_max = _json_number(np.max(values))
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        total += float(np.sum(values, dtype=np.float64))
        finite_count += chunk_finite
    mean = total / finite_count if finite_count else None
    return {
        "status": "computed",
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean if mean is None or math.isfinite(mean) else None,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
    }


def _array_chunks(
    array: np.ndarray[Any, Any], chunk_voxels: int
) -> Iterator[np.ndarray[Any, Any]]:
    """Yield storage-order chunks without copying a full Fortran-order volume."""
    if array.flags.c_contiguous or array.flags.f_contiguous:
        flat = array.ravel(order="K")
        for start in range(0, int(flat.size), chunk_voxels):
            yield flat[start : start + chunk_voxels]
        return
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="K",
        buffersize=chunk_voxels,
    )
    yield from iterator


def _unknown_statistics() -> dict[str, str]:
    return {
        "status": "not_computed",
        "minimum": UNKNOWN,
        "maximum": UNKNOWN,
        "mean": UNKNOWN,
        "finite_count": UNKNOWN,
        "nonfinite_count": UNKNOWN,
    }


def _unknown_spacing() -> dict[str, Any]:
    return {
        axis: {
            "value": UNKNOWN,
            "unit": UNKNOWN,
            "provenance": {"source": UNKNOWN, "field": UNKNOWN, "raw_value": UNKNOWN},
        }
        for axis in ("z", "y", "x")
    }


def _ome_spacing(ome_xml: str | None) -> dict[str, Any] | None:
    if not ome_xml:
        return None
    try:
        root = ElementTree.fromstring(ome_xml)
    except ElementTree.ParseError:
        return None
    pixels = next((element for element in root.iter() if element.tag.endswith("Pixels")), None)
    if pixels is None:
        return None
    spacing = _unknown_spacing()
    found = False
    for axis in ("x", "y", "z"):
        field = f"PhysicalSize{axis.upper()}"
        raw = pixels.attrib.get(field)
        if raw is None:
            continue
        try:
            value: float | str = float(raw)
        except ValueError:
            continue
        found = True
        spacing[axis] = {
            "value": value,
            "unit": pixels.attrib.get(f"{field}Unit", UNKNOWN),
            "provenance": {
                "source": "ome_xml",
                "field": f"Pixels.{field}",
                "raw_value": raw,
            },
        }
    return spacing if found else None


def _imagej_spacing(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata or "spacing" not in metadata:
        return None
    raw = metadata["spacing"]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    spacing = _unknown_spacing()
    spacing["z"] = {
        "value": value,
        "unit": str(metadata.get("unit", UNKNOWN)),
        "provenance": {
            "source": "imagej_metadata",
            "field": "spacing",
            "raw_value": raw,
        },
    }
    return spacing


def _resolution_value(value: Any) -> float | None:
    try:
        if isinstance(value, tuple):
            numerator, denominator = value
            return float(numerator) / float(denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _tiff_resolution_spacing(tif: tifffile.TiffFile) -> dict[str, Any] | None:
    if not tif.pages:
        return None
    tags = tif.pages[0].tags
    unit_tag = tags.get("ResolutionUnit")
    unit_name = str(unit_tag.value).upper() if unit_tag is not None else ""
    scale_and_unit: tuple[float, str] | None = None
    if "INCH" in unit_name or unit_name == "2":
        scale_and_unit = (25.4, "mm")
    elif "CENTIMETER" in unit_name or unit_name == "3":
        scale_and_unit = (10.0, "mm")
    if scale_and_unit is None:
        return None
    spacing = _unknown_spacing()
    found = False
    for axis, tag_name in (("x", "XResolution"), ("y", "YResolution")):
        tag = tags.get(tag_name)
        pixels_per_unit = _resolution_value(tag.value) if tag is not None else None
        if pixels_per_unit is None or pixels_per_unit <= 0:
            continue
        found = True
        scale, normalized_unit = scale_and_unit
        spacing[axis] = {
            "value": scale / pixels_per_unit,
            "unit": normalized_unit,
            "provenance": {
                "source": "tiff_tag",
                "field": tag_name,
                "raw_value": str(tag.value),
                "resolution_unit": str(unit_tag.value),
            },
        }
    return spacing if found else None


def _merge_spacing(primary: dict[str, Any], fallback: dict[str, Any] | None) -> dict[str, Any]:
    if fallback is None:
        return primary
    merged = json.loads(json.dumps(primary))
    for axis in ("z", "y", "x"):
        if merged[axis]["value"] == UNKNOWN and fallback[axis]["value"] != UNKNOWN:
            merged[axis] = fallback[axis]
    return merged


def _inspect_npy(
    path: Path, *, header_only: bool, chunk_voxels: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    dtype = np.dtype(array.dtype)
    _normalized_dtype(dtype)
    metadata = {
        "format": "npy",
        "shape": [int(value) for value in array.shape],
        "ndim": int(array.ndim),
        "dtype": _normalized_dtype(dtype),
        "dtype_string": dtype.str,
        "byte_order": _byte_order(dtype),
        "axes": UNKNOWN,
        "voxel_count": int(array.size),
        "array_bytes": int(array.nbytes),
        "voxel_spacing": _unknown_spacing(),
    }
    statistics = (
        _unknown_statistics()
        if header_only
        else _statistics(_array_chunks(array, chunk_voxels))
    )
    return metadata, statistics


def _tiff_chunks(
    path: Path, shape: tuple[int, ...], chunk_voxels: int
) -> Iterator[np.ndarray[Any, Any]]:
    try:
        mapped = tifffile.memmap(path, mode="r")
    except (OSError, ValueError, TypeError):
        mapped = None
    if mapped is not None:
        yield from _array_chunks(mapped, chunk_voxels)
        return
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        count = 0
        for page in series.pages:
            array = page.asarray()
            count += int(array.size)
            yield from _array_chunks(array, chunk_voxels)
    expected = int(np.prod(shape, dtype=np.int64))
    if count != expected:
        raise VolumeMetadataError(
            f"TIFF pages yielded {count} voxels, expected {expected} from shape {shape}"
        )


def _inspect_tiff(
    path: Path, *, header_only: bool, chunk_voxels: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tifffile.TiffFile(path) as tif:
        if not tif.series:
            raise VolumeMetadataError(f"TIFF contains no image series: {path}")
        series = tif.series[0]
        shape = tuple(int(value) for value in series.shape)
        dtype = np.dtype(series.dtype)
        _normalized_dtype(dtype)
        axes = str(series.axes) if series.axes else UNKNOWN
        byte_order = _tiff_byte_order(dtype, tif.byteorder)
        spacing = _ome_spacing(tif.ome_metadata) or _unknown_spacing()
        spacing = _merge_spacing(spacing, _imagej_spacing(tif.imagej_metadata))
        spacing = _merge_spacing(spacing, _tiff_resolution_spacing(tif))
    voxel_count = int(np.prod(shape, dtype=np.int64))
    metadata = {
        "format": "tiff",
        "shape": list(shape),
        "ndim": len(shape),
        "dtype": _normalized_dtype(dtype),
        "dtype_string": dtype.str,
        "byte_order": byte_order,
        "axes": axes,
        "voxel_count": voxel_count,
        "array_bytes": int(voxel_count * dtype.itemsize),
        "voxel_spacing": spacing,
    }
    statistics = (
        _unknown_statistics()
        if header_only
        else _statistics(_tiff_chunks(path, shape, chunk_voxels))
    )
    return metadata, statistics


def inspect_volume(
    path: Path,
    *,
    repository_root: Path,
    header_only: bool = False,
    include_sha256: bool = True,
    chunk_voxels: int = 8 * 1024 * 1024,
    retention: str = "external",
) -> dict[str, Any]:
    """Inspect one supported volume and return deterministic manifest-ready data."""
    if chunk_voxels <= 0:
        raise VolumeMetadataError("chunk_voxels must be positive")
    if retention not in {"committed", "external", "regenerable"}:
        raise VolumeMetadataError(f"Unsupported retention policy: {retention!r}")
    resolved, relative_path = _resolve_repository_path(path, repository_root)
    volume_format = SUPPORTED_SUFFIXES.get(resolved.suffix.lower())
    if volume_format is None:
        raise VolumeMetadataError(
            f"Expected a .npy, .tif, or .tiff file: {relative_path}"
        )
    if volume_format == "npy":
        metadata, statistics = _inspect_npy(
            resolved, header_only=header_only, chunk_voxels=chunk_voxels
        )
    else:
        metadata, statistics = _inspect_tiff(
            resolved, header_only=header_only, chunk_voxels=chunk_voxels
        )
    digest = _sha256_file(resolved) if include_sha256 else UNKNOWN
    array_axes: list[str] | str
    axes = metadata["axes"]
    if isinstance(axes, str) and axes != UNKNOWN and len(axes) == metadata["ndim"]:
        array_axes = [axis.lower() for axis in axes]
    else:
        array_axes = UNKNOWN
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "path": relative_path,
        "sha256": digest,
        "file_bytes": int(resolved.stat().st_size),
        **metadata,
        "statistics": statistics,
        "manifest_fragment": {
            "ct_volume": {
                "path": relative_path,
                "sha256": digest,
                "role": "ct_volume",
                "retention": retention,
            },
            "ct_metadata": {
                "format": metadata["format"],
                "shape": metadata["shape"],
                "dtype": metadata["dtype"],
                "byte_order": metadata["byte_order"],
                "array_axes": array_axes,
                "voxel_spacing": metadata["voxel_spacing"],
            },
        },
    }


def inspect_volume_envelope(
    path: Path,
    *,
    repository_root: Path,
    header_only: bool = False,
    include_sha256: bool = True,
    chunk_voxels: int = 8 * 1024 * 1024,
    retention: str = "external",
) -> dict[str, Any]:
    """Return one inspection with shared MCP/CLI authority metadata."""
    inspection = inspect_volume(
        path,
        repository_root=repository_root,
        header_only=header_only,
        include_sha256=include_sha256,
        chunk_voxels=chunk_voxels,
        retention=retention,
    )
    return {
        "status": "ok",
        "authoritative": include_sha256,
        "inspection_mode": (
            "header_only" if header_only else "streaming_statistics"
        ),
        **inspection,
    }


def inspect_volumes(
    paths: Iterable[Path],
    *,
    repository_root: Path,
    header_only: bool = False,
    include_sha256: bool = True,
    chunk_voxels: int = 8 * 1024 * 1024,
    retention: str = "external",
) -> dict[str, Any]:
    """Inspect multiple volumes in caller order."""
    return {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "header_only": header_only,
        "sha256_included": include_sha256,
        "volumes": [
            inspect_volume_envelope(
                path,
                repository_root=repository_root,
                header_only=header_only,
                include_sha256=include_sha256,
                chunk_voxels=chunk_voxels,
                retention=retention,
            )
            for path in paths
        ],
    }
