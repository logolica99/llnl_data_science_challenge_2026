"""Memory-mapped TIFF/NPY loading and the canonical CT axis mapping."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import numpy as np
import tifffile


SUPPORTED_SUFFIXES = {".npy": "npy", ".tif": "tiff", ".tiff": "tiff"}
AXIS_MAPPING = {
    "coordinate_order": ["x", "y", "z"],
    "array_axes": ["z", "y", "x"],
    "numpy_index_expression": "volume[round(z), round(y), round(x)]",
}


class VolumeLoadError(ValueError):
    """Raised when a volume cannot be safely exposed as a 3-D mapped array."""


@dataclass(frozen=True)
class VolumeView:
    """A memory-mapped 3-D volume and its storage metadata."""

    path: Path
    array: np.ndarray[Any, Any]
    format: str
    source_axes: str
    byte_order: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.array.shape)  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.array.dtype)

    @property
    def is_memory_mapped(self) -> bool:
        return isinstance(self.array, np.memmap)


def _normalized_byte_order(dtype: np.dtype[Any], *, tiff_order: str | None = None) -> str:
    if dtype.itemsize == 1:
        return "not_applicable"
    if tiff_order is not None:
        if tiff_order == "<":
            return "little"
        if tiff_order == ">":
            return "big"
        raise VolumeLoadError(f"Unsupported TIFF byte order {tiff_order!r}")
    if dtype.byteorder == "=":
        return sys.byteorder
    if dtype.byteorder == "<":
        return "little"
    if dtype.byteorder == ">":
        return "big"
    return "not_applicable"


def _validate_volume_array(path: Path, array: np.ndarray[Any, Any]) -> None:
    dtype = np.dtype(array.dtype)
    if array.ndim != 3:
        raise VolumeLoadError(
            f"Expected a 3-D CT volume, but {path} has shape {array.shape}"
        )
    if dtype.fields or dtype.subdtype or dtype.kind not in "buif":
        raise VolumeLoadError(
            f"Expected a real numeric CT volume, but {path} has dtype {dtype}"
        )


def load_volume(path: str | Path) -> VolumeView:
    """Memory-map one TIFF or NPY volume without converting the full array.

    Three-dimensional inputs are interpreted as ``(Z, Y, X)``.  TIFF inputs
    must be memory-mappable; compressed TIFFs are intentionally rejected
    instead of being silently decoded into a full in-memory array.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise VolumeLoadError(f"Volume does not exist: {resolved}")
    format_name = SUPPORTED_SUFFIXES.get(resolved.suffix.lower())
    if format_name is None:
        raise VolumeLoadError(
            f"Expected a .npy, .tif, or .tiff volume: {resolved}"
        )

    if format_name == "npy":
        try:
            array = np.load(
                resolved,
                mmap_mode="r",
                allow_pickle=False,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise VolumeLoadError(f"Unable to memory-map NPY volume {resolved}: {exc}") from exc
        source_axes = "unknown"
        byte_order = _normalized_byte_order(np.dtype(array.dtype))
    else:
        try:
            with tifffile.TiffFile(resolved) as tif:
                if not tif.series:
                    raise VolumeLoadError(f"TIFF contains no image series: {resolved}")
                source_axes = str(tif.series[0].axes or "unknown")
                tiff_order = tif.byteorder
            array = tifffile.memmap(resolved, mode="r")
        except VolumeLoadError:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise VolumeLoadError(
                "TIFF is not directly memory-mappable; use an uncompressed "
                f"TIFF or NPY artifact: {resolved}: {exc}"
            ) from exc
        byte_order = _normalized_byte_order(
            np.dtype(array.dtype),
            tiff_order=tiff_order,
        )

    _validate_volume_array(resolved, array)
    return VolumeView(
        path=resolved,
        array=array,
        format=format_name,
        source_axes=source_axes,
        byte_order=byte_order,
    )


def volume_metadata(volume: VolumeView) -> dict[str, Any]:
    """Return compact metadata without inspecting voxel values."""

    dtype = volume.dtype
    return {
        "path": str(volume.path),
        "format": volume.format,
        "shape": list(volume.shape),
        "ndim": 3,
        "dtype": dtype.name,
        "dtype_string": dtype.str,
        "byte_order": volume.byte_order,
        "source_axes": volume.source_axes,
        "voxel_count": int(volume.array.size),
        "array_bytes": int(volume.array.nbytes),
        "memory_mapped": volume.is_memory_mapped,
        "axis_mapping": AXIS_MAPPING,
    }


def iter_array_chunks(
    array: np.ndarray[Any, Any],
    chunk_voxels: int,
) -> Iterator[np.ndarray[Any, Any]]:
    """Yield storage-order views with bounded conversion working sets."""

    if chunk_voxels <= 0:
        raise VolumeLoadError("chunk_voxels must be positive")
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


def xyz_to_zyx_indices(
    xyz: Sequence[float],
    shape: Sequence[int],
) -> tuple[int, int, int]:
    """Map graph ``[x, y, z]`` coordinates to NumPy ``[z, y, x]`` indices."""

    if len(xyz) != 3 or len(shape) != 3:
        raise VolumeLoadError("xyz coordinates and volume shape must both have length 3")
    values = [float(value) for value in xyz]
    if not all(math.isfinite(value) for value in values):
        raise VolumeLoadError(f"xyz coordinates must be finite: {list(xyz)!r}")
    x, y, z = (int(np.rint(value)) for value in values)
    indices = (z, y, x)
    if any(index < 0 or index >= int(limit) for index, limit in zip(indices, shape)):
        raise VolumeLoadError(
            f"xyz coordinate {list(xyz)!r} maps to out-of-bounds zyx "
            f"index {indices} for shape {tuple(shape)}"
        )
    return indices


def sample_xyz(volume: VolumeView | np.ndarray[Any, Any], xyz: Sequence[float]) -> Any:
    """Sample one value using the pinned ``[x,y,z] -> volume[z,y,x]`` map."""

    array = volume.array if isinstance(volume, VolumeView) else volume
    indices = xyz_to_zyx_indices(xyz, array.shape)
    value = array[indices]
    return value.item() if isinstance(value, np.generic) else value
