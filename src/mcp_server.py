from contextlib import redirect_stdout
import hashlib
from pathlib import Path
import sys
from typing import Any, Callable, Literal

import matplotlib
import numpy as np
from fastmcp import FastMCP

try:
    from .part2_core import (
        error_response as _error_response,
        load_volume as _load_volume,
        normalize_lattice_graph as _normalize_lattice_graph,
        replay_exact_otsu as _replay_exact_otsu,
        success_response as _success_response,
        volume_metadata as _volume_metadata,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from .skeletonization import skeletonize_mask
    from .volume_artifacts import (
        compare_segmentation_masks as _compare_segmentation_masks,
        render_volume_3d as _render_volume_3d,
        summarize_nde_artifacts as _summarize_nde_artifacts,
    )
    from .volume_metadata import inspect_volume_envelope
except ImportError:
    from part2_core import (
        error_response as _error_response,
        load_volume as _load_volume,
        normalize_lattice_graph as _normalize_lattice_graph,
        replay_exact_otsu as _replay_exact_otsu,
        success_response as _success_response,
        volume_metadata as _volume_metadata,
        write_otsu_artifacts as _write_otsu_artifacts,
    )
    from skeletonization import skeletonize_mask
    from volume_artifacts import (
        compare_segmentation_masks as _compare_segmentation_masks,
        render_volume_3d as _render_volume_3d,
        summarize_nde_artifacts as _summarize_nde_artifacts,
    )
    from volume_metadata import inspect_volume_envelope


# MCP servers may run without a display, so use Matplotlib's non-interactive
# backend before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@mcp.tool()
def inspect_volume_metadata(
    input_filepath: str,
    header_only: bool = True,
    include_sha256: bool = True,
    retention: Literal["committed", "external", "regenerable"] = "external",
) -> dict[str, Any]:
    """Inspect one repository CT volume and return manifest-ready metadata.

    Use header-only mode for specimen intake. It reads the NPY/TIFF header and
    streams the file for SHA-256 without decoding voxel intensities. Set
    include_sha256 to false only for a non-authoritative preview. Inputs are
    constrained to this repository and are never modified.
    """
    return inspect_volume_envelope(
        Path(input_filepath),
        repository_root=REPOSITORY_ROOT,
        header_only=header_only,
        include_sha256=include_sha256,
        retention=retention,
    )


def _input_npy_path(filepath: str) -> Path:
    """Resolve and validate an input NumPy volume path."""
    path = Path(filepath).expanduser().resolve()
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Input file must use the .npy extension: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return path


def _repository_path(
    filepath: str,
    *,
    must_exist: bool,
    expected_suffixes: set[str] | None = None,
) -> tuple[Path, str]:
    """Resolve one new Part 2 tool path without allowing repository escape."""

    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Input file does not exist: {relative.as_posix()}")
    if expected_suffixes and resolved.suffix.lower() not in expected_suffixes:
        choices = ", ".join(sorted(expected_suffixes))
        raise ValueError(
            f"Expected one of [{choices}], found {relative.as_posix()}"
        )
    return resolved, relative.as_posix()


def _repository_output_directory(filepath: str) -> tuple[Path, str]:
    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(
            f"Output directory is an existing file: {relative.as_posix()}"
        )
    return resolved, relative.as_posix()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _structured_failure(tool: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, FileNotFoundError):
        code = "input_not_found"
    elif isinstance(exc, FileExistsError):
        code = "artifact_exists"
    elif isinstance(exc, (ValueError, TypeError, IndexError)):
        code = "invalid_input"
    else:
        code = "tool_execution_failed"
    return _error_response(
        tool=tool,
        code=code,
        error_type=type(exc).__name__,
        message=str(exc),
    )


def _run_structured_tool(
    tool: str,
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        return operation()
    except Exception as exc:
        return _structured_failure(tool, exc)


def _output_path(filepath: str, expected_suffix: str | None = None) -> Path:
    """Resolve an output path and create its parent directory."""
    path = Path(filepath).expanduser().resolve()
    if expected_suffix and path.suffix.lower() != expected_suffix:
        raise ValueError(
            f"Output file must use the {expected_suffix} extension: {path}"
        )
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Output path is a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_3d_array(filepath: str) -> tuple[Path, np.ndarray]:
    """Use the shared memory-mapped TIFF/NPY loader for legacy tools."""
    volume = _load_volume(filepath)
    return volume.path, volume.array


@mcp.tool()
def volume_info(
    input_filepath: str,
    include_sha256: bool = True,
) -> dict[str, Any]:
    """Return compact shared-loader metadata for a TIFF or NPY CT volume."""

    def operation() -> dict[str, Any]:
        path, relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        volume = _load_volume(path)
        result = _volume_metadata(volume)
        result["path"] = relative
        digest = _sha256_file(path) if include_sha256 else ""
        warnings = [] if include_sha256 else ["input SHA-256 was explicitly omitted"]
        return _success_response(
            tool="volume_info",
            gate="pass",
            summary=f"Loaded 3-D {result['format']} volume {relative}",
            result=result,
            artifacts={
                "input": {
                    "path": relative,
                    "role": "ct_volume",
                    "retention": "external",
                }
            },
            hashes={"input_sha256": digest} if digest else {},
            warnings=warnings,
        )

    return _run_structured_tool("volume_info", operation)


@mcp.tool()
def load_lattice_graph(
    input_filepath: str,
    output_filepath: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Normalize a lattice JSON to NPZ with explicit node/edge/cell ID maps."""

    def operation() -> dict[str, Any]:
        source, source_relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".json"},
        )
        output, output_relative = _repository_path(
            output_filepath,
            must_exist=False,
            expected_suffixes={".npz"},
        )
        result = _normalize_lattice_graph(
            source,
            output,
            overwrite=overwrite,
        )
        result["source_path"] = source_relative
        result["output_path"] = output_relative
        warnings = list(result["warnings"])
        gate: Literal["pass", "manual_review"] = (
            "pass" if not warnings else "manual_review"
        )
        return _success_response(
            tool="load_lattice_graph",
            gate=gate,
            summary=(
                f"Normalized {result['counts']['nodes']} nodes, "
                f"{result['counts']['edges']} edges, and "
                f"{result['counts']['cells']} cells"
            ),
            result=result,
            artifacts={
                "normalized_graph": {
                    "path": output_relative,
                    "sha256": result["artifact_sha256"],
                    "role": "normalized_lattice_graph",
                    "retention": "regenerable",
                }
            },
            hashes={
                "input_sha256": result["source_sha256"],
                "artifact_sha256": result["artifact_sha256"],
            },
            warnings=warnings,
        )

    return _run_structured_tool("load_lattice_graph", operation)


@mcp.tool()
def replay_exact_otsu(
    input_filepath: str,
    output_directory: str,
    histogram_encoding: Literal[
        "auto", "native_uint16", "full_volume_affine_uint16"
    ] = "auto",
    edge_slices_excluded: int = 0,
    chunk_voxels: int = 8 * 1024 * 1024,
    coarse_bins: int = 1024,
    peak_smoothing_sigma_bins: float = 2.0,
    peak_prominence_fraction: float = 0.003,
    minimum_significant_peaks: int = 2,
    minimum_foreground_fraction: float = 0.01,
    maximum_foreground_fraction: float = 0.35,
    minimum_otsu_separability: float = 0.45,
    minimum_class_mean_separation_sigma: float = 0.75,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Replay per-scan exact Otsu and persist its histogram and diagnostics."""

    def operation() -> dict[str, Any]:
        source, source_relative = _repository_path(
            input_filepath,
            must_exist=True,
            expected_suffixes={".npy", ".tif", ".tiff"},
        )
        output, _ = _repository_output_directory(output_directory)
        recipe = {
            "histogram_encoding": histogram_encoding,
            "edge_slices_excluded": edge_slices_excluded,
            "chunk_voxels": chunk_voxels,
            "coarse_bins": coarse_bins,
            "peak_smoothing_sigma_bins": peak_smoothing_sigma_bins,
            "peak_prominence_fraction": peak_prominence_fraction,
            "minimum_significant_peaks": minimum_significant_peaks,
            "minimum_foreground_fraction": minimum_foreground_fraction,
            "maximum_foreground_fraction": maximum_foreground_fraction,
            "minimum_otsu_separability": minimum_otsu_separability,
            "minimum_class_mean_separation_sigma": (
                minimum_class_mean_separation_sigma
            ),
        }
        result, histogram = _replay_exact_otsu(source, recipe=recipe)
        result["source_path"] = source_relative
        artifacts = _write_otsu_artifacts(
            output,
            result,
            histogram,
            overwrite=overwrite,
        )
        for artifact in artifacts.values():
            artifact_path = Path(artifact["path"])
            artifact["path"] = artifact_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix()
        failed_gates = sorted(
            name for name, passed in result["gates"].items() if not passed
        )
        gate: Literal["pass", "halt"] = (
            "pass" if result["overall_pass"] else "halt"
        )
        warnings = (
            []
            if not failed_gates
            else ["histogram rejection gates failed: " + ", ".join(failed_gates)]
        )
        input_hash = _sha256_file(source)
        return _success_response(
            tool="replay_exact_otsu",
            gate=gate,
            summary=(
                f"Replayed Otsu threshold {result['threshold']} for "
                f"{source_relative}"
            ),
            result=result,
            artifacts=artifacts,
            hashes={
                "input_sha256": input_hash,
                "histogram_sha256": result["histogram_sha256"],
                "histogram_artifact_sha256": artifacts["histogram"]["sha256"],
                "report_artifact_sha256": artifacts["report"]["sha256"],
            },
            warnings=warnings,
        )

    return _run_structured_tool("replay_exact_otsu", operation)


@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path, volume = _load_3d_array(input_filepath)
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be a finite number.")

        output_path = _output_path(output_filepath, ".npy")
        segmented = (volume >= threshold).astype(np.uint8)
        np.save(output_path, segmented, allow_pickle=False)

        foreground_voxels = int(np.count_nonzero(segmented))
        return (
            f"Segmented {input_path} at threshold {threshold}. "
            f"Saved {foreground_voxels} foreground voxels out of "
            f"{segmented.size} total voxels to {output_path}."
        )
    except Exception as exc:
        return f"Error segmenting CT dataset: {exc}"


@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    figure = None
    try:
        input_path, volume = _load_3d_array(input_filepath)
        if axis not in (0, 1, 2):
            raise ValueError(f"Axis must be 0, 1, or 2; received {axis}.")
        if not 0 <= slice_index < volume.shape[axis]:
            raise IndexError(
                f"Slice index {slice_index} is outside axis {axis}, which has "
                f"valid indices 0 through {volume.shape[axis] - 1}."
            )

        output_path = _output_path(output_filepath)
        image = np.take(volume, slice_index, axis=axis)

        figure, axes = plt.subplots(figsize=(8, 8))
        axes.imshow(image, cmap="gray")
        axes.set_title(f"{input_path.name}: axis {axis}, slice {slice_index}")
        axes.axis("off")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")

        return (
            f"Saved axis {axis}, slice {slice_index} from {input_path} "
            f"to {output_path}."
        )
    except Exception as exc:
        return f"Error visualizing CT slice: {exc}"
    finally:
        if figure is not None:
            plt.close(figure)


@mcp.tool()
def compare_segmentation_masks(
    raw_filepath: str,
    mask_filepaths: list[str],
    thresholds: list[float],
) -> dict[str, Any]:
    """Compare threshold masks without returning voxel arrays.

    The mask and threshold lists are positional pairs. Every mask must be a
    three-dimensional boolean or integer NPY array with the same shape as the
    raw volume.
    """
    return _compare_segmentation_masks(
        raw_filepath,
        mask_filepaths,
        thresholds,
    )


@mcp.tool()
def summarize_nde_artifacts(
    raw_filepath: str,
    mask_filepath: str,
    skeleton_filepath: str | None = None,
) -> dict[str, Any]:
    """Summarize aligned raw, mask, and optional skeleton NPY artifacts.

    The tool returns report-ready scalar metrics and never returns voxel arrays.
    Skeleton endpoints and branch points use a 26-connected neighborhood.
    """
    return _summarize_nde_artifacts(
        raw_filepath,
        mask_filepath,
        skeleton_filepath,
    )


@mcp.tool()
def render_volume_3d(
    input_filepath: str,
    output_filepath: str,
    surface_level: float = 0.5,
    downsample_factor: int = 2,
    elevation: float = 30.0,
    azimuth: float = 45.0,
    skeleton_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Render a volume isosurface and optional skeleton overlay to PNG.

    surface_level is normalized to the downsampled volume range and must be
    strictly between zero and one. The tool writes the image and returns only
    compact render metadata.
    """
    return _render_volume_3d(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        surface_level=surface_level,
        downsample_factor=downsample_factor,
        elevation=elevation,
        azimuth=azimuth,
        skeleton_filepath=skeleton_filepath,
        overwrite=overwrite,
    )


@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = _input_npy_path(input_filepath)
        mask = np.load(input_path, mmap_mode="r", allow_pickle=False)
        if mask.ndim != 3:
            raise ValueError(
                f"Expected a 3D mask, but {input_path} has shape {mask.shape}."
            )

        output_path = _output_path(output_filepath, ".npy")

        # skeletonize_mask reports progress with print(). Redirect that output to
        # stderr so it cannot interfere with MCP's JSON-RPC messages on stdout.
        with redirect_stdout(sys.stderr):
            skeleton = skeletonize_mask(str(input_path), str(output_path))

        if skeleton is None or not output_path.is_file():
            raise RuntimeError("Skeletonization did not produce an output file.")

        skeleton_voxels = int(np.count_nonzero(skeleton))
        return (
            f"Skeletonized {input_path}. Saved {skeleton_voxels} skeleton "
            f"voxels to {output_path}."
        )
    except Exception as exc:
        return f"Error skeletonizing segmentation mask: {exc}"


if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
