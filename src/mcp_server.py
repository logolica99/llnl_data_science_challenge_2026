import os
import site
import sys

# The project keeps its scientific/MCP runtime under .python_packages. Using
# addsitedir (rather than PYTHONPATH alone) processes pywin32's .pth bootstrap
# on Windows, which FastMCP's stdio transport requires.
_REPOSITORY_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)
_LOCAL_PACKAGES = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", ".python_packages")
)
if os.path.isdir(_LOCAL_PACKAGES):
    site.addsitedir(_LOCAL_PACKAGES)

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache" if os.name != "nt" else os.path.join(os.environ["TEMP"], "matplotlib_cache")
from fastmcp import FastMCP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from research.skeletonization import skeletonize_mask
from strut_defect_pipeline import (
    classify_struts as classify_struts_artifacts,
    compute_strut_metrics as compute_strut_metrics_artifacts,
    render_strut_evidence as render_strut_evidence_artifacts,
    run_pipeline as run_strut_defect_pipeline_artifacts,
)

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")


REPOSITORY_ROOT = _REPOSITORY_ROOT


def _repository_path(value: str, must_exist: bool = False) -> str:
    """Resolve one MCP path while refusing access outside this repository."""
    resolved = os.path.realpath(value)
    try:
        common = os.path.commonpath([REPOSITORY_ROOT, resolved])
    except ValueError as exc:
        raise ValueError(f"Path is not on the repository drive: {value}") from exc
    if common != REPOSITORY_ROOT:
        raise ValueError(f"Path is outside the repository: {value}")
    if must_exist and not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    return resolved

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
        volume = np.load(input_filepath)
    except FileNotFoundError:
        return f"Error: could not find input file at {input_filepath}"
    except Exception as e:
        return f"Error loading {input_filepath}: {e}"

    mask = (volume >= threshold).astype(np.uint8)

    try:
        np.save(output_filepath, mask)
    except Exception as e:
        return f"Error saving output to {output_filepath}: {e}"

    return f"Segmentation complete. Saved to {output_filepath}"

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
    try:
        volume = np.load(input_filepath)
    except FileNotFoundError:
        return f"Error: could not find input file at {input_filepath}"
    except Exception as e:
        return f"Error loading {input_filepath}: {e}"

    if axis not in (0, 1, 2):
        return f"Error: axis must be 0, 1, or 2 (got {axis})"

    if slice_index < 0 or slice_index >= volume.shape[axis]:
        return f"Error: slice_index {slice_index} out of range for axis {axis} (size {volume.shape[axis]})"

    slice_2d = np.take(volume, slice_index, axis=axis)

    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(slice_2d, cmap="gray")
        plt.axis("off")
        plt.savefig(output_filepath, bbox_inches="tight", pad_inches=0)
        plt.close()
    except Exception as e:
        return f"Error saving image to {output_filepath}: {e}"

    return f"Slice visualization complete. Saved to {output_filepath}"

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
        result = skeletonize_mask(file_path=input_filepath, output_path=output_filepath)
    except Exception as e:
        return f"Error skeletonizing mask: {e}"

    if result is None:
        return f"Error: skeletonization failed for {input_filepath} (check that the file exists and is a valid mask)"

    return f"Skeletonization complete. Saved to {output_filepath}"


@mcp.tool()
def compute_strut_metrics(
    input_tiff: str,
    registered_json: str,
    output_dir: str,
    threshold: float,
    positions: int = 21,
    tracking_radius_voxels: float = 6.0,
    voxel_size_mm: float | None = None,
    strut_ids: list[int] | None = None,
    max_struts: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Measure file-backed radius and centerline profiles for registered CT struts.

    The registered JSON is used only as a local spatial prior. Large measurements
    are written under ``output_dir``; the tool returns only a compact receipt.
    """
    return compute_strut_metrics_artifacts(
        _repository_path(input_tiff, must_exist=True),
        _repository_path(registered_json, must_exist=True),
        _repository_path(output_dir),
        threshold,
        positions=positions,
        tracking_radius_voxels=tracking_radius_voxels,
        voxel_size_mm=voxel_size_mm,
        strut_ids=strut_ids,
        max_struts=max_struts,
        overwrite=overwrite,
    )


@mcp.tool()
def classify_struts(
    strut_summary_csv: str,
    strut_sections_csv: str,
    output_dir: str,
    thresholds_json: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Apply frozen thin/thick/bent policy to existing metric artifacts."""
    return classify_struts_artifacts(
        _repository_path(strut_summary_csv, must_exist=True),
        _repository_path(strut_sections_csv, must_exist=True),
        _repository_path(output_dir),
        thresholds_json=(
            _repository_path(thresholds_json, must_exist=True)
            if thresholds_json else None
        ),
        overwrite=overwrite,
    )


@mcp.tool()
def render_strut_evidence(
    classified_struts_csv: str,
    strut_sections_csv: str,
    output_dir: str,
    thresholds_json: str,
    overwrite: bool = False,
) -> dict:
    """Render radius plots for thin/thick and centerline plots for bent calls."""
    return render_strut_evidence_artifacts(
        _repository_path(classified_struts_csv, must_exist=True),
        _repository_path(strut_sections_csv, must_exist=True),
        _repository_path(output_dir),
        thresholds_json=_repository_path(thresholds_json, must_exist=True),
        overwrite=overwrite,
    )


@mcp.tool()
def run_thin_thick_bent_pipeline(
    input_tiff: str,
    registered_json: str,
    output_dir: str,
    threshold: float,
    thresholds_json: str | None = None,
    positions: int = 21,
    tracking_radius_voxels: float = 6.0,
    voxel_size_mm: float | None = None,
    strut_ids: list[int] | None = None,
    max_struts: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Run the complete independently testable thin/thick/bent artifact flow."""
    return run_strut_defect_pipeline_artifacts(
        _repository_path(input_tiff, must_exist=True),
        _repository_path(registered_json, must_exist=True),
        _repository_path(output_dir),
        threshold,
        thresholds_json=(
            _repository_path(thresholds_json, must_exist=True)
            if thresholds_json else None
        ),
        positions=positions,
        tracking_radius_voxels=tracking_radius_voxels,
        voxel_size_mm=voxel_size_mm,
        strut_ids=strut_ids,
        max_struts=max_struts,
        overwrite=overwrite,
    )

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
