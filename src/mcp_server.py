import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastmcp import FastMCP

from skeletonization import skeletonize_mask

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")

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
        if not os.path.exists(input_filepath):
            return f"Error: Input file not found at {input_filepath}"

        volume = np.load(input_filepath)
        mask = (volume >= threshold).astype(np.uint8)

        output_dir = os.path.dirname(os.path.abspath(output_filepath))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        np.save(output_filepath, mask)
        foreground = int(np.count_nonzero(mask))
        total = int(mask.size)
        return (
            f"Saved segmentation to {output_filepath} "
            f"(shape={mask.shape}, threshold={threshold}, "
            f"foreground={foreground}/{total})"
        )
    except Exception as exc:
        return f"Error segmenting dataset: {exc}"

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
        if not os.path.exists(input_filepath):
            return f"Error: Input file not found at {input_filepath}"
        if axis not in (0, 1, 2):
            return f"Error: axis must be 0, 1, or 2 (got {axis})"

        volume = np.load(input_filepath)
        if volume.ndim != 3:
            return f"Error: Expected a 3D array, got shape {volume.shape}"

        axis_size = volume.shape[axis]
        if slice_index < 0 or slice_index >= axis_size:
            return (
                f"Error: slice_index {slice_index} out of range for axis {axis} "
                f"(valid: 0–{axis_size - 1})"
            )

        slice_2d = np.take(volume, slice_index, axis=axis)

        output_dir = os.path.dirname(os.path.abspath(output_filepath))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_2d, cmap="gray", origin="lower")
        ax.set_title(f"Slice {slice_index} (axis={axis})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return (
            f"Saved slice visualization to {output_filepath} "
            f"(volume shape={volume.shape}, axis={axis}, slice_index={slice_index}, "
            f"slice shape={slice_2d.shape})"
        )
    except Exception as exc:
        return f"Error visualizing slice: {exc}"

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
        if not os.path.exists(input_filepath):
            return f"Error: Input file not found at {input_filepath}"

        output_dir = os.path.dirname(os.path.abspath(output_filepath))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Thin wrapper around the existing skeletonization API
        skeleton = skeletonize_mask(input_filepath, output_filepath)
        if skeleton is None:
            return f"Error: skeletonize_mask failed for {input_filepath}"

        return (
            f"Saved skeleton to {output_filepath} "
            f"(shape={skeleton.shape}, non-zero voxels={int(np.count_nonzero(skeleton))})"
        )
    except Exception as exc:
        return f"Error skeletonizing mask: {exc}"

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
