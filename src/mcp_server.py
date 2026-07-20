import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache" if os.name != "nt" else os.path.join(os.environ["TEMP"], "matplotlib_cache")
from fastmcp import FastMCP
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
