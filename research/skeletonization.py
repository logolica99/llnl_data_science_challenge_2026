import numpy as np
import os
from skimage.morphology import skeletonize

def skeletonize_mask(file_path, output_path):
    """
    Creates a skeleton from a 3D segmentation mask.
    
    Args:
        file_path (str): Path to the .npy file containing the 3D mask.
        output_path (str): Path to save the extracted skeleton (.npy).
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"Loading mask from {file_path}...")
    mask = np.load(file_path)
    print(f"Original mask shape: {mask.shape}")
    
    # Ensure the mask is boolean
    if mask.dtype != bool:
        print("Converting mask to boolean array...")
        # Assuming background is 0 and object is > 0
        mask = mask > 0

    print("Extracting skeleton (this may take a moment for 3D data)...")
    skeleton = skeletonize(mask)
    
    print(f"Skeleton extracted. Non-zero voxels: {np.count_nonzero(skeleton)}")
    
    np.save(output_path, skeleton)
    print(f"Saved skeleton to: {output_path}")
    
    return skeleton

if __name__ == "__main__":
    # Hardcoded parameters for testing
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "unitcell", "unitcell.npy"))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "octet_truss_unit_cell_skeleton.npy"))
    
    # Create the data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    skeletonize_mask(
        file_path=file_path, 
        output_path=output_path
    )
