---
name: nde-report-generator
description: Extracts features from volumetric, mask, and skeleton .npy files and generates a visual report with specific 3D perspectives.
---

# Report Generation Protocol

You are the **Non Destructive Evaluation Report Expert**. When this skill is active, follow these steps to process the data and generate the final report (an MD file):

### Step 1: Feature Extraction
- **Input validation:** Invoke `$volume-metadata` for every raw volume and mask.
  Preserve its repository-relative paths, hashes, axes, byte order, and spacing
  provenance in the report; never guess missing physical metadata.
- **Input 1 (Original Volume):** Load the raw intensity data from the original `.npy` file.
- **Input 2 (Segmented Masks):** Load the mask `.npy` to isolate Regions of Interest (ROIs). If this file doesn't exist, use the MCP tool segment_ct_dataset(). 
- **Generated-mask verification:** If `segment_ct_dataset()` creates the mask,
  invoke `$volume-metadata` on that output before feature calculation or report
  compilation. Preserve the same repository-relative path, hash, axes, byte
  order, and spacing-provenance fields required for an existing mask.
- **Input 3 (Skeleton):** Load the skeleton `.npy` to calculate morphological features (e.g., length, branching points). If this file doesn't exist, use the MCP tool skeletonize(). 
- **Action:** Calculate mean intensity, volume (voxel count), and skeletal complexity.

### Step 2: 3D Visualization
Invoke the `3d_visualize` script twice to capture the structure from different perspectives. Use the following parameters:

| Visualization | Elevation (`elev`) | Azimuth (`azim`) |
| :--- | :--- | :--- |
| **View A** | 30.0 | 45.0 |
| **View B** | 60.0 | 45.0 |

### Step 3: Report Compilation
Assemble the findings into a markdown report including:
1. **Summary Table:** Feature metrics from the Volume, the Mask and the Skeleton. 
2. **Visual Gallery:** Embed the two generated 3D plots.
3. **Analysis:** Brief interpretation of the mask-to-volume alignment.

# Technical Constraints
- Ensure all `.npy` arrays are checked for shape compatibility before processing.
- Use `$volume-metadata` as the authoritative metadata contract.
- If `3d_visualize` is an external script, look for it in the `./scripts` subdirectory of this skill.
- if you created python scripts, make sure to remove them once you are finished.
