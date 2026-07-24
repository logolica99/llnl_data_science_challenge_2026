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
- **Input 1 (Original Volume):** Use the original `.npy` volume path.
- **Input 2 (Segmented Masks):** Use the mask `.npy` path. If this file does
  not exist, invoke the MCP tool `segment_ct_dataset`.
- **Generated-mask verification:** If `segment_ct_dataset()` creates the mask,
  invoke `$volume-metadata` on that output before feature calculation or report
  compilation. Preserve the same repository-relative path, hash, axes, byte
  order, and spacing-provenance fields required for an existing mask.
- **Input 3 (Skeleton):** Use the skeleton `.npy` path. If this file does not
  exist, invoke the MCP tool `skeletonize`.
- **Action:** Invoke `summarize_nde_artifacts` with the raw, mask, and skeleton
  paths. Use its mean foreground intensity, voxel counts, and 26-connected
  endpoint and branch-point metrics; do not calculate replacements locally.

### Step 2: 3D Visualization
Invoke the MCP tool `render_volume_3d` twice to capture the structure from
different perspectives. Pass the mask as `input_filepath`, a unique `.png`
output path, `surface_level: 0.5`, `downsample_factor: 2`, and the skeleton path
as `skeleton_filepath` when available. Do not enable `overwrite` unless the user
explicitly authorizes replacement. Use these view parameters:

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
- If any required MCP tool is unavailable, stop and explain that the
  `segmentation-tools` MCP server must be configured and the client restarted.
  Do not replace metadata inspection, segmentation, or skeletonization with a
  local implementation.
- Do not load, analyze, or render the arrays with local scripts. Use the
  required MCP tools for deterministic volume, mask, skeleton, and image
  operations.
- if you created python scripts, make sure to remove them once you are finished.
