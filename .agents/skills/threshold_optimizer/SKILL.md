---
name: threshold-optimizer
description: Segments a CT .npy volume at multiple thresholds via segment_ct_dataset() and saves comparable mask outputs.
---

# Threshold Optimizer Protocol

You are the **Threshold Optimizer**. When this skill is active, compare several density thresholds for segmenting a CT volume.

### Step 1: Choose Input and Output Location
- **Input:** the CT `.npy` path from the user (default: `data/unitcell/unitcell.npy`).
- **Output directory:** `data/<dataset_name>/threshold_sweep/` (create it if needed), or a path the user specifies.

### Step 2: Inspect Intensity Range First
Before choosing thresholds, inspect the volume range. Prefer the **metadata-extractor** skill / script:

```bash
python .agents/skills/metadata_extractor/scripts/extract_metadata.py <input_npy>
```

Use the printed `min` / `max` to decide whether the data looks normalized to `[0, 1]` or not.

### Step 3: Pick Thresholds
- If intensities are roughly in `[0, 1]`, use: **0.3, 0.5, 0.7**.
- If intensities are much smaller (e.g. unitcell ~`[-0.003, 0.015]`), do **not** use 0.3/0.5/0.7 (those yield empty masks). Instead use three values spanning the useful range, for example **0.001, 0.002, 0.005**, or ask the user.
- Always tell the user which thresholds you chose and why.

### Step 4: Segment with MCP
Call the MCP tool `segment_ct_dataset()` once per threshold. Save each mask with a clear name, e.g.:

| Threshold | Output path |
| :--- | :--- |
| `t` | `<out_dir>/mask_thresh_<t>.npy` |

Replace dots in filenames as needed (`0.002` → `0p002`).

### Step 5: Optional Visual Comparison
If `visualize_slice` is available, save a mid-volume slice for each mask (same `slice_index` and `axis` for every threshold) into the same output directory.

### Step 6: Summarize
Present a comparison table with:
1. threshold
2. output mask path
3. foreground voxel count / fraction (from the MCP status string or metadata script)
4. brief note on which threshold looks most reasonable (connected lattice, not empty, not overly filled)

# Technical Constraints
- Prefer calling `segment_ct_dataset()` via MCP rather than reimplementing thresholding.
- Do not overwrite unrelated files; only write into the chosen sweep directory.
- If MCP is unavailable, fall back to running equivalent Python with NumPy and say that you used a fallback.
- If you create temporary helper scripts outside this skill folder, delete them when finished.
