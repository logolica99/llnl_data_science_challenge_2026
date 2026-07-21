---
name: threshold-optimizer
description: Use this skill when the user wants to compare multiple segmentation thresholds for a CT dataset, find a good threshold value, or asks to "optimize" or "sweep" thresholds. Calls the segment_ct_dataset MCP tool repeatedly with different threshold values and reports the results for comparison.
---

# Threshold Optimizer

This skill helps identify a good segmentation threshold for a CT dataset by
running segmentation at several candidate thresholds and comparing results.

## Instructions

1. Identify the input `.npy` file the user wants to analyze. If not specified,
   ask which file to use, or look in `./data` for a likely candidate.

2. Before choosing threshold values, inspect the data's actual min/max range
   (e.g. using a short Python snippet or existing metadata tool). Do not
   assume thresholds like 0.3/0.5/0.7 are appropriate — pick threshold
   candidates that are actually within the data's real value range. A good
   default is to pick 4-5 values spread across the data's min-to-max range,
   including one near the midpoint.

3. For each candidate threshold, call the `segment_ct_dataset` MCP tool with:
   - `input_filepath`: the source file
   - `output_filepath`: a unique name per threshold, e.g.
     `<original_name>_thresh_<value>.npy`
   - `threshold`: the candidate value

4. After each call, report the resulting foreground voxel count and
   foreground fraction (foreground voxels / total voxels) for that threshold.

5. Present a summary table comparing all thresholds tried, with columns:
   threshold | foreground voxels | foreground fraction | output filename

6. Recommend which threshold looks most reasonable, briefly explaining why
   (e.g. "closest to X% material fraction" or "sits in a low-density valley
   between background and material peaks, if that information is available").