---
name: ct-threshold-optimizer
description: Compare binary segmentations of a 3D CT .npy volume across several density thresholds by repeatedly invoking the segment_ct_dataset MCP tool. Use when asked to tune, sweep, test, optimize, or compare CT segmentation thresholds and save separate candidate masks.
---

# CT Threshold Optimizer

Generate a small, reproducible threshold sweep through the MCP server, then compare the resulting masks. This workflow ranks candidates for human inspection; it does not prove segmentation accuracy without ground truth.

## Workflow

1. Resolve the input `.npy` path and confirm that it is a 3D CT volume. Invoke
   `$volume-metadata` with statistics enabled (never `--header-only`) and inspect
   its shape, dtype, axes, spacing provenance, minimum, and maximum. Stop if
   either intensity bound is `unknown` or non-finite; do not infer missing
   values or proceed with a threshold sweep.
2. Choose three to seven finite, distinct thresholds:
   - Use thresholds supplied by the user.
   - Otherwise, use values at 30%, 50%, and 70% of the observed intensity range: `minimum + fraction * (maximum - minimum)`.
   - If the intensity range is zero, stop because no meaningful sweep is possible.
3. Create a dedicated output directory next to the input unless the user provides one. Name masks `<input-stem>_threshold_<value>.npy`, replacing filename-unsafe characters in the value. Never overwrite existing masks without explicit permission; select a new directory or filename instead.
4. Invoke the MCP tool `segment_ct_dataset` once per threshold with:
   - `input_filepath`: absolute input path
   - `output_filepath`: unique absolute mask path
   - `threshold`: the absolute density threshold
5. Treat any tool response beginning with `Error` or any missing output as a failed candidate. If the MCP tool is unavailable, stop and explain that the project MCP server must be registered and the Codex session restarted. Do not silently replace the MCP call with a local implementation.
6. Compare successful masks with:

   ```bash
   python .agents/skills/ct-threshold-optimizer/scripts/compare_masks.py \
     --raw <input.npy> \
     --mask '<threshold>=<mask.npy>' [--mask '<threshold>=<mask.npy>' ...]
   ```

7. Present a table containing threshold, output path, foreground voxels, total voxels, and foreground percentage. Recommend visual inspection of representative slices in Napari or with `visualize_slice` before choosing a final threshold.

## Selection Guidance

- A foreground percentage that changes abruptly between adjacent thresholds can signal sensitivity to noise or partial-volume effects.
- Prefer a stable candidate that preserves expected struts without filling voids, based on slice inspection and specimen knowledge.
- If a reference mask exists, compare against it explicitly. Otherwise label any recommendation as provisional.
- Keep every candidate output so the user can reproduce the comparison.

## Constraints

- Use absolute paths for MCP arguments.
- Do not use normalized fractions as literal thresholds unless the volume itself is normalized.
- Do not run an unbounded optimization loop.
- Do not claim that foreground percentage alone measures segmentation quality.
