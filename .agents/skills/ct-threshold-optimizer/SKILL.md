---
name: ct-threshold-optimizer
description: Replay or compare bounded CT segmentation thresholds through segmentation-tools MCP for TIFF or NPY volumes. Use for exact per-scan Otsu production segmentation or explicitly exploratory threshold candidates without loading voxel arrays into context.
---

# CT Threshold Optimizer

Use `segmentation-tools` MCP only. This skill contains no executable scripts
and never imports the MCP implementation or falls back to a local CLI.

## Production path

1. Invoke `volume_info` for TIFF/NPY metadata, endian, ZYX/XYZ mapping, and hash.
2. Invoke `replay_exact_otsu` once. Require the exact 65,536-bin per-scan
   histogram, persisted input/config/method hashes, and all plausibility gates.
3. For the reference scan, enforce threshold 40054 and 58,653,410 foreground
   voxels as a replay check. Never tune toward that fraction.
4. Invoke `segment_ct_dataset` at the accepted threshold to write the canonical
   uint8 ZYX mask. Pin its path, role, dtype, shape, retention, and SHA-256.
5. Invoke `compare_segmentation_masks` with an explicit report path to verify
   shape and compact foreground statistics. Use `visualize_slice` only for a
   bounded representative PNG.

## Exploratory path

Only when explicitly requested, use a small predeclared list of finite
thresholds. Write every candidate to a unique path and compare them once. Mark
all candidates provisional; never choose by labels, target foreground fraction,
ground-truth segmentation, or open-ended search.

Require `part2-mcp-response/1.0.0` with status, gate, artifact metadata/hashes,
counts, warnings, and structured errors. Never return voxel arrays. If the MCP
server or a required tool is unavailable or incompatible, stop with a
structured halt and state that no fallback was used.
