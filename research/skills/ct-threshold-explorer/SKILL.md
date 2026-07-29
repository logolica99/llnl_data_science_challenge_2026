---
name: ct-threshold-explorer
description: Compare a small explicit set of Otsu-centered CT segmentation thresholds through the disabled research MCP server. Use only for offline research, diagnostics, or method development outside production Stage 0–4 runs; never use it to select a production threshold or optimize against labels.
---

# CT Threshold Explorer

Use only the `segmentation-tools-research` MCP server and require outputs under
`research/runs/`. Never write a production manifest, handoff, receipt, mask,
threshold, classification, or report.

1. Require an explicit repository CT path, research output directory, and one
   to nine finite unique threshold offsets.
2. Invoke `explore_ct_thresholds`. Let the MCP tool replay exact Otsu, write
   each provisional mask and representative slice, and compare every candidate
   against its declared threshold.
3. Report the persisted candidate paths, hashes, compact foreground counts,
   and comparison artifact. Keep every result marked `research_only`.

Never choose a candidate using labels, known defect counts, target foreground
fractions, or ground-truth segmentation. Never copy a research threshold into
a frozen production run. Missing or disabled research MCP dependencies are a
hard stop; do not fall back to a CLI, script, direct import, or local array
processing.
