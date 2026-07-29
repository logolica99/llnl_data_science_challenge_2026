---
name: strut-defect-analyzer
description: Compute deterministic, label-blind Stage 2 per-strut evidence from a verified Stage 1 handoff. Use when the production Part 2 pipeline needs batched padded rotated-cuboid connectivity, axial foreground profiles, supplementary junction-masked collar evidence, local EDT radius, or curvature before Stage 3 defect classification.
---

# Stage 2 Strut Metrics

Use only the `segmentation-tools.compute_strut_metrics` MCP tool. Do not run a script, import its implementation, recompute registration, select a threshold, or construct substitute artifacts.

## Required handoff

Require the attempt-scoped canonical Stage 2 handoff for `strut_metrics`. Require its verified Stage 1 predecessor receipt hash and exactly these input artifact roles:

- `analysis_ready_specimen_manifest`
- `data_prep_completion_receipt`
- `analysis_config`
- `ct_volume`
- `localized_graph`
- `registration_qa`
- `otsu_report`
- `canonical_segmentation_mask`
- `segmentation_mask_comparison`

Reject extra roles and any role or path exposing labels, development data, sealed data, ground truth, or classifications. If the MCP server or tool is missing, unhealthy, or unavailable, stop and return `halt`. Treat an incompatible response schema, failed Stage 1 gate, hash mismatch, stale contract, or stale handoff as `halt`; never use a fallback.

Require the analysis-ready manifest and data-prep completion receipt to agree
on specimen, registration mode, requested analysis scope, canonical mask, and
authorized/unauthorized outputs. Under `roi_screening`, direct dimensional
metrology must remain unauthorized.

## Compute measurements

Pass the handoff path, all nine exact artifact paths, and only the canonical output directory `analysis/<specimen_id>/struts` to `compute_strut_metrics`. Require the frozen configuration to satisfy `analysis/schema/strut_metrics_input.schema.json`, including the exact Stage 1 Otsu threshold, 20% total axial padding, interpolation batch size, fixed endpoint/collar windows, junction-mask radius, and bounded label-blind corridor bootstrap policy.

Preserve the returned measurement hierarchy:

1. Treat `same_material_component_connects_a_to_b` as Claire's primary connectivity measurement: one identical 26-neighbor foreground component in the full calibrated corridor intersects the A and B endpoint windows.
2. Treat `same_component_connects_collar_a_to_b` as supplementary junction-masked evidence at fixed `0.20L` and `0.80L` collar slabs.
3. Treat axial foreground fractions, minimum foreground fraction, EDT radius, and curvature only as measurements.

Do not replace the primary endpoint result with the masked collar result. Do not infer a defect class from either value.

Accept only the versioned `part2-mcp-response/1.0.0` envelope. On `pass`, verify the returned artifact hashes and report these files:

- `corridor_calibration.json`
- `per_strut_metrics.csv`
- `per_strut_profiles.json`
- `metrics_report.json`

The four files are one immutable bundle. Reject overwrite; permit only an exact
idempotent replay of a byte-identical complete bundle. A partial or differing
output directory is `halt`, never an instruction to repair files locally.

Treat `manual_review` as a 20%-padded ROI boundary condition. Preserve the returned evidence and stop for explicit scientist disposition. Treat `halt` or an error envelope as terminal for deterministic Stage 2; do not retry or modify inputs.

## Boundaries

Treat all outputs only as measurements. Do not assign missing, broken, thin, or bent labels or tune classification cutoffs. Stage 3 owns classification. Do not change localized nodes, the canonical mask, the Otsu threshold, registration artifacts, or frozen configuration.
