---
name: strut-defect-analyzer
description: Compute deterministic label-blind Stage 2 strut measurements and perform strictly handoff-bound Stage 3 defect analysis. Use for batched rotated-cuboid connectivity/profiles or for missing, broken, thin, and bent specialist findings, deterministic classification merge, evidence rendering, and independent verification.
---

# Strut Measurement and Defect Analysis

Select exactly one workflow from the verified handoff identity. Never allow
Stage 2 measurement authority and Stage 3 classification authority to overlap.

## Stage 2 Strut Metrics

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

1. Treat `same_material_component_connects_a_to_b` as Claire's primary connectivity measurement: one identical 26-neighbor foreground component in the full 20%-padded calibrated corridor intersects the unchanged nominal A and B endpoint windows.
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

## Stage 3 Defect Analysis

Require owner `defect_lead`, stage 3, the verified Stage 2 predecessor receipt,
and exactly the artifact roles declared by
`analysis/contracts/defect_analysis.json`. Rehash the frozen analysis config,
corridor calibration, metrics, profiles, localized graph, and CT before every
MCP call. Never recompute Stage 1 or Stage 2 science.

Pass `stage_2_completion_receipt_filepath` to every Stage 3 MCP call. Require
the canonical passing `part2-stage-receipt/1.1.0` at its attempt-scoped path,
verify its self-hash and Stage 2 contract/config identity, rehash its Stage 2
input handoff and complete four-artifact measurement bundle, and require its
canonical receipt hash to equal the Stage 3 predecessor binding. A receipt hash
string without the verified receipt artifact is not an authorized handoff.

Use `classify_struts` only with the operation authorized for the caller:

- `missing_strut_agent`: `analyze_missing`
- `broken_strut_agent`: `analyze_broken`
- `thin_strut_agent`: `analyze_thin` and `analyze_bent`
- `defect_lead`: `merge`
- `classifier_verifier`: `verify`

Require every specialist to emit `part2-specialist-findings/1.0.0`. Missing
requires primary A-to-B disconnection and a central (20%-80%) present-slice
fraction of at most 10%, where a slice is material-bearing at foreground
fraction ≥ 0.05 (Claire recovered standalone 96/78 rule). Broken uses the
central `0.20L`-`0.80L` profile, P90 reference, `0.50 * P90` deficient-slice
cutoff, 15% deficient fraction or a three-slice run, 0.05 collar support, and
at least 500 voxels in each endpoint-to-collar component. Missing has
precedence. Connected bite cases may be broken; unresolved disconnections
require review. The specimen-specific `y=18` crop filter is viewer-only CSV
postprocessing and must never change scientific findings.

Thin and bent are teammate-owned. If either implementation is `deferred`,
preserve missing/broken development results as `manual_review`; never claim a
production pass or unlock Stage 4. In production require all four findings
complete, merge with `missing > broken > thin > present`, and record bent as a
separate attribute.

Invoke `render_strut_evidence` for every non-present or bent result. Require
local A-to-B-z CT views, the axial profile, supplied metrics/classification,
policy, and exact hashes. Rendering may resample CT for display but must not
recompute metrics or change a class.

Run `classifier_verifier` last. It must not participate in classification. It
must verify complete nominal coverage, precedence, evidence coverage, decision
log, frozen cutoffs, and artifact hashes before emitting a pass report.

Optional CSV viewing and validation are outside this production skill. After
exporting copies of the frozen artifacts into `research/runs/`, a scientist may
use the disabled-by-default
`segmentation-tools-research.export_stage3_validation_csvs` tool. Never include
those CSVs in a Stage 3 handoff, receipt, verifier input, or Stage 4 production
input, and never let the viewer filter modify a finding or classification.

## Boundaries

Under a Stage 2 handoff, treat all outputs only as measurements and never
classify. Under a Stage 3 handoff, consume those frozen measurements without
recomputation and never modify nodes, masks, Otsu, registration, calibration,
or profiles. A missing/unhealthy MCP dependency, schema mismatch, stale
handoff, hash mismatch, forbidden label path, or incomplete production
specialist set is `halt`. Never invoke a CLI, import an MCP implementation,
write substitute artifacts, or overwrite a non-identical artifact.
