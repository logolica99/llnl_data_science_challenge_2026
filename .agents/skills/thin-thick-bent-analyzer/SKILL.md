---
name: thin-thick-bent-analyzer
description: Measure and classify thin, thick, and bent struts in a registered X-ray CT lattice using file-backed MCP computation. Use for TIFF plus registered-JSON strut inspection, radius-profile comparisons, centerline-bending analysis, peer-baseline classification, evidence PNG generation, or production of a verified Stage 4 thin/thick/bent hand-off.
---

# Thin/Thick/Bent Analyzer

Produce auditable thin, thick, and bent findings from an aligned CT TIFF and
registered lattice JSON. Treat the JSON coordinates as a soft spatial prior.

## Required boundary

Require the repository CT MCP server and these tools:

1. `compute_strut_metrics`
2. `classify_struts`
3. `render_strut_evidence`

Use `run_thin_thick_bent_pipeline` only when the user requests the complete
default flow without an agent threshold-calibration step.

Halt if a required tool is missing, unhealthy, or schema-incompatible. Never
replace an unavailable MCP tool with a local script, direct Python import, or
ad hoc array computation. Pass paths and compact receipts between steps; never
place CT arrays or per-section tables in context.

## Workflow

1. Accept exactly one CT TIFF, one registered JSON, one frozen CT threshold,
   and one output directory. Accept physical voxel spacing only from
   authoritative metadata; otherwise retain voxel units.
2. Confirm the input association and output location. Never modify the TIFF,
   registered JSON, existing HTML viewer, or `standalone_strut_viewer`.
3. Invoke `compute_strut_metrics`. Require a ready receipt and verify the
   `measurement_manifest.json` hand-off before continuing.
4. Review only compact population summaries needed to choose or approve
   thresholds. Keep scope limited to thin, thick, and bent.
5. Freeze all chosen cutoffs in a thresholds JSON. Do not change them after
   inspecting individual candidate identities.
6. Invoke `classify_struts` with the measurement CSV paths and frozen
   thresholds. Preserve independent thin, thick, and bent flags; do not force
   combined findings into one lossy label.
7. Invoke `render_strut_evidence`. Use radius profiles for thin/thick and
   centerline-deviation plus curvature profiles for bent. Do not use a radius
   graph as the primary bent evidence.
8. Verify all artifact hashes, finding counts, and required paths. Emit or
   consume `classification_handoff.json` as the downstream contract.

Read [artifact-contract.md](references/artifact-contract.md) when validating
inputs, outputs, defect rules, or downstream hand-offs.

## Scientific policy

- Form peer groups from `unit_cell_edge_idx` design families in the registered
  JSON. Use only CT measurements passing coverage, tracking-confidence,
  junction, dense-boundary, and radius-variation gates.
- Use robust median/MAD population baselines. Thin/thick decisions must satisfy
  both a radius-ratio cutoff and robust-score cutoff.
- Diagnose bending from tracked CT-centerline deviation relative to its
  best-fit straight CT line, with adjacent-section support and curvature.
  A coherent registration offset or rigid tilt is not bending.
- Treat all cutoffs as screening thresholds until scientifically validated.
- Return `uncertain` when peer support or tracking quality is insufficient.

## Limits and verification

- Analyze at most one specimen per invocation.
- Permit at most two corrections for a failed artifact/gate, then halt with the
  failing receipt and reason.
- Verify that later agents can operate from files without reopening the TIFF.
- Report thresholds, units, finding counts, hand-off path, and limitations.
