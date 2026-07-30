---
name: part2-pipeline-runbook
description: Orchestrate or resume the production nominal-graph + CT NDE pipeline through strict Stage 0–4 hash-sealed artifact handoffs.
---

# Part 2 Production Pipeline Runbook

For the seal-free hackathon demo, invoke `$hackathon-nde-pipeline` and
dispatch stage subagents through `segmentation-tools` MCP. Do **not** use
`scripts/hackathon_pipeline.py` when MCP is available.

Operate only the control plane for the **hash-sealed production** path. Invoke
bounded agents and the required `segmentation-tools` MCP server; never perform
scientific algorithms locally.

## Production boundary

Accept one scientist-confirmed nominal lattice graph JSON and one specimen CT
TIFF/NPY. Do not request CAD/STL variants, an aligned graph, design-deletion
labels, development/sealed splits, or ground truth. Those belong to separate
research evaluation, not production analysis.

## Preflight

1. Read `AGENTS.md`, the current pipeline manifest, the current stage contract,
   [stages.md](references/stages.md), and
   [control-policy.md](references/control-policy.md).
2. Require the exact declared agents and MCP schemas. Missing, unhealthy, or
   incompatible dependencies are a structured `halt`. If MCP is unavailable,
   stop immediately; do not substitute local computation.
3. Use `scripts/manage_part2_pipeline.py` for every state transition. Never
   edit manifests, handoffs, attempts, or receipts manually.

## Stage order

`0 intake + graph validation → 1 segmentation/registration/QA → 2 strut metrics → 3 defect analysis/verifier → 4 report`

- Stage 0 calls both `inspect_volume_metadata` and `load_lattice_graph`.
- Stage 1 registers the nominal graph to CT autonomously and freezes the CT-only
  fit before independent node localization.
- Stage 2 performs deterministic, label-free measurements.
- Stage 3 performs label-free classification under frozen policy and independent
  verification. A missing nominal strut is reported as `missing`; production
  does not infer intent.
- Stage 4 reports only hash-verified committed values.

## Terminal-state policy

- `pass`: verify every artifact and receipt, then unlock the next stage.
- `manual_review`: stop and preserve evidence; resume only with an explicit,
  hashed same-stage resolution.
- `halt`: fail closed for the run.

Agent/judgment work has at most two attempts. Deterministic gate failures are
never retried. Training/evaluation labels are forbidden in every production
stage. Research benchmarking is a separate workflow and cannot mutate a frozen
production run.

Before accepting Stage 4, require the number crosscheck and the presentation
maintenance checklist in
[presentation-checklist.md](references/presentation-checklist.md).
