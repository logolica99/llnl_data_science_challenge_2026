# Production stage contracts and artifact routes

`analysis/contracts/*.json` is authoritative. This file is the maintainer’s
map of the five-stage production path.

## Stage 0 — intake and nominal-graph validation

- Owner: `specimen_ingest`.
- Inputs: confirmed specimen/design IDs, scope, nominal graph JSON, and CT.
- MCP: `inspect_volume_metadata`, then `load_lattice_graph`.
- Outputs: normalized nominal graph plus the CT metadata response/call receipt,
  ingest request, specimen manifest, ingest receipt, and data-prep handoff.
- Gate: unique graph IDs, valid references, finite coordinates, topology hash,
  3-D CT metadata, input hashes, and confirmed association all pass.

## Stage 1 — data preparation

- Owner: `data_prep`.
- Inputs: Stage 0 bundle, nominal/normalized graph, and CT.
- Outputs: exact Otsu evidence, canonical mask, autonomous registration,
  independently localized graph, QA, and analysis-ready manifest.
- Gate: segmentation, coarse registration, localization, padded-ROI, and
  requested-scope authorization pass. No aligned graph or labels are accepted.

## Stage 2 — per-strut measurement

- Owner: `strut_metrics`.
- Inputs: Stage 1 mask, localized graph, CT, QA, and frozen configuration.
- Outputs: corridor calibration, metrics CSV, profiles, and report.
- Gate: one complete provenance-bound row per nominal strut.

## Stage 3 — label-free defect analysis

- Owner: `defect_lead`; specialists cover missing, thin/bent, and broken, with
  `classifier_verifier` last.
- Inputs: Stage 2 measurements and Stage 1 evidence; no training/eval labels.
- Outputs: specialist findings, frozen thresholds, classifications, evidence,
  decision log, and verifier report.
- Gate: fixed precedence, evidence for every non-present call, complete nominal
  strut coverage, and independent verifier pass.

## Stage 4 — report

- Owner: `report_agent`.
- Skill: `nde-report-generator`.
- MCP: `get_strut_report`, `compute_spatial_stats`, and `render_lattice_3d`.
- Inputs: committed classifications, metrics, evidence, and QA.
- Outputs: spatial statistics, graph-based 3-D render, NDE report, number
  crosscheck, and presentation checklist.
- Gate: all numbers trace to frozen artifacts and no intentional-versus-
  unintentional attribution is claimed.

## Separate research evaluation

Design-variant comparison and labeled benchmarking are not production stages.
They may consume copies of frozen outputs in an isolated research workspace,
but may not write production manifests, thresholds, classifications, receipts,
or reports.
