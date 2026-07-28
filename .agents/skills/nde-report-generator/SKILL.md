---
name: nde-report-generator
description: Assemble the production Stage 4 lattice NDE report from frozen graph, classification, strut-metric, evidence, and QA artifacts. Use when report_agent must create graph-aware spatial statistics, a classified 3D lattice render, number crosschecks, and a traceable final report without recomputing science or reading labels.
---

# NDE Report Generator

Consume only the immutable Stage 4 handoff and its hash-verified Stage 3
predecessor artifacts. Treat the handoff's specimen/design IDs, paths, roles,
hashes, config hash, scope, and output paths as authoritative.

## Workflow

1. Preflight `get_strut_report`, `compute_spatial_stats`, and
   `render_lattice_3d` on the required `segmentation-tools` MCP server at
   `part2-mcp-response/1.0.0`. Missing, unavailable, or incompatible
   dependencies require a structured `halt` and an immediate stop; never
   substitute a CLI, direct import, or local science.
2. Rehash every allowed input and reconcile it with the Stage 4 handoff and
   verified Stage 3 receipt. Reject undeclared, stale, or contradictory files.
3. Invoke `compute_spatial_stats` with the committed localized graph,
   classifications, and per-strut metrics. Write only the contract-declared
   JSON and PNG paths. Require complete explicit strut-ID coverage, matching
   graph endpoints, artifact hashes, and `gate: pass`.
4. Invoke `render_lattice_3d` with the same localized graph and classification
   hashes. Require every nominal strut exactly once and preserve the frozen
   `missing`, `broken`, `thin`, and `present` classes; keep bent as a separate
   attribute.
5. Invoke `get_strut_report` for every non-present strut cited in the report.
   Supply an evidence manifest when one is declared. Use only its committed
   metrics, reasons, thresholds, evidence links, and hashes.
6. Write `number_crosscheck.json` by comparing already-persisted totals and
   class counts across classifications, spatial statistics, the lattice
   render, and cited strut records. Do not derive a new scientific metric.
7. Compile `nde_report.md` with specimen identity, requested scope,
   registration/ROI QA status, class totals, spatial clusters, flagged-strut
   evidence, the two Stage 4 figures, provenance, and limitations. Report
   missing geometry relative to the nominal graph only; never infer design
   intent.
8. Write `presentation_checklist.json` against the runbook checklist, bind all
   required Stage 4 artifacts and SHA-256 values, and return exactly one stage
   completion receipt.

## Isolation and replay

Never read CAD/STL, aligned graphs, training/evaluation labels, deleted-edge
labels, ground-truth segmentation, or unrestricted research manifests. Never
recompute classification, registration, ROI measurements, spatial statistics,
or rendering in agent code. Refuse overwrite except exact idempotent replay.
If the number crosscheck fails, return `halt`; if presentation disposition is
the only unresolved item, return `manual_review` with bounded evidence.
