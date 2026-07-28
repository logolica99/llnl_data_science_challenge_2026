# Stage contracts and artifact routes

Treat `analysis/contracts/*.json` as authoritative machine-readable contracts.
This reference explains their intended routing; it does not replace them.

## Stage 0 — specimen intake

- Owner: `specimen_ingest`.
- Inputs: the scientist-confirmed specimen ID, CAD STL, nominal graph, CT, and
  optional authorized aligned JSON in challenge mode.
- Outputs: `analysis/<specimen_id>/config/{ingest_request,specimen_manifest,ingest_receipt,data_prep_handoff}.json`.
- Gate: confirmed association and conventions, authoritative MCP volume
  metadata, valid manifest/receipt hashes, and a ready handoff.
- Prohibitions: labels, segmentation, registration, scientific QA, and filename
  inference.

## Stage 1 — design-only labels

- Owner: `design_diff`.
- Inputs: nominal graph plus `0.stl`, `0.1.stl`, `0.5.stl`, and `1.stl`; no CT
  or aligned graph.
- Outputs: normalized explicit ID map, orientation evidence, three intentional-deletion lists,
  `labels/dev_split.json`, `evals/labels/sealed_split.json`, and
  `labels/label_report.md`.
- Gate: independently validated counts 18/93/186, triangle-deficit ratio 170–180, valid
  IDs, unambiguous orientation/edges, and a disjoint stratified 30/70 split.

## Stage 2 — data preparation

- Owner: `data_prep`.
- Inputs: verified Stage 0 data-prep handoff, CT, nominal graph, and frozen
  configuration. No Stage 1 label artifact enters the handoff.
- Outputs: analysis config; exact Otsu histogram/report; canonical uint8 mask
  and bounded mask comparison; registered and
  independently localized graphs/reports; registration QA and figures; data
  prep result/completion receipt.
- Gate: histogram, topology, registration-image support, independent node
  refinement, coarse-capture, and padded-ROI checks pass. Record the stricter
  metrology result separately; a metrology-only failure requires explicit
  ROI-only review disposition.
- Autonomous branch: freeze and hash CT-only registration outputs before any
  optional aligned-JSON validation.

## Stage 3 — per-strut measurement

- Owner: `strut_metrics`.
- Inputs: CT, frozen analysis config/Otsu result, canonical mask contract,
  localized graph, and QA only.
- Outputs: corridor calibration, `per_strut_metrics.csv`, profiles, and metrics
  report.
- Gate: empirical radius bootstrap, unique/exhaustive IDs, one row per nominal
  strut, valid ROI bounds, and complete provenance. This is deterministic and
  has no retry after a gate failure.

## Stage 4 — blind defect classification

- Owner: `defect_lead`; bounded specialists are `missing_strut_agent`,
  `thin_strut_agent`, and `broken_strut_agent`; run `classifier_verifier` last.
- Only the missing specialist receives dev labels. The lead and verifier use a
  hashed calibration attestation, not raw dev IDs. Nobody receives sealed IDs.
- Produce the closed `missing-calibration-attestation/1.0.0` and
  `classifier-verifier-report/1.0.0` documents exactly as declared in
  `defect_analysis.json` under `output_document_schemas`. The attestation binds
  the missing specialist's scoped handoff, metrics, and findings; the verifier
  binds all specialist outputs and the canonical hash of the complete, sorted
  evidence set.
- Outputs: specialist findings, thresholds, one classification per strut,
  decision log, evidence index/packets, and independent verifier report.
- Gate: fixed precedence, bent as a non-competing attribute, logged
  adjudications, hash-bound evidence, and verifier `pass` bound to the exact
  classifications, thresholds, evidence, and decision log.

## Stage 5 — one-shot sealed reporting

- Owner: `eval_agent`; sole consumer of the sealed split.
- Inputs: frozen Stage 4 artifacts/verifier, intentional labels for attribution,
  sealed split, and evidence.
- Outputs: timestamped detection metrics, attributed struts, triage results,
  and a human spot-check list.
- Gate: integrity and completeness only—strict/lenient recall with Wilson CI,
  confusion matrix, attribution, and triage hygiene. Do not gate on recall or
  report precision/F1 against intentional deletions.

## Stage 6 — report and presentation

- Owner: `report_agent`; presentation artifacts remain orchestrator-owned.
- Inputs: verified upstream artifacts, attribution, triage, and evidence; no
  raw dev/sealed split.
- Outputs: spatial statistics/figures, graph-based 3D defect render, NDE report,
  number crosscheck, prose judge result, demo manifest, and presentation
  checklist.
- Gate: deterministic number crosscheck passes and every value cites a frozen
  artifact. The prose judge evaluates only structure, clarity, and honest
  uncertainty.
