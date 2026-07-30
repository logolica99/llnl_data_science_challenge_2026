---
name: hackathon-nde-pipeline
description: Orchestrate the seal-free hackathon NDE pipeline by dispatching stage subagents that call segmentation-tools MCP only (metadata → registration → defect agents/CSVs → materials NDE report). Use when the user wants an agentic MCP/subagent workflow instead of scripts or hash-sealed production receipts.
---

# Hackathon NDE Pipeline (agentic MCP)

You are the **control-plane orchestrator**. Do not run science yourself. Do not
call `scripts/hackathon_pipeline.py`. Do not import `llnl_nde.core` directly.
Dispatch the stage owner subagents below and require every scientific step to go
through the healthy `segmentation-tools` MCP server.

## Stage order (immutable)

`0 metadata → 1 registration → 2 defect agents + CSVs → 3 NDE report`

## Preflight

1. Confirm `segmentation-tools` MCP is available and tools respond with
   `response_schema_version: part2-mcp-response/1.0.0`.
2. Require explicit inputs from the user or run config:
   - `specimen_id`
   - CT path (`.tif` / `.tiff` / `.npy`)
   - nominal lattice graph JSON
   - output root, default `analysis/<specimen_id>/`
3. If MCP is missing, unhealthy, or schema-incompatible: **halt**. Never fall
   back to a Python script, CLI, or direct core import.

## Subagent dispatch map

| Stage | Owner subagent | Skill (if useful) | MCP tools |
|---|---|---|---|
| 0 | `specimen_ingest` (metadata mode) | `$volume-metadata` | `inspect_volume_metadata`, optional `load_lattice_graph` |
| 1 | `data_prep` (registration mode) | `$ct-registration` | `register_lattice_to_ct`, `hackathon_localize_lattice_nodes` |
| 2 | `defect_lead` | `$strut-defect-analyzer` | metrics + defect tools below |
| 2a | `missing_strut_agent` | | `hackathon_analyze_defect` (`missing`) |
| 2b | `broken_strut_agent` | | `hackathon_analyze_defect` (`broken`) |
| 2c | `thin_strut_agent` | | `hackathon_analyze_defect` (`thin` and `bent`) |
| 3 | `report_agent` | `$nde-report-generator` | spatial/report tools + markdown assembly |

Future defect agents: add a subagent + one `hackathon_analyze_defect` kind, then
require that findings file in the merge step.

## Canonical artifact layout

```text
analysis/<specimen_id>/
  stage0/ct_metadata_response.json
  stage0/ct_metadata_mcp_call_receipt.json
  stage1/registered_graph.json
  stage1/registration_report.json
  stage2/localized_graph.json
  stage2/localization_report.json
  stage2/per_strut_metrics.csv
  stage2/per_strut_profiles.json
  stage2/metrics_report.json
  stage2/findings_{missing,broken,thin,bent}.json
  stage2/classified_struts.json
  stage2/classified_struts_report.json
  stage2/thresholds.json
  stage2/decision_log.md
  stage2/csv/missing_struts.csv
  stage2/csv/broken_struts.csv
  stage3/spatial_statistics.json
  stage3/spatial_statistics.png
  stage3/lattice_3d.png
  stage3/nde_report.md
```

## Stage 0 — metadata

Dispatch `specimen_ingest` with:

- `inspect_volume_metadata`
  - `input_filepath` = CT
  - `output_filepath` = `analysis/<specimen_id>/stage0/ct_metadata_response.json`
  - `call_receipt_filepath` = `analysis/<specimen_id>/stage0/ct_metadata_mcp_call_receipt.json`
  - `header_only: true`, `include_sha256: true`
- Optional: `load_lattice_graph` on the nominal JSON for topology sanity

Accept only `status: ok` and `gate: pass`. Record shape/dtype/path; unlock Stage 1.

## Stage 1 — registration

Dispatch `data_prep` with:

1. `register_lattice_to_ct`
   - `registration_mode: autonomous_v2`
   - `nominal_graph_filepath`, `ct_filepath`
   - outputs under `stage1/registered_graph.json` + `stage1/registration_report.json`
   - omit aligned graph
   - soft registration gates may be `halt` in the report; for hackathon continue
     only if the registered JSON was written and the user did not demand strict
     gates
2. Read Otsu `threshold` from the registration report `mode_details.threshold`
3. `hackathon_localize_lattice_nodes` with that threshold → `stage2/` localized
   graph (localization is prerequisite for defects; keep it under stage2)

Do not invent transforms. Do not read challenge `registered_jsons/`.

## Stage 2 — defect agents + CSVs

Dispatch `defect_lead` as coordinator:

1. `hackathon_compute_strut_metrics` → metrics/profiles/report under `stage2/`
2. In parallel, dispatch specialists:
   - `missing_strut_agent` → `hackathon_analyze_defect(..., defect_kind=missing)`
   - `broken_strut_agent` → `hackathon_analyze_defect(..., defect_kind=broken)`
   - `thin_strut_agent` → `thin` then `bent`
3. After all four findings exist:
   - `hackathon_merge_defect_classifications`
   - `hackathon_analyze_defect(..., overwrite=true)` for each specialist when
     regenerating under an existing run directory
   - `hackathon_merge_defect_classifications(..., overwrite=true)`
   - `hackathon_export_defect_csvs` → `stage2/csv/` including
     `missing_struts_viewer_filtered.csv` and `broken_struts_viewer_filtered.csv`
     (broken filtered = not crop-plane AND A-B disconnected; connected bites
     are excluded to match Claire’s final-78 scope)
   - `hackathon_prepare_report_classifications` with nominal graph + metrics +
     crop plane `y=18` + `overwrite=true` → remaps deferred, crop-face
     missing/broken, and connected-bite broken to present for reporting only

`hackathon_compute_strut_metrics(..., overwrite=true)` must rewrite metrics,
profiles, calibration, and metrics_report in seal-free mode.

Thin/bent may return deferred findings; that is expected. Do not block the
hackathon report solely because thin/bent are deferred.

## Stage 3 — materials-scientist NDE report

Dispatch `report_agent` with:

1. `compute_spatial_stats` on localized graph + **filtered** report
   classifications (`classified_struts_report.json`) + metrics
2. `render_lattice_3d` with the same filtered classifications
3. `get_strut_report` only for remaining missing/broken/thin after crop filter;
   prefer `csv/*_viewer_filtered.csv`
4. Write `stage3/nde_report.md` using **filtered** class totals. Note unfiltered
   scientific counts in `classified_struts.json` as a limitation. Never claim
   intentional design deletions. Never recompute metrics in agent code.

## Control-plane return

After each stage, return a compact summary:

- specimen_id, stage, gate
- MCP tools called
- artifact paths
- next subagent action or `complete`

On MCP failure: structured halt naming the missing/incompatible tool. Never
substitute local science.
