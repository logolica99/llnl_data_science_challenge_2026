# Thin/thick/bent strut pipeline

This pipeline measures actual CT cross-sections near each registered design edge,
classifies only thin, thick, and bent candidates, and writes file-backed evidence
for later agents. The registered JSON is a soft spatial prior; the TIFF and JSON
are never modified.

Measurement uses a two-stage 3D tracker. A continuity-constrained plane pass
bootstraps the CT path inside a bounded tube around the registered edge. The
final radius samples are then recentered on a smoothed 3D CT centerline and
resampled perpendicular to its local tangent. Centerline deviation is the
orthogonal distance from each tracked CT center to a robust best-fit straight
3D CT line, not distance from the registered JSON line.

## Run the Brian specimen

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_thin_thick_bent_pipeline.ps1
```

For a bounded smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_thin_thick_bent_pipeline.ps1 `
  -OutputDir .\data\missing_struts\analysis\thin_thick_bent_smoke `
  -MaxStruts 50
```

The launcher refuses an existing non-empty output directory. Use `-Overwrite`
only for an intentional rerun.

## Run without the launcher

Set `PYTHONPATH` to include `.python_packages` and `src`, then run:

```powershell
python src/strut_defect_pipeline.py input.tif registered.json output_dir `
  --threshold 40129 `
  --thresholds-json configs/thin_thick_bent_thresholds.json
```

Useful developer options include `--max-struts 50`, `--strut-ids 12 18 25`,
`--positions 21`, and `--voxel-size-mm <known-spacing>`. Physical-unit fields
remain blank unless authoritative voxel spacing is supplied.

## Output contract

The root `classification_handoff.json` is the hand-off consumed by later
agents. It hashes and indexes:

```text
metrics/
  measurement_manifest.json
  strut_section_measurements.csv
  strut_summary.csv
classification/
  classified_struts.csv
  findings_thin.json
  findings_thick.json
  findings_bent.json
  thresholds.json
  peer_baselines.json
  decision_log.md
evidence/
  evidence_manifest.json
  thin/*_radius.png
  thick/*_radius.png
  bent/*_centerline.png
pipeline_receipt.json
classification_handoff.json
```

`classified_struts.csv` preserves separate `is_thin`, `is_thick`, and
`is_bent` columns, so combined findings are retained. When a radius defect
passes and adjacent-section bend evidence is within 10% of or stronger than
the radius evidence, `classification` uses `bent` as its primary label. This
also covers competitive bend evidence just below the strict standalone bend
gate. The independent flags and both evidence strengths remain available to
later agents. Bent plots show
best-fit-centerline deviation and curvature; they do not use radius as primary
bending evidence.

Each class-specific findings JSON embeds the exact pipeline measurement profile
for every finding: sampled radius, tracked center, best-fit-line deviation,
curvature, validity, exclusion reason, confidence, CT threshold, and the
section-measurement artifact hash. This allows review tools to render the
classification evidence without reopening or recomputing the TIFF.

Tracking-quality gates and radius-quality gates are evaluated separately.
Reliable bending evidence can therefore be classified even when radius
variation is too high for a thin/thick decision. Short one- or two-section gaps
are recovered only when a unique, continuous CT bridge exists between
confident tracked sections.

The 21-position policy requires at least 12 valid samples. Bent evidence needs
at least three adjacent high-deviation samples, preserving approximately the
same physical support length used by the earlier 11-position/two-sample policy.

## Export result IDs to review CSV

Convert one or more class-specific findings JSON files to the existing review
CSV header format:

```powershell
python src/export_finding_ids_csv.py `
  output/classification/findings_bent.json `
  -o output/classification/findings_bent.csv
```

Only `strut_id` is populated. Multiple JSON inputs can be supplied to create a
deduplicated combined CSV; the remaining review columns are intentionally
blank.

## Test

```powershell
$env:PYTHONPATH = "$PWD\.python_packages;$PWD\src"
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  -m unittest tests.test_registration_tolerant_screening `
    tests.test_thin_thick_bent_pipeline `
    tests.test_export_finding_ids_csv -v
```

The tests use synthetic TIFF/JSON fixtures and do not require any agent,
orchestrator, or Brian full-volume run.

## Enable the agentic MCP flow

Configure the repository server in `~/.codex/config.toml` using absolute paths:

```toml
[mcp_servers.segmentation-tools]
command = "C:\\path\\to\\python.exe"
args = ["C:\\path\\to\\llnl_data_science_challenge_2026\\src\\mcp_server.py"]
env = {}
```

Restart Codex after changing the configuration and verify that
`segmentation-tools` exposes:

```text
compute_strut_metrics
classify_struts
render_strut_evidence
run_thin_thick_bent_pipeline
```

The project skill is
`.agents/skills/thin-thick-bent-analyzer/SKILL.md`, and the bounded specialist
configuration is `.codex/agents/thin_thick_bent_agent.toml`. Production agent
runs must use the MCP tools and file receipts; the Python entry point is for
independent developer testing.
