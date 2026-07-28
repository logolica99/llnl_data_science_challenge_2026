# Production pipeline maintenance

## Canonical sources

When changing the workflow, update these together:

1. `src/part2_orchestration.py` stage constants and transition rules;
2. `analysis/contracts/*.json` production contracts;
3. `.codex/agents/*.toml` owner instructions;
4. `.agents/skills/part2-pipeline-runbook/` policy and stage routing;
5. `notes/PART2_DESIGN.md` and both SVG diagrams;
6. tests covering contracts, orchestration, intake, MCP boundaries, and docs.

The complete production contract set is `specimen_ingest.json`, `data_prep.json`,
`strut_metrics.json`, `defect_analysis.json`, and `nde_report.json`. Do not place
research workflow contracts in `analysis/contracts/`.

The production MCP server is `src/mcp_server.py`. Its registered tool set must
equal the union of `required_dependencies.mcp_tools` in the five contracts.
Research-only tools belong in `research/mcp_server.py`, which is disabled and
non-required in `.codex/config.toml`; research outputs must stay beneath
`research/runs/`.

## Current production input contract

- Required: nominal lattice graph JSON and specimen CT TIFF/NPY.
- Scientist-declared: specimen ID, design ID, requested scope, graph axes, and
  CT array axes when metadata is insufficient.
- Forbidden: CAD/STL variants, aligned graphs, training/evaluation labels,
  known-defect lists, and ground-truth segmentation.

## Safe change checklist

1. Keep stage numbers contiguous from zero and update `STAGE_NAMES`.
2. Ensure every stage declares at least one agent and one required MCP tool.
3. Update handoff/review paths when renumbering a stage.
4. Keep Stage 0 graph normalization bound to `load_lattice_graph` response and
   artifact hashes.
5. Keep Stage 4 bound to `report_agent`, `nde-report-generator`,
   `get_strut_report`, `compute_spatial_stats`, and `render_lattice_3d`.
6. Compare registered production tools with the exact contract dependency
   union; extra production registrations are a boundary failure.
7. Search for stale stage names, stage numbers, and research-only inputs.
8. Regenerate SVG/PNG diagrams after workflow changes.
9. Run syntax, contract, focused, research-boundary, and full tests before merging.

## Verification commands

```bash
python -m compileall -q src scripts
python -m json.tool analysis/contracts/specimen_ingest.json >/dev/null
python -m pytest tests/test_specimen_ingest.py tests/test_part2_orchestration.py
python -m pytest tests/test_stage4_reporting.py tests/test_research_mcp_tools.py
python -m pytest
```

Repository-wide stale-reference audit:

```bash
rg -n "design_diff|Design Diff|stage_5|stage_6|Stage 5|Stage 6|dev_split|sealed_split" \
  AGENTS.md .codex .agents analysis/contracts notes src scripts tests demo
```

Any remaining match must be explicitly marked research-only or historical and
must not be reachable from a production contract, agent, or skill.

The following names must not be registered by `src/mcp_server.py`:
`compute_detection_metrics`, `summarize_nde_artifacts`, `render_volume_3d`,
`skeletonize`, and `explore_ct_thresholds`.

Low-level registration primitives retain historical aligned-graph support for
offline research tests. Production contracts, CLI choices, agent prompts, and
MCP schemas expose only `autonomous_v2`. The pipeline manifest also retains an
empty `sealed_evaluation` compatibility marker for old readers; production
validation requires `consumed` to remain `false`.

## Compatibility policy

Do not silently reinterpret an existing run manifest after contract hashes or
stage numbering change. Existing runs remain historical evidence. Start a new
run under the current five-stage contracts.
