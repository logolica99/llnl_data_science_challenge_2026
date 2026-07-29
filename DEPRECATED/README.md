# Deprecated and historical material

Everything below this directory is retained for provenance, reproduction, or
historical challenge context. It is not part of the production Part 2 agent
pipeline and must not be imported, invoked, or treated as an active agent,
skill, MCP implementation, test, or runtime artifact.

## Production replacements

| Archived material | Production replacement |
|---|---|
| `poc/ct_registration/` and `poc/ct_registration_v2/` | `src/llnl_nde/core/registration.py`, the `register_lattice_to_ct` MCP tool, and the `ct-registration` skill |
| `poc/tube_emptiness_test/` and `scripts/missing_strut_heatmap.py` | Research-only design comparison outside the production nominal-graph + CT workflow |
| `agents/segmentation_agent.toml` | `.codex/agents/data_prep.toml` |
| `scripts/ct-threshold-optimizer/` | `research/skills/ct-threshold-explorer/` and the disabled `segmentation-tools-research` MCP server |
| Other files under `scripts/` | Production MCP tools, agent workflows, and QA artifacts |

The POC result directories are snapshots. Their numbers and generated files
may describe older thresholds, coordinate assumptions, or evaluation runs.
They must not be used as current pipeline receipts.

## Part 1 archive

`part1/` contains the unit-cell data, tutorial images, and segmentation-eval
records used by the first half of the challenge. They remain available for
historical reproduction but are not inputs to the production Part 2 pipeline.

## Historical generated output

`historical/9x9x9_octet_lattice/segmentation/` contains the tracked portion of
an earlier segmentation-agent run. Large local masks, run directories, and
other regenerable output are intentionally not committed.

The following local paths are generated data rather than source and remain
ignored instead of being copied into Git history:

- `analysis/brian_tran_9x9x9_runtime_*`
- `analysis/lawrence_registration/`
- `demo/part2-orchestrator/proof-evidence/`
- `demo/part2-orchestrator/runtime-evidence/`
- `data/missing_struts/failed_unwanted/`
- `data/missing_struts/reconstruction/`
- caches and editor/OS metadata

## Research-only replacements

- The production Stage 4 skill is `.agents/skills/nde-report-generator/` and
  uses graph-aware reporting tools.
- Historical skeletonization, voxel rendering, exploratory thresholds, and
  labeled evaluation are available only under `research/` and are not
  registered on the production MCP server.
