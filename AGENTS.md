# Repository agent requirements

## MCP-required skills

- Every project skill under `.agents/skills/` must declare at least one required
  MCP dependency in `agents/openai.yaml`.
- Treat a missing, disabled, unhealthy, or schema-incompatible MCP server or
  tool as a hard stop. Explain which dependency is unavailable and that the
  client must be configured or restarted.
- Never replace an MCP-owned operation by invoking a bundled CLI, importing its
  implementation directly, or writing a new local substitute.
- Project skills must not bundle executable scripts for deterministic volume,
  mask, skeleton, or visualization operations. Add those capabilities to the
  required MCP server and invoke them through an MCP client.
- Keep large arrays and artifacts out of model context. MCP tools should write
  them to repository paths and return compact structured status, hashes, and
  artifact paths.
- Validate MCP-backed changes through an MCP client, not only by calling the
  underlying Python function directly.

## Hackathon pipeline (default demo / agentic MCP path)

For hackathon / demo runs that do **not** need hash-sealed receipts, the active
main agent is the control-plane orchestrator and must:

1. Invoke `$hackathon-nde-pipeline`.
2. Read and follow `.codex/agents/hackathon_orchestrator.toml` (or the
   hackathon section of `.codex/agents/orchestrator.toml`).
3. Dispatch only these stage owners as subagents, in order:
   `specimen_ingest` → `data_prep` → `defect_lead` (+ `missing_strut_agent`,
   `broken_strut_agent`, `thin_strut_agent`) → `report_agent`.
4. Require every scientific step to go through the healthy
   `segmentation-tools` MCP server. Missing MCP is a hard stop.

Stages (immutable for this path):

`0 metadata → 1 registration (registered JSON) → 2 defect agents + CSVs →
3 materials-scientist NDE report`

Seal-free MCP tools for this path include `hackathon_localize_lattice_nodes`,
`hackathon_compute_strut_metrics`, `hackathon_analyze_defect`,
`hackathon_merge_defect_classifications`, `hackathon_export_defect_csvs`, and
`hackathon_prepare_report_classifications`, plus existing metadata /
registration / reporting tools.

Never replace those MCP calls with `scripts/hackathon_pipeline.py`, a direct
`llnl_nde.core` import, or a local substitute when MCP is available. The script
is offline-only fallback for humans, not an agent path.

Add future defect agents by extending `DEFECT_KINDS` / specialist MCP coverage
and registering a new bounded subagent under `.codex/agents/`.

## Production Part 2 pipeline orchestration (hash-sealed)

Only when the user explicitly asks for the hash-sealed production Part 2
pipeline or any Stage 0–4 contract operation, the active main agent is the
control-plane orchestrator.

Before performing that scientific work or dispatching any agent, it must:

1. Invoke `$part2-pipeline-runbook`.
2. Read and follow `.codex/agents/orchestrator.toml`.
3. Read the current pipeline manifest and applicable
   `analysis/contracts/*.json` stage contract.
4. Use the deterministic orchestration-state CLI for initialization, preflight,
   stage transitions, receipts, halts, manual-review resolution, and validation.
5. Dispatch only the exact contract-declared stage owner and bounded subagents.

The production input is one scientist-confirmed nominal graph JSON and its
specimen CT volume. CAD/STL variants and design-label splits are research
inputs and must not enter a production handoff.

The production stage identities and order are immutable:

`0 specimen_ingest/graph validation → 1 data_prep → 2 strut_metrics →
3 defect_lead/verifier → 4 report_agent`

Never rename, reinterpret, combine, skip, or replace these production stages
with an ad-hoc workflow.

Treat a missing or unreadable runbook, contract, agent definition, skill, MCP
server, MCP tool, or schema-compatible interface as a hard dependency failure.
Record a structured `halt` through the deterministic orchestration-state CLI,
name every missing or incompatible dependency, and leave all downstream stages
locked. Never use an undeclared agent or tool, weaken a contract, import an MCP
implementation directly, invoke a bundled scientific CLI as fallback, or write
a local substitute.
