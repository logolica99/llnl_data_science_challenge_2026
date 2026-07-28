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

## Part 2 pipeline orchestration

For every request that runs, resumes, validates, or inspects the LLNL Part 2
pipeline or any Stage 0–4 operation, the active main agent is the control-plane
orchestrator.

Before performing scientific work or dispatching any agent, it must:

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

The stage identities and order are immutable:

`0 specimen_ingest/graph validation → 1 data_prep → 2 strut_metrics →
3 defect_lead/verifier → 4 report_agent`

Never rename, reinterpret, combine, skip, or replace these stages with an
ad-hoc workflow.

Treat a missing or unreadable runbook, contract, agent definition, skill, MCP
server, MCP tool, or schema-compatible interface as a hard dependency failure.
Record a structured `halt` through the deterministic orchestration-state CLI,
name every missing or incompatible dependency, and leave all downstream stages
locked. Never use an undeclared agent or tool, weaken a contract, import an MCP
implementation directly, invoke a bundled scientific CLI as fallback, or write
a local substitute.
