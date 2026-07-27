---
name: part2-pipeline-runbook
description: Orchestrate or resume the LLNL missing-strut Part 2 NDE pipeline through strict Stage 0–6, hash-sealed artifact handoffs. Use for pipeline preflight, stage dispatch, receipt verification, registration-mode isolation, label-access enforcement, retry/manual-review decisions, one-shot sealed evaluation, and final NDE presentation assembly.
---

# Part 2 Pipeline Runbook

Operate only the control plane. Invoke bounded agents and the required
`segmentation-tools` MCP server; never perform CT, registration, ROI,
classification, rendering, spatial-statistics, or evaluation algorithms.

## Preflight

1. Read `AGENTS.md`, `analysis/<specimen_id>/manifest.json`, and the current
   `analysis/contracts/*.json` contract. Read [stages.md](references/stages.md)
   for artifact routing and [control-policy.md](references/control-policy.md)
   before dispatching or resuming a stage.
2. Require a healthy `segmentation-tools` MCP server and every tool schema and
   bounded agent contract declared by the stage contract. Treat a missing,
   disabled, unhealthy, or schema-incompatible dependency as `halt`.
3. Record the structured dependency halt through the checked-in deterministic
   orchestration-state interface. Explain which dependency is unavailable and
   that the MCP client must be configured or restarted.
4. Never replace an MCP operation with a CLI, direct implementation import,
   bundled script, or local approximation.

## Run the state machine

Use `src/part2_orchestration.py` through its checked-in CLI for every state
change. Never edit the pipeline manifest, handoff, receipt, attempt counter, or
one-shot marker by hand.

1. Validate the manifest self-hash, frozen config hash, contract hashes, and
   all active artifact hashes.
2. Start only the unique `ready` stage. Supply exact role/path/SHA-256 records;
   the control plane writes an attempt-scoped sanitized handoff.
3. Dispatch only the contract owner and declared bounded subagents. Do not give
   an agent the unrestricted pipeline manifest or another agent's scoped input.
   Stage 4 returns a dev-blind shared handoff plus a separate dev-label handoff
   that only `missing_strut_agent` may receive.
4. Accept only a current, self-hashed completion receipt bound to the specimen,
   stage, attempt token, contract, config, predecessor receipt, handoff, and
   exact output hashes.
5. Let `pass` unlock the next numeric stage. Stop immediately on
   `manual_review` or `halt`.

The immutable order is:

`0 intake → 1 design labels → 2 registration/QA → 3 ROI metrics → 4 classification/verifier → 5 sealed reporting → 6 spatial/render/report`

Stage 2 cannot start after Stage 0 until Stage 1 passes. If a planned agent or
tool is absent, keep downstream stages locked and emit the structured halt;
never fabricate its artifact.

## Apply terminal-state policy

- `pass`: verify receipt and every live artifact before unlocking the declared
  next stage.
- `manual_review`: stop automation, preserve the attempt handoff, receipt,
  outputs, and evidence. Resume only through an explicit hashed resolution.
- `halt`: fail closed permanently for this run; never retry or unlock a
  downstream stage.

Allow at most two starts for agent/judgment stages. Never retry a deterministic
gate failure. Stage 3 has one deterministic attempt. Reserve Stage 5 before
exposing the sealed split; it has exactly one attempt even after crash,
`manual_review`, or `halt`. See [control-policy.md](references/control-policy.md)
for resume and exhaustion rules.

## Enforce registration and label isolation

- In `challenge_aligned_json` mode, pass only the scientist-authorized aligned
  JSON declared at intake.
- In `autonomous_v2` mode, start Stage 2 with CT and nominal graph only. Record
  and verify the CT-only registration-freeze receipt before authorizing an
  optional aligned-JSON validator handoff. Never alter the frozen fit hashes.
- Never pass defect labels to Stage 2 or Stage 3.
- Pass the development split only to `missing_strut_agent`; the defect lead,
  thin/broken specialists, and independent verifier receive no raw dev labels.
- Pass the sealed split only to `eval_agent` in Stage 5. Stage 6 consumes the
  attribution and evaluation artifacts, never raw label splits.
- Enforce `missing > broken > thin > present`; record bent separately.

## Finish the deliverable

Treat Stage 5 as one-shot reporting, not a performance gate or optimization
loop. Stage 6 may cite only committed or hash-verified artifact values and must
not recompute them. The orchestrator owns the demo manifest and presentation
checklist. Read and satisfy
[presentation-checklist.md](references/presentation-checklist.md) before
accepting the Stage 6 receipt.
