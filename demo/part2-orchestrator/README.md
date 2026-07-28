# Part 2 NDE Orchestration Proof

## Real callable-runtime proof

The default demonstration uses no synthetic specialist artifacts. A localhost
Python backend starts the installed `codex app-server`, creates a real Codex
thread with the checked-in orchestrator instructions, and streams structured
runtime events to the browser and terminal. After the scientist explicitly
confirms the displayed association, the turn asks the orchestrator to invoke
the project `specimen_ingest` subagent for Stage 0 only.

The backend does not accept browser-supplied filesystem paths, grant interactive
approvals, fabricate receipts, or continue to Stage 1. Missing inputs, tools,
agent dispatch, or receipt gates remain visible as real failures.

```bash
npm run demo
```

Open <http://localhost:3000/>. Confirm the displayed inputs, then select
**Attempt real Stage 0**. Stop the backend with `Ctrl-C`. Each run stores its
local transcript under `runtime-evidence/<run_id>/`.

## Fixture walkthrough (not proof)

This local web app lets a team watch the LLNL missing-strut Part 2 control
plane advance through Stages 0–6. Production orchestration code runs live to
create and validate the manifest, handoffs, receipts, hashes, access boundaries,
retry state, and terminal transitions.

Every downstream specialist action, MCP capability response, verifier report,
and scientific artifact is a deterministic fixture simulation. The demo does
not invoke a live Codex model or execute CT, registration, ROI, classification,
rendering, spatial-statistics, or evaluation algorithms, and it does not display
real specimen findings.

The richer Stage 0–6 walkthrough remains available only as a clearly separated
UI fixture test. It must not be used as evidence that agents or scientific
tools ran.

## Run the fixture walkthrough

From this directory:

```bash
npm install
npm run demo:fixtures
```

Open <http://localhost:3000>. The single command starts both the localhost-only
Python control-plane adapter and the frontend. Stop both with `Ctrl-C`.

The **One check** control sends one mutation request and performs one legal
control-plane transition: a `ready` stage starts, or a `running` stage completes
or stops. The page's live terminal shows the same redacted event lines that the
Python process flushes to stdout, so the UI can be compared directly with the
launching terminal without exposing raw label paths or payloads.

## Demonstration scenarios

- **Verified walkthrough:** real Stage 0–6 state transitions using fixture
  specialist artifacts.
- **Manual review:** Stage 2 pauses, preserves evidence, and resumes only after
  a hashed scientist resolution.
- **Tampered receipt:** a changed receipt self-hash is rejected while all
  downstream stages remain locked.
- **Missing dependency:** preflight records a structured `halt`, uses no
  fallback, and consumes no attempt.

Both `autonomous_v2` and `challenge_aligned_json` registration branches are
available. Autonomous mode visibly freezes CT-only registration artifacts
before permitting optional aligned-reference validation.

## Safety boundary

Every browser session gets a server-owned temporary repository outside the
checkout. Browser requests cannot select filesystem paths, capability
inventories, artifact payloads, stage numbers, terminal states, or receipts.
The API returns only a redacted, hash-verified projection and never exposes raw
development/sealed label records or scoped handoffs.

The hosted frontend runtime cannot execute the repository's Python control
plane. This demonstrator is intentionally local; a future hosted version would
need a separately authenticated orchestration service.

## Verify it

```bash
npm test
npm run test:server
```
