# Control, retry, and access policy

## Hash and handoff invariants

- Freeze the pipeline config hash and all seven contract hashes at manifest
  creation. A changed config starts a distinct run; never edit/reset this run.
- Bind each attempt to a unique token derived from specimen, stage, attempt,
  timestamp, config, contract, predecessor receipt, and exact input artifacts.
- Give agents only the attempt-scoped handoff. Keep predecessor receipts and
  sensitive-path registries in orchestrator scope.
- Re-hash handoff inputs, outputs, completion receipt, active prior artifacts,
  and verifier/freeze checkpoints before transition.
- Treat exact receipt replay as a byte-identical no-op, but still re-hash live
  artifacts. Reject a replay whose artifact was deleted or changed.
- Reject absolute, escaping, symlink-aliased, duplicate, stale, or colliding
  artifact records.

`specimen_manifest.json` has one declared lineage exception: Stage 2 may replace
the Stage 0 version only when its output record binds the exact prior hash and
the data-prep completion receipt binds prior and final hashes. No other
same-path replacement is legal.

## Transitions and attempts

- Legal: `locked → ready → running → pass|manual_review|halt`.
- Only verified predecessor `pass` performs `locked → ready`.
- `pass` and `halt` are immutable.
- Resume `manual_review → ready` only with a hashed resolution under
  `analysis/<specimen_id>/reviews/`, a reason, an audit timestamp, and remaining
  judgment attempts. Preserve all prior evidence.
- Put review outputs under the attempt-scoped
  `reviews/stage_<n>_attempt_<attempt>/` path; never overwrite canonical pass
  artifacts during a correction attempt.
- A second `manual_review` at the two-attempt limit becomes `halt`.
- A deterministic gate failure always returns `halt`; transport or judgment
  correction is the only retryable class.
- Stage 5 is reserved atomically at start. It cannot be re-opened after
  success, crash, review, or halt.

## Access checks

Authorize using the combination of canonical path, role, SHA-256, stage,
consumer, and phase. Compare hashes against the sensitive registry so renaming
a label file does not declassify it.

- Stage 2/3: no defect-label hash, role, or path.
- Stage 4: dev split only for `missing_strut_agent`; full intentional labels and
  sealed split forbidden; the shared handoff is dev-blind and only the separate
  consumer-scoped handoff may name or hash the development split.
- Stage 5: sealed split only for `eval_agent` during sealed evaluation.
- Stage 6: no raw dev/sealed split; use evaluation/attribution artifacts.
- Autonomous Stage 2: aligned JSON only in a supplemental post-freeze validator
  handoff tied to the CT-only freeze receipt.
- Stage 2→3: bind the canonical mask's exact path, role, dtype, ZYX shape,
  retention, and SHA-256 in both handoffs; neither handoff may contain any
  label path, role, hash, count, or content.

## Fail-closed dependency behavior

Preflight every declared agent and MCP tool against its exact contract/response
schema. On missing, unhealthy, disabled, or incompatible capability, record a
structured `halt` with zero fabricated outputs and `fallback_used: false`.
Current unimplemented downstream capabilities are expected to halt rather than
being approximated locally.
