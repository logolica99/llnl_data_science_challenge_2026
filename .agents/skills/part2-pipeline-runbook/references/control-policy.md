# Control, retry, and access policy

## Integrity invariants

- Freeze the configuration and five contract hashes when the run is created.
- Bind every attempt to the specimen, stage, timestamp, config, contract,
  predecessor receipt, and exact input artifacts.
- Rehash live inputs, outputs, handoffs, and receipts before every transition.
- Reject absolute, escaping, stale, duplicate, or path-colliding artifacts.
- Exact receipt replay is byte-identical and must not alter state.

Stage 1 may replace the Stage 0 specimen manifest only through the declared
replacement binding and data-prep completion receipt.

## Transitions and retries

Legal transitions are `locked → ready → running → pass|manual_review|halt`.
Only predecessor `pass` unlocks the next stage. Agent/judgment stages allow two
attempts; deterministic Stage 2 allows one. Deterministic gate failure is a
non-retryable `halt`.

## Access policy

- Stage 0: nominal graph + CT only; graph normalization is an MCP output.
- Stage 1: Stage 0 graph/CT artifacts only; no CAD/STL, aligned graph, or labels.
- Stage 2: measurements only; no classification or labels.
- Stage 3: metrics/evidence only; no training, development, sealed, or ground-
  truth labels.
- Stage 4: committed production artifacts only; no research-label artifacts.

Research evaluation operates on exported copies and cannot unlock stages,
change thresholds, revise classifications, or write a production receipt.
