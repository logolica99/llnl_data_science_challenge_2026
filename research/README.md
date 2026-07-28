# Research-only surfaces

Nothing under this directory is part of the production Stage 0–4 pipeline.
Research tools may read repository inputs, but they may write only beneath
`research/runs/` and cannot mutate production manifests, handoffs, thresholds,
classifications, receipts, or reports.

`research/mcp_server.py` contains the disabled-by-default
`segmentation-tools-research` MCP server. It owns labeled evaluation,
exploratory threshold comparison, and the historical voxel/skeleton reporting
tools removed from the production MCP surface. Enable it only for an explicit
research task and disable it again before production operation.

The `research/skills/ct-threshold-explorer/` skill is not auto-discovered with
the production skills. Invoke it explicitly by path only for bounded offline
threshold comparisons. `research/scripts/` contains historical standalone
research utilities and is never an MCP or production fallback.
