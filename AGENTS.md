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
