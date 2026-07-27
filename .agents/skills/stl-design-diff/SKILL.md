---
name: stl-design-diff
description: Resolve nominal-lattice-to-STL orientation and label intentional CAD strut deletions using triangle-supported tube emptiness. Use for Stage 1 design-only comparison of the 0, 0.1, 0.5, and 1 percent STL variants, including sealed deterministic dev/evaluation splitting.
---

# STL Design Diff

Operate only on the attempt-scoped Stage 1 handoff. Delegate graph parsing,
orientation search, mesh analysis, labeling, splitting, hashing, and artifact
writes to `segmentation-tools` MCP. Never read CT, aligned coordinates,
segmentation, registration, prior labels, or unrestricted manifests.

## Workflow

1. Preflight `load_lattice_graph`, `resolve_cad_graph_orientation`, and
   `label_deleted_edges` at response schema `part2-mcp-response/1.0.0`. A
   missing or unavailable server/tool or incompatible schema means stop with a structured `halt`; do not
   use a CLI, script, direct import, or local substitute.
2. Invoke `load_lattice_graph` once and retain its explicit ID↔row map.
3. Invoke `resolve_cad_graph_orientation` on the nominal graph and 0 percent
   STL. Require origin-centered millimetres, scale preservation, support from
   lattice geometry, and exactly one winning hypothesis. Do not use a
   whole-mesh bounding box as the primary method. Equivalent hypotheses mean
   `manual_review`; never select one by ordering.
4. Invoke `label_deleted_edges` once for the frozen orientation and all three
   variants. The tool must load meshes sequentially, test every nominal strut,
   calibrate the tube radius on the full design, and use triangle support.
5. Accept `pass` only when every gate in [references/gates.md](references/gates.md)
   passes. Return artifact metadata and the stage receipt, never label arrays
   or sensitive label contents in chat.

The 0.5 percent labels are evaluation-only as a complete artifact. Only its
deterministic 30 percent development split may be scoped to the Stage 4 missing
specialist; only the sealed 70 percent split and full 0.5 labels may be scoped
to the Stage 5 evaluator. Stage 2 and Stage 3 handoffs must contain no label
path, role, hash, count, or content.

## Constraints

- Preserve nominal IDs; never use list positions as identities.
- Do not use exact floating-point coordinate differencing or clustering as the
  primary deletion method.
- Do not guess orientation or edge assignments.
- Refuse overwrite unless the MCP response verifies an exact idempotent replay.
- Report access truthfully; this skill is MCP-backed, not fully autonomous.
