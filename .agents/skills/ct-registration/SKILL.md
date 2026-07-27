---
name: ct-registration
description: Prepare one CT scan for strut analysis with exact 65,536-bin Otsu segmentation, declared-mode lattice registration, independent full-resolution node localization, and all-edge registration QA. Use for production Stage 2 in challenge-aligned or CT-only autonomous-v2 mode.
---

# CT Registration

Consume only the immutable Stage 2 handoff and its verified Stage 0 data-prep
handoff. All volume, mask, registration, localization, QA, visualization, and
comparison work belongs to the declared `segmentation-tools` MCP interfaces.

## Workflow

1. Preflight every tool in [references/tool-contract.md](references/tool-contract.md)
   at `part2-mcp-response/1.0.0`. Missing or unavailable dependencies mean stop with a structured
   `halt`; never use a CLI, script, direct import, or local implementation.
2. Invoke `volume_info`, then `replay_exact_otsu`. Require a native uint16
   65,536-bin histogram for this scan, persisted method/recipe/input/config
   hashes, and plausible histogram gates. The reference replay is threshold
   40054 and 58,653,410 foreground voxels. Do not tune toward that fraction,
   labels, or ground truth.
3. Invoke `segment_ct_dataset` once at the accepted threshold to publish the
   canonical uint8 ZYX mask. Pin path, role, dtype, shape, retention, and hash;
   verify it through bounded `compare_segmentation_masks`.
4. Follow exactly one registration mode:

   - `challenge_aligned_json`: validate the authorized Stage 0 aligned graph's
     schema, counts, topology, axes, bounds, and hash. Record the mode in every
     result. This is an authorized shortcut, not a claim of autonomous fit.
   - `autonomous_v2`: expose only CT and nominal topology to fitting. Require
     holdout evidence, 21 deterministic multistarts, synthetic recovery, and
     bounded threshold/EDT/trim robustness. Freeze and hash CT-only fit
     artifacts before any optional aligned-reference validation.

5. Invoke `localize_lattice_nodes` for independent full-resolution local
   recentering of every node. Retain each accepted position and each coarse
   fallback with ambiguity evidence; never globally refit afterward.
6. Invoke `compute_registration_qa` over every node and all 18,468 edges. Keep
   coarse capture, padded-ROI capture, and metrology as separate gates. Emit
   slice 380 and XYZ bias figures. If coarse and padded ROI pass but metrology
   fails, return `manual_review` requiring explicit ROI-only authorization.
7. Publish the config, exact histogram/report, canonical mask/comparison,
   registered and localized graph/reports, QA and figures, data-prep result,
   completion receipt, and analysis-ready manifest replacement.

## Isolation and replay

Never read defect labels, development labels, sealed labels, Stage 3 metrics,
or ground-truth segmentation. Stage 2/3 handoffs must expose no label path,
role, hash, count, or content. Reject existing output paths unless the tool
confirms exact idempotent replay. MCP responses remain compact: status, gate,
schema, artifact paths/hashes/counts, warnings, and structured failure only;
voxel arrays never enter context.
