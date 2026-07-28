---
name: ct-registration
description: Prepare one CT scan for strut analysis with exact 65,536-bin Otsu segmentation, declared-mode lattice registration, independent full-resolution node localization, and all-edge registration QA. Use for production Stage 2 in challenge-aligned or CT-only autonomous-v2 mode.
---

# CT Registration

Consume only the immutable Stage 2 handoff and its verified Stage 0 data-prep
handoff. All volume, mask, registration, localization, QA, visualization, and
comparison work belongs to the declared `segmentation-tools` MCP interfaces.
The handoff's hash-bound `requested_analysis_scope`, localization policy, QA
policy, specimen ID, and design ID are authoritative; never accept replacements
from free-form agent arguments.

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
   verify it through bounded `compare_segmentation_masks`, then invoke
   `verify_canonical_segmentation`. Require its closed, atomically persisted
   specimen-scoped evidence to bind the frozen manifest, CT, exact-Otsu report,
   canonical mask, comparison report, normalized request, and every SHA-256.
   The verifier independently replays exact Otsu and `raw >= threshold`; the
   agent must not reproduce either computation locally.
4. Follow exactly one registration mode:

   - `challenge_aligned_json`: validate the authorized Stage 0 aligned graph's
     schema, counts, topology, axes, bounds, and hash. Record the mode in every
     result. This is an authorized shortcut, not a claim of autonomous fit.
   - `autonomous_v2`: expose only CT and nominal topology to fitting. Require
     holdout evidence, 21 deterministic multistarts, synthetic recovery, and
     bounded threshold/EDT/trim robustness. Freeze and hash CT-only fit
     artifacts before any optional aligned-reference validation.

5. Invoke `localize_lattice_nodes` for independent full-resolution local
   recentering of every node. Require deterministic seven-seed convergence on
   the Gaussian-smoothed canonical foreground, robust median support along the
   registered incident-edge directions, and a non-degrading CT-support check.
   Treat the incident-edge score robustly so one missing strut cannot bias its
   junction. Retain a coarse coordinate when CT support does not improve and
   preserve a fallback when support is insufficient; never globally refit.
   Keep primary, stable-coarse, fallback, ambiguous, rejected, and
   boundary-limited classes distinct. Apply only the frozen aggregate
   thresholds, and propagate each node's quality/provenance to incident edges.
   Bounded fallback may pass; fallback, ambiguity, or rejection must never be
   relabeled as a primary match.
6. Invoke `compute_registration_qa` over every node and all 18,468 edges. Keep
   coarse capture, padded-ROI capture, and metrology as separate gates. Emit
   a color-coded slice 380 status overlay and XYZ bias figures. QA must derive
   displacement and repeatability from the hashed localization report; the
   agent must not supply an uncertainty scalar. Under `roi_screening`, passing
   segmentation/localization/image/coarse/padded-ROI gates yields `pass` while
   metrology is explicitly `not_authorized`; direct dimensional outputs remain
   forbidden. Under `direct_metrology`, require passing artifact-backed
   absolute uncertainty; missing or excessive evidence yields `manual_review`.
7. Publish the config, exact histogram/report, canonical mask/comparison,
   persisted segmentation-verification MCP evidence,
   registered and localized graph/reports, QA and figures, data-prep result,
   completion receipt, and analysis-ready manifest replacement.
   Each binds scope, authorization lists, ROI results, metrology status,
   localization counts, reason codes, and report paths/hashes.

## Isolation and replay

Never read defect labels, development labels, sealed labels, Stage 3 metrics,
or ground-truth segmentation. Stage 2/3 handoffs must expose no label path,
role, hash, count, or content. Reject existing output paths unless the tool
confirms exact idempotent replay. MCP responses remain compact: status, gate,
schema, artifact paths/hashes/counts, warnings, and structured failure only;
voxel arrays never enter context.
