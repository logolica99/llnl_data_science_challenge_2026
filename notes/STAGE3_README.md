# Stage 3 defect analysis: implementation and teammate handoff

## Purpose and current status

Stage 3 converts the immutable, label-blind measurements produced by Stage 2
into per-strut defect findings. It does **not** resample struts for scientific
measurement, change the Stage 1 Otsu threshold, alter registration, or revise
Stage 2 connectivity.

The current development implementation includes:

- a complete missing-strut specialist based on Claire's original
  node-connectivity method;
- a complete broken-strut specialist based on Claire's axial
  foreground-fraction and material-loss method;
- deterministic merging with fixed class precedence;
- hash-bound evidence rendering for every non-present or bent result;
- an independent production verifier;
- schema-identical extension points for the teammate-owned thin and bent
  specialists.

Thin and bent are intentionally marked `deferred` in the default policy.
Therefore, current missing/broken results are suitable for development and
validation, but a partial run returns `manual_review`. It cannot claim a
production Stage 3 pass, unlock Stage 4, or issue a Stage 3 completion receipt.

## Production boundary

The pipeline order is immutable:

```text
Stage 0 specimen ingest
  -> Stage 1 data preparation
  -> Stage 2 strut metrics
  -> Stage 3 defect analysis
  -> Stage 4 report
```

Stage 3 starts only from a verified, attempt-scoped Stage 3 handoff whose
predecessor is a passing Stage 2 receipt. The handoff must bind the exact
artifact hashes and canonical paths for:

- the Stage 2 completion receipt itself;
- `analysis_config.json`;
- `corridor_calibration.json`;
- `per_strut_metrics.csv`;
- `per_strut_profiles.json`;
- `localized_graph.json`;
- the specimen CT volume.

Classification consumes the frozen analysis config, metrics, and profiles;
the receipt and corridor calibration are verified provenance inputs. The
localized graph and CT are provided to `render_strut_evidence` only; they are
not used to recompute Stage 2 measurements or classification features.

Every Stage 3 MCP request includes the canonical Stage 2 completion-receipt
path. The adapter verifies the receipt self-hash, Stage 2 handoff, frozen
control-config and contract bindings, required passing assertions, and all four
Stage 2 output hashes before allowing classification or evidence rendering.

Development labels, evaluation labels, intentional-deletion labels, ground
truth, STL/CAD variants, and aligned substitute graphs are forbidden production
inputs. Intentional versus unintentional absence is not inferred. A nominal
strut with qualifying absent geometry is simply reported as `missing`.

## Agent, skill, and MCP workflow

```text
orchestrator
  -> defect_lead
       -> missing_strut_agent -> classify_struts(analyze_missing)
       -> broken_strut_agent  -> classify_struts(analyze_broken)
       -> thin_strut_agent    -> classify_struts(analyze_thin)
                              -> classify_struts(analyze_bent)
       -> defect_lead         -> classify_struts(merge)
                              -> render_strut_evidence(...) per required strut
       -> classifier_verifier -> classify_struts(verify)
```

All Stage 3 agents use the shared `strut-defect-analyzer` skill. The verified
handoff identity selects its Stage 3 mode; Stage 2 measurement and Stage 3
classification authority must never overlap.

The `segmentation-tools` MCP server owns both Stage 3 tools:

- `classify_struts` supports `analyze_missing`, `analyze_broken`,
  `analyze_thin`, `analyze_bent`, `merge`, and `verify`;
- `render_strut_evidence` creates the local-frame evidence packet for one
  classified strut.

The disabled-by-default `segmentation-tools-research` server separately
exposes `export_stage3_validation_csvs`. It is not a production Stage 3 tool,
contract output, handoff input, or receipt artifact.

Agents orchestrate and interpret the bounded operations. They must not import
the deterministic implementation directly, invoke a scientific CLI as a
fallback, or create substitute artifacts if MCP is unavailable.

## What Stage 2 contributes

Stage 2 already performed the expensive batched cuboid interpolation,
corridor restriction, connected-component measurement, EDT radius
measurement, and curvature measurement. Stage 3 reads these frozen values.

The missing/broken implementation depends principally on:

- `same_material_component_connects_a_to_b`: Claire's authoritative
  connectivity measurement. It is true only if one identical 26-neighbor
  foreground component in the full 20%-padded calibrated cylindrical corridor
  intersects the unchanged nominal endpoint windows at `z=0` and `z=L`;
- `a_collar_foreground_fraction` and `b_collar_foreground_fraction`;
- `endpoint0_to_collar_component_voxel_count_in_corridor` and
  `endpoint1_to_collar_component_voxel_count_in_corridor`, measured separately
  on the full unmasked corridor;
- `both_endpoint_segments_observed`;
- the per-strut `axial_t` coordinate profile;
- the matching per-strut `foreground_fraction` profile.

Junction-masked collar connectivity remains supplementary evidence. It does
not replace `same_material_component_connects_a_to_b` and does not decide the
missing or broken class.

### Stage 2 compatibility note

The audited disconnected-fragment rule requires the two separate
endpoint-to-collar component-size columns listed above. A previously generated
Stage 2 CSV that lacks those columns is not schema-compatible with this Stage 3
implementation. Do not patch or overwrite an immutable old bundle. Produce a
new Stage 2 bundle through `compute_strut_metrics` under a valid fresh pipeline
attempt/run, then let the orchestrator seal its new hashes into the Stage 3
handoff.

## Shared axial features

For each strut, Stage 3 selects the nominal central span:

```text
0.20 <= axial_t <= 0.80
```

Let the selected foreground-fraction samples be
`f[1], ..., f[n]`. The implementation derives:

```text
central_p90 = quantile(f, 0.90)
deficit_cutoff = 0.50 * central_p90
deficient[i] = f[i] < deficit_cutoff
deficit_fraction = count(deficient) / n
longest_deficit_run = longest consecutive run of deficient samples
smoothed[j] = mean(f[j], f[j+1], f[j+2])
smoothed_minimum = min(smoothed)
minimum_collar_fraction = min(A collar fraction, B collar fraction)
```

The output features also retain the raw central minimum, the smoothed minimum
relative to P90, and the mean clipped material-loss severity:

```text
material_loss_fraction = mean(clip(1 - f[i] / central_p90, 0, 1))
```

If `central_p90` is zero, material-loss fraction is defined as `1.0` and the
relative minimum is `null`. These values are recorded as evidence features;
the class triggers are the frozen rules below.

## Missing-strut implementation

A strut is a positive missing finding only when both conditions hold:

1. `same_material_component_connects_a_to_b` is false; and
2. the minimum of the three-sample-smoothed central foreground profile is
   exactly `0.0`.

In formula form:

```text
missing = (not primary_A_to_B_connected) and (smoothed_minimum <= 0.0)
```

The three-sample average prevents a single empty axial sample from being
treated as complete absence without neighboring support. The central
`0.20L`-to-`0.80L` restriction avoids allowing the junction bodies to dominate
the decision. The primary connectivity value remains exactly the unmasked
same-material-component result transferred from Claire's node-connectivity
code.

Every missing positive records:

- `primary_disconnected`;
- `three_slice_smoothed_central_minimum_is_zero`;
- all shared axial features and frozen input hashes;
- `evidence_required: true`.

## Broken-strut implementation

Broken represents a material-loss defect, including a gap or one or more
substantial "bites," without requiring the strut to be completely absent.
Missing has priority, so a positive missing finding is always negative for the
broken specialist.

For a non-missing strut, the material-loss condition is:

```text
material_loss =
    deficit_fraction >= 0.15
    or longest_deficit_run >= 3 samples
```

The endpoint-fragment support condition is:

```text
endpoint_support =
    minimum_collar_fraction >= 0.05
    and endpoint0_to_collar_component_observed
    and endpoint1_to_collar_component_observed
    and min(endpoint0_component_voxels, endpoint1_component_voxels) >= 500
```

The final rule is:

```text
broken = (not missing) and material_loss and endpoint_support
```

This intentionally includes both:

- a disconnected-fragment case, where material remains near both sides but no
  identical primary component reaches from A to B; and
- a connected-bite case, where an identical component still reaches from A to
  B but the axial profile shows qualifying localized material loss.

A primary disconnection that is neither missing nor sufficiently supported by
the broken rule is returned as `review`; it is not forced into a defect class.
This preserves uncertainty rather than confusing sparse/thin geometry with a
confirmed break.

### Borderline material loss and thin-specialist ownership

The broken specialist does not classify every visible reduction in material
as broken:

- a connected bite that satisfies the frozen material-loss and endpoint-support
  rules is `broken`;
- a disconnected, non-missing strut that lacks sufficient broken evidence is
  `review`, with evidence required;
- a connected strut below the broken material-loss cutoff is negative for the
  broken specialist. While thin remains deferred, the merged primary class is
  also `deferred` and the Stage 3 gate remains `manual_review`;
- the teammate-owned thin specialist may later determine that the last case is
  thin. It must not weaken or reinterpret the frozen broken formula.

This preserves Claire's established missing/broken decisions while avoiding a
forced broken label for struts that may instead have a radial thinness defect.

## Merge behavior and precedence

Each specialist emits one finding for every nominal strut using the same
`part2-specialist-findings/1.0.0` schema. A finding disposition is one of
`positive`, `negative`, `review`, or `deferred`, with reasons, scalar features,
evidence requirements, input hashes, policy hash, coverage counts, and
label-access provenance.

The primary class precedence is fixed:

```text
missing > broken > thin > present
```

Bent is a non-competing attribute, so a strut may have a primary class and
also `bent: true`. The lead agent may not silently override a specialist. It
merges the schema-compatible findings deterministically and writes:

- `findings_missing.json`;
- `findings_broken.json`;
- `findings_thin.json`;
- `findings_bent.json`;
- `thresholds.json`;
- `classified_struts.json`;
- `decision_log.md`.

If thin or bent remains deferred, unclaimed primary results remain
`class: deferred`, bent remains `null`, and the merge gate is `manual_review`.

## Evidence rendering

Every missing, broken, thin, or bent positive—and every review result—must have
a canonical evidence packet under:

```text
analysis/<specimen_id>/evidence/strut_<id>/
```

The renderer uses the localized endpoints to orient the display with local
`z` running from node A to node B. It may resample CT intensity for display,
but it cannot recompute the measurements or change the classification. The
packet includes aligned XY/XZ/YZ views, the axial foreground profile, supplied
metrics and classification, applied policy, artifact hashes, and a manifest
stating that measurement and classification were not recomputed.

## Optional validation CSV export

After the Stage 3 missing/broken findings and merged classification exist,
`segmentation-tools-research.export_stage3_validation_csvs` can write
spreadsheet-friendly validation views under:

```text
research/runs/stage3_validation/<specimen_id>/
```

It produces:

- `missing_struts.csv`: every strut whose merged primary class is `missing`;
- `broken_struts.csv`: every strut whose merged primary class is `broken`;
- `missing_struts_viewer_filtered.csv`: the missing rows after removing the
  known nominal crop-plane contacts;
- `stage3_validation_export_manifest.json`: exact input/output hashes, counts,
  filter definition, and the non-authoritative declaration.

For the Brian Tran octet-truss validation data, the frozen viewing filter is:

```text
exclude from the filtered CSV when either nominal endpoint has y = 18.0
```

The filter reads `junctions[].position` from the supplied nominal graph. It
does not use registered CT coordinates and does not change
`findings_missing.json`, `classified_struts.json`, evidence, the verifier, or
the production receipt. The unfiltered missing CSV always preserves the full
Stage 3 result. This utility is only for checking and viewing the known
premature build-plate truncation case. To maintain the production/research
firewall, copy the frozen classifications, missing findings, broken findings,
metrics, and nominal graph under `research/runs/.../inputs/` before invoking
the research MCP.

## Independent verifier and the meaning of production complete

`classifier_verifier` runs last and did not participate in classification. It
requires:

- all four specialist findings to have status `complete`;
- every nominal strut to appear exactly once in every required finding set and
  exactly once in the merged classification;
- the fixed precedence and separate bent attribute to be applied correctly;
- evidence manifests to cover exactly every non-present, bent, or review call;
- the decision log and frozen thresholds to exist;
- policy, metric, profile, finding, evidence, handoff, contract, config, and
  predecessor-receipt hashes to match;
- provenance confirming that no labels were read and no Stage 2 metric was
  recomputed.

Only a passing verifier can produce the production Stage 3 pass needed to
unlock Stage 4.

## Teammate implementation: thin

Thin should follow the established specialist pattern rather than modify the
missing/broken branches. Stage 2 already supplies radial evidence, including
`edt_radius_median_voxels` and the per-strut interior EDT-radius profile.

The teammate should:

1. Define a label-blind radial reference and decision policy. Freeze all
   thresholds and normalization rules in `stage_3_defect_analysis.thin`.
2. Implement the `analyze_thin` branch in
   `src/llnl_nde/core/defect_analysis.py` using only frozen Stage 2 values.
3. Emit one common-schema finding per nominal strut, including the radial
   reference, observed radius statistic, normalized deviation, threshold,
   reasons, and evidence requirement.
4. Add thin-positive, thin-negative, boundary, repeatability, stale-hash, and
   full-coverage tests.
5. Ensure evidence rendering exposes the radial measurements needed to review
   every thin call.
6. Change thin's `implementation_status` to `complete` only after the
   implementation, schema, MCP-client tests, and review policy are complete.

The policy should account for the fact that raw voxel radius may vary with scan
resolution and nominal strut family. Do not invent or tune a threshold from
production defect labels. If normalization by scan-wide or strut-family
statistics is required, define it explicitly and freeze it before
classification.

## Radial reuse for inflated struts

Thin and inflated can share the same radial feature-extraction and reference
machinery: thin examines a sufficiently low radial deviation, while inflated
examines a sufficiently high radial deviation. Their direction and policy
thresholds differ, but both can consume the Stage 2 EDT-radius measurements.

Inflated is **not currently declared** in the Stage 3 design contract, common
findings enum, merge precedence, output roles, or verifier. It must not be
silently added to a production run. Before implementing it, the team must
decide whether inflated is a competing primary class or a separate attribute,
then update and review together:

- `notes/PART2_DESIGN.md`;
- `analysis/contracts/defect_analysis.json`;
- the Stage 3 configuration and output schemas;
- the responsible agent and MCP operation;
- merge precedence or attribute rules;
- evidence requirements, verifier checks, and Stage 4 mappings.

Because the analysis config and stage contracts are hash-frozen, such a
contract change requires a fresh pipeline run namespace/manifest. Do not mutate
an already frozen run.

## Teammate implementation: bent

Bent remains a separate attribute. Stage 2 supplies
`centerline_curvature_rms_voxels` and its curvature profile where available.
The teammate should define and freeze a label-blind curvature policy, implement
`analyze_bent`, emit the common findings schema for every nominal strut, add
curvature evidence to the render packet, and test both `bent: true` and
`bent: false` while preserving every primary class.

No curvature cutoff has been invented in the current implementation. A bent
threshold must be scientifically justified and frozen through the contract
policy before the specialist is marked complete.

## Files that define or support Stage 3

### Agent definitions

- `.codex/agents/defect_lead.toml`: owns Stage 3 sequencing, deterministic
  merge, evidence requests, and handoff to the verifier.
- `.codex/agents/missing_strut_agent.toml`: bounded missing specialist.
- `.codex/agents/broken_strut_agent.toml`: bounded broken/material-loss
  specialist.
- `.codex/agents/thin_strut_agent.toml`: declared owner for teammate thin and
  bent work.
- `.codex/agents/classifier_verifier.toml`: independent final gate.

### Skill and design documentation

- `.agents/skills/strut-defect-analyzer/SKILL.md`: stage-gated Stage 2 and
  Stage 3 operating policy.
- `.agents/skills/strut-defect-analyzer/agents/openai.yaml`: skill metadata and
  required MCP dependency declaration.
- `notes/PART2_DESIGN.md`: records incremental specialist delivery while
  preserving the complete production team requirement.
- `notes/STAGE3_README.md`: this implementation and extension handoff.

### Contracts and schemas

- `analysis/contracts/defect_analysis.json`: authoritative Stage 3 owners,
  inputs, outputs, gates, and forbidden operations.
- `analysis/schema/defect_analysis_input.schema.json`: freezes missing/broken
  policy and requires complete thin/bent implementations outside development
  mode.
- `analysis/schema/specialist_findings.schema.json`: common specialist output.
- `analysis/schema/classified_struts.schema.json`: merged classification and
  provenance contract.

### Deterministic core and MCP boundary

- `src/llnl_nde/core/strut_metrics.py`: preserves the authoritative unmasked
  A-to-B measurement and additionally records each endpoint-to-collar
  component size needed by the original disconnected-fragment rule.
- `src/llnl_nde/core/defect_analysis.py`: shared feature derivation,
  missing/broken analysis, deferred teammate branches, fixed-precedence merge,
  and independent verifier logic.
- `src/llnl_nde/core/evidence.py`: hash-bound local-A-to-B evidence rendering.
- `src/llnl_nde/mcp_tools/defect_analysis_stage3.py`: strict production `classify_struts` and
  `render_strut_evidence` interfaces only.
- `research/mcp_server.py`: disabled-by-default, non-authoritative
  `export_stage3_validation_csvs` research interface.
- `src/llnl_nde/core/registration.py`: places the default Stage 3 policy in the
  frozen analysis config without granting Stage 2 classification authority.
- `src/llnl_nde/orchestration/pipeline.py`: declares exact Stage 3 MCP arguments and
  artifact routes.
- `src/llnl_nde/core/__init__.py`: exposes the deterministic Stage 3 core to its
  MCP server package.

### Tests

- `tests/test_stage3_defect_analysis.py`: verifies the exact missing/broken
  rules, missing precedence, connected-bite handling, deferred-team
  `manual_review`, common schemas, local evidence, stale-handoff rejection,
  disconnected endpoint-fragment evidence, ambiguous-disconnection review,
  y=18 validation filtering through the research MCP, and complete-team
  verifier behavior.
- `tests/test_research_mcp_tools.py`: verifies that the optional CSV exporter
  is registered only on the isolated research surface.
- `tests/test_part2_production_mcp.py`: verifies that the research CSV exporter
  is absent from the production MCP surface and that production tools exactly
  match the declared contracts.
- `tests/test_stage1_stage2_agents.py`: verifies the updated agent/skill and
  cross-stage contract declarations.

At this audit checkpoint, 46 focused Stage 1/2/3, production-MCP,
research-MCP, and evidence/core tests passed. The desktop client must be
restarted after MCP schema/code changes so its active `segmentation-tools`
process advertises the current tool interface.

## Safe teammate completion checklist

Before claiming Stage 3 complete:

- preserve the missing and broken formulas and their frozen policy fields;
- do not replace primary unmasked A-to-B connectivity with junction-masked
  collar connectivity;
- implement thin and bent only through their bounded specialist operations;
- use Stage 2 metrics/profiles without CT measurement recomputation;
- keep one finding per nominal strut, including negative findings;
- keep `missing > broken > thin > present` and bent non-competing;
- render evidence for every required call;
- run schema tests and MCP-client tests, not only direct Python unit tests;
- set `development_mode: false` only when thin and bent are both complete;
- create a fresh manifest if a contract or frozen policy changes;
- require `classifier_verifier` to pass before Stage 4 is unlocked.

Any missing or incompatible agent, skill, MCP server/tool, schema, handoff,
receipt, artifact, or hash is a hard halt. Do not weaken the contract or create
a fallback implementation to make a partial run appear complete.
