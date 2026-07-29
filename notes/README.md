# Thin/thick/bent CT strut pipeline

## Overview

This agentic pipeline screens registered CT lattice struts for **thin**,
**thick**, and **bent** defects. It accepts one CT TIFF, one registered lattice
JSON, a frozen CT threshold, and an output directory.

The registered JSON supplies strut IDs, topology, approximate coordinates, and
`unit_cell_edge_idx` design families. Its coordinates are only a spatial prior;
the final centerline and radius measurements come from CT material.

The agent controls the workflow while deterministic MCP tools perform the
heavy computation:

1. `compute_strut_metrics` tracks and measures CT struts.
2. `classify_struts` builds peer baselines and applies frozen rules.
3. `render_strut_evidence` creates radius or centerline plots.
4. `classification_handoff.json` seals the results for later agents.

The scope is limited to thin, thick, and bent. Broken, missing, dross, and
teammate-review labels are handled separately.

## CT measurement

Each strut is sampled at 21 positions from 10% to 90% of its registered length.
The default CT settings are:

| Setting | Value |
|---|---:|
| CT threshold for Brian | 40129 |
| Samples per strut | 21 |
| Tracking radius | 6 voxels |
| Cross-section plane | 49 × 49 |
| Plane extent | ±12 voxels |
| Per-section confidence gate | 0.45 |

Tracking has two stages:

1. A continuity-constrained search around the registered edge bootstraps the
   CT path. Candidate components are scored using position, area, circularity,
   and continuity so the tracker does not independently jump to a neighboring
   strut.
2. The bootstrap centers form a smoothed 3D CT centerline. Final cross-sections
   are centered on that path and sampled perpendicular to its local tangent.

One- or two-section gaps are recovered only when a unique, smooth,
area-consistent bridge exists between confident tracked sections. Ambiguous
gaps remain untracked.

Sections near junctions or the dense build-plate/boundary region are excluded.
The junction exclusion distance is:

```text
max(3 × interior radius, 0.15 × strut length)
```

with a maximum of 25% of the length from either endpoint.

The measured radius is the area-equivalent radius of the segmented CT
cross-section:

```text
radius = sqrt(segmented area / π)
```

It is not a fitted circle.

## Metrics

The per-section CSV records:

- radius and segmented area;
- tracked global CT center;
- sampling-plane center and local tangent;
- deviation from the best-fit straight CT centerline;
- curvature;
- tracking confidence and recovery status;
- validity and exclusion reason.

Possible exclusion reasons are `near_junction`, `junction_contaminated`,
`dense_boundary`, `tracking_lost`, and `low_tracking_confidence`.

The per-strut summary includes:

- valid sample count;
- median, minimum, and maximum radius;
- radius coefficient of variation;
- tracking coverage and mean confidence;
- junction and boundary interference fractions;
- median registration offset;
- centerline RMS and maximum deviation;
- RMS curvature and maximum turn angle;
- peer-group ID and measurement quality.

Centerline deviation is measured from the tracked CT centers to their robust
best-fit straight **3D CT line**, not to the registered JSON line. Therefore, a
coherent registration shift or rigid tilt is not considered bending.

## Quality gates

Tracking must satisfy every gate below:

| Gate | Threshold |
|---|---:|
| Valid samples | ≥ 12 |
| Tracking coverage | ≥ 0.75 |
| Mean tracking confidence | ≥ 0.60 |
| Junction contamination | ≤ 0.20 |
| Boundary interference | ≤ 0.15 |

Thin/thick classification additionally requires:

| Gate | Threshold |
|---|---:|
| Interior radius coefficient of variation | ≤ 0.25 |

This separation allows reliable centerline evidence to support a bent finding
even when radius variation is too noisy for thin/thick classification.

## Peer baselines

Equivalent design edges are grouped by `unit_cell_edge_idx`. Only struts
passing all radius-quality gates enter the baseline.

Each peer group must contain at least 20 usable struts. Its baseline uses a
robust median and median absolute deviation:

```text
robust scale = max(1.4826 × MAD, 0.10 × peer median, 0.20 voxels)
```

Initial outliers farther than 4.5 robust scales are removed before the final
peer median and scale are calculated.

For a target strut:

```text
radius ratio = target median radius / peer median radius
robust z = (target median radius - peer median radius) / robust scale
```

## Classification rules

### Thin

A strut is thin only when tracking/radius quality passes, a peer baseline
exists, and both conditions hold:

```text
radius ratio ≤ 0.78
robust z ≤ -3.5
```

### Thick

A strut is thick only when tracking/radius quality passes, a peer baseline
exists, and both conditions hold:

```text
radius ratio ≥ 1.30
robust z ≥ +3.5
```

Thin/thick currently uses the **median valid radius**. A short local neck or
bulge appears in the radius profile but may not change the classification if
the rest of the strut keeps the median normal. The rule intentionally favors
persistent size changes over isolated one-section fluctuations.

### Bent

Bent uses 3D centerline shape, not radius. It requires:

```text
tracking quality passes
AND at least 3 adjacent valid sections have deviation ≥ 0.75 voxels
AND maximum deviation ≥ 1.50 voxels
AND (
    RMS deviation ≥ 0.75 voxels
    OR RMS curvature ≥ 0.15 inverse voxels
)
```

Adjacent support prevents one noisy center estimate from producing a bent
finding.

### Bent priority

Thin, thick, and bent flags are retained independently. If radius and bend
evidence coexist, bent becomes the primary label when:

```text
bend evidence strength ≥ 0.90 × radius evidence strength
```

This is the requested 10% priority margin. The independent flags remain in the
CSV even when the displayed primary classification is `bent`.

### Normal and uncertain

A strut is `normal` only if all required quality and peer gates pass and no
defect rule passes.

A strut is `uncertain` when the pipeline cannot make a high-confidence
decision because of:

- insufficient tracking quality;
- excessive radius variation;
- insufficient peer support.

Uncertain is an intentional abstention, not a defect class.

## Outputs

```text
output/
  metrics/
    measurement_manifest.json
    strut_section_measurements.csv
    strut_summary.csv
  classification/
    classified_struts.csv
    findings_thin.json
    findings_thick.json
    findings_bent.json
    thresholds.json
    peer_baselines.json
    decision_log.md
  evidence/
    thin/*_radius.png
    thick/*_radius.png
    bent/*_centerline.png
    evidence_manifest.json
  pipeline_receipt.json
  classification_handoff.json
```

`classified_struts.csv` is the main downstream table. The class-specific JSON
files also embed the exact sampled radius, tracked center, tangent, deviation,
curvature, validity, confidence, CT threshold, and measurement hash used for
each decision. The standalone viewer can render graphs from this JSON without
recomputing the TIFF; four-view CT crops still load on demand.

## Current Brian result

The latest verified run is under:

```text
data/missing_struts/analysis/thin_thick_bent/
```

| Classification | Count |
|---|---:|
| Normal | 14,966 |
| Uncertain | 3,291 |
| Bent | 202 |
| Thin | 5 |
| Thick | 4 |

These are conservative screening results under the current frozen thresholds,
not experimentally validated ground-truth labels.

## Main files

- `.agents/skills/thin-thick-bent-analyzer/SKILL.md`
- `src/strut_cross_section_viewer.py`
- `src/strut_defect_pipeline.py`
- `src/mcp_server.py`
- `configs/thin_thick_bent_thresholds.json`
- `docs/thin_thick_bent_pipeline.md`
