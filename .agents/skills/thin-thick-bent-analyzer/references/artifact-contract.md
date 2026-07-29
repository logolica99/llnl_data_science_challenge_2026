# Artifact contract

## Inputs

| Input | Type | Requirement |
|---|---|---|
| CT volume | `.tif`/`.tiff` | Associated with exactly one specimen |
| Registered graph | `.json` | `junctions` and `struts`; stable strut IDs |
| CT threshold | number | Frozen and provenance-recorded |
| Threshold policy | `.json` | Optional overrides using known schema keys |
| Voxel spacing | number | Optional; authoritative millimeters per voxel |

The registered graph supplies topology, `unit_cell_edge_idx` peer families,
and approximate coordinates. It is not ground truth for the observed
centerline.

## Measurement artifacts

`strut_section_measurements.csv` contains one row per sampled cross-section:
strut/sample IDs, normalized position, equivalent-area radius, tracked CT
center, best-fit-centerline deviation, curvature, confidence, validity, and
exclusion reason. Recovered samples also preserve a bounded
`tracking_recovery_reason`; component-guided recovery must identify the same
26-neighbor material component on both sides of an isolated gap. Its final
fallback samples raw TIFF intensities at fixed 4x transverse resolution with
trilinear interpolation, while the component label limits which material can
contribute to the measured area.

The registered-axis plane pass is a bounded bootstrap only. Final radii are
measured on planes centered on the tracked 3D CT path and perpendicular to its
smoothed local tangent. Per-section rows preserve the sampling-plane center,
local tangent, and `tracking_method`. Best-fit-centerline deviation is computed
from the tracked global CT centers against a robust straight 3D CT line.

Valid exclusion reasons are:

- `near_junction`
- `junction_contaminated`
- `dense_boundary`
- `tracking_lost`
- `low_tracking_confidence`

`strut_summary.csv` contains one row per strut with peer identity, robust radius
summaries, coverage/confidence, centerline RMS/max deviation, curvature RMS,
measurement quality, and junction-safe tracked-tube continuity. Continuity
fields include status, shared-component result, both collar support fractions,
maximum axial gap, corridor radius, and sample count.

## Classification artifacts

`classified_struts.csv` is the primary tabular hand-off. Preserve:

- `classification`
- `is_thin`, `is_thick`, `is_bent`
- confidence and decision status
- `continuous_for_shape_classification`, continuity evidence, and any shape
  exclusion reason
- radius and bend evidence strengths plus any primary-label priority reason
- target and peer radius statistics
- bend statistics
- reason and evidence path

Class-specific `findings_thin.json`, `findings_thick.json`, and
`findings_bent.json` support bounded downstream consumers. `thresholds.json`
and `decision_log.md` make the judgments reproducible.

Every class-specific finding includes `measurement_profile`, containing the
compact per-section radius, tracked center, centerline deviation, curvature,
validity, exclusion reason, confidence, CT threshold, and
`section_measurements_sha256`. This is the preferred graph source for viewers;
it contains no TIFF arrays, crops, masks, or contours.

Thin, thick, and bent flags must all be false unless one CT material component
connects the junction-safe A/B collars and both collar-support fractions pass
the frozen threshold. A nonconnected or unresolved strut remains outside this
specialist's findings so the broken/missing stage can assign its own label.

## Evidence

- `evidence/thin/*_radius.png`
- `evidence/thick/*_radius.png`
- `evidence/bent/*_centerline.png`
- `evidence/evidence_manifest.json`

## Downstream gate

Require `classification_handoff.json` with `status: ready`, scope exactly
`thin`, `thick`, and `bent`, all required artifact hashes, and
`self_verification.all_required_artifacts_exist: true`.
