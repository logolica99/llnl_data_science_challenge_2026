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
exclusion reason.

Valid exclusion reasons are:

- `near_junction`
- `junction_contaminated`
- `dense_boundary`
- `tracking_lost`
- `low_tracking_confidence`

`strut_summary.csv` contains one row per strut with peer identity, robust radius
summaries, coverage/confidence, centerline RMS/max deviation, curvature RMS,
and measurement quality.

## Classification artifacts

`classified_struts.csv` is the primary tabular hand-off. Preserve:

- `classification`
- `is_thin`, `is_thick`, `is_bent`
- confidence and decision status
- radius and bend evidence strengths plus any primary-label priority reason
- target and peer radius statistics
- bend statistics
- reason and evidence path

Class-specific `findings_thin.json`, `findings_thick.json`, and
`findings_bent.json` support bounded downstream consumers. `thresholds.json`
and `decision_log.md` make the judgments reproducible.

## Evidence

- `evidence/thin/*_radius.png`
- `evidence/thick/*_radius.png`
- `evidence/bent/*_centerline.png`
- `evidence/evidence_manifest.json`

## Downstream gate

Require `classification_handoff.json` with `status: ready`, scope exactly
`thin`, `thick`, and `bent`, all required artifact hashes, and
`self_verification.all_required_artifacts_exist: true`.
