# Method and trust model

## 1. Scope and coordinate convention

The method estimates one 3D similarity transform

```text
ct_xyz = scale * (design_xyz @ rotation_matrix.T) + translation
```

from the ideal graph and CT intensities. It does not estimate non-rigid
deformation. The design contains 10,206 node records, 3,430 unique positions,
and 18,468 struts. Repeated node records are preserved in the output by ID.

Ground truth is outside the fitting trust boundary. `fit_registration.py` has
no ground-truth argument. It freezes the transform and registered graph, hashes
the complete CT-only artifact set, writes `fit_manifest.json`, then writes
`FIT_COMPLETE.json` last. The separate validator verifies those hashes before
its first reference-file read.

## 2. Per-scan threshold

The TIFF is memory-mapped and processed in depth chunks. A 65,536-bin
`uint16` histogram is accumulated with `numpy.bincount`; no sampled or
down-binned histogram is used to compute Otsu. The selected threshold and exact
histogram are persisted.

The threshold is rejected before registration if any diagnostic fails:

- foreground fraction is outside 1%–35%;
- Otsu between-class separability is below 0.45;
- foreground/background class means are separated by less than 0.75 pooled
  standard deviations;
- the smoothed 1,024-bin diagnostic histogram has fewer than two significant
  modes.

Peak prominence is 0.3% of the dominant mode. That setting is diagnostic only;
it does not change the exact-histogram Otsu computation. It retains the two
stable modes in the supplied scan while the unimodal Gaussian test remains
rejected.

## 3. CT node candidates and isolation

The binary mask is downsampled by two, followed by a Euclidean distance
transform. Connected components of local EDT maxima become candidate junction
centroids. A data-derived central Z margin excludes boundary structures; no
reference coordinates are used.

Candidates are deterministically split 80%/20%. The 20% holdout is never
passed to ICP or refinement. Its distance to predicted design junctions is an
independent image-space check.

## 4. Similarity fit and multi-start agreement

The initial scale and translation come from the ideal-design span and detected
foreground bounds. ICP uses nearest CT candidates, retains a configured
fraction of the smallest residuals, and solves the similarity transform by
SVD. Degenerate point sets are rejected.

Twenty-one starts combine three scale multipliers with identity and ±1°
rotations about each axis. All starts within 5% of the best objective form the
near-optimal set. At least three must exist, at least half must converge, and
their P95 prediction spread must be at most one voxel.

The winning transform is refined at full CT resolution by local EDT peaks
inside bounded patches. Refinement is intensity-derived and still has no access
to the registered reference.

## 5. Synthetic recovery

Before real fitting, the design is transformed by known random similarities.
Each case adds Gaussian localization noise, removes 20% of nodes, and adds 25%
outliers. Recovery is checked against the known transform:

- relative scale error ≤ 1%;
- rotation error ≤ 0.5°;
- translation error ≤ 2 voxels;
- the multi-start agreement gate also passes.

At least 90% of 12 deterministic cases must pass. These tests establish
algorithmic recovery under controlled corruption; they do not establish
generalization to another scanner or lattice.

## 6. Robustness sweep

The pipeline repeats registration around the chosen operating point:

- Otsu threshold offsets: −80, 0, +80 intensity units;
- downsampled EDT peak thresholds: 1.75, 2.00, 2.25 voxels;
- ICP trim fractions: 0.65, 0.70, 0.75;
- four paired bootstrap samples containing 80% of the baseline trimmed
  correspondences.

Bootstrap resampling is paired: a design node and its baseline CT
correspondence are sampled together. Independent point-cloud resampling would
destroy correspondence and measure artificial missing-data bias.

For every case, displacement of all design predictions from the baseline is
summarized. The largest P95 displacement must be ≤2 voxels. Each individual
case remains in the report so a failure cannot be hidden by averaging.

The displacement comparison is end to end: every perturbed coarse solution is
passed through the same bounded full-resolution CT refinement as the baseline
before its prediction shift is measured. The report preserves both the
pre-refinement transform and final transform. This distinction matters because
a candidate-setting perturbation may create a poor coarse solution that the
image-space refinement can either reject/recover or fail to recover. Comparing
unfinished coarse transforms would measure an intermediate state rather than
the transform actually delivered downstream.

## 7. Independent image-space validation

Validation uses the frozen CT-only transform:

- all 10,206 node records are checked in 5×5×5 voxel patches;
- the 20% held-out candidate set is compared with predicted unique junctions;
- every one of the 18,468 strut corridors is sampled at nine axial positions,
  radii 0–6 voxels, and eight angles per nonzero radius;
- complete axial gaps and per-edge foreground occupancy are recorded;
- radial foreground probability estimates a conservative strut radius;
- corridor occupancy is partitioned into five bins along X, Y, and Z to expose
  spatial drift.

Acceptance requires mean junction foreground ≥0.85, median corridor occupancy
≥0.08, maximum axis-bin median range ≤0.25, held-out candidate median distance
≤6 voxels, and a positive measured radius.

These measures establish CT support but are not alone sufficient: a slightly
shifted transform can still overlap thick struts.

## 8. Downstream tolerance gate

Registration uncertainty is the maximum of:

- multi-start P95 spread;
- worst robustness-sweep P95 displacement;
- held-out-candidate median distance.

It must be no larger than both the recommended corridor margin and the measured
strut radius times the configured ratio. The default corridor margin is the
larger of two voxels and one measured radius. Failure halts defect
classification.

The post-fit ground-truth validator also requires the held-out P95 node error
to fit within measured radius plus corridor margin and the median to fit within
one measured radius. These validation-only criteria never influence fitting.

## 9. External validation

Each external case must have:

- an independently completed CT-only fit manifest proving reference isolation;
- a subsequent validation result;
- a geometry identifier.

The default release gate requires at least three unique CT hashes, at least two
geometries, and a ≥90% passing-case fraction. Byte-identical duplicate TIFFs do
not count as independent scans.

Only one unique scan/geometry is present in the supplied workspace, so external
generalization is currently unproven by design.

## 10. Known limitations

- A global similarity cannot model CT warping or specimen deformation.
- Otsu assumes useful global intensity separation; multi-material or strongly
  varying illumination may be rejected.
- Candidate detection and radial radius estimates are voxel-resolution
  measurements.
- Reference validation is not an independent manual landmark study unless the
  provenance of that reference is independently verified.
- Gate values are explicit starting acceptance criteria. Changing them requires
  a new config hash and a fresh frozen fit; limits must not be tuned against
  the held-out result and then reported as held out.
