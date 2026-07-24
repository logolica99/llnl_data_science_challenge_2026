# Supplied-scan verification result

Run date: 2026-07-23

Verdict: **HALT — not trustworthy enough for defect classification.**

The CT-only fit was completed and frozen before the supplied registered JSON
was read. The fit used SHA-256 CT
`1dea75b7a9882065cc52d4eb137b7d2cdc86d3ad928543e751ae4c811c466b79`
and design
`af7bbd51657735f0a0af07ea8ede2007416c61821b7860936457ab0b76fad6a2`.

## CT-only evidence

| Check | Result | Evidence |
|---|---:|---|
| Exact-histogram Otsu | Pass | threshold 40,054; foreground 58,653,410 / 519,119,955 = 11.30% |
| Histogram quality | Pass | separability 0.814; class separation 4.32σ; two significant modes |
| Synthetic recovery | Pass | 12/12 cases |
| Candidate isolation | Pass | 3,525 detected; 2,820 fit; 705 held out |
| Multi-start | Pass | 21/21 near-optimal starts agree; P95 spread 0.000 voxels |
| Robustness | **Fail** | worst P95 shift 3.056 voxels; allowed 2.000 |
| Image-space validation | Pass | all 10,206 node records and 18,468 corridors checked |
| Downstream tolerance | **Fail** | uncertainty/radius = 3.056/2.000 = 1.528 |

The robustness failure is specifically the 0.75 ICP trim case. Other P95
shifts were:

- threshold perturbations: 0.000–0.912 voxels;
- EDT perturbations: 0.000–0.975 voxels;
- 0.65 and 0.70 trim: 0.600 and 0.000 voxels;
- paired bootstraps: 0.188–0.408 voxels.

The frozen CT-only transform is:

```text
scale       39.1787400584
rotation    0.3980255548 degrees
translation [60.9224224943, 55.6823213287, 29.6567095293]
```

Image evidence was strong but not decisive: mean 5×5×5 junction foreground was
0.923, median corridor occupancy was 0.385, and X/Y/Z corridor-bin median
ranges were 0.039/0.034/0.084. Thick corridor overlap can coexist with a
registration offset, which is why the tolerance gate remains mandatory.

## Post-fit comparison

Only after the fit manifest and frozen-artifact hashes verified, the validator
read the supplied registered JSON (SHA-256
`8e422e1ceece719bdc1d0aa1b6f42273c4dea5bff6ee2ebba76e85ffddba99c7`).

| Metric | Result |
|---|---:|
| Median node error | 3.705 voxels |
| P95 node error | 5.710 voxels |
| Maximum node error | 7.542 voxels |
| Relative scale error | 0.785% |
| Rotation-matrix difference | 0.185° |
| Translation difference | 4.996 voxels |

The absolute held-out component limits passed, but the downstream-relative
gate failed: median error exceeded the measured 2-voxel strut radius, and P95
exceeded radius plus the 2-voxel margin. Error also drifted spatially; median
error rose from 2.56 to 5.14 voxels across X quintiles.

## External evidence

External validation is incomplete: the workspace contains one unique CT scan
and one geometry, while the configured minimum is three scans and two
geometries. The only available case also fails the downstream-relative
held-out gate. Therefore `classification_allowed` remains `false`.

Primary machine-readable records:

- `results/current/fit_manifest.json`
- `results/current/robustness_report.json`
- `results/current/image_validation.json`
- `results/current/downstream_tolerance.json`
- `results/current/held_out_validation.json`
- `results/external_validation_summary.json`

