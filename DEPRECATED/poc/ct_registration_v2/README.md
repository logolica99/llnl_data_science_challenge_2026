# Trust-gated CT registration v2

This directory registers an ideal lattice graph into CT voxel coordinates and
refuses downstream defect classification when the registration evidence is not
strong enough. The supplied registered JSON is never accepted by the fitting
program and is read only by a separate, integrity-checked validation program.

The current supplied scan is **not approved for defect classification**. Its
CT-only result passes histogram, synthetic recovery, multi-start, and
independent image-space checks, but fails the robustness and downstream
tolerance gates. See [RESULTS.md](RESULTS.md).

## Programs

- `fit_registration.py`: CT + ideal design only. Computes Otsu, detects CT
  candidates, fits, sweeps perturbations, validates all nodes/struts, writes a
  hash-sealed completion marker, and returns `0` only if every internal gate
  passes.
- `validate_against_ground_truth.py`: evaluation only. It refuses to open the
  registered reference until `FIT_COMPLETE.json` and the hashes of the frozen
  CT-only outputs verify.
- `run_external_validation.py`: aggregates independently fitted and validated
  cases. It requires at least three unique CT scans and two geometries by
  default.
- `registration_core.py`: numerical and validation implementation.
- `config.default.json`: all sampling choices and acceptance limits.
- `config.pacificvis.json`: settings for the 1200³ float32 PacificVis volume.

The detailed algorithms and rationale are in [METHOD.md](METHOD.md). Test
coverage and the release protocol are in [TEST_PLAN.md](TEST_PLAN.md).

## Reproduce

From the repository root:

```bash
python3.11 -m pip install -r poc/ct_registration_v2/requirements.txt
python3.11 -m unittest discover -s poc/ct_registration_v2/tests -v
python3.11 poc/ct_registration_v2/fit_registration.py
```

The supplied scan currently exits with code `2`, intentionally, because an
internal trust gate fails. The fit is still frozen and may be evaluated:

```bash
python3.11 poc/ct_registration_v2/validate_against_ground_truth.py
python3.11 poc/ct_registration_v2/run_external_validation.py
```

Those two commands currently exit with codes `3` and `4`: the held-out
downstream-relative gate fails, and there are not enough external cases.

## Inputs and outputs

Default fit inputs:

- CT: `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- ideal graph: `data/missing_struts/octet_truss_9x9x9.json`

Both TIFF/uint16 and NPY/float CT volumes are supported. Native uint16 inputs
use their exact 65,536-level histogram. Floating-point inputs are scanned in
full, mapped deterministically from the finite source minimum/maximum into
65,536 bins, and counted without voxel sampling. The report records the affine
mapping and converts the selected Otsu threshold back to native intensity
units.

The fitting CLI deliberately has no ground-truth option. Its outputs are under
`results/current/`, including:

- `histogram_report.json` and `exact_histogram_uint16.npy`
- detected, fit, and candidate-holdout arrays
- `multistart_report.json`, `robustness_report.json`
- `image_validation.json` and all-edge corridor occupancies
- `downstream_tolerance.json`
- `fitted_transform.json` and `our_registered.json`
- `fit_manifest.json` and the last-written `FIT_COMPLETE.json`

Only the validator reads:

- `data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json`

It writes `held_out_validation.json` and
`held_out_error_histogram.png`.

## Classification policy

`our_registered.json` is a diagnostic result, not automatic authorization to
classify defects. Classification is allowed only when:

1. every internal gate passes;
2. the registration uncertainty is within the measured strut radius/corridor
   budget; and
3. an external-evidence summary meeting the configured scan/geometry coverage
   is supplied to the fit.

Missing evidence or a failed gate produces a nonzero exit and
`classification_allowed: false`.

## PacificVis blind run

The repository also contains a CT-only run against the physically simulated
PacificVis 8×8×8 five-defect volume. The fit uses
`data/octet_truss_8x8x8/octet_truss_8x8x8.json`; it does not use either
PacificVis OBJ mesh and no registered JSON exists in that dataset directory.

See
[`results/pacificvis_8x8x8_v2run/RUN_SUMMARY.md`](results/pacificvis_8x8x8_v2run/RUN_SUMMARY.md).
The strict end-to-end robustness sweep passes. A nearby downsampled-EDT setting
first produces a visibly incorrect coarse transform, but the standard
full-resolution image refinement rejects that coarse placement and recovers to
within 1.66 voxels P95 of the baseline. External validation is still pending,
so this run remains diagnostic rather than authorization for defect
classification.
