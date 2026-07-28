# Verification and release test plan

## Test layers

### Fast unit tests

Run:

```bash
python3.11 -m unittest discover -s poc/ct_registration_v2/tests -v
```

The suite checks:

- exact recovery of scale, rotation, and translation;
- rejection of degenerate similarity inputs;
- acceptance of a separated bimodal histogram;
- rejection of a unimodal histogram;
- absence of any ground-truth path/argument in the fitting executable;
- disjoint fit and holdout candidate indices;
- classification halt when uncertainty exceeds measured radius;
- validator refusal when the completion marker is absent;
- validator refusal after a frozen artifact is tampered with;
- synthetic recovery with noise, missing nodes, and outliers.

### Per-fit self-tests

Every real fit executes the configured 12-case synthetic suite before CT
registration. It then produces machine-readable reports for histogram quality,
multi-start agreement, parameter robustness, image-space validation, and the
downstream tolerance gate.

Expected invariants:

- the fitting CLI exposes CT, design, config, output, and optional prior
  external evidence, but no ground-truth input;
- `ground_truth_used_for_fit` is `false` in transform, manifest, and completion
  marker;
- `FIT_COMPLETE.json` is written only after all CT-only artifacts;
- all paths and SHA-256 hashes in the manifest verify;
- corridor `edge_count` equals 18,468;
- node validation `record_count` equals 10,206;
- fit and holdout indices are disjoint and cover all detected candidates.

### Held-out validation

Only after the CT-only fit is frozen:

```bash
python3.11 poc/ct_registration_v2/validate_against_ground_truth.py
```

The validator must refuse execution if the completion marker is missing,
isolation flags are not exact booleans, or any frozen fit hash differs. It
compares nodes by ID, fits the reference transform for component diagnostics,
and reports error by X/Y/Z quintile.

The held-out result is pass/fail evidence, not permission to tune the frozen
fit. If the method changes after viewing it, use a new output directory and
treat the old reference as development data.

### External generalization

Add independently frozen case records to `external_cases.json`, then run:

```bash
python3.11 poc/ct_registration_v2/run_external_validation.py
```

Before release, include at minimum:

- three unique scan hashes;
- two lattice geometry identifiers;
- different acquisition sessions where possible;
- at least one manually verified landmark set independent of algorithm
  development;
- documented scanner, reconstruction, voxel-spacing, and material variation.

The aggregator exits nonzero until all configured coverage and pass-fraction
requirements are met.

## Failure handling

No failing gate may be converted to a warning in a production classification
run. The allowed responses are:

1. halt and request manual registration/landmark review;
2. diagnose the failing perturbation or spatial region;
3. change the method for a scientifically stated reason;
4. create a new config/version and rerun all layers;
5. acquire independent external cases.

Do not relax a limit solely because the held-out registered JSON failed it.

## Current execution

The commands above were executed on the supplied scan on 2026-07-23. All ten
fast tests passed. The detailed immutable numeric result is summarized in
[RESULTS.md](RESULTS.md) and retained as JSON under `results/`.
