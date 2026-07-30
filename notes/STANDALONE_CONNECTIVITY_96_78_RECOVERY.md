# Recovered standalone strut-connectivity workflow

> **Ported into this branch:** missing/broken decision rules now live in
> `src/llnl_nde/core/defect_analysis.py` (present-slice missing + material-loss
> broken). Historical scripts are retained under
> `research/standalone_connectivity_96_78/` for provenance. Prefer MCP /
> `defect_analysis` over re-running the standalone scripts.
>
> Source branch:
> [codex/recover-standalone-connectivity-96-78](https://github.com/logolica99/llnl_data_science_challenge_2026/blob/codex/recover-standalone-connectivity-96-78/notes/STANDALONE_CONNECTIVITY_96_78_RECOVERY.md)

This branch preserves Claire's standalone Python workflow from immediately
before the connectivity code was reorganized into staged agents, skills, and
MCP tools. It is intentionally a historical/research implementation. It does
not use the later Stage 0-4 contracts.

## Provenance

The committed baseline is Git commit `5637bc643166f099616d21e3aafee2e5e9c53711`
(`Add strut node connectivity baseline`). The recovered
`scripts/test_strut_node_connectivity.py` also includes the exact subsequent
edits preserved in the July 27 Codex session history. Those edits added:

- compact axial-profile output;
- inclusive strut-ID checkpoint ranges;
- optional suppression of full cuboid artifacts and overview rendering;
- per-batch garbage collection;
- an explicit frozen threshold argument.

The standalone postprocessing scripts were present locally but had never been
committed. They are preserved here so the measurement and the historical
96-missing/78-broken result can be inspected together.

## Complete file inventory

### Python implementation files

These are the complete standalone implementation files a teammate should read
when porting the workflow:

- `scripts/test_strut_node_connectivity.py`: rotated-cuboid sampling, batched
  trilinear interpolation, corridor calibration, same-component A-to-B
  connectivity, endpoint/collar measurements, and axial foreground profiles.
- `scripts/run_compact_profile_checkpoints.py`: runs the full 18,468-edge graph
  in bounded 600-strut checkpoints and then starts the material-loss pass.
- `scripts/postprocess_missing_struts.py`: removes the explicitly known
  registered `y=18` truncation from a copied viewer/research candidate list.
- `scripts/append_connectivity_metrics.py`: joins the A/B endpoint rows from
  the generated `connection_metrics.csv` files onto the filtered candidates.
- `scripts/classify_missing_broken_struts.py`: performs the initial
  failed-connectivity missing/broken/review split from saved failed-strut
  cuboids and endpoint measurements.
- `scripts/classify_material_loss_struts.py`: calculates all-strut relative
  foreground-fraction/bite features and preserves the missing labels.
- `scripts/write_true_missing_broken_struts.py`: creates the final viewer CSVs
  containing the minimum smoothed foreground fraction.
- `tests/test_strut_node_connectivity.py`: original deterministic unit tests
  for cuboid sampling and connectivity behavior.

### Required scientific inputs for a complete rerun

The historical 0.5-scan rerun uses these exact files:

- `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
  is the three-dimensional CT volume. It is tracked through Git LFS as a
  1,038,433,319-byte object. A clone must run `git lfs pull` before analysis;
  the small text pointer is not a usable TIFF.
- `poc/ct_registration/test_results/our_registered.json` is the registered
  graph actually consumed by the connectivity runner. It contains strut IDs,
  graph endpoint IDs, and registered CT-space node coordinates.
- `outputs/strut_node_connectivity_test/analysis_config.json` is the historical
  frozen scan-level configuration, including the calibrated corridor radius
  and input identity. The absolute path strings recorded inside it are
  provenance from the original machine; reruns in another clone may require a
  deliberately created compatible configuration rather than blindly reusing
  machine-specific paths.
- `data/missing_struts/analysis/0_5_stl_heatmap/missing_struts_copy_Full.csv`
  preserves the 1,217-row complete nonconnected viewer/geometry list used
  before the specimen-specific boundary filter.
- `requirements.txt` lists the standalone runtime dependencies. The relevant
  packages are NumPy, SciPy, tifffile, and Matplotlib.

The nominal design graph
`data/missing_struts/octet_truss_9x9x9.json` is useful provenance and was used
upstream to create the registered graph and viewer coordinates. The
connectivity runner itself reads `our_registered.json`, so it does not need the
nominal JSON as an additional command-line input.

The fixed CT foreground threshold is `40129` by default. A scan-specific value
can be supplied with `--threshold`, but changing it will no longer reproduce
the historical 96/78 result.

### Generated connectivity intermediates

These files are produced by `test_strut_node_connectivity.py` and are required
by the later classifiers:

- `connection_metrics.csv` in every run/checkpoint directory;
- `connection_summary.json` in every run/checkpoint directory;
- `all_strut_axial_profiles.npz` in each compact-profile checkpoint;
- `strut_<ID>_cuboid.npz` for saved failed cases when cuboid artifacts are not
  suppressed;
- `analysis_config.json` in the selected output directory;
- optional `connection_overview.png` and the generated output README.

The historical checkpoint convention is
`outputs/node_connectivity/strut_node_connectivity_profiles_<FIRST>_<LAST>/`.
The large generated checkpoint, cuboid, and plot directories are not part of
the recovery commit. They must be regenerated from the TIFF and registered
graph to execute the postprocessors from scratch. The preserved CSVs allow the
historical results and schemas to be inspected without regenerating them.

### Preserved classifier inputs and outputs

All paths below are under
`data/missing_struts/analysis/0_5_stl_heatmap/`:

- `missing_struts_copy_Full.csv`: preserved 1,217-row raw nonconnected input;
- `true_missing_struts.csv`: 241 filtered candidates with endpoint evidence;
- `missing_broken_classification.csv`: initial 241-candidate classification;
- `missing_strut_candidates.csv`: final 96 missing candidates;
- `review_strut_candidates.csv`: saved uncertain/review candidates;
- `all_strut_material_loss_features.csv`: foreground-fraction features for all
  18,468 struts;
- `true_missing_broken_struts.csv`: combined final missing/broken viewer rows;
- `broken_strut_candidates.csv`: final 78 broken/material-loss rows.

`broken_strut_candidates_all.csv` is an optional broad research export of
material-loss evidence across all struts. It is not required to understand or
reproduce the final 78-row direct-nonconnected broken list and is not included
in the focused recovery commit.

## Connectivity measurement

For each graph edge with endpoints A and B, the runner constructs an
orthonormal local coordinate system whose local z-axis is the normalized A-to-B
direction. It samples a rotated cuboid with trilinear interpolation. Sampling
coordinates have the form

```text
p = c + x*u + y*v + z*w
```

where `c` is the cuboid center, `u` and `v` span the transverse plane, and `w`
is the A-to-B unit vector. Foreground is restricted to the calibrated
cylindrical corridor. A strut is connected only when one identical 26-neighbor
foreground component intersects both endpoint windows. Neighboring material
that belongs to another component cannot satisfy this test.

Multiple rotated cuboids are concatenated into each SciPy interpolation call.
Connected-component labeling remains independent per strut. This preserves the
decision logic while avoiding one interpolation setup per graph edge.

Important outputs are:

- `connection_metrics.csv`: endpoint and same-component measurements;
- `connection_summary.json`: configuration, provenance, and run summary;
- `all_strut_axial_profiles.npz`: compact A-to-B foreground-fraction profiles;
- optional failed-strut cuboid NPZ files and `connection_overview.png`.

## Historical postprocessing sequence

The original 0.5 scan workflow used these scripts in order:

1. `scripts/test_strut_node_connectivity.py`
   computes batched connectivity, endpoint metrics, and axial profiles.
2. `scripts/postprocess_missing_struts.py`
   copies the complete nonconnected list and excludes the known intentional
   registered `y=18` edge truncation for viewer/research analysis. It does not
   modify connectivity.
3. `scripts/append_connectivity_metrics.py`
   adds A/B collar fractions and shared-component voxel counts to the filtered
   candidates.
4. `scripts/classify_missing_broken_struts.py`
   creates the initial failed-connectivity split. A missing candidate has
   material-bearing slices in no more than 10% of the central 20%-80% span,
   where a slice is material-bearing at foreground fraction at least 0.05.
5. `scripts/classify_material_loss_struts.py`
   computes the final foreground-fraction/bite features from all compact
   profiles. For the central 20%-80% span, it calculates the strut's own P90
   foreground reference. A slice is deficient when its foreground fraction is
   below `0.50 * P90`. Material-loss evidence requires either at least 15% of
   central slices to be deficient or a run of at least three deficient slices.
   Endpoint support additionally requires a minimum collar fraction of 0.05
   and at least 500 shared-component voxels at each end. Existing missing IDs
   retain the missing label. The historical final broken export retains the
   bite/material-loss cases from the filtered direct-nonconnected candidate
   set.
6. `scripts/write_true_missing_broken_struts.py`
   writes the viewer-oriented combined and broken CSVs with
   `minimum_foreground_fraction`, using the minimum three-slice-smoothed central
   foreground fraction.

`scripts/run_compact_profile_checkpoints.py` is the bounded full-graph driver
used to create 600-strut checkpoints before material-loss classification.

## Preserved reference results

- `missing_struts_copy_Full.csv`: 1,217 raw nonconnected detections from the
  historical full scan.
- `true_missing_struts.csv`: 241 candidates after excluding the known edge
  truncation and appending the endpoint evidence used by classification.
- `missing_strut_candidates.csv`: 96 missing candidates.
- `broken_strut_candidates.csv`: 78 final broken/material-loss candidates.
- `true_missing_broken_struts.csv`: 174 combined missing and broken candidates.
- `all_strut_material_loss_features.csv`: features for all 18,468 struts.

The intermediate `missing_broken_classification.csv` contains an earlier
failed-connectivity split (96 missing, 79 broken, and 66 review). The final
78-row broken CSV comes from the later all-profile foreground-fraction pass and
the retained direct-nonconnected candidate set. That distinction should be
preserved when reintegrating the code.

## Scope warning

The known `y=18` removal is postprocessing for this particular cropped scan,
not part of the general connectivity decision. A reusable implementation should
always preserve the complete nonconnected result and make specimen-specific
viewer filtering a separate, explicit operation.
