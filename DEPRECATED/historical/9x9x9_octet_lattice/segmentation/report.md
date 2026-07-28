# CT lattice segmentation report

## Input

- Path: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`
- Shape (ZYX): `761 × 815 × 837`
- Data type: `uint16`
- Total voxels: `519,119,955`
- Required review plane: axis-0 slice `380` (present)

The source TIFF was read only. No ground-truth image was inspected or used.

## Final method and parameters

The final mask uses a global Triangle threshold derived from the exact streamed
uint16 intensity histogram. Foreground is `input >= 34,963`; background is below
that threshold. TIFF pages were decoded and thresholded incrementally, and the
binary `uint8` result was written with zlib compression. The estimated uncompressed
output size is below the classic TIFF limit, so BigTIFF was not required.

The choice used quantitative and visual evidence together. At 4× downsampling in
all three axes, 99.979999% of foreground occupied the largest 26-connected
component and only 0.020001% belonged to components smaller than 27 sampled
voxels. On slice 380, the largest 8-connected component held 60.935851% of
foreground. The visual preview retained appreciably more oblique struts than
Otsu without turning broad background texture into foreground.

## Final voxel statistics

- Foreground: `100,399,672` voxels (`19.3403607457%`)
- Background: `418,720,283` voxels (`80.6596392543%`)
- Sum: `519,119,955` voxels (`100%`)

These are segmentation statistics, not accuracy measurements.

## Iteration history

1. **Environment probe — failed.** The default Python lacked `tifffile`. This
   failed command counted as an attempt. Decision: use the project's existing
   `dssi_env`; the input was untouched.
2. **Exact-histogram Otsu — successful, provisional.** Threshold `40,049`;
   foreground `58,675,274` (`11.302835%`); slice-380 foreground `4.946530%`;
   255 slice components, largest-component share `5.850102%`; sampled 3D largest-
   component share `99.929555%`, small-component share `0.070445%`. Visual review
   showed clean background but visibly thinned or missing oblique struts.
3. **Triangle through scikit-image histogram API — failed.** The installed
   `threshold_triangle` did not accept a `hist` argument. The failure counted as
   an attempt. Decision: implement the published Triangle geometry directly on
   the same exact histogram, preserving memory-aware operation.
4. **Exact-histogram Triangle — successful, selected.** Threshold `34,963`;
   foreground `100,399,672` (`19.340361%`); slice-380 foreground `11.115362%`;
   207 slice components, largest-component share `60.935851%`; sampled 3D largest-
   component share `99.979999%`, small-component share `0.020001%`. Connectivity
   and visual strut preservation improved over Otsu while background remained
   largely suppressed.
5. **Exact-histogram Yen — successful command, rejected result.** Threshold
   `65,514`; foreground only `2` voxels (`0.0000003853%`) and no foreground on
   slice 380 or in the downsampled 3D sample. This clearly destroyed the lattice.

## Stopping reason

Converged after 5 bounded optimization attempts. Triangle was the best successful
result among the data-derived criteria tested; Otsu lost visible struts and Yen
collapsed the foreground. Further experimentation was not supported by the
available evidence, so the selected result was materialized and optimization
stopped before the 10-attempt cap (with no run of three consecutive failures).

## Limitations

- No ground truth was used, so accuracy, precision, recall, and boundary error
  cannot be claimed.
- The threshold is global; spatially varying beam-hardening or attenuation can
  make faint struts less complete in some regions.
- Full-resolution 3D connected-component labeling would require substantially
  more working memory; connectivity diagnostics used a 4× sample, supplemented
  by full-resolution slice-380 statistics and visual review.
- The output intentionally contains no topology repair or learned prior; it
  reflects only evidence available from this input volume.

## Artifacts

- Program: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/segment_ct.py`
- Binary mask: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/mask.tif`
- Slice-380 visualization: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/slice_380.png`
- Report: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/report.md`
