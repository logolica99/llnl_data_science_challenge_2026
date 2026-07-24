# Complete CT nonconnected-strut viewer labels

`data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv` contains all
1,217 registered struts that failed the direct CT connectivity test across the
complete graph of 18,468 IDs, 0 through 18,467.

The CSV is a viewer-ready list of CT **nonconnected candidates**. It is not a
list of 1,217 confirmed missing struts.

## Complete-run result

| Measurement | Count |
|---|---:|
| Registered graph struts tested | 18,468 |
| Connected by the CT component rule | 17,251 |
| Not connected by the CT component rule | 1,217 |

## Actions taken

1. Used `our_registered.json` and Brian Tran's TIFF as the fixed inputs.
2. Reused the frozen scan configuration: threshold 40,129 and cylindrical
   corridor radius 8 voxels, fixed and determined from the label-blind calibration.
    - Corridor radius calculation: 
      - Selection of 24 spatially distributed struts with viable endpoints: 14 = calibration half-width (12 voxels) + 2-voxel safety buffer. using deterministic farthest-point sampling of their registered midpoints, from a deterministic edge, then repeatedly chooses the candidate whose midpoint is furthest from the already selected midpoints.
      - For each strut, it created a larger rotated calibration cuboid with a 12-voxel half width.
      - ignores the first and last 20% of the strut length, using the central 60% so junction material does not inflate the measured strut width.
      - On each central local-z slice, it thresholded the CT at 40,129 and found 2D foreground components.
      - For each accepted slice, it measured the component’s furthest radial foreground pixel from the local axis. For each strut, it used the median slice extent. a strut was accepted only if at least 75% of its central slices yielded valid measurements &17 of 24 sampled struts passed
      - 90 percentile median radial extent across the viable struts was 6.841 voxels with manual 0.5 voxel safety added = approx 8 voxels

3. For every graph edge, built a rotated local cuboid with node A on the first
   local-z slice and node B on the last; sampled the TIFF with trilinear
   interpolation.
4. Thresholded each cuboid, restricted material to its cylindrical corridor,
   and independently labeled 26-neighbor foreground components.
5. Marked a strut not connected when no one corridor-local component intersected
   both endpoint windows.
6. Ran the graph in ten checkpointed ID ranges, then verified the summaries
   cover IDs 0 through 18,467 exactly once with no duplicate IDs.
7. Added only those 1,217 not-connected registered IDs to `missing_struts.csv`.
8. Recomputed every CSV row's source-edge midpoint gap, nominal endpoints, and
   STL endpoints using the documented inverse cube rotation; no old coordinate
   fields were copied into new rows.

## inputs

- CT TIFF:
  `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
- Registered graph used by both the test and viewer:
  `poc/ct_registration/test_results/our_registered.json`
- Frozen connectivity configuration:
  `outputs/strut_node_connectivity_test/analysis_config.json`
- Nominal source graph:
  `data/missing_struts/octet_truss_9x9x9.json`
- Source STL:
  `data/missing_struts/stls/0.5.stl`


## Registered-ID to source-STL mapping

The source STL and registered CT graph use different cube-orientation
conventions. The validated source-to-registered rotation about nominal center
`(9, 9, 9)` is:

```text
source (x, y, z) -> registered (18-x, 18-z, 18-y)
```

The rotation is its own inverse. For each registered CT failure, the CSV maps its nominal endpoints back through this rotation, looks up the corresponding source edge, and derives the remaining fields from that source edge.

## CSV field contract
The original 14-column schema is retained for viewer compatibility.

- `strut_id` is the registered CT graph ID consumed by the viewer.
- `midpoint_surface_gap_mm` is the nearest `0.5.stl` triangle-centroid distance
  at the corresponding source edge midpoint.
- `x0_json` through `z1_json` are the corresponding source nominal endpoints.
- `x0_stl_mm` through `z1_stl_mm` are derived with
  `stl_mm = (json_coordinate - 9) * 2.28`.

Thus `strut_id` uses the registered viewer frame, while the remaining columns
document the inverse-rotated source STL edge. No connected strut is included,
and all registered IDs are unique and sorted.

## CT connectivity

Each registered A-B edge is rotated into a cuboid whose local z-axis follows
A to B. The TIFF is sampled with trilinear interpolation and thresholded at
40,129. Foreground is restricted to the frozen 8-voxel-radius cylindrical
corridor. A strut is connected only when one 26-neighbor foreground component
intersects both endpoint windows.

## Run the viewer

Use the same registered graph that produced the connectivity IDs:

```bash
python scripts/overlay_brian_ct_registered_ideal.py \
  --ideal poc/ct_registration/test_results/our_registered.json \
  --missing-struts data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv
```

Do not apply another orientation remapping in the viewer. The CSV `strut_id`
column already contains registered viewer IDs.
