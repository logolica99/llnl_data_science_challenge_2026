# Brief: Tube-Emptiness Strut-Deletion Test (Proof of Concept)

**Task:** Build a proof-of-concept "tube-emptiness" test that detects deliberately-deleted
struts in an octet-truss lattice by comparing design STL meshes against the ideal design
graph. This brief is self-contained — you do not need any external conversation context.

## Context

This is the LLNL 2026 Data Science Challenge (Part 2). We have an ideal lattice design graph
(JSON: 10,206 junctions, 18,468 struts) and four STL meshes where LLNL deliberately removed
0% / 0.1% / 0.5% / 1% of struts. The goal is to recover *which* strut IDs were deleted, from
the STL + JSON alone (no CT scan is involved in this step). This produces the ground-truth
"answer key" that later CT-based defect detection is scored against.

## Inputs (paths relative to repo root)

- **Ideal graph:** `data/missing_struts/octet_truss_9x9x9.json`
  - `junctions`: list of `{id, position:[x,y,z], indices:[i,j,k]}`
  - `struts`: list of `{id, unit_cell_edge_idx, junction0, junction1, thickness}`
  - `unit_cells`: list of `{id, struts:[...], indices}`
  - Coordinates span 0–18 per axis. All 18,468 struts have `thickness = 0.1`.
- **Meshes:** `data/missing_struts/stls/0.stl`, `0.1.stl`, `0.5.stl`, `1.stl`
  - Binary STL, ~167 MB each, units are **millimeters, origin-centered**.
  - `0.stl` is the baseline with **no** deletions (use as negative control).

## Deliverables

Put everything in this directory (`poc/tube_emptiness_test/`) — do not modify files elsewhere:

- `tube_test.py` — single script that runs the whole test.
- `results/deleted_struts_0.json`, `..._0.1.json`, `..._0.5.json`, `..._1.json`
  — each a list of deleted strut IDs plus a count.
- `results/summary.json` — counts table for all four STLs, the final transform (scale/center)
  and radius used, and pass/fail for each gate below.
- `results/deleted_centerlines_0.5.png` — 3D scatter of the deleted strut centerlines
  (visual proof).
- `README.md` — method, transform, how to run, and the results.

## Method

1. Load ideal JSON → `junctions[id] -> (x,y,z)` and `struts[id] -> (junction0, junction1)`.
2. Parse each binary STL **directly with numpy** (no trimesh required). Format:
   80-byte header, then a `uint32` triangle count, then per triangle a record with
   `normal <f4[3]`, `vertices <f4[3][3]`, `attribute <u2`. Reduce triangles to a point
   cloud — start with triangle centroids; fall back to all 3 vertices per triangle if thin
   struts get missed.
3. Transform JSON coords into the STL mm frame: `p_mm = (p_json - 9.0) * scale`.
   The center is 9.0 (midpoint of the 0–18 span). Calibrate `scale` empirically near
   **2.30** (reference value 2.3052 mm/design-unit; a prior method used 2.28).
4. For each strut: sample K points along the centerline A->B; for each sample, query the
   point cloud for any STL point within radius `r`. If **no** sample along the whole strut
   has material within `r`, the tube is empty -> strut deleted. Sampling the full centerline
   (not just the midpoint) is deliberate: it must catch partial deletions.
5. Use `scipy.cKDTree` for radius queries (~3.5M points, a few seconds).
   **Environment note:** the local `python3.11` has numpy but not scipy. Either run
   `python3.11 -m pip install scipy`, or implement a pure-numpy voxel-grid spatial hash
   instead of cKDTree. scipy is already listed in the project `requirements.txt`.

## Calibration and validation gates

Calibrate `r` and `scale` on `0.stl` (negative control), then the result MUST pass all four:

| Gate | Expected |
|---|---|
| `0.stl` deleted count (negative control) | ~= 0 |
| `0.5.stl` deleted count | 88–96 (nominal 92; a prior KD-tree method got 93) |
| Monotonic across designs | `0 < 0.1 < 0.5 < 1` |
| Triangle-deficit cross-check | (baseline tri count − variant tri count) / deleted count ~= 170–180 |

## Independent validation reference (already computed)

A different method — KD-tree nearest-surface gap at strut midpoints, gap threshold 0.5 mm —
previously found **93 missing struts (0.50%)** on `0.5.stl`
(see `data/missing_struts/analysis/0_5_stl_heatmap/summary.json`). The tube test should land
in the same neighborhood. If it does **and** the triangle-deficit ratio (~175 triangles per
deleted strut) checks out, the POC is proven.

## Constraints

- Do not modify any existing files outside `poc/tube_emptiness_test/`.
- Load one STL at a time (memory: 16 GB machine, meshes are ~167 MB / ~3.5M triangles each).
- Report the final `scale` and `radius` you settled on, plus all four counts, in
  `results/summary.json`.
