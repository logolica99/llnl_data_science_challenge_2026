# Brief: Autonomous CT Registration — Produce Our Own Registered JSON (Proof of Concept)

**Task:** Align the ideal octet-lattice design graph to the actual X-ray CT scan **from
scratch**, using CT node-peak detection + point-set fitting — then produce a registered
lattice JSON and validate it against the held-out ground-truth registration. This brief is
self-contained; you do not need any external conversation context.

## Context

LLNL 2026 Data Science Challenge, Part 2. The challenge README states registration is "a
problem by itself" and provides a pre-aligned JSON so participants *can* skip it. This is the
**optional stretch** attempt to solve registration ourselves: derive the design→CT alignment
purely from the CT intensities, without peeking at the provided registered file.

**Hard rule:** the provided registered JSON is **ground truth for validation ONLY**. It must
never be read as an input to detection or fitting. Reading it during fitting defeats the
entire purpose (proving we can produce it ourselves).

> **What "ground truth" means here:** it is the ground truth for **node positions** (the
> correct place each design node lands in CT space) — NOT a list of defects. This file
> contains all 18,468 struts fully present; it does not encode missing/thin/broken struts and
> is not a defect label set. It answers "did we register to the right place?", not "which
> struts are missing?". Missing-strut ground truth is a separate artifact (the deleted-strut
> ID list derived from the STL diff, e.g. `poc/tube_emptiness_test/results/deleted_struts_0.5.json`)
> and belongs to the detection task, not this registration task.

## Inputs (paths relative to repo root)

- **CT volume:** `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`
  - Binary TIFF, ~990 MiB, **761 pages × 815 rows × 837 cols**, dtype **uint16 big-endian**,
    ImageJ-written, series axes **ZYX**. No voxel-size metadata.
  - Segmentation threshold is frozen at the Part 2 exact-histogram Otsu value **40049**
    (>= 40049 is metal; expected foreground **58,675,274 voxels**). Axis mapping:
    design `[x,y,z]` samples the array as `vol[z, y, x]`.
- **Ideal design graph:** `data/missing_struts/octet_truss_9x9x9.json`
  - `junctions`: `{id, position:[x,y,z], indices}` — **10,206 nodes**, coords span 0–18 per axis.
  - `struts`: `{id, junction0, junction1, thickness}` — 18,468 edges.
- **GROUND TRUTH (validation only — DO NOT use during fitting):**
  `data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json`
  - Same 10,206 nodes by ID, but `position` is in **CT voxel coordinates**:
    X 58.76–773.74, Y 48.57–764.94, Z 24.50–737.85.

## Deliverables — all in `poc/ct_registration/`

- `register_ct.py` — single script.
- `results_otsu40049/detected_nodes.npy` — detected CT node positions (N×3, x,y,z voxel coords).
- `results_otsu40049/fitted_transform.json` — `{scale, rotation_matrix (3×3), translation (3,), rotation_deg}`.
- `results_otsu40049/our_registered.json` — all 10,206 design nodes mapped into CT space by OUR transform
  (same schema as the provided registered JSON).
- `results_otsu40049/validation.json` — error stats vs ground truth + transform comparison (see gates).
- `results_otsu40049/registration_error_hist.png` — per-node error histogram.
- `README.md` — method, fitted transform, validation results.

## Method

### 1. Load + segment the CT
- Read with `tifffile` (memmap if possible). **Do not cast the whole volume to float** (would
  be 2–4 GB). Threshold at 40049 → boolean mask (Z,Y,X).

### 2. Detect node candidates in the CT
Octet junctions are where up to 12 struts converge → locally **thicker** material than the
thin struts. Exploit that:
- Compute the Euclidean distance transform of the mask (`scipy.ndimage.distance_transform_edt`).
  Nodes sit at **local maxima** of the EDT (the thickest points); struts have small EDT.
- Extract peaks: threshold the EDT above a node-scale radius, `scipy.ndimage.label` the
  result, take `center_of_mass` of each component → candidate node centroids (x,y,z voxels).
- Downsampling the volume (factor 2–4) for this detection is fine and recommended for speed
  and memory; scale the detected coordinates back up.
- Expect **thousands** of candidates — not exactly 10,206 (some nodes merge, some are missed).
  Outliers and missing nodes are expected; the fitting step must be robust to them.

### 3. Coarse initialization
The part is nearly axis-aligned (true rotation ~0.3°), so a coarse init is easy and reliable:
- `scale_init` ≈ (detected coordinate span) / (design span = 18) ≈ ~39.5 vox/design-unit.
- `translation_init` ≈ detected centroid − scale_init × design centroid.
- `rotation_init` ≈ identity.

### 4. Point-set fitting (correspondence-free): trimmed ICP with a 7-DOF similarity transform
Correspondences between design nodes and detected CT nodes are unknown, so iterate:
- Transform all design nodes by the current estimate.
- For each transformed design node, find its nearest detected CT node (`scipy.spatial.cKDTree`).
- **Trim** the worst X% of pairs (e.g. keep the best 80%) to reject outliers / missing nodes.
- Solve the best **similarity** transform (uniform scale + rotation + translation) for the kept
  pairs via **Umeyama / Kabsch-with-scale**.
- Repeat until the mean residual stops improving.

### 5. Produce `our_registered.json`
Apply the final fitted transform to **all 10,206** design nodes → CT-space positions. Write a
JSON with the same structure as the provided registered file but with OUR positions. (Struts
and unit_cells copy through unchanged — only node positions are transformed.)

### 6. Validate against the held-out ground truth
Only now open the provided registered JSON. Node IDs correspond directly, so:
- Per-node error (voxels) = `our_position - provided_position`; report median / mean / p95.
- Compare fitted transform to the known-good values (from an exact fit of the provided file):
  **scale ≈ 39.4888 vox/unit, rotation ≈ 0.335°, translation ≈ (59.34, 52.18, 26.46)**.
- Sample the CT mask at OUR node positions; report the 5×5×5 foreground hit-rate.

## Validation gates

| Gate | Target |
|---|---|
| Median per-node error (ours vs provided) | < ~5 voxels (ideally < 2) |
| Fitted scale | within ~1% of 39.4888 vox/unit |
| Fitted rotation | within ~0.2° of 0.335° |
| Fitted translation | within a few voxels of (59.34, 52.18, 26.46) |
| CT-node foreground hit-rate (5×5×5) | >= ~85% |

Passing these proves we recovered the alignment **from the CT alone**, matching the supplied
registration we were never allowed to look at during fitting.

## Environment

- `python3.11` has **numpy + scipy** already.
- **Install `tifffile`** to read the CT TIFF: `python3.11 -m pip install tifffile`.
- `scikit-image` is optional — `scipy.ndimage` (EDT, label, center_of_mass, maximum_filter)
  covers everything needed.

## Constraints

- **Never read the provided registered JSON except in the final validation step (§6).**
- Do not modify any files outside `poc/ct_registration/`.
- Memory: memmap the TIFF, keep the mask boolean, downsample for detection; target machine is 16 GB.
- Quote the space-containing TIFF path.
- Report the final transform, node counts, and all error stats in
  `results_otsu40049/validation.json`.
