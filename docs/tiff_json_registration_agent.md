# 9x9x9 Octet Lattice: JSON-to-TIFF Registration Process

This document records the complete registration investigation for the 9x9x9
octet-lattice data. It explains what was tested, why the nominal JSON did not
initially align to the CT TIFF, how the verified registration was obtained,
and how to reproduce and validate the result.

## Executive summary

The nominal `octet_truss_9x9x9.json` is a lattice design graph in abstract
design coordinates. Its junction positions range from `[0, 0, 0]` to
`[18, 18, 18]`; those are **not** CT voxel coordinates.

The supplied `9x9x9_octet_lattice.tif` is byte-for-byte identical to the
Brian Tran TIFF named `210127_Brian_Tran_strut_lattices_0point5dash1 1
Slices.tif`. Brian's corresponding JSON is already registered to that TIFF.
We verified the TIFF identity with SHA-256, fitted the nominal-to-Brian
position transform using matching junction IDs, and applied it to the nominal
JSON without changing its graph topology.

The final registered output is:

- [Registered 9x9x9 JSON](../outputs/9x9x9_registration/octet_truss_9x9x9_registered.json)
- [Nominal-to-TIFF transform](../outputs/9x9x9_registration/nominal_to_tiff_transform.json)

## Input files and their roles

| File | Role | Coordinate system |
|---|---|---|
| `octet_truss_9x9x9.json` | Nominal lattice graph; all expected struts | Abstract nominal design coordinates |
| `9x9x9_octet_lattice.tif` | Raw X-ray CT intensity volume | TIFF voxel coordinates |
| `210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json` | Nominal graph already registered to the paired CT volume | TIFF voxel coordinates |
| `210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif` | Brian Tran CT intensity volume | TIFF voxel coordinates |
| `0.stl`, `0.1.stl`, `0.5.stl`, `1.stl` | CAD design meshes for missing-strut variants | Separate CAD frame; not directly aligned |

The key result is that the two TIFF files are not merely similar; they have
the same SHA-256 hash:

```text
1DEA75B7A9882065CC52D4EB137B7D2CDC86D3AD928543E751AE4C811C466B79
```

Therefore the Brian registered JSON is a valid registered reference for the
9x9x9 TIFF.

## JSON structure

The nominal and Brian JSON files have the same graph schema:

| Top-level array | Count | Meaning |
|---|---:|---|
| `junctions` | 10,206 | Endpoint/node records |
| `struts` | 18,468 | Graph edges representing expected struts |
| `unit_cells` | 729 | A 9 x 9 x 9 grouping of the graph |

### Junctions

```json
{
  "id": 0,
  "position": [0.0, 0.0, 0.0],
  "indices": [0.0, 0.0, 0.0]
}
```

- `id` is a graph identifier. Struts use it through `junction0` and
  `junction1`; it is not an image index.
- `position` is the 3D geometry location. In the nominal file it is in design
  coordinates. In Brian's registered file it is `[x, y, z]` in CT voxel space.
- `indices` describes the junction's local position within its canonical unit
  cell. Values such as `0`, `0.5`, and `1` identify corners and face centers.
  Registration does not change these values.

The graph stores junction records per unit cell, so different IDs may share a
geometric location at a unit-cell boundary. Treat `id` as a graph key, not as
a globally unique physical location.

### Struts

```json
{
  "id": 0,
  "unit_cell_edge_idx": 1,
  "junction0": 0,
  "junction1": 9,
  "thickness": 0.1
}
```

`junction0` and `junction1` define a nominal strut centerline. That centerline
is what should be inspected in the CT after registration. `unit_cell_edge_idx`
is the strut's canonical edge type inside the octet template. The graph's
`thickness` value is not automatically a CT-voxel or millimeter radius.

### Unit cells

```json
{
  "id": 728,
  "struts": [18432, 18433],
  "indices": [8, 8, 8]
}
```

`unit_cells.indices` is the global cell-grid index `[cell_x, cell_y, cell_z]`.
Each axis ranges from 0 through 8. Registration leaves all unit-cell records
unchanged.

## TIFF voxel frame

The TIFF stack has:

```text
shape: (Z, Y, X) = (761, 815, 837)
dtype: uint16
pages: 761
axes: ZYX
```

The valid voxel frame written in JSON-style coordinates is:

```text
[x, y, z] = [0, 0, 0] through [836, 814, 760]
```

The crucial order conversion is:

```python
# JSON geometry coordinate
x, y, z = junction["position"]

# TIFF / NumPy array access
intensity = volume[z, y, x]
```

Transposing an array from `ZYX` to `XYZ` changes its indexing convention but
does **not** register an unregistered geometry:

```python
volume_xyz = volume_zyx.transpose(2, 1, 0)
```

Registration is a geometric transform; axis reordering is only an array-memory
operation.

## Why direct nominal JSON lookup failed

The nominal JSON's junction positions span only:

```text
[0, 0, 0] through [18, 18, 18]
```

All those coordinates are technically inside the TIFF frame, but they occupy
only a small corner of the CT volume. Being in bounds is not evidence of
registration.

We used `scripts/inspect_tiff_voxel.py --test-json` to sample every nominal
junction at its untransformed coordinate:

| Statistic | Direct nominal JSON positions | Global TIFF sample |
|---|---:|---:|
| Median intensity | 27,989 | 32,429 |
| 90th-percentile intensity | 28,623 | 42,453 |

The nominal positions sampled background-level intensity and were therefore
not registered.

## An initial bounding-box transform was tested and rejected

A rough estimate based on the segmented CT bounds was tested:

```text
x_voxel = 41.2 * x_json + 40.5
y_voxel = 40.5 * y_json + 37.5
z_voxel = 42.0 * z_json + 1.5
```

This makes the nominal graph span approximately the same CT bounding box, but
it does not account for rotation or a precise origin. Its all-junction median
intensity was only 33,354, close to the global median of 32,429. It was
discarded rather than reported as a final registration.

This is an important general lesson: matching bounding boxes is an initial
guess, not a registration result.

## Verified reference transfer

The registered Brian JSON has identical graph IDs and topology to the nominal
JSON. Let a nominal position be a row vector `[x, y, z, 1]`. We fit the affine
relation:

```text
target_xyz = [x, y, z, 1] @ C
```

where `target_xyz` is the Brian JSON's registered TIFF-voxel position and:

```text
C = [
  [ 39.4882854238, -0.2007955790,  0.0327223347],
  [  0.2008854037, 39.4881474118, -0.1092445060],
  [ -0.0321662916,  0.1094095195, 39.4886448262],
  [ 59.3396041862, 52.1828934867, 26.4617319582]
]
```

Equivalently:

```text
x' = 39.4882854*x + 0.2008854*y - 0.0321663*z + 59.3396042
y' = -0.2007956*x + 39.4881474*y + 0.1094095*z + 52.1828935
z' = 0.0327223*x - 0.1092445*y + 39.4886448*z + 26.4617320
```

The 3 x 3 portion is essentially uniform scale plus a small rotation. The
scale is approximately 39.4888 CT voxels per nominal JSON unit, or about
78.98 voxels per two-unit cell pitch.

The fit residual against all 10,206 matching junction records was:

```text
mean error: 1.466e-12 voxels
maximum error: 3.387e-12 voxels
```

This residual is numerical roundoff, so the source nominal graph and Brian
graph differ by this transform alone.

## Final validation

Applying the transform produced a registered JSON whose junction positions
span:

```text
[58.7606, 48.5686, 24.4953] through
[773.7447, 764.9389, 737.8463]
```

All 10,206 positions are inside the TIFF voxel frame. Sampling their nearest
CT voxels produced:

| Statistic | Registered JSON positions | Global TIFF sample |
|---|---:|---:|
| Median intensity | 52,089 | 32,429 |
| 90th-percentile intensity | 56,188 | 42,453 |

The registered points have substantially higher CT intensity than the volume
baseline, consistent with their landing on dense lattice junctions.

The visual validation is:

![Registered JSON versus deliberately shifted control](../outputs/brian_registration_proof/registered_vs_shifted_overlay.png)

The top row overlays registered JSON nodes on CT slices. The lower row applies
an intentionally wrong in-plane shift of `[+38, -31, 0]` voxels. Registered
nodes coincide with bright junctions; shifted nodes do not.

## Reproducible commands

Run these commands from the repository root. Replace paths if the input files
are elsewhere.

### Inspect the TIFF voxel frame or one JSON node

```powershell
python scripts\inspect_tiff_voxel.py `
  --tif "C:\Users\Claire\Downloads\9x9x9_octet_lattice.tif" `
  --json "C:\Users\Claire\Downloads\octet_truss_9x9x9.json" `
  --junction-id 0
```

### Verify a full JSON-to-TIFF registration

```powershell
python scripts\inspect_tiff_voxel.py `
  --tif "C:\Users\Claire\Downloads\9x9x9_octet_lattice.tif" `
  --json "outputs\9x9x9_registration\octet_truss_9x9x9_registered.json" `
  --test-json
```

### Recreate the registered JSON from the verified reference pair

```powershell
python scripts\register_json_with_reference.py `
  --tif "C:\Users\Claire\Downloads\9x9x9_octet_lattice.tif" `
  --nominal-json "C:\Users\Claire\Downloads\octet_truss_9x9x9.json" `
  --reference-tif "C:\Users\Claire\Downloads\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif" `
  --reference-registered-json "C:\Users\Claire\Downloads\210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json" `
  --output-json "outputs\registered.json" `
  --report-json "outputs\registered_report.json"
```

The script refuses to transfer a reference transform unless the target and
reference TIFF SHA-256 hashes match. This prevents accidentally applying a
registration from a different specimen or scan.

## Reusable subagent

The registration subagent configuration is:

- [tiff_json_registration_agent.toml](../.codex/agents/tiff_json_registration_agent.toml)

Its usage guide is:

- [tiff_json_registration_agent.md](tiff_json_registration_agent.md)

The subagent has two operating modes:

1. **Verified-reference mode:** transfer an existing registration only after
   proving the TIFFs are identical. This is the mode used here.
2. **No-reference mode:** estimate a global similarity transform from landmarks
   and CT segmentation, then refine and validate it. A bounding-box transform
   alone must be labeled provisional.

## General registration workflow when no reference exists

For a new TIFF without a byte-identical registered reference, use:

```text
nominal graph
    -> coarse similarity transform (scale, rotation, translation)
    -> local node refinement in CT
    -> strut-to-material validation
    -> registered JSON plus validation report
```

The global transform is:

```text
p_voxel = s * R * p_nominal + t
```

where `s` is scale, `R` is rotation, and `t` is translation. Obtain the initial
estimate from several visible node correspondences, or a cautious bounding-box
estimate. Use least squares if corresponding points are available.

The papers reviewed for this project provide specific guidance:

- *LatticeAnalytics* uses a nominal spatial graph and XCT volume. After coarse
  alignment, it fine-registers each node by searching a local Gaussian-smoothed
  CT window for the peak density; its window width is the nominal strut length.
- *Virtual Inspection of Additively Manufactured Parts* uses coarse CAD-to-CT
  isosurface alignment, scale adjustment, and control-point deformation. It
  recommends correspondence pairs plus least squares for an initial alignment,
  and evaluates fit from multiple angles and slice overlays.

For automated node refinement, retain safeguards: limit the local search
radius, reject implausibly large node motion, preserve graph connectivity, and
evaluate strut centerlines or tubes rather than trusting a single voxel. A
missing or defective strut is expected to disagree with CT material; the
registration objective must therefore be robust to outliers.

## What registration changes and what it does not

Registration changes only:

```text
junction.position
```

It must not alter:

```text
junction.id
junction.indices
strut.id
strut.junction0 / strut.junction1
strut.unit_cell_edge_idx
unit_cell.id / unit_cell.indices / unit_cell.struts
```

The registered graph remains a nominal expectation. It does not declare that
each physical strut is present. The TIFF supplies the as-built evidence needed
to classify struts as present, missing, thin, bent, broken, or disconnected.
