# PacificVis simulated-defect dataset

This directory contains the local working copy of **Physically simulated x-ray
CT data of additively manufactured octet lattice structures with common
defects** (Miao et al., 2022):

- DOI: <https://doi.org/10.6075/J0GM87G6>
- Collection: Lawrence Livermore National Laboratory Open Data Initiative

The downloaded binary dataset is intentionally excluded from Git. Its total
size is approximately 7.1 GB, and
`8x8x8 octet lattice with defects/five_defects_1200_xray_recon.npy` is 6.44
GiB, exceeding GitHub LFS's maximum per-file size.

## Local contents

The local directory contains 15 source files organized into two groups:

- `octet unit cell with defects/`: six `256 x 256 x 256` float32 CT volumes,
  paired with OBJ meshes for the defect-free, bent-strut, broken-strut,
  inflated-strut, missing-strut, and thin-strut cases.
- `8x8x8 octet lattice with defects/`: one `1200 x 1200 x 1200` float32 CT
  volume containing five defects, plus defect-free and defective OBJ meshes.

## Restoring and verifying the data

Download and extract the dataset from the DOI above, then copy the two extracted
dataset directories into this directory. From the repository root, verify the
files with:

```bash
shasum -a 256 -c data/pacificvis/SHA256SUMS
```

The checksum manifest records the exact local files used for this challenge.
