---
name: volume-metadata
description: Inspect NumPy .npy and TIFF .tif/.tiff CT volumes with bounded memory, streaming SHA-256, header-only metadata, explicit axes and spacing provenance, and manifest-ready JSON. Use before specimen ingestion, segmentation, registration, or NDE reporting.
---

# Volume Metadata

Inspect CT inputs without modifying them or guessing physical metadata. The
bundled script constrains inputs to the repository, emits repository-relative
paths, and produces a stable fragment for the specimen-manifest builder.

## Workflow

1. Require the scientist to identify each CT file explicitly. Do not associate
   it with a CAD or graph from filename similarity.
2. For intake, inspect the header and calculate the required streaming hash:

   ```bash
   python .agents/skills/volume-metadata/scripts/extract_metadata.py \
     --header-only <volume.npy-or-tif>
   ```

   Header-only mode does not decode voxel values. SHA-256 still reads the file
   once because integrity hashing is required for intake. Add `--skip-hash` only
   for a non-authoritative metadata preview.
3. When exact intensity and finite-value statistics are required, omit
   `--header-only`. NumPy data is memory-mapped; TIFF data is memory-mapped when
   possible and otherwise processed page by page.
4. Consume `manifest_fragment.ct_volume`, `manifest_fragment.ct_metadata`, and
   the top-level spacing provenance directly. Treat a non-3D CT as invalid for
   specimen intake.

## Output Contract

The `volume-metadata/1.0.0` JSON output includes repository-relative path,
SHA-256, format, shape, normalized dtype, explicit byte order, axes, voxel and
byte counts, TIFF spacing with per-axis source fields, and finite/non-finite
counts when statistics are requested.

Axes and spacing unavailable from the file are the string `unknown`. Never
replace them with an assumption derived from prose, array shape, CAD bounds, or
another specimen.

## Constraints

- Accept only real numeric `.npy`, `.tif`, and `.tiff` files.
- Never enable NumPy pickle loading.
- Reject paths outside the configured repository root.
- Treat inputs as read-only.
- Do not infer physical dimensions or voxel spacing.
- Do not compute or select a segmentation threshold during metadata extraction.
