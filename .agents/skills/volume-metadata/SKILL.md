---
name: volume-metadata
description: Inspect NumPy .npy and TIFF .tif/.tiff CT volumes through the inspect_volume_metadata MCP tool with bounded memory, streaming SHA-256, explicit axes and spacing provenance, and manifest-ready JSON. Use before specimen ingestion, segmentation, registration, or NDE reporting.
---

# Volume Metadata

Inspect explicit CT inputs without modifying them or guessing scientific
metadata. Keep reasoning here; delegate file inspection and hashing to the
deterministic `inspect_volume_metadata` MCP tool.

## Workflow

1. Require an explicit CT path. Never associate a CT with a CAD or graph from
   filename similarity.
2. For authoritative intake, call `inspect_volume_metadata` with:

   - `input_filepath`: the explicit repository path;
   - `header_only: true`;
   - `include_sha256: true`;
   - the requested retention policy.
3. Require `status: ok`, `authoritative: true`, a 64-character SHA-256, and a
   three-dimensional shape before intake.
4. Consume `manifest_fragment.ct_volume` and
   `manifest_fragment.ct_metadata` directly.
5. Use `header_only: false` only when the user explicitly requests exact
   intensity or finite-value statistics.

Header-only mode avoids voxel decoding but still streams the file once for its
authoritative hash.

## CLI fallback

Only when the MCP tool is unavailable, run:

```bash
python .agents/skills/volume-metadata/scripts/extract_metadata.py \
  --header-only <volume.npy-or-tif>
```

For one file, the CLI emits the same authoritative envelope and
`volume-metadata/1.0.0` contract as the MCP tool. Report that fallback was used.

## Constraints

- Accept only real numeric `.npy`, `.tif`, and `.tiff` files.
- Treat inputs as read-only and reject paths outside the repository.
- Preserve unavailable axes and spacing as `unknown`.
- Never infer metadata from prose, array shape, CAD bounds, or another specimen.
- Do not compute or select a segmentation threshold during metadata extraction.
