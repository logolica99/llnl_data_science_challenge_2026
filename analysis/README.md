# Part 2 analysis contract

Part 2 is an agent-based system, but its scientific hand-offs are deterministic
files. The `orchestrator` owns each specimen manifest, the `data_prep` agent
supplies scan-derived values, and downstream agents and tools read the manifest
instead of scraping thresholds, paths, counts, slice indices, radii, or axis
conventions from prose.

## Specimen manifests

Every specimen has a committed
`analysis/<specimen_id>/config/specimen_manifest.json` validated by
`analysis/schema/specimen_manifest.schema.json`. A manifest pins:

- CT, design graph, aligned graph, and CAD paths and SHA-256 hashes;
- CT shape, dtype, byte order, graph/array axes, and voxel-spacing provenance;
- registration mode (`challenge_aligned_json` or `autonomous_v2`);
- coordinate-independent graph counts and topology hashes;
- a per-scan segmentation recipe and its derived diagnostics;
- ROI, metrology, retry budgets, and downstream artifact schema versions.

New CT inputs are inspected through `.agents/skills/volume-metadata`. Its
`volume-metadata/1.0.0` output supplies repository-relative paths, streaming
SHA-256, normalized dtype and byte order, axes, and per-axis spacing provenance
to specimen ingestion. Missing axes or spacing remain `unknown`; they are never
reconstructed from notes or array shape. Intake requires hashing-enabled output
and rejects `sha256: unknown`; `--skip-hash` is only for a non-authoritative
preview and cannot supply a specimen manifest.

Derived records use a common envelope containing `method`, `method_version`,
and provenance with input and canonical analysis-parameter hashes. Changing an
analysis parameter invalidates all derived records until they are recomputed.

Validate both committed examples:

```bash
python scripts/validate_specimen_manifests.py
```

Add `--verify-files` to hash locally available inputs. External and regenerable
artifacts may be absent; add `--require-all-files` when a fully restored dataset
is required.

Replay the supplied scan's exact-histogram Otsu gate:

```bash
python scripts/replay_specimen_segmentation.py \
  analysis/brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json
```

The replay must produce threshold `40054`, foreground
`58,653,410 / 519,119,955`, and the frozen histogram hash. The threshold and
count are results for this scan, never defaults for another scan. Float CT
volumes use the manifest's full-volume affine uint16 histogram encoding and
record the native scale needed to map the threshold back to float units.

## Reproducible runtime

Python 3.12 and every direct runtime dependency are pinned in
`pyproject.toml`; `uv.lock` freezes the transitive dependency graph and artifact
hashes. Create the environment without re-resolving versions:

```bash
uv sync --frozen
```

`requirements.txt` repeats the exact direct pins for tools that only accept a
pip-style requirements file. The locked runtime includes the Part 2 scientific
stack: scipy, trimesh, pandas, pyvista, scikit-image, tifffile, and NumPy.

## Retention policy

| Artifact | Git policy | Reason |
|---|---|---|
| Specimen manifests and schemas | Commit | Auditable contract and provenance |
| Raw challenge inputs already tracked | Commit | Canonical challenge dataset |
| External PacificVis volumes/meshes | Ignore | Multi-gigabyte source data; restore from checksums |
| Segmentation masks and exact histograms | Ignore | Deterministically regenerated from manifest |
| Per-strut profiles and evidence crops | Ignore | Large derived evidence; regenerate by config hash |
| Registration outputs and POC results | Ignore | Regenerable experiment artifacts |
| Compact final tables, reports, and declared presentation assets | Commit deliberately | Reviewable milestone outputs |

Do not put a production value in a note and teach an agent to read it. Update
the manifest, recompute its canonical parameter hash and affected derived
records, validate, and then run the relevant deterministic gate.
