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

## Manifest lifecycle

Schema version 2 separates intake from scientific readiness:

- `provisional` records hashed inputs and declared conventions while preserving
  ambiguous fields in `unresolved_fields`; autonomous registration may omit the
  future aligned graph and its hash.
- `ready_for_data_prep` has no unresolved intake fields and is the deterministic
  hand-off for Otsu, registration, local recentering, and QA.
- `analysis_ready` requires the aligned graph, topology agreement, segmentation
  pass, registration pass, local recentering, and ROI/metrology gates.

Challenge mode requires the scientist-supplied aligned JSON at intake.
Autonomous mode does not allow a fabricated aligned-graph artifact in the
provisional contract. Downstream ROI, classification, and reporting code must
call `specimen_manifest.require_analysis_ready` before reading scientific
fields.

## Deterministic intake core

`scripts/ingest_specimen.py` accepts only explicitly associated CAD, nominal
graph, and CT paths. It constrains them to configured repository data roots,
streams every SHA-256, validates graph IDs/references, inspects STL bounds, and
uses `volume-metadata` for the CT header. It does not accept a threshold and
does not execute segmentation, registration, or defect labeling.

Each run writes:

- `analysis/<specimen_id>/config/ingest_request.json`;
- `analysis/<specimen_id>/config/specimen_manifest.json`;
- `analysis/<specimen_id>/config/ingest_receipt.json`.

The receipt records input and canonical artifact hashes, warnings, unresolved
fields, lifecycle state, method versions, and structured self-verification.
Unchanged inputs and declarations reproduce byte-identical artifacts. A changed
input hash changes the manifest and receipt hashes.

Example autonomous intake:

```bash
python scripts/ingest_specimen.py \
  --specimen-id new_specimen \
  --cad data/new/design.stl \
  --design-graph data/new/design.json \
  --ct data/new/scan.tiff \
  --registration-mode autonomous_v2 \
  --confirm-association \
  --cad-units millimeter \
  --cad-units-provenance "scientist declaration" \
  --array-axes zyx \
  --aligned-graph-units voxel
```

## Agent and data-prep hand-off

The orchestrator invokes `.codex/agents/specimen_ingest.toml` under the
machine-readable contract in `analysis/contracts/specimen_ingest.json`. The
agent may ask for missing declarations and invoke the deterministic intake
scripts, but it is forbidden to inspect labels or run segmentation,
registration, node refinement, or scientific QA. It stops after two failed
correction attempts.

After intake, seal the next-stage envelope:

```bash
python scripts/prepare_data_prep_handoff.py \
  analysis/<specimen_id>/config/specimen_manifest.json \
  analysis/<specimen_id>/config/ingest_receipt.json
```

A ready hand-off allowlists exact input paths/hashes and the registration mode.
A provisional hand-off is an explicit `halt` containing unresolved fields.
Tampered or stale manifests/receipts cannot unlock `data_prep`.

Stage 2 writes a `data-prep-result/1.0.0` envelope containing its aligned graph,
four derived records, and mandatory Otsu/registration/local-node/ROI/metrology
self-verification. The boundary adapter atomically validates and advances the
manifest:

```bash
python scripts/apply_data_prep_result.py \
  analysis/<specimen_id>/config/specimen_manifest.json \
  analysis/<specimen_id>/config/data_prep_result.json
```

The completion receipt is published before the manifest becomes
`analysis_ready`; downstream stages must verify both hashes and call
`require_analysis_ready`.

Validate both committed examples:

```bash
python scripts/validate_specimen_manifests.py
```

Add `--verify-files` to hash locally available inputs. External and regenerable
artifacts may be absent; add `--require-all-files` when a fully restored dataset
is required. Add `--require-analysis-ready` at downstream stage boundaries.

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
