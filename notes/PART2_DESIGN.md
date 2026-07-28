# Part 2 Design — Agentic Missing-Strut NDE Pipeline

**Goal:** a multi-agent system that goes from the `data/missing_struts/` dataset (CT TIFF + nominal lattice graph + design STLs) to a per-strut defect table, a traceable NDE report, and a 3D defect visualization, with a closed evaluation loop. Registration has two supported modes: **(A) challenge mode**, which uses the LLNL-supplied aligned JSON exactly as the challenge README permits, and **(B) autonomous mode**, which uses the production registration core as a CT-only coarse ROI locator followed by independent local node recentering. The aligned JSON contains coordinates and nominal topology, not defect labels, so using it is not answer leakage. The earlier registration POC is archived under `DEPRECATED/poc/ct_registration_v2/`. This document is the frozen design: architecture, pipeline stages, MCP tools, subagent contracts, skills, and the eval layer.

This design was produced by reviewing every existing component (MCP server, skills, subagents, notes, evals), **verifying the data itself by running code against it** (Section 1), and then passing the draft through an adversarial design review whose fixes are folded in below.

**Repository execution policy:** project skills own scientific policy and MCP
call sequencing; deterministic volume, mask, skeleton, comparison, metric, and
rendering operations execute through declared MCP tools. If a required server
or compatible tool schema is unavailable, the skill halts and requests MCP
configuration or a client restart. Skills do not fall back to bundled scripts,
direct implementation imports, or improvised local replacements.

---

## 1. Verified data facts (ground truth for this design)

All numbers below were measured directly from the files in this repo (not taken from notes or briefings).

### 1.1 The CT volume

- `data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif`: **761 pages × (815 rows × 837 cols), dtype uint16, big-endian (`>u2`), ImageJ-written, series axes `ZYX`**, ~990 MiB. No voxel-size metadata in the TIFF.
- **It is md5-identical to `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`** — the repo carries the same scan twice, and all Part 1 segmentation work on the latter applies verbatim to the former. **Despite the "octet lattice" folder name, this scan is the 0.5 %-deleted specimen**: the filename (`…0point5dash1…`) and the single provided registered JSON both correspond to `0.5.stl`, so scoring the 0.5 % deletion set against this CT (Stage 5) is coherent — it is not the full/defect-free lattice.
- Full uint16 load ≈ 7 s, ~1.04 GB in RAM. Never cast the whole volume to float (2.1–4.2 GB).
- Edge slices are atypical (slice 0 mean 49,093 and slice 760 mean 44,759 vs slice 380 mean 33,068) — exclude edges from global statistics.

### 1.2 Per-scan exact-histogram Otsu: **40054** for this scan

Registration v2 recomputes Otsu from the complete 65,536-bin uint16 histogram for every scan. On this volume it selects **40054** → foreground **58,653,410 / 519,119,955 voxels (11.2986 %)**. The older Part 1/v1 record selected 40049 → 58,675,274 foreground voxels; the five-intensity-unit difference is operationally negligible, but **40054 is the authoritative v2 value and is never hard-coded for a new scan**. Part 1 selected the lower Triangle threshold 34963 for a visualization-oriented segmentation because Otsu made some struts appear thinner.

- **Frozen rule for Part 2:** recompute exact-histogram Otsu per scan, then persist the threshold, exact histogram, foreground count/fraction, class means, Otsu separability, and significant modes. For this scan, v2 replay must reproduce 40054 and 58,653,410 foreground voxels.
- **Histogram rejection:** v2 halts before fitting if the foreground fraction is implausible, Otsu separability or class separation is weak, or the smoothed diagnostic histogram lacks evidence of multiple modes. This scan passes with separability 0.814, class separation 4.32 pooled standard deviations, and significant modes near 32,288 and 48,992.
- **Why Part 2 overrides Part 1's Triangle pick.** The goals differ. Part 1 wanted faithful struts, so it chose the *lower* Triangle threshold, which renders struts thicker. Part 2 detects *absences*: a lower threshold thickens struts and back-fills genuinely-deleted ones with noise/neighbor bleed, so a deletion still reads as occupied → **false negatives** (missed deletions). A defect-detection threshold must be **blind to how much material "should" be there** — the higher Otsu operating point is what makes a real deletion read as empty. For the same reason we do **not** tune the threshold to hit a target foreground fraction: this specimen is the 0.5 %-deleted part, so fraction-matching would *lower* the threshold to refill the very deletions we are trying to find.
- **Freeze the method, record the value.** These uint16 intensities are uncalibrated reconstruction units, so the exact integer separating titanium and air can drift with acquisition and reconstruction. The config freezes the recipe and acceptance diagnostics, not a universal integer. Stage 2 may compare the resulting ~11.3 % foreground against the design's expected metal fraction as corroboration, never as a threshold-selection knob.
- The mask itself (`mask.tif`, ~495 MB) is **gitignored and absent**: the frozen segmentation is a *recipe*, not a file. Stage 2 regenerates it deterministically.

### 1.3 ⚠ Axis mapping (silent-failure hazard)

Registered-JSON positions are `[x, y, z]`; the numpy array is `(Z, Y, X) = (761, 815, 837)`. **Correct sampling is `vol[round(z), round(y), round(x)]`.** The supplied registration validates this mapping, and v2 independently achieves a **92.31 % mean 5×5×5 foreground fraction** over all 10,206 node records at its per-scan Otsu threshold (§1.7). Every plausible wrong mapping scores far worse (2.1–67.1 %), and `vol[x][y][z]` **goes out of bounds** (x max 773.7 > 761 pages). The briefing's "837×815×761" is X-by-Y-by-Z, i.e. reversed from numpy order. This mapping is pinned in the config and inside registration, QA, and corridor-sampling tools — no agent ever indexes the raw array from prose instructions.

### 1.4 The lattice graphs

- `octet_truss_9x9x9.json`: keys `junctions` / `struts` / `unit_cells` (note: `unit_cells`, not `cells`); **10,206 junctions, 18,468 struts, 729 unit cells**; junction ids 0–10,205 and strut ids 0–18,467 sequential; all junction pairs unique; nominal coordinates span exactly [0, 18] per axis; every strut has `thickness = 0.1` (design units).
- The nominal graph, the v1/v2 CT-derived graphs, and the LLNL-supplied aligned graph preserve the same **10,206 / 18,468 / 729 structure and topology field-by-field**; only junction `position` values change. **Therefore strut labels derived in design space transfer 1:1 to CT space by strut ID.** The supplied graph contains only `junctions`, `struts`, and `unit_cells`; it contains no defect, missing, broken, disconnected, label, or ground-truth fields. In challenge mode it is the canonical production graph. In autonomous mode, Stage 2 writes `analysis/registration/our_registered.json` from v2 and records which mode produced every downstream artifact.

### 1.5 The STLs

- All four are **binary** STL: `0.stl` = 3,514,642 triangles; `0.1.stl` = 3,511,458; `0.5.stl` = 3,498,656; `1.stl` = 3,482,368. Triangle deficits vs `0.stl` (3,184 / 15,986 / 32,274) are **~172–177 triangles per removed strut** against the archived tube-emptiness POC's measured **18 / 93 / 186** removed struts (0.1 / 0.5 / 1 % of 18,468; `DEPRECATED/poc/tube_emptiness_test/results/`) — a highly consistent ratio (176.9 / 171.9 / 173.5) that serves as a built-in consistency check.
- `trimesh` loads one in ~5 s with `process=False`. Units are **mm, origin-centered** (bounds ± ~20.7 mm in X/Z), unlike the 0–18 nominal frame. **The Y extent is 51.04 mm vs 41.49 mm in X/Z — the STL contains ~9.5 mm of extra non-lattice geometry (tabs/plates) along Y.** Bounding-box registration is therefore forbidden; the mm→design transform must be fitted on lattice geometry only.
- Authoritative graph-to-STL centerline scale: the challenge brief declares a
  **4.56 mm unit cell** and the nominal graph encodes two design units per cell,
  so the frozen value is **2.28 mm per design unit**. The 18-unit centerline
  span is therefore 41.04 mm. Dividing the STL's ~41.493 mm material envelope
  by 18 gives the misleading 2.3052 value because that envelope includes the
  finite tube surface beyond both endpoint centerlines. A reproducible
  autocorrelation of baseline-STL centroid projections over the lattice region
  peaks at 2.279–2.280 mm; correlation collapses near 2.3052. Stage 1 therefore
  rejects 2.3052 declarations and records the 2.28 value plus frozen geometry
  tolerances. This scale decision is separate from CT voxel-spacing provenance.

### 1.6 ⚠ Strut diameter ambiguity

JSON `thickness = 0.1` design units ≈ **231 µm**, which matches neither the README's 350 µm (≈ 6.03 voxels) nor the paper-derived 424 µm noted in `notes/CHALLENGE_NOTES.md`. **Never use JSON thickness as a physical diameter.** The pipeline derives the nominal radius empirically (Stage 3 bootstrap, §4) and reports all three candidates.

### 1.7 CT registration v2: robust coarse ROI localization, not metrology

The production registration core, promoted from the archived
`DEPRECATED/poc/ct_registration_v2/fit_registration.py`, recovers a global
7-DOF similarity transform from the **nominal graph plus CT intensities only**.
Its interface has no ground-truth input. It computes per-scan exact-histogram
Otsu; detects factor-two EDT node candidates; withholds 20 % of candidates from
fitting; runs 21 independent rotation/scale starts; performs full-resolution
local EDT refinement; sweeps nearby thresholds, EDT settings, ICP trim
fractions, and paired bootstrap samples; validates all 10,206 node records and
all 18,468 strut corridors in image space; hashes the frozen artifacts; and
writes completion evidence last. Only the separate validator may then open the
supplied aligned JSON.

**V2 CT-derived transform:**

- **Uniform scale 39.178740 voxels/design unit**
- **Rotation 0.398026°**
- **Translation (60.922422, 55.682321, 29.656710)** voxels in (x, y, z)

**Internal evidence:** 12/12 synthetic recovery cases pass despite noise, 20 % missing nodes, and 25 % outliers; all 21 near-optimal starts converge to the same solution; mean 5×5×5 junction foreground is **92.31 %**; median all-edge corridor occupancy is **0.385**; and X/Y/Z corridor-bin median ranges are 0.039/0.034/0.084. Threshold, EDT, and paired-bootstrap perturbations remain within 0.975 voxels. The 0.75 ICP trim case moves predictions by P95 **3.056 voxels**, so v2 correctly fails its stricter **2-voxel metrology/direct-corridor gate**.

**Post-fit held-out evidence:** median node error **3.705 voxels**, mean 3.627, P95 5.710, maximum 7.542; relative scale error 0.785 %; rotation-matrix difference 0.185°; translation difference 4.996 voxels. The registered struts are uniformly **55.846 voxels** long. V2's maximum held-out error (7.542) fits inside its 8-voxel local-search radius, and its worst internal robustness displacement (3.056) leaves substantial capture margin. The maximum error is 13.5 % of one strut length; for longitudinal context, the LatticeAnalytics paper's 20 % padded length would be 11.169 voxels. The paper does not state whether its 20 % is an axis-specific error tolerance, however, so that number is **not** treated as a lateral or metrology bound. Thus:

- **Coarse-capture gate: PASS for this scan.** V2 is sufficient to put expected nodes inside the local recentering window. Padded strut ROIs are accepted only after independently recentered endpoints pass CT image-support and ambiguity checks.
- **Metrology/direct narrow-corridor gate: FAIL for this evidence.** V2 must not classify defects from an unrefined two-voxel corridor or claim precise dimensional accuracy. A `roi_screening` intake may still pass when every ROI/localization gate passes, with metrology recorded as `not_authorized`; `direct_metrology` remains in review.
- **Local refinement requirement:** Stage 2 retains each locally recentered node position independently; it must not collapse those positions back into one global similarity transform before extracting strut ROIs.
- **Challenge default:** because the Part 2 README note explicitly provides the aligned JSON for participants who skip registration, challenge mode uses that graph for maximum defect-detection confidence. Autonomous mode remains an alternative path and a demonstration of agent/tool autonomy.

---

## 2. Current-state review (what exists, what's missing)

| Component | Exists today | Part 2 gap |
|---|---|---|
| **MCP server** (`src/mcp_server.py`) | Production Stage 1/2 tools are implemented: normalized graph loading, design-only CAD orientation and deletion labeling, exact Otsu, slab-wise canonical masks, bounded mask comparison/visualization, declared-mode registration, independent node localization, and separate all-edge QA gates. TIFF/NPY operations share the memory-aware loader and all new interfaces return `part2-mcp-response/1.0.0`. | Stage 3–6 reporting and specialist primitives remain the downstream implementation gap. |
| **Archived registration POCs** (`DEPRECATED/poc/ct_registration/`, `DEPRECATED/poc/ct_registration_v2/`) | V2 is promoted into `part2_core`: per-scan Otsu rejection, holdout, 21 starts, seeded synthetic recovery, bounded threshold/downsample/trim/EDT robustness, CT-only freeze, independent local recentering, and challenge-aligned validation. | Optional post-freeze aligned-reference validation remains control-plane scoped; the autonomous fit never opens it. |
| **Skills** (`.agents/skills/`) | `stl-design-diff`, `ct-registration`, `ct-threshold-optimizer`, `volume-metadata`, and reporting/runbook skills all declare `segmentation-tools`, fail closed when MCP is unavailable, and contain no executable scientific scripts. | Stage 3–6 skills must preserve the same MCP-only boundary. |
| **Subagents** (`.codex/agents/`) | `specimen_ingest`, production `design_diff`, production `data_prep`, and the earlier bounded `segmentation_agent` are checked in with explicit contracts. The production agents use `gpt-5.6-sol` and immutable handoffs. | Stage 3–6 bounded agents remain to be implemented. |
| **Evals** (`evals/`) | One LLM-judge rubric (well-written) + one single-sample result (4/5); `verification.json` checks format only | No objective metrics at all; no repeats/calibration/metadata; single-slice coverage; **the dataset's one exact ground truth (STL diff → intentional deletions) is unused** |
| **Notes** | `notes/CHALLENGE_NOTES.md` is current and good | Root `STUDY_NOTES.md` is a stale truncation that dies mid-heading — a context-poisoning hazard; replace with a pointer |
| **Deps** (`pyproject.toml` + `uv.lock`) | FastMCP and the scientific stack are pinned, including NumPy, SciPy, tifffile, scikit-image, trimesh, pandas, Matplotlib, and PyVista. | Keep the lock synchronized as planned graph/STL/registration tools move behind MCP; no separate skill-script dependency path is permitted. |

### 2.1 Relation to LatticeAnalytics (Miao et al., IEEE TVCG 2025)

The most directly relevant prior art is **LatticeAnalytics** [Miao, Narain, Chheang, Hooten, Seede, Klacansky, Bertsch, Guss, Giera, Bremer — *"LatticeAnalytics: Strut-Level Visualization and Inspection of Additively Manufactured Lattice Structures,"* IEEE TVCG 31(10):9266–9283, Oct 2025, doi:10.1109/TVCG.2025.3593230] — from the same LLNL group behind this challenge. It is a **semi-automated, human-in-the-loop** system, and it solves the *substrate* our pipeline assumes while deliberately leaving the *decision layer* to a human. Reading it tightens our scope rather than changing it:

- **What it already solves — use as design precedent.** (a) *Data management*: OpenVisus multi-resolution volumes on a NAS, Docker deployment, per-strut subvolume queries so the full volume is never loaded — this is our per-corridor / memmap principle (§6 memory row), validated at 91 GB. (b) *Registration*: VR coarse alignment + Gaussian-peak fine node registration. Our autonomous analogue is v2 CT EDT candidates + multi-start trimmed similarity ICP + independent label-blind, topology-supported CT node recentering (§1.7). In challenge mode the README-authorized aligned JSON replaces the coarse step, but local recentering and ROI QA still run. (c) *Per-strut primitive*: rotated-cuboid subvolume (+20 % margin), z-normalized, Otsu-segmented, **ellipse-fit-per-cross-section → Laplacian-smoothed centerline polyline** → curvature/aspect-ratio metrics, plus a Python API for custom metrics. Their published custom-metric example — fraction of the centerline density profile below the Otsu threshold, used to flag broken/missing struts — is essentially our occupancy-profile / max-axial-gap idea.
- **Specimen and tolerance relevance.** The paper's closest large-scale case is also a 9×9×9 octet `AMTruss` with **18,468 struts**. Its table lists a 2900×2900×2600, 91.33 GB volume, whereas the challenge TIFF is 837×815×761 and 1.04 GB, so they must not be treated as the same acquisition resolution. The workflow principle still applies directly: coarse graph alignment only needs to place each node inside a local fine-registration window; the paper sets that window width to one nominal strut length, recenters the node from Gaussian-smoothed CT intensity, and then adds **20 % padding** to the normalized strut cuboid for imaging and registration artifacts. V2's 7.542-voxel maximum held-out error is below its own 8-voxel local-search radius and only 13.5 % of the 55.846-voxel strut length, so it satisfies the coarse-capture role on this scan. The paper does not publish an axis-specific alignment-error tolerance and explicitly calls objective alignment-error evaluation future work; therefore final ROI acceptance comes from local image support, not from treating 20 % as a universal voxel-error bound.
- **What it explicitly does NOT do — this is precisely our Part 2 contribution.** It computes metrics and then a **human expert** reads histograms, selects outlier bins, and visually adjudicates each candidate ("any artifact would bias our metrics toward false positives — which the expert can quickly disregard"). There is **no automated classification with calibrated/justified cutoffs, no blinded or objective evaluation** (their validation is 5 known defects in a simulated lattice + expert interviews — no recall/CI), **no intentional-vs-unintentional attribution** (they never diff the design STLs — our Stage 1), and **no generated NDE report**. Our Stages 4–6 (the defect team + verifier, sealed-recall scoring with Wilson CI, judge triage, recompute-free report) are exactly the human-in-the-loop steps they left manual. The lead author's stated future direction — "autonomous visualization agents to accelerate defect detection" — is the layer we build.
- **What we adopt from it.** The **ellipse-fit centerline → curvature** metric (folded into `compute_strut_metrics`, §3.3), because their RMS-curvature histogram cleanly isolated *bent* struts — a deformation class our original occupancy/gap/EDT/connectivity set could not see (see §5.2; bent is triage-only, no sealed GT). Framing: we position this pipeline as the **autonomous analyst layer on top of a LatticeAnalytics-style substrate** (their pipeline ≈ our Stages 1–3; our contribution is Stages 4–6), not a competitor to it.

---

## 3. Architecture

**Principle: deterministic heavy compute lives in MCP tools; judgment lives in bounded subagents; skills define policy and tool sequencing; all hand-offs are files.** Required MCP dependencies are fail-closed: skills never replace an unavailable tool with a local script, direct import, or ad hoc computation. Agents never pass arrays through context. Control-plane contracts and receipts live under `analysis/<specimen_id>/`; large derived evidence lives at manifest-declared artifact paths (the challenge specimen currently uses `data/missing_struts/analysis/`). The orchestrator gates each stage on the previous stage's verified hand-off.

### 3.1 Agent workflow

[![Part 2 agent workflow: seven bounded agent stages over a required MCP
boundary and verified file hand-offs](assets/part2-agent-workflow.svg)](assets/part2-agent-workflow.svg)

GitHub renders the SVG inline and opens the full-size visual when selected. The
editable source is
[`part2-agent-workflow.excalidraw`](part2-agent-workflow.excalidraw), and a
review-ready screenshot is available as
[`assets/part2-agent-workflow.png`](assets/part2-agent-workflow.png).

#### Agent capability map

[![Part 2 map from agents to policy skills and deterministic MCP
tools](assets/part2-agent-capability-map.svg)](assets/part2-agent-capability-map.svg)

Solid outlines are implemented; dashed outlines are planned. The implemented
MCP layer includes `volume_info`, `load_lattice_graph`, and
`replay_exact_otsu`, merged for Issue #13 in commit `515d88c`. The remaining
dashed MCP groups are the unimplemented Issue #13 and downstream Part 2
contracts. The map was created on **2026-07-24**. Its editable source is
[`part2-agent-capability-map.excalidraw`](part2-agent-capability-map.excalidraw),
and its review screenshot is
[`assets/part2-agent-capability-map.png`](assets/part2-agent-capability-map.png).

**In graph terms:** intentional missing = G_full − G_0.5-design (Stage 1); candidate unintentional defects = G_0.5-design − G_observed-CT after removing the intentional set (Stage 5). The topology identity verified in §1.4 is what makes Stage 1's labels transfer to CT space by strut ID alone.

### 3.2 Subagents and contracts

Every production agent gets an immutable, attempt-scoped hand-off with exact
inputs, a never-touch list, enumerated output artifacts, iteration/failure
limits, and mandatory self-verification. An agent result is not accepted merely
because the agent returned: the orchestrator must build and apply the
attempt-bound completion receipt, rehash every declared artifact, and verify
the predecessor receipt, frozen config, contract, hand-off, and terminal gate.

`specimen_ingest.toml`, `orchestrator.toml`, `design_diff.toml`, and
`data_prep.toml` are implemented production contracts. The archived
`DEPRECATED/agents/segmentation_agent.toml` is historical context and must not
be reused because it references retired script/fallback behavior. The Stage
3–6 agent owners below remain planned even where their deterministic MCP
primitives already exist.

The roster is deliberately lean everywhere **except Stage 4**, the scientific
judgment point. A dedicated `defect_lead` coordinates narrowly scoped class
specialists, but receives the same development-blind shared hand-off as the
thin, broken, and independent verifier roles. Only `missing_strut_agent`
receives the separate development-label hand-off. No Stage 4 participant ever
receives the sealed split. Deterministic calculation remains in MCP tools or
the allowlisted intake/control-plane cores; agents own policy, bounded
judgment, reconciliation, and explicit abstention.

| Agent | Stage | Contract highlights |
|---|---|---|
| `specimen_ingest` | 0 | **Implemented bounded intake agent.** Accepts exactly one closed, self-hashed scientist request binding the specimen/design identity, requested scope, registration mode, declarations, and current CAD/graph/CT hashes, plus an optional authorized aligned graph. It invokes `$volume-metadata` → `inspect_volume_metadata` in authoritative header-only mode and halts if MCP is unavailable; it never substitutes a CLI or direct import. The MCP call writes `ct_metadata_response.json` and a separate `ct_metadata_mcp_call_receipt.json`, then the allowlisted intake and hand-off cores emit `ingest_request.json`, `specimen_manifest.json`, `ingest_receipt.json`, and `data_prep_handoff.json`. The receipt provides a closed integrity/lineage chain, not cryptographic authentication of process identity. The orchestrator semantically reopens all six outputs and anchors them to the scientist request before accepting Stage 0. The agent never reads labels, thresholds, masks, registration outputs, or another specimen manifest; after at most two correction attempts it returns a verified `ready` hand-off or explicit `halt`. |
| `orchestrator` | all | **Implemented control-plane agent.** Sequences stages strictly through the deterministic state CLI and immutable file hand-offs; maintains `analysis/<specimen_id>/manifest.json`; verifies every predecessor receipt, contract/config hash, input/output path, artifact hash, self-verification assertion, and terminal gate before unlocking the next stage. A `manual_review` stops automation and keeps downstream stages locked until an explicit, hashed same-stage resolution is recorded. Standalone or prior-run outputs cannot be borrowed without the exact hand-off and predecessor receipt. Agent/judgment work has at most two attempts; deterministic gate failures halt rather than retry. Owns the final presentation/demo assets but performs no scientific computation. |
| `design_diff` | 1 | **Implemented design-only agent.** Invokes `$stl-design-diff` and the MCP sequence `load_lattice_graph` → `resolve_cad_graph_orientation` → `label_deleted_edges`. Reads only the nominal graph and the 0, 0.1, 0.5, and 1 percent STLs; CT, aligned coordinates, registration, and prior labels are forbidden. Bounding-box registration is forbidden. Requires unique IDs/topology, finite all-edge support, independent 18/93/186 deletion counts, and triangle-deficit support; it does **not** require deletion sets from different specimens to be nested. Symmetric geometry passes only with an intake-hashed transform declaration that independently verifies against the graph and STL; otherwise orientation ambiguity produces `manual_review` and the agent never guesses. Emits the normalized ID map, orientation evidence, label artifacts, deterministic development/sealed split, report, and receipt. |
| `data_prep` | 2 | **Implemented label-blind agent.** Consumes the immutable Stage 2 hand-off, verified Stage 0 `data_prep_handoff.json`, and passing Stage 1 predecessor receipt. Invokes `$ct-registration`, which owns `volume_info` → `replay_exact_otsu` → `segment_ct_dataset` → `compare_segmentation_masks` → `verify_canonical_segmentation` → declared-mode registration → independent node localization → all-node/all-edge QA. `$ct-threshold-optimizer` is optional exploratory support and is not on the production critical path. **Challenge mode:** verifies the authorized aligned graph without claiming autonomous fit. **Autonomous mode:** fits from nominal graph + CT only, freezes/hashes the CT-only fit before optional aligned validation, then independently recenters nodes without a later global refit. It reproduces exact Otsu (40054 / 58,653,410 here), publishes a canonical uint8 ZYX mask, preserves primary/stable-coarse/fallback/ambiguous/rejected/boundary quality, and emits registration/localization/QA artifacts. A hash-bound `roi_screening` request may pass when every ROI gate passes while metrology is explicitly `not_authorized`; `direct_metrology` requires artifact-backed absolute uncertainty or returns `manual_review`. Defect, development, sealed, and ground-truth files are forbidden. |
| `strut_metrics` | 3 | **Planned agent; MCP primitive implemented.** Receives no label artifact; runs the corridor-radius bootstrap and `compute_strut_metrics`, then verifies exactly 18,468 artifact-backed rows. |
| `defect_lead` | 4 | **Planned dedicated defect-analysis agent** coordinating the specialists below. Receives metrics and a development-blind shared hand-off; it never receives development or sealed labels. Adjudicates conflicts under fixed precedence (**missing > broken > thin > present**), merges findings into `classified_struts.json`, generates evidence for every non-present strut, and writes `decision_log.md`. It never overrides a specialist silently: every adjudication is logged. |
| ├ `missing_strut_agent` | 4a | Sole subagent allowed to read `dev_split.json`. Calibrates the missing/present occupancy boundary on the ~28 dev positives; outputs `findings_missing.json` + its own decision log; forbidden to touch thin/broken cutoffs |
| ├ `thin_strut_agent` | 4b | **No labels exist for these classes** — owns **thin** and **bent** (both purely geometric, distribution-derived, no sealed ground truth, triage-only per §5.2). Thin: percentiles of median/min EDT radius vs the Stage 3 empirical nominal radius. Bent: percentile of the centerline-curvature RMS (§2.1) — a present-but-deformed strut, so it never overrides missing/broken. Outputs `findings_thin.json` + `findings_bent.json` with justification; forbidden to read any label file |
| ├ `broken_strut_agent` | 4c | Owns broken **and disconnected-at-joint** (the Tran et al. class): max-axial-gap profile + corridor-local connectivity with junction spheres masked out; distribution-derived cutoffs; outputs `findings_broken.json`; forbidden to read any label file |
| ├ `classifier_verifier` | 4d | **Planned independent process check, run last before one-shot sealed scoring.** Sees only metrics, per-class findings, evidence packets, `decision_log.md`, and `thresholds.json`; development and sealed labels are forbidden, and it did not participate in classification. Verifies evidence support, label-free cutoff provenance, fixed precedence, logged adjudications, and agreement with `classify_struts`. Writes `struts/verifier_report.json` plus self-verification; Stage 5 remains locked until this receipt passes. |
| `eval_agent` | 5 | **Planned sole owner of the sealed split.** Stage 5 is reserved/consumed before disclosure and runs once for the frozen configuration, even if scoring fails or reports low/zero recall. Scores sealed recall, produces the intentional-vs-unintentional attribution table, and runs judge triage; raw sealed labels never flow to Stage 6. |
| `report_agent` | 6 | **Planned report agent.** Consumes committed Stage 5 attribution/scoring artifacts but no raw development or sealed labels. Computes seeded spatial statistics and rendering through declared MCP tools, and compiles a recompute-free report whose numbers come from committed artifacts. The artifact-backed number crosscheck runs before the judge rubric. |

Across all stages, `pass`, `manual_review`, and `halt` are control-plane states,
not prose suggestions. `manual_review` preserves attempt evidence and locks the
current/downstream stages; resumption requires an explicit resolution artifact
and a legal same-stage transition. `halt` is terminal. An exact replay of an
already accepted receipt is an idempotent no-op and must not change artifacts,
attempts, timestamps, events, or downstream state.

### 3.3 MCP server v2 (`src/mcp_server.py` extended)

The `segmentation-tools` server now exposes the production Stage 1/2 tools in
addition to the earlier reporting utilities. These are authoritative MCP
boundaries; agents and skills never import their implementations.

| Current tool | Input support | Contract |
|---|---|---|
| `inspect_volume_metadata` | TIFF/NPY | Authoritative, repository-constrained metadata, streaming SHA-256, axes and spacing provenance, and optional bounded statistics. |
| `volume_info` | TIFF/NPY | Compact structured metadata through the shared memory-mapped loader, including format, axes, spacing provenance, repository path, and optional input SHA-256. |
| `load_lattice_graph` | JSON | Normalizes nodes, edges, and cells to NPZ with explicit ID maps, count/topology warnings, artifact hashes, and a structured pass/manual-review gate. |
| `replay_exact_otsu` | TIFF/NPY | Replays deterministic per-scan histogram/Otsu analysis, writes hashed histogram and report artifacts, and returns explicit pass/halt diagnostics. |
| `resolve_cad_graph_orientation` / `label_deleted_edges` | JSON + binary STL | Resolve scale-preserving design orientation with ambiguity abstention, then label every nominal ID through sequential mesh tube-emptiness evidence. |
| `segment_ct_dataset` | TIFF/NPY | Writes the canonical uint8 ZYX mask in slabs and returns its pinned contract. |
| `visualize_slice` | TIFF/NPY | Writes one bounded 2D slice image. |
| `compare_segmentation_masks` | TIFF/NPY | Validates aligned masks, persists a compact report, and returns foreground statistics. |
| `verify_canonical_segmentation` | TIFF/NPY + JSON | Independently replays the frozen exact-Otsu recipe and mask comparison, then atomically persists closed specimen-scoped, path/SHA-bound verification evidence. |
| `register_lattice_to_ct` / `localize_lattice_nodes` / `compute_registration_qa` | JSON + TIFF/NPY | Execute the declared registration branch, preserve independent local positions, and emit separate coarse-capture, padded-ROI, and metrology gates with figures. |
| `summarize_nde_artifacts` | NPY | Returns report-ready raw/mask/skeleton scalar metrics using bounded slab processing. |
| `render_volume_3d` | NPY | Writes a PNG isosurface with an optional skeleton overlay; refuses implicit overwrite. |
| `skeletonize` | NPY | Writes a skeleton artifact for optional QA and Part 1 reporting. |

Part 2 extends this server rather than adding executable processing scripts to
skills. `segment_ct_dataset`, `visualize_slice`, comparison, summary, and
rendering should share a TIFF/NPY loader where Part 2 needs both formats.
`skeletonize` is demoted from the critical path because corridor sampling
replaces whole-volume skeletonization; retain it only as an MCP-exposed
subvolume QA operation. Every new tool must write large outputs to artifact
paths, return compact structured metadata and hashes, and surface failures
through MCP so the calling skill can halt.

Production Stage 1/2 skills and planned downstream skills:

| Tool | Purpose |
|---|---|
| `resolve_cad_graph_orientation` | Resolve the 2.28 mm/design-unit CAD-to-graph orientation/translation hypotheses needed before tube-emptiness labeling. It may consume an intake-hashed declaration, but independently verifies source/provenance, specimen/design, graph/STL hashes, finite dimensions, orthonormality/handedness, translation, and all-edge correspondence at frozen tolerances. Symmetry without a valid declaration remains review; a bad declaration halts. |
| `register_lattice_to_ct` | Runs the production core promoted from `DEPRECATED/poc/ct_registration_v2/fit_registration.py`: per-scan exact-histogram Otsu + histogram rejection; factor-two EDT candidates; deterministic 80/20 candidate holdout; 21 rotation/scale starts; trimmed similarity ICP over 3,430 unique positions; full-resolution local EDT refinement; threshold/EDT/trim/paired-bootstrap sweeps; all-node/all-corridor image QA; downstream ROI and metrology gates. It writes/hashes all CT-only artifacts before any validation path can be opened. |
| `localize_lattice_nodes` | Starting from either the supplied graph or v2 coarse predictions, independently recenter every unique junction inside a bounded CT window using seven deterministic mean-shift starts over Gaussian-smoothed canonical foreground. Cluster converged starts, score incident edges with a robust median so a missing strut cannot bias its node, accept only non-degrading CT support, and otherwise retain the coarse coordinate or abstain. Emit displacement, repeatability, support, status, and out-of-window evidence without fitting another global similarity. |
| `label_deleted_edges` | Step (a) engine — **tube-emptiness test, not a mesh diff** (§4 Stage 1): for each of the 18,468 design edges, test whether the k% STL has any triangles within a small tube of the scaled centerline; a strut is deleted iff its tube is empty. It revalidates the frozen 2.28 mm/unit orientation artifact before opening answer-bearing variants—no clustering, no exact-float triangle matching, and no label-assisted orientation choice. Triangle-deficit counts (§1.5) remain the independent consistency gate. |
| `compute_strut_metrics` | **The core Part 2 primitive.** Per registered edge: corridor occupancy profile, max axial gap, EDT local radius, **corridor-local connectivity** — connected components computed on the per-strut subvolume with junction spheres masked out (a whole-mask CC check is useless: the lattice is one giant component) — and **centerline curvature** (§2.1): per-slice cross-section centroids along the corridor form a polyline, Laplacian-smoothed, whose RMS deviation from the straight design axis is the bent-strut signal LatticeAnalytics uses. **Axis map `[x,y,z]→vol[z,y,x]` pinned inside the tool.** Corridor radius comes from the Stage 3 bootstrap, not a hard-coded guess. |
| `classify_struts` | Applies agent-chosen cutoffs deterministically; records thresholds verbatim in the output |
| `compute_registration_qa` | Three explicit outputs: (1) production image QA on the selected coordinate graph over all node/edge records with separate primary/stable-coarse/fallback/ambiguous/rejected/boundary quality; (2) **ROI gates** using displacement/repeatability from the hashed localization report plus image support on 20 %-padded ROIs; and (3) a stricter metrology gate requiring artifact-backed absolute uncertainty. Scope comes only from the hashed manifest: `roi_screening` may pass with metrology `not_authorized`, while `direct_metrology` requires passing uncertainty. The MCP boundary accepts neither an agent-selected scope nor uncertainty scalar. |
| `compute_spatial_stats` | Cell-shell defect rates, cKDTree nearest-neighbor + seeded permutation tests |
| `render_strut_evidence` | Evidence packets: three orthogonal CT crops centered on a strut + occupancy/intensity profile plot (generated at Stage 4; consumed by Stage 5 triage and embedded by Stage 6) |
| `render_lattice_3d` | Offscreen PyVista render of the **design graph** (18k line segments — not the 1 GB volume) colored by defect class; the report hero figure and demo asset, answering the README's visualization emphasis |
| `compute_detection_metrics` | Eval-side **recall only** (strict + lenient, with Wilson CI) + 4-class confusion matrix vs sealed labels; invoked only by `eval_agent` (§5.3 explains why precision/F1 are not computed) |
| `get_strut_report` | Return a compact, artifact-backed report payload for one strut ID: final class, attribution, metrics, thresholds, evidence paths, and provenance hashes. It must not recompute metrics or expose sealed labels outside `eval_agent`. |

### 3.4 Skills

Every current and planned project skill must declare its required MCP server in
`agents/openai.yaml`. Missing, unhealthy, or schema-incompatible MCP tooling is
a hard stop; a skill must not invoke a bundled CLI, import the implementation,
or create a substitute. Deterministic volume and artifact processing belongs
in §3.3, while skills retain scientific policy, stage ordering, acceptance
gates, and interpretation guidance.

- **`strut-defect-analyzer` (NEW):** owns Stages 3–4; documents the registered-JSON schema, the pinned axis mapping, the corridor-radius bootstrap, the corridor-local connectivity method (and its limitation: sub-voxel lack-of-fusion is undetectable at 58 µm/voxel), and dev/sealed label discipline.
- **`ct-registration` (implemented):** owns Stage 2; documents challenge vs autonomous mode, CT-only isolation for v2, automatic Otsu diagnostics, candidate holdout, multi-start ICP, robustness sweeps, independent local recentering, transform convention, ROI-vs-metrology gates, the registered-graph schema, and the hard prohibition on reading the aligned JSON before an autonomous fit is frozen.
- **`stl-design-diff` (implemented):** owns Stage 1; documents that STLs are mm, origin-centered, unregistered, with extra Y geometry; the tube-emptiness method; labels transfer **by edge ID only**.
- **`volume-metadata`:** memory-aware TIFF/NPY headers, hashes and optional
  statistics through `inspect_volume_metadata`; file metadata is the only
  sanctioned source of voxel spacing, and unavailable axes/spacing remain
  explicitly `unknown`. No CLI fallback exists.
- **`ct-threshold-optimizer` (implemented, off critical path):** defaults to TIFF/NPY exact per-scan Otsu replay and canonical-mask verification. Explicit exploratory candidates remain bounded and provisional; unavailable MCP tooling halts the skill without fallback.
- **`nde-report-generator` (current `nde_report_expert`; Part 2 upgrade planned):** currently calls `summarize_nde_artifacts` and `render_volume_3d`, with `segment_ct_dataset` and `skeletonize` only when artifacts are absent. The Part 2 template adds the per-strut findings table, blind-findings vs attribution appendix (§5.1), spatial statistics, 3D figure, and methods/provenance pinned to the config hash. Rendering and metric extraction remain MCP-owned; no `3d_visualize.py` skill script is retained.
- **`part2-pipeline-runbook` (NEW, orchestrator-facing):** stage order, artifact paths, gate conditions, retry policy, presentation checklist.

### 3.5 Non-goals: interaction surfaces outside the autonomous loop

After the scientist-confirmed Stage 0 intake produces a verified `ready`
hand-off, the primary goal is a closed, auditable Stage 1–6 loop:
measurement → classification → independent verification → sealed scoring →
reporting. The orchestrator gates every transition on immutable artifacts and
completion receipts; deterministic tools and bounded agents replace the manual
adjudication used in an interactive inspection workflow.

The following LatticeAnalytics capabilities remain explicit non-goals for this
pipeline. They are human-facing interaction or infrastructure choices, not
missing scientific primitives:

| Non-goal | Why it is excluded |
|---|---|
| **VR-guided alignment** | Requires manual alignment for each specimen. Challenge mode instead consumes the README-authorized aligned graph; autonomous mode uses CT-only v2 coarse localization followed by independent local node recentering in Stage 2. |
| **Contour-view or roughness-map dashboard** | Its purpose is human histogram browsing and candidate adjudication. Stage 4 replaces that decision loop with calibrated cutoffs, evidence packets, `classifier_verifier`, and logged triage. |
| **OpenVisus multi-resolution streaming** | It supports interactive browsing of very large network-hosted volumes. The current headless pipeline processes local specimen files with memory mapping, bounded chunks, and per-corridor subvolumes. |
| **Interactive Plotly/Dash application** | The deliverables are machine-generated artifacts—a metrics table, classified findings, evidence, a traceable report, and static 3D figures—not a continuously operated application. |

These exclusions apply to interaction surfaces, not to the underlying
capabilities. The pipeline retains the relevant registration, per-strut
sampling, centerline, curvature, and defect-analysis methods described in
§2.1.

---

## 4. Pipeline stages, artifacts, and milestones

Specimen manifests, receipts, stage hand-offs, and canonical configuration live
under `analysis/<specimen_id>/`. Large derived challenge artifacts live under
the manifest-declared `data/missing_struts/analysis/` tree; eval-owned artifacts
(sealed split, rubrics, results, harness) live under repo-root `evals/` to keep
them physically separate from what upstream agents touch. Large binaries are
gitignored and regenerable; everything else is committed. Milestones are
numbered in execution order and each is independently demoable.

| Stage | Milestone | In → Out (key artifacts) | Gate |
|---|---|---|---|
| 0 scientist-confirmed specimen intake | — | closed self-hashed scientist request + CAD STL + nominal graph JSON + CT TIFF/NPY + optional authorized aligned JSON → `analysis/<specimen_id>/config/{ct_metadata_response,ct_metadata_mcp_call_receipt,ingest_request,specimen_manifest,ingest_receipt,data_prep_handoff}.json` | `specimen_ingest` confirms association/units/axes/provenance, receives header-only `inspect_volume_metadata` output plus its integrity receipt, validates all six documents and current source bytes against the scientist request, then emits `ready` or explicit `halt`; metadata evidence is a Stage 0 output rather than a pre-start input, and its hash chain is not cryptographic origin authentication; no filename inference or defect/segmentation/registration access |
| 1 intentional-deletion labels | **M1** — "here are the ~93 struts LLNL deleted, from design files alone" | `0.5.stl` (+`0.stl`, `0.1.stl`, `1.stl` independent cross-check specimens) + nominal JSON → `labels/intentional_deletions_{0p1,0p5,1p0}.json`, `label_report.md`; then a **stratified** 30/70 split (by x-bin and z-shell) → `labels/dev_split.json` + `evals/labels/sealed_split.json` | independent counts == 18/93/186, tri-deficit ratio 170–180 |
| 2 registration mode + local refinement + QA | **M2** — "every expected strut maps to a defensible CT ROI" + junction overlay | **Challenge mode:** supplied aligned JSON → schema/topology verification. **Autonomous mode:** nominal JSON + TIFF → v2 CT-only artifacts, frozen hashes, `our_registered.json`, then optional post-fit validation. Either mode → `localize_lattice_nodes` → independently refined graph; TIFF → per-scan Otsu/histogram report, `config/analysis_config.json`, `slice_380.png`, `qa/registration_qa.json`, `qa/bias_by_xyz.png` | Otsu replay == 40054 / 58,653,410 here; 10,206/18,468/729 schema; fit/holdout disjoint; histogram, synthetic, multi-start, image QA pass; accepted-node displacement P95 must stay within the 8-voxel local capture radius; independently recentered/stable-coarse nodes and 20 %-padded ROIs pass support/bounds checks. Metrology is reported separately and remains unavailable in challenge mode without absolute CT-only uncertainty evidence. |
| 3 padded ROI extraction + metrics | **M3** — sortable metrics table, worst-20 struts visualized | mask + locally refined graph + config → 20 %-padded, orientation-normalized strut ROIs; `struts/corridor_calibration.json`; `struts/per_strut_metrics.csv` with axial occupancy, longest gap, endpoint support, local connectivity, radius, and curvature | exactly 18,468 rows; every ROI has provenance and valid bounds |
| 4 blind classification (defect team) | — | metrics + dev labels + config → per-class `struts/findings_{missing,thin,broken}.json` (from the three class subagents), merged `struts/classified_struts.json` + `thresholds.json` + `decision_log.md` (from `defect_lead`, precedence missing > broken > thin > present), `evidence/strut_<id>/` packets for every non-present strut, `struts/verifier_report.json` (from `classifier_verifier`) | every strut labeled exactly once; every lead adjudication logged; `classifier_verifier` sign-off (sealed-split-blind evidence/cutoff/merge/log audit) required before Stage 5 unlocks |
| 5 sealed scoring + attribution + triage | **M4** — confusion matrix: "found X of 65 sealed deletions (CI), plus Y candidate unintentional defects with CT evidence" | classifications + sealed labels → `evals/results/<timestamp>.json` (recall + Wilson CI + confusion matrix), `struts/attributed_struts.json` (final intentional/unintentional table), `triage/triage_results.json` | **one-shot protocol** (§5.4); reporting, not pass/fail |
| 6 spatial stats + 3D viz + NDE report | **M5** — the deliverable | all artifacts (read-only) → `spatial/spatial_stats.json` + figures, `report/lattice_defects_3d.png`, `report/nde_report.md` | number-crosscheck script passes; judge rubric on report prose |

**Explicit MVP cuts:** full dense/non-rigid deformation registration (bounded independent node recentering is retained), spatially adaptive segmentation (bias is measured and normalized; per-scan global Otsu remains frozen), whole-volume skeletonization, open-ended threshold optimization, full analysis of the 0.1 %/1 % designs (count cross-checks only — no CT exists for them in this repo), and an interactive dashboard (static 3D render + evidence views suffice for the demo). Autonomous v2 registration is an optional path, not a blocker for the README-authorized challenge mode.

---

## 5. Evaluation layer

**Objective wherever ground truth exists; LLM-as-judge only where it doesn't.** The supplied aligned JSON has two explicitly separated roles: it is an authorized coordinate/topology input in challenge mode, or a post-fit node-position reference in autonomous-mode evaluation—never both in the same registration claim. It contains no defect labels. The STL-derived intentional-deletion list is defect ground truth used in Stage 5 and labels only the *missing* class. The `0.5.stl`-derived labels never enter registration or blind per-strut measurement, and the manifest records every reference access.

### 5.1 What is blinded, and what the split proves

The labels' *existence* is public (Stage 1 commits `label_report.md`); what is blinded is **classifier calibration**: within the Stage 4 defect team, only `missing_strut_agent` reads the 30 % dev split (the thin/broken subagents and the lead are label-free by contract), and the sealed 70 % (~65 struts) is read once, by `eval_agent`, at Stage 5. Blinding is **procedural, not cryptographic** — any agent could re-derive the labels — so the contracts state the prohibition explicitly and the manifest records which agent read which label file. Consequences for the report: Stage 6's report is two-part — *blind findings* (what the pipeline detected, before unsealing) and an *attribution appendix* (the final intentional-vs-unintentional table, produced by `eval_agent` at Stage 5). This avoids the trap of a "candidate unintentional" table that is 70 % sealed intentional deletions.

### 5.2 What supervises which class

The per-class subagent split of Stage 4 mirrors this supervision asymmetry directly:

- **missing / present boundary** (`missing_strut_agent`): calibrated on the ~28 dev positives; scored on the ~65 sealed positives.
- **thin / broken / disconnected / bent** (`thin_strut_agent`, `broken_strut_agent`): **no ground truth exists.** Cutoffs are distribution-derived (percentiles of EDT radius, max-gap, and centerline-curvature RMS over the population), justified in each subagent's findings file and the merged `decision_log.md`, and validated only via judge triage of evidence packets — never via detection metrics. **Bent** is a present-but-deformed attribute (it does not remove material), so it is reported in triage and the report but never competes in the missing > broken > thin > present precedence.

### 5.3 Objective metrics: recall, not precision

`compute_detection_metrics` reports **recall** (strict = missing only; lenient = missing ∪ broken, because deleted struts can partially print) with a **Wilson 95 % CI** (n≈65 → roughly ±0.07), plus the 4-class confusion matrix over sealed struts. **Precision and F1 are deliberately not computed**: the paper itself warns measured missing rates exceed nominal, so a detection outside the sealed list is a *candidate unintentional defect*, not a false positive — precision against sealed labels is undefined by the design's own logic. Precision is instead *estimated* via judge-triage confidence over the candidate set plus a human spot-check list.

### 5.4 One-shot sealed protocol

Stage 5 is a **reporting stage, not a pass/fail gate** (a retry would re-run the same deterministic metrics — and a hard "recall ≥ 0.90" gate would pressure post-hoc threshold loosening, which is exactly the circularity this design avoids). The protocol, pre-committed in the config: upstream thresholds are frozen before the first sealed evaluation; re-evaluation requires a logged config bump and is reported as run 2, not a replacement. Target recall (lenient ≥ 0.90) is stated as an *expectation with CI*, discussed in the report either way.

### 5.5 LLM-as-judge

One rubric doing real work, one lightweight check:

- `evals/rubric_defect_triage.md` — judges each *candidate unintentional defect's* evidence packet (3 orthogonal crops + occupancy profile): material truly absent? ring/artifact? neighbors intact? Per-criterion subscores + 0–5 plausibility. Counts reported judge-confidence-weighted with a human spot-check list — never as ground truth.
- Report prose: a **number-crosscheck script** verifies every figure in `nde_report.md` against committed artifacts (this is the real gate); a short judge rubric then grades only structure/clarity/honesty-about-uncertainty.

**Judge hygiene** (fixing every deficiency of the single-sample Part 1 result): N ≥ 5 repeats, median + spread, calibration items each run (known-good must score max, blank must score 0, harness fails on miscalibration), alternated attachment order, model ID/date/prompt+input hashes recorded in every result JSON.

**Harness:** `evals/run_evals.py` runs gates → detection metrics → triage → report checks and writes one aggregated timestamped result; invoked by `eval_agent`, never by hand.

---

## 6. Top risks and mitigations

| Risk | Mitigation |
|---|---|
| Threshold provenance conflict (v1 recorded 40049; v2 recomputes 40054) | Freeze the **per-scan exact-histogram Otsu method**, not either integer; persist diagnostics and require 40054 / 58,653,410 only as this scan's replay result. Pass the same recorded threshold into segmentation, registration QA, and ROI measurements. |
| Registration-mode ambiguity or reference leakage | Manifest declares `challenge_aligned_json` or `autonomous_v2`. Challenge mode may use the aligned JSON because it contains no defect labels and the Part 2 README note authorizes it. Autonomous mode accepts CT + nominal graph only and hashes all fit artifacts before the aligned JSON becomes available to the optional validator. Never claim the challenge-mode graph was autonomously recovered. |
| Coarse coordinates are mistaken for metrology-grade registration | Maintain separate gates. ROI mode uses artifact-derived displacement/repeatability, independently refined or CT-stable endpoints, and supported padded ROIs. Direct narrow-corridor/metrology requires an artifact-backed absolute-registration uncertainty and remains blocked when that evidence is unavailable or exceeds the measured radius. The agent cannot supply this scalar, and the paper's 20 % padding is not misreported as a universal lateral-error limit. |
| A few bounded fallback assignments force global review or get mislabeled as primary | Freeze aggregate fractions in the hashed manifest; retain separate primary, stable-coarse, fallback, ambiguous, rejected, and boundary counts; propagate node quality to incident edges. Fallback alone may pass only within all thresholds, while ambiguity/rejection/boundary failures remain explicit. |
| Local CT artifacts or normal junction branches move a node to the wrong peak | Use deterministic perturbed starts and require one spatial convergence cluster; score the junction core plus a robust median over incident-edge directions so one missing strut cannot pull the estimate; accept a move only when CT support improves, otherwise retain the coarse coordinate or abstain. |
| Axis-order silent failure (wrong mappings still return plausible 28–67 % foreground) | Mapping pinned in config **and inside registration, QA, and ROI tools**; Stage 2 must reproduce v2's 92.31 % mean 5×5×5 foreground fraction, 18,468-edge coverage, and schema gates before autonomous-mode outputs proceed. |
| Right-side intensity falloff biases exactly the per-strut signal (notes §11: struts captured left, mostly nodes right) | Stage 2 x-binned occupancy curve, **before** classification; recorded position normalization if material; disclosed as a limitation either way |
| STL label extraction errors (junction re-tessellation, float-exact diffs, cluster→edge assignment) | **Method avoids all three**: per-edge tube-emptiness test needs no triangle matching and no clustering; each specimen's triangle deficit and expected 18/93/186 count are independent gates; ambiguous edges flagged, never guessed |
| Label leakage / circular evaluation | Stratified 30/70 dev/sealed split; one-shot sealed protocol; procedural blinding stated honestly (§5.1) |
| Deleted struts partially print; measured rates exceed nominal (per Tran et al.) | Strict + lenient recall; no precision claimed against sealed labels; extras routed to judge triage with evidence packets |
| Disconnected-at-joint struts evade naive line sampling | Corridor-local CC with junction spheres masked + max-axial-gap profile; `broken` is a first-class label; sub-voxel lack-of-fusion documented as undetectable at this resolution |
| Strut-diameter ambiguity (231 / 350 / 424 µm) poisons `thin` and the corridor radius itself | Stage 3 bootstrap: empirical EDT radius from high-occupancy struts sets the corridor radius and nominal-thickness reference; all diameter candidates reported |
| Memory blowups (1.04 GB volume, 495 MB mask, 175 MB STLs on 16 GB) | uint16/bool only; memmap reads; registration EDT on a factor-two central sample; full-resolution registration only in local patches; per-corridor EDT subvolumes; one STL at a time; `process=False` |
| Stale context (truncated `STUDY_NOTES.md`, duplicate 1 GB scan, Codex-only subagent TOML) | Stage 2 notes hygiene; duplicate recorded in manifest (symlink instead of re-read); contract skeleton ported, not the TOML |
| Sampling noise on n≈65 sealed positives | Wilson CI reported alongside recall; stratified split; expectation-with-CI instead of hard gate |
| Operational footguns | Quote all space-containing paths; tifffile handles the big-endian byte order (documented for raw-memmap consumers); keep `pyproject.toml` and `uv.lock` synchronized as MCP tools add dependencies |

---

## 7. Immediate next steps (implementation order)

1. **Stage 0 intake:** use the implemented `specimen_ingest` agent to confirm the specimen association, produce the provisional manifest and receipt, and gate entry into `data_prep` on a verified `ready` hand-off.
2. **Stage 1 and Stage 2 are implemented:** run them through the immutable orchestration handoffs; require every deterministic gate and the autonomous CT-only freeze before Stage 3.
3. **M1 (complete):** `resolve_cad_graph_orientation`, `label_deleted_edges`, normalized ID map, sealed split, agent, skill, contracts, and MCP-client tests.
4. **M2 (complete):** exact Otsu/canonical mask, both registration modes, v2 robustness, independent node localization, separate all-edge QA gates/figures, data-prep handoff/manifest contract, agents, skills, and reference replay tests.
5. **M3 (Stage 3):** 20 %-padded normalized ROI extraction + `compute_strut_metrics` — the core primitive.
6. **M4–M5 (Stages 4–6):** defect team (`defect_lead` + the three class subagents), sealed scoring + triage, spatial stats, `render_lattice_3d`, NDE report.
7. **Presentation (owner: orchestrator/us):** demo workflow walks Stage 0 and M1→M5; show both registration modes, v2's ROI-vs-metrology gate result, the locally refined junction overlay, the 3D defect render, and the confusion matrix.

---

## 8. References

- Miao, H., Narain, A., Chheang, V., Hooten, J., Seede, R., Klacansky, P.,
  Bertsch, K., Guss, G., Giera, B., and Bremer, P.-T.,
  [“LatticeAnalytics: Strut-Level Visualization and Inspection of Additively
  Manufactured Lattice
  Structures”](https://doi.org/10.1109/TVCG.2025.3593230), *IEEE Transactions
  on Visualization and Computer Graphics* 31(10), 2025.
- Tran, B., Fisher, K. A., Wang, J., Divin, C., Balensiefer, G. J., Townsend,
  A. P.,
  [“Resonant ultrasound spectroscopy measurement and modeling of additively
  manufactured octet truss lattice
  cubes”](https://www.osti.gov/pages/biblio/2246722), *NDT&E International*
  138 (2023) 102870.
