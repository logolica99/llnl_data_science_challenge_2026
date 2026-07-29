# LLNL 2026 Data Science Challenge — Complete Study Notes
### Agentic AI for Materials Science

> Merged from my lecture notes + the repo README + inspection of the actual data files.
> Mentor: **Haichao Miao** (miao1@llnl.gov), Research Scientist at LLNL — Additive Manufacturing / 3D printing.
> Repo: `mhaichao/llnl_data_science_challenge_2026` (this repo is our fork/copy of it).

---

## 1. The Big Picture (one paragraph)

We 3D-print metal lattice parts using **laser powder bed fusion (LPBF)**. After printing, the only way to check the *inside* of the part without destroying it is an **X-ray CT scan**, which gives a 3D volume of density values. Today, a human materials scientist manually inspects that volume for defects — this is the **bottleneck** of the whole manufacturing pipeline, and LLNL wants it automated (eventually in real time). Our job: build an **agentic AI workflow** — an LLM equipped with tools (MCP), domain instructions (skills), and specialized workers (subagents) — that can segment the CT volume, extract the lattice structure, compare it against the intended design, find defects (missing / broken / thin / bent struts, dross), and write a professional **NDE (non-destructive evaluation) report**, all with traceable, reproducible outputs.

The tech stack progression from the kickoff talk: **OpenCV/image processing → LLM agents → materials science application (LPBF inspection)**.

---

## 2. Materials Science Background — Filling the Gaps

### What is LPBF?
**Laser Powder Bed Fusion**: a thin layer of metal powder is spread, a laser melts/fuses the cross-section of the part, the bed drops, a new powder layer is spread, repeat. It can build geometries impossible to machine — like internal lattices — but the process is failure-prone: incomplete fusion, warping, and missing/malformed features. Note the acronym is **LPBF** (my original notes had "LBPF" — the README itself typos it once too).

### What is a lattice structure?
Instead of printing solid metal, you fill the part's interior with a repeating **unit cell** — a small strut-based geometric motif tiled in 3D (here: 9×9×9 cells). You get most of the stiffness at a fraction of the weight.

### What is an octet unit cell, and *why are we so interested in it?* ✅ (open question answered)
An **octet cell** is a strut-based cell made of interconnected diagonal beams; tiled, it forms the **octet truss**. Reasons it's the star of this challenge:

1. **It's mechanically near-optimal.** The octet truss is *stretch-dominated*: under load, struts carry tension/compression along their axis (efficient) rather than bending (inefficient). This gives outstanding **stiffness-to-weight and strength-to-weight** ratios — my notes said "strength at low intensity"; the right word is **low density**.
2. **It's what LLNL actually prints.** Lightweight structures, energy absorption (crash/impact mitigation), thermal applications. The Part 2 dataset is a real LLNL/Tran et al. print in **Ti5553 titanium alloy**.
3. **It's a great inspection testbed.** The geometry is perfectly regular and known in advance, so a *missing* or *thin* strut is well defined — you can insert defects deliberately and measure whether your algorithm catches them.
4. **Defects matter here.** Because the truss is stretch-dominated, every strut is a load path. **A missing strut breaks a load path** — locally the structure loses strength disproportionately. That's why the defect classes are strut-level: missing, broken, thin, bent, plus **dross** (excess re-solidified material blobs, an LPBF-specific defect).

### What is X-ray CT data, concretely?
The object is rotated in an X-ray beam; material **absorbs** radiation and a reconstruction algorithm builds a **3D scalar field** — a 3D array where each **voxel** (3D pixel) holds a grayscale density value (in our simulated data, normalized to [0, 1]). High value ≈ metal, low value ≈ air. It is *not* a mesh or a point cloud — it's basically a 3D image, which is why image-processing tools (OpenCV, scikit-image) apply.

### Ground truth — correcting my earlier note
My notes said "GroundTruth is the CT scan." More precisely, it depends on the question:
- **For segmentation quality (Part 1, Task 7):** the ground truth is the provided reference mask image `data/9x9x9_octet_lattice/ground_truth_segmentation_slice_380.png`. We compare *our* segmentation of slice 380 against it.
- **For defect detection (Part 2):** the CT scan is the **measurement** (what was actually printed); the **design** (STL mesh and/or JSON graph of nodes+edges) is the **intent**. Defects = deviations of measurement from intent. So: align STL/JSON with the TIF, then find design struts with no material in the CT → missing struts.

---

## 3. The Analysis Pipeline (why each step exists)

The three-stage image pipeline, all shown on the same slice in `images/`:

| Stage | Input → Output | Why |
|---|---|---|
| **Segmentation** | density volume → binary mask (material=1, background=0) | Raw CT is fuzzy grayscale; thresholding turns "how dense is this voxel" into "is this metal or not." Simplest method: `mask = volume >= threshold`. Choosing the threshold well is the whole game (histograms, Otsu's method, visual feedback loops). |
| **Slicing** | 3D volume → one 2D cross-section | You can't eyeball a 3D array. A CT volume is literally a stack of 2D images; viewing slice *i* along an axis is how you (and the agent) inspect data and debug segmentations. |
| **Skeletonization** | binary mask → 1-voxel-thick centerline | See below. |

### *Why skeletonization?* ✅ (open question answered)
Skeletonization (`skimage.morphology.skeletonize`) erodes the segmented mask down to a **thin centerline that preserves shape and connectivity**. The point:

1. **It converts pixels into structure.** A blob of ~millions of foreground voxels becomes a sparse curve network whose branches ≈ **struts** and junction points ≈ **nodes**. That's the same vocabulary as the design's JSON graph (junctions + struts) — so now you can compare *printed topology* vs *designed topology* graph-to-graph.
2. **Defects become graph queries.** Missing strut = an edge in the design graph with no corresponding skeleton branch. Broken strut = a skeleton branch with a gap/disconnected component. Thin strut = small local radius of the mask around the skeleton centerline. Bent strut = centerline deviates from the straight line between its two junctions.
3. **It's something an LLM agent can reason about.** An agent can't "look at" 10⁸ voxels, but it *can* reason over "junction 412 has 11 of 12 expected struts." Segmentation + skeletonization turn density into a symbolic structure that fits in a context window. (This was the punchline of the lecture: *"segmentation and skeletonization turn density into a structure an agent can reason about."*)

---

## 4. *What's the role of AI in materials science?* ✅ (open question answered)

- **Old paradigm (traditional ML):** isolated point tasks — predict one property, segment one image. A human still glues every step together.
- **New paradigm (agentic AI):** materials science is now **data-rich and workflow-heavy** — the bottleneck isn't a single prediction, it's running the whole multi-step analysis. Agents = LLMs + tools + planning + environment access = **autonomous AI research assistants** that execute the entire workflow.
- **The agent loop** (memorize this — it's the mental model for everything we build):

  > **Plan → write & call code → inspect artifacts (slices, masks, reports, logs, failures) → revise → repeat**

  The "inspect artifacts" step is what makes it *agentic* rather than a script: the agent looks at its own outputs (e.g., a rendered slice of its segmentation) and self-corrects.

---

## 5. Agentic AI Building Blocks (the four pillars)

### 5.1 MCP — Model Context Protocol
The standardized way to let an LLM call our Python functions. Before MCP, every tool × every LLM needed custom glue code; MCP is one open protocol.

- **Architecture:** *MCP server* (our Python script hosting the tools) ↔ *MCP client* (Codex CLI) speaking **JSON-RPC**.
- **Why it's good for science:** (1) decouples our scientific code from the LLM, (2) tools are auto-discovered, (3) security — a controlled layer defining exactly what the LLM may do.
- **We use FastMCP.** The rules that matter (this is what the "well-written descriptions" note was about — the docstring *is* the tool's UI for the agent):
  - `@mcp.tool()` exposes a function; the **function name becomes the tool name**.
  - The **docstring becomes the tool description** — write it for the agent: what it does, what each arg means, what it returns.
  - **Type annotations define the input schema** (`threshold: float`); params **without defaults are required**, with defaults are optional (`axis: int = 0`).
  - **No `*args`/`**kwargs`** — FastMCP needs an explicit signature to build the JSON schema.
  - Return short, useful status strings (or structured data when needed downstream).
- **Registration:** add the server to `~/.codex/config.toml` under `[mcp_servers.<name>]` with absolute paths to the Python exe and `src/llnl_nde/server.py`. **Restart Codex CLI after every config change** (it does not hot-reload). Verify with `/mcp`.

### 5.2 Skills
MCP's weakness: many tools up front **bloats the context window** and hurts **tool selection** reliability. A **skill** is a focused, on-demand instruction package (a `SKILL.md` + optional scripts) for a specific workflow — loaded only when relevant. My one-liner from the talk holds: *"skills package domain instructions so the agent writes like a materials scientist."*

The original `nde_report_expert` exercise demonstrated scripts, MCP calls, and
report instructions. That historical voxel/skeleton workflow now lives on the
disabled research MCP surface. Production uses
`.agents/skills/nde-report-generator/`, which calls graph-aware Stage 4 MCP
tools and never bundles scientific scripts.

Skills only load if Codex CLI is **started from the repo root** (it looks for `.agents/skills/`), and — same as MCP — **restart after adding/editing a skill**.

### 5.3 Subagents
Independent workers in a multi-agent system, each with its **own context window**. Why that matters: instead of crowding one agent's memory with loading + segmenting + reporting instructions, each subagent gets one bounded job — less confusion, deeper iteration, specialized instructions. My note "bounded loops for optimization and traceable outputs" is exactly the design requirement: give the subagent **explicit termination limits** (e.g., max 10 iterations or 3 failed attempts) and make it **save everything** (script, mask, plots, report) so its work is auditable.

Codex subagents are TOML files in `.codex/agents/` with `name`, `description`, `model`, `model_reasoning_effort`, `sandbox_mode`, and `developer_instructions`.

### 5.4 LLM Evals
How do we know the agent's output is any good? **LLM-as-judge**: give an LLM the result image + ground-truth image + a **rubric**, and have it return structured JSON (`{"reasoning": ..., "score": 0-5}`). Reasoning-then-score ordering matters (the score should follow from the reasoning). This is a *subjective* eval — a complement to (not a replacement for) objective pixel metrics like IoU/Dice, which would make a strong addition.

---

## 6. What's Actually in the Repo (verified by inspection)

```
DATA_SCIENCE_CHALLENGE_2026.pdf   ← full challenge instructions (same content as README)
README.md                          ← merged instructions (read this, it's canonical)
requirements.txt                   ← numpy, matplotlib, fastmcp, scikit-image, tifffile
src/
  llnl_nde/
    server.py                      ← production Stage 0–4 FastMCP server
    mcp_tools/                     ← thin wrappers grouped by immutable stage
    core/                          ← deterministic scientific implementations
    orchestration/                 ← contracts, receipts, and pipeline state
    cli/                           ← user-facing command-line entry points
.agents/skills/nde-report-generator/ ← production Stage 4 reporting skill
research/
  mcp_server.py                    ← disabled legacy/evaluation/research surface
  skeletonization.py               ← historical skeletonization implementation
images/                            ← slice.png, segmentation.png, skeleton.png examples
presentation/                      ← intro slides PDF
data/
  unitcell/                        ← 1×1×1 octet cell: unitcell.npy (simulated CT, values [0,1]),
                                      polyhedron_1x1x1.json (design graph), ground-truth seg image
  octet_truss_8x8x8/               ← design graph JSON only (no CT volume)
  9x9x9_octet_lattice/             ← 9x9x9_octet_lattice.tif (simulated CT, Git LFS!),
                                      ground_truth_segmentation_slice_380.png (Task 7 GT)
  data/missing_struts/             ← PART 2 real data (Git LFS!):
    tif_stacks/…Slices.tif         ← real X-ray CT of printed lattice
    stls/0.stl, 0.1.stl, 0.5.stl, 1.stl  ← design meshes at 0/0.1/0.5/1% missing struts
    registered_jsons/…Slices.json  ← design graph ALREADY ALIGNED to the tif of same name
    octet_truss_9x9x9.json         ← unregistered design graph
```

### ⚠️ Git LFS gotcha (bit me already)
The `.tif` volumes and the registered JSON are **Git LFS pointers** until you run:
```bash
git lfs install && git lfs pull
```
If `np.load`/`tifffile.imread`/`json.load` explodes on a file that's ~130 bytes, you're reading a pointer, not data.

### The design-graph JSON schema (verified from `octet_truss_8x8x8.json`)
Three top-level keys — this is the "intent" side of defect detection:
```jsonc
{
  "junctions": [ {"id": 0, "position": [x,y,z], "indices": [i,j,k]}, ... ],   // 7168 nodes (8x8x8)
  "struts":    [ {"id": 0, "unit_cell_edge_idx": 1,
                  "junction0": 0, "junction1": 9, "thickness": 0.1}, ... ],   // 13056 edges
  "unit_cells":[ {"id": 0, "struts": [0..23], "indices": [i,j,k]}, ... ]      // 512 cells, 24 struts each
}
```
So **missing-strut detection = for each strut (junction0→junction1 line segment), sample the CT/mask along that segment and test whether material is present.** The JSON gives us the exhaustive checklist.

### Registration (Part 2's hidden hard problem)
The **STL is NOT aligned** with the TIF/JSON — different coordinate frames, scales, orientations. Aligning them (registration) is a research problem in itself. Escape hatch: `registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json` **is already aligned** with the TIF of the same name. Recommendation: use the registered JSON, treat STL registration as a stretch goal.

### Part 2 dataset facts (Tran et al., NDT&E International 138 (2023) 102870)
- 9×9×9 octet lattices, LPBF-printed in **Ti5553**, from LLNL's Open Data Initiative.
- Intentionally missing struts at **0%, 0.1%, 0.5%, 1%** (that's what the STL filenames mean).
- Unit cell **4.56 mm**, ~10% relative density, **350 µm** strut diameter.
- Two primary defect classes in the real data: **missing struts** and **disconnected struts**.
- **Caveat for validation:** measured missing-strut percentages **may exceed nominal** (the printer failed extra struts beyond the designed ones), and disconnected struts are common. So "nominal 0.5%" is not a hard ground-truth count — don't score our detector as wrong just because it finds more than 0.5%.

### Environment
```bash
conda create -n dssi_env python=3.11 -y
conda activate dssi_env
pip install -r requirements.txt
```
Helpful extra software: **Napari** (interactive 3D/slice viewer — great for sanity-checking masks) and **MeshLab** (view STL meshes). Both were name-dropped in the talk.

---

## 7. Part 1 — Task-by-Task Cheat Sheet

| # | Task | What we build | Definition of done |
|---|---|---|---|
| 1 | Tool calling with MCP | Implement `segment_ct_dataset(input, output, threshold)` in the applicable `src/llnl_nde/mcp_tools/*_stage#.py` module; register `src/llnl_nde/server.py` in `~/.codex/config.toml` | `/mcp` shows the server; the archived Part 1 input is `DEPRECATED/part1/data/unitcell/unitcell.npy` |
| 2 | Multiple tools | Add `visualize_slice(input, output, slice_index, axis=0)` | Agent chains segment → visualize in one conversation |
| 3 | MCP as API wrapper | Historical `skeletonize` exercise, now isolated on `segmentation-tools-research` | Research agent runs segment → visualize → skeletonize without altering production |
| 4 | Skills | Historical report exercise; production replacement is `nde-report-generator` | Stage 4 produces graph-aware, hash-bound report artifacts |
| 5 | Custom skill | Our own `.agents/skills/<name>/SKILL.md` (ideas: metadata extractor; threshold optimizer sweeping 0.3/0.5/0.7) | Skill triggers and runs after CLI restart |
| 6 | Subagent | `.codex/agents/*.toml` **Segmentation Subagent** | Segments a `.tif`; closed-loop optimization with visual feedback; saves script + mask (`.tif`) + slice-380 PNG + MD report w/ fg/bg voxel counts, all into `segmentation/` subfolder next to the input; terminates at 10 iterations / 3 failed attempts |
| 7 | LLM evals | `DEPRECATED/part1/evals/rubric_segmentation_1.md` | `codex -i <gt.png> -i <result.png> "...apply rubric..."` returns JSON `{reasoning, score}` |

**Task 7 rubric criteria (0–5 scale):** structural integrity (strut connectivity vs GT), false positives/negatives (over/under-segmentation), topology (junctions preserved), noise/artifacts. 5 = identical to GT … 0 = blank/unrelated.

**Budget discipline (explicit rule in the README):** Codex usage is not unlimited. Frontier/high-reasoning models only for genuinely hard things (designing multi-step workflows, reviewing scientific assumptions); cheap models/low reasoning for routine coding, formatting, docs, test iteration. **Coordinate as a team to spread API-key usage evenly.**

---

## 8. Part 2 — Open-Ended Project

Goal: a multi-agent system that can **visualize, analyze, and reason about** the `missing_struts` dataset with more autonomy than Part 1. Three suggested tracks (or propose our own):

1. **Autonomous Data Explorer** — agents do EDA: one explores the directory, others pick feature-extraction methods for `.tif`/`.stl`/graph JSON; includes a Literature Research Agent + Coding Agent; output = analysis reports.
2. **Visual Reasoner** — put a renderer (**PyVista / ParaView / Napari**) inside the agent loop so the agent looks at rendered 3D views and reasons about anomalies from what it "sees."
3. **Interactive Co-Pilot & Dashboard** — 3D dashboard + chat; agent sees the current viewport and runs analyses on demand ("analyze connectivity in the region I'm looking at").

**My take on the highest-value core (fits any track):** a **missing-strut detector** — load registered JSON + TIF, segment the TIF, then for each of the design struts sample the mask along the junction0→junction1 segment and flag struts below a material-fraction threshold; cross-check flagged struts against skeleton connectivity; render flagged regions for the visual gallery; write an NDE report ranking suspect struts. That's a complete, demoable, quantifiable pipeline that directly answers the sponsor's problem statement.

---

## 9. Suggested Split for Our 4-Person Team

Everyone does Part 1 Tasks 1–3 individually (it's how you learn MCP), then split:

| Person | Part 1 ownership | Part 2 ownership |
|---|---|---|
| A | Task 4 + 5 (skills) | Detection logic: strut sampling vs design graph, defect classification |
| B | Task 6 (segmentation subagent) | Segmentation & skeletonization tuning on the *real* (noisier) CT |
| C | Task 7 (evals + rubric) | Evals & validation: metrics (IoU/Dice), rubrics, benchmarking vs nominal % |
| D | MCP server polish + docs | Visualization/dashboard (PyVista/Napari) + final NDE report generation |

Shared: registration decision (use registered JSON), budget coordination, final presentation.

---

## 10. Glossary (quick reference)

- **AM** — additive manufacturing (3D printing).
- **LPBF** — laser powder bed fusion; laser melts metal powder layer by layer.
- **X-ray CT** — computed tomography; rotational X-ray absorption → reconstructed 3D density volume.
- **Voxel** — 3D pixel; one cell of the volume array.
- **Octet truss** — stretch-dominated lattice of diagonal struts; high stiffness/strength at low density.
- **Strut / junction (node)** — beam element / point where struts meet (design JSON: `struts`, `junctions`).
- **Dross** — defect: excess re-solidified material attached to the part.
- **Segmentation** — foreground/background labeling; here, thresholding density into a binary mask.
- **Skeletonization** — reducing a mask to a 1-voxel centerline preserving topology.
- **Registration** — aligning two datasets into one coordinate frame (STL ↔ CT).
- **NDE** — non-destructive evaluation: inspecting a part without damaging it.
- **MCP** — Model Context Protocol; standard client-server (JSON-RPC) protocol for LLM tool use.
- **FastMCP** — Python library turning decorated functions into MCP tools.
- **Skill** — on-demand instruction package (`SKILL.md`) for a specific workflow.
- **Subagent** — independent agent with its own context window and bounded task.
- **LLM eval / LLM-as-judge** — scoring outputs with an LLM against a rubric, returning structured JSON.
- **Ground truth** — the reference you score against (provided seg images in Part 1; the design graph in Part 2).
- **Git LFS** — large file storage; run `git lfs pull` or the big files are just pointer stubs.

---

## 11. Dataset Deep-Dive (measured, not assumed)

Every file in `data/` was loaded and inspected. Key numbers:

| File | What it actually is | Measured facts |
|---|---|---|
| `unitcell/unitcell.npy` | Simulated CT recon of ONE octet unit cell | `(256,256,256)` float32, 67 MB. **Values are NOT [0,1]** — actual range ≈ **[-0.003, +0.015]** (attenuation-style recon, with negative noise and streak artifacts). Threshold ≈ **0.005** gives a clean mask (~4.3% foreground). A "0.5" threshold segments *nothing*. |
| `unitcell/polyhedron_1x1x1.json` | Design graph of the unit cell | 14 junctions, 12 struts, 1 unit cell; bbox (0,0,0)–(2,2,2) → **one unit cell = 2 design units**. Strut thickness 0.1. |
| `unitcell/ground_truth_segmentation_image.png` | A 3D **render** (1765×1838 RGBA) of the correctly segmented cell — visual reference, not a pixel mask | Grayscale-ish render with antialiasing (values 0–255, not binary). |
| `octet_truss_8x8x8/octet_truss_8x8x8.json` | Design graph only (no CT volume for it) | 7,168 junctions / 13,056 struts / 512 cells; bbox 0–16 (8 cells × 2 units). Good for practicing graph parsing. |
| `9x9x9_octet_lattice/9x9x9_octet_lattice.tif` | **Real** X-ray CT scan (LFS, ~1.04 GB) | `(761, 815, 837)` **uint16**, values ≈ 29k–60k, uncompressed multipage TIF. **Identical LFS hash to the Part 2 missing_struts TIF — it is literally the same scan** (Brian Tran 0.5%-missing specimen #1). So Part 1 Task 6 already runs on real Part 2 data. |
| `9x9x9_octet_lattice/ground_truth_segmentation_slice_380.png` | Task 7 ground truth | 800×800 RGBA — it is a **matplotlib figure** (axes + colorbar included), not a raw mask, and not pixel-aligned to the 815×837 CT slice. The LLM judge compares figures, not arrays. |
| `missing_struts/tif_stacks/*.tif` | Real CT (same file as above) | Slice 380 shows intensity falloff across the field: a single global threshold (e.g. 40000) captures struts on the left but only nodes on the right → **adaptive/local thresholding will be needed**. |
| `missing_struts/octet_truss_9x9x9.json` | Nominal design graph (0% missing), unregistered | 10,206 junctions / **18,468 struts** / 729 cells; bbox 0–18 design units. |
| `missing_struts/registered_jsons/…Slices.json` | Same graph **transformed into the TIF's voxel coordinates** | Same counts (10,206/18,468/729); junction positions span ≈ (58,48,24)–(774,765,738), i.e. inside the (837,815,761) voxel grid. **Verified by overlay: junctions land on bright lattice nodes in slice 380.** Position order is (x, y, z) = (col, row, slice). |
| `missing_struts/stls/{0,0.1,0.5,1}.stl` | Binary STL design meshes at each missing-strut % | ~175 MB, ~3.5 M triangles each (0%: 3,514,642 → 1%: 3,482,368 — fewer struts = fewer triangles). Coordinates in **mm**, centered near origin (± ~20–25 mm) → completely different frame from the TIF (unregistered, as warned). |
| `missing_struts/file_names.txt` | Metadata for the full published dataset | The full dataset has **3 physical replicate specimens per missing-% level** (same STL, different build-plate positions); this repo ships only the `0point5dash1` scan. Explicitly confirms STL/JSON/TIF live in different coordinate systems. |

**Consequences for our pipeline:**
1. Thresholds are dataset-specific: ~0.005 for `unitcell.npy` (float recon), ~40k for the uint16 TIF — motivates histogram/Otsu-based auto-thresholding rather than hard-coding.
2. The registered JSON + TIF pair is analysis-ready **today**: for each of the 18,468 design struts, sample the segmented volume along junction0→junction1 and flag low-material struts. No registration work needed.
3. The intensity gradient across the real scan means the Task 6 subagent's "iterative optimization with visual feedback" is genuinely necessary, not busywork.
4. Task 7's "ground truth" is a rendered figure — our result image should be rendered similarly (same slice, similar framing) for a fair LLM-judge comparison.

## 12. Key Paper: Tran et al. 2023 (the source of the Part 2 dataset)

**Tran, B., Fisher, K.A., Wang, J., Divin, C., Balensiefer, G.J., Townsend, A.P.** (all LLNL), "Resonant ultrasound spectroscopy measurement and modeling of additively manufactured octet truss lattice cubes," *NDT&E International* 138 (2023) 102870. [OSTI record](https://www.osti.gov/pages/biblio/2246722).

**Study design:** LPBF-printed Ti-5553 octet lattice cubes (9×9×9 cells, 4.56 mm unit cell, 10% relative density) with missing struts *designed in* at 0/0.5/1% as controlled defects, 3 replicate specimens per level. Actual defect state verified by X-ray CT (= our dataset); global elastic response measured by **resonant ultrasound spectroscopy (RUS)** — sweeping frequencies and recording the cube's natural resonances — and modeled with 3D FEM + a homogeneous anisotropic continuum model.

**Findings that matter for our project:**
1. **Missing struts ↑ → mass ↓, stiffness ↓, resonance frequencies ↓** — a monotonic global signature of strut-level damage.
2. **CT found more defects than designed:** measured missing-strut % exceeded nominal, and **disconnected struts** (printed but not fused at a junction — material present, no load path) were common. → Two defect classes: *missing* (density absent along design strut) and *disconnected* (density present but gapped at the joint; naive line-sampling passes it).
3. **Outlier specimen:** lower effective modulus despite *higher* density (dross adds mass, not stiffness). → Density alone cannot certify structural health; **topology/connectivity is the signal** — validates the skeleton-vs-design-graph approach.
4. The simple continuum model matched RUS well → lattices behave like an effective bulk material globally, but only CT localizes *which* strut is bad.

**Framing for our presentation:** RUS is the fast/cheap *global* screen ("something is wrong"); our agentic CT pipeline is the *local* diagnosis ("strut #4,312 between junctions 2101–2110 is missing"). Validation target for the `0point5dash1` scan: ≥ ~0.5% of 18,468 struts (~92+) missing — report measured vs nominal, don't treat 0.5% as exact.

**Open discrepancy:** paper-derived sources quote **424 µm** strut diameter at 10% relative density; the challenge README says **350 µm**. Likely design vs as-built (LPBF prints struts thick). Verify against the PDF.

## 13. Top Gotchas Checklist

- [ ] `git lfs pull` before touching the `.tif` volumes or registered JSON.
- [ ] Restart Codex CLI after **any** change to `~/.codex/config.toml`, skills, or subagents.
- [ ] Start Codex CLI **from the repo root** or project skills won't be found.
- [ ] Use **absolute paths** in the MCP server config.
- [ ] Threshold convention: voxels **≥ threshold → 1** (per the docstring).
- [ ] Task 6 outputs go in a `segmentation/` folder **next to the input tif**, slice index **380**, and the loop must terminate (10 iters / 3 fails).
- [ ] STL is unregistered — use the pre-registered JSON unless we deliberately take on registration.
- [ ] Real data: measured missing % can exceed nominal; disconnected struts are common — don't treat nominal % as exact ground truth.
- [ ] Spend model budget wisely; coordinate keys across the 4 of us.
