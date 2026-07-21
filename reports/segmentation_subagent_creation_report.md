# Segmentation Subagent Creation Process Report

**Project:** LLNL Data Science Challenge 2026 — Agentic AI for Materials Science  
**Focus:** Task 6 (Subagents) → Task 7 (LLM Evals) feedback loop  
**Artifact:** `.codex/agents/my_segmentation_agent.toml`  
**Date:** 2026-07-21  

---

## 1. Executive Summary

This report documents how a Codex **Segmentation Subagent** was created, tested, evaluated, and refined for X-ray CT lattice segmentation. The agent was not “finished” after the first TOML draft. Instead, it was improved through a closed loop:

1. Define the agent (Task 6 requirements)  
2. Run segmentation on `9x9x9_octet_lattice.tif`  
3. Score the result with an LLM rubric (Task 7)  
4. Translate score-3 failure modes into clearer `developer_instructions`  
5. Re-run and re-evaluate  

The final agent encodes scientific method choices (hysteresis thresholding, light morphology, strut-vs-node tradeoffs) learned from repeated LLM evaluations against slice-380 ground truth.

---

## 2. Challenge Context

### Task 6 — Subagents
Build a specialized Codex subagent that can:
- Segment a `.tif` / `.tiff` CT lattice volume  
- Iterate with visual feedback  
- Save reproducible code, a full mask, slice-380 visualization, and a Markdown report  
- Stop safely (max 10 iterations or 3 failed attempts without improvement)  

All outputs must live in a `segmentation/` folder next to the input TIF.

### Task 7 — LLM Evals
Evaluate `segmentation/slice_380.png` against  
`ground_truth_segmentation_slice_380.png` using  
`evals/rubric_segmentation_1.md`, returning JSON with `reasoning` and `score` (0–5).

---

## 3. Phase A — Initial Agent Creation

### 3.1 Location and schema
Created:

```text
.codex/agents/my_segmentation_agent.toml
```

Required Codex fields:
- `name`  
- `description`  
- `developer_instructions`  

Optional settings used:
- `model = "gpt-5.4"`  
- `model_reasoning_effort = "high"`  
- `sandbox_mode = "workspace-write"`  
- `nickname_candidates`  

### 3.2 Baseline instructions (v1)
The first version mirrored the challenge checklist:
- Default input: `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`  
- Required outputs: `segment_lattice.py`, `mask.tif`, `slice_380.png`, `report.md`, feedback plots  
- Closed-loop optimize → run → visualize → evaluate  
- Safety limits: 10 iterations / 3 no-improvement stops  
- Notes: large uint16 volume; prefer percentiles over naive 0.5 thresholds; compare to GT slice 380  

**Limitation of v1:** It told the agent *what to produce*, but not *how to fix* the specific physics/image-processing failure modes of this lattice CT (thin dim struts, intensity gradient, node thickening).

---

## 4. Phase B — First Operational Run and Supporting Tooling

### 4.1 Data characteristics discovered
Inspection of the CT volume established:
- Shape ≈ `(761, 815, 837)`, `uint16`  
- Intensities not normalized to `[0, 1]`  
- Strong material signal in high percentiles (roughly p90–p99)  
- Noticeable intensity nonuniformity across the field of view  

### 4.2 Deliverables produced
Typical Task 6 outputs under:

```text
data/9x9x9_octet_lattice/segmentation/
```

Including:
- Full-volume mask: `mask.tif` (and earlier/alternate `.npy` copy)  
- Eval preview: `slice_380.png`  
- Process log: `report.md`  
- Iteration plots / histograms  

### 4.3 Visualization lesson (Napari)
The mask is binary `uint8` with values `{0, 1}`. Opening it as an **Image** layer with default `0–255` contrast often appears blank. Reliable viewing:
- open as **Labels**, or  
- set contrast limits to `0–1`, or  
- display a `×255` copy  

This was a viewer issue, not an empty segmentation.

---

## 5. Phase C — Task 7 Evaluation Loop

### 5.1 Rubric
Created `evals/rubric_segmentation_1.md` with criteria:
1. Structural integrity  
2. False positives / false negatives  
3. Topology (nodes/junctions)  
4. Noise and artifacts  

Scoring 0–5, JSON-only output: `{ "reasoning", "score" }`.

### 5.2 Typical early eval result
Repeated scores around **3**, with reasoning like:
- Main node grid preserved  
- Left-side diamond topology partly preserved  
- Thin diagonal struts broken/missing (false negatives)  
- Later: nodes thicker than GT (localized over-segmentation)  
- Little random background noise  

Interpretation: **topology OK, fine structure and boundary quality not yet “excellent.”**

---

## 6. Phase D — Agent Modification Round 1  
### Goal: recover thin struts

### 6.1 Scientific diagnosis
A single global threshold tends to:
- keep bright **nodes**  
- drop dimmer **struts**  
especially under left/right intensity imbalance.

### 6.2 Instruction changes (v2)
Updated `developer_instructions` to prefer:
- **Hysteresis thresholding**
  - high cut (~p97) for strong cores  
  - low cut (~p92) for weak strut candidates  
  - keep weak regions only if connected to strong regions  
- **Light morphological closing** to bridge tiny gaps  
- Explicit warning against large dilations  
- Eval checklist focused on left-edge strut continuity vs GT  
- Hard rule: never use threshold `0.5` on this uint16 volume  

### 6.3 Effect
Strut recovery improved in places, but lowering sensitivity / adding closing risked **fatter nodes** — a new failure mode visible in later evals.

---

## 7. Phase E — Agent Modification Round 2  
### Goal: balance strut continuity vs node thickness

### 7.1 New eval diagnosis
Score remained ~3, but reasoning shifted to a **dual problem**:
- thin/fragmented struts still missing  
- nodes thicker/larger than ground truth  

Important insight encoded into the agent:  
**“Lower the threshold more” is not a valid universal fix** — it can improve connectivity while worsening over-segmentation at junctions.

### 7.2 Instruction changes (v3 — current)
Current agent behavior emphasizes a **balanced dual objective**:

| Priority | Requirement |
|---|---|
| 1 | Correct node grid |
| 2 | Thin strut continuity where GT also shows connections |
| 3 | Node thickness close to GT |
| 4 | Low false-positive noise |

Method guidance updated to:
- Wider hi/lo gap (hi ≈ p97–p98, lo ≈ p91–p93)  
- **Close then open** (reconnect, then shrink blobs)  
- Escalation options: Frangi/Sato, adaptive thresholding, two-branch merge (strict nodes OR thin struts)  
- Decision rules for parameter updates based on whether nodes are already too fat  
- Improvement definition: strut gains must not clearly worsen node thickness (and vice versa)  

Also clarified not to invent right-side diagonals if GT mostly shows isolated nodes on slice 380 (slice geometry, not necessarily a full 3D missing-strut map).

---

## 8. What Changed vs What Stayed Fixed

### Changed across versions
- Quality priorities and failure-mode language  
- Preferred segmentation algorithm family  
- Parameter tuning decision tree  
- How “improvement” is judged each iteration  
- Agent `description` text (now mentions hysteresis / thin-strut focus)  

### Kept constant (Task 6 contract)
- Output directory and required filenames  
- Safety limits (10 / 3)  
- Sandbox / model settings  
- Stop after segmentation deliverables unless asked for more  

---

## 9. Validation Beyond Slice 380

Only one official ground-truth image exists (slice 380). That creates a risk of **slice-specific overfitting**.

Mitigation used in practice:
- Full-volume mask inspected across many frames in Napari (Labels mode)  
- Qualitative check that other slices still show coherent lattice structure  

Conclusion from that review: **other frames also looked strong**, reducing concern that the pipeline only “solved” slice 380.  
Caveat: without additional annotated GT slices, Task 7 score remains a **single-slice** official metric.

---

## 10. Final Artifacts

| Artifact | Path | Role |
|---|---|---|
| Subagent definition | `.codex/agents/my_segmentation_agent.toml` | Codex worker instructions |
| Eval rubric | `evals/rubric_segmentation_1.md` | Task 7 scoring protocol |
| Full mask | `data/9x9x9_octet_lattice/segmentation/mask.tif` | All 761 slices |
| Eval slice | `data/9x9x9_octet_lattice/segmentation/slice_380.png` | Task 7 result image |
| Ground truth | `data/9x9x9_octet_lattice/ground_truth_segmentation_slice_380.png` | Task 7 reference |
| Run report | `data/9x9x9_octet_lattice/segmentation/report.md` | Per-run technical log |

---

## 11. Lessons Learned

1. **Subagent quality is mostly in `developer_instructions`.**  
   The TOML is the scientific playbook the model follows when writing code.

2. **Eval feedback should be compiled into the agent, not only discussed in chat.**  
   Score-3 reasoning became durable rules (hysteresis, close→open, dual objective).

3. **Image-processing tradeoffs must be stated explicitly.**  
   Strut recall vs node thickness is a Pareto problem; unconstrained “make it more sensitive” instructions fail.

4. **Viewer settings can fake failure.**  
   `{0,1}` masks need Labels / correct contrast; otherwise teams debug the wrong problem.

5. **One GT slice is enough for the challenge score, not enough for full generalization proof.**  
   Multi-slice visual QA is the practical complement.

6. **Agent creation is iterative engineering.**  
   v1 satisfied the assignment structure; v2/v3 made the agent competent on this dataset’s failure modes.

---

## 12. Recommended Next Steps

1. Re-run Task 7 after each substantive TOML change; track score + reasoning themes.  
2. Add a subagent rule to validate ≥3 non-GT slices via CT overlay before stopping.  
3. Optionally log parameter sets that improved struts without fattening nodes as a “known-good” preset.  
4. Transfer the same agent to `data/missing_struts` TIFs (Part 2) without retuning only to slice 380.

---

## 13. Conclusion

The Segmentation Subagent was created to satisfy Task 6, then matured through Task 7’s LLM evaluation loop. The decisive modifications were not changes to Codex infrastructure, but richer domain instructions: hysteresis segmentation, morphology constraints, and an explicit strut/node tradeoff policy derived from repeated score-3 critiques. The resulting full-volume `mask.tif` appears coherent across frames, while official scoring remains anchored to slice 380 by challenge design.
