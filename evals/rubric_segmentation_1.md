# Segmentation Evaluation Rubric (`rubric_segmentation_1`)

You are an expert evaluator for X-ray CT lattice segmentation. Compare the **Result Image** against the **Ground Truth Image** for the same lattice slice.

## Inputs
- **Ground Truth Image:** first attached image (clean reference segmentation of the lattice at slice 380).
- **Result Image:** second attached image (segmentation produced by the agent / pipeline).

Evaluate only what is visible in these two images. Do not invent details that are not observable.

## Criteria
Score the result using all of the following:

1. **Structural Integrity**  
   Does the result capture the connectivity of the lattice struts compared to the ground truth? Are diagonal/strut pathways present where the GT shows them?

2. **False Positives / False Negatives**  
   Identify over-segmentation (extra noise, spurious blobs) or under-segmentation (missing struts, broken segments, missing nodes).

3. **Topology**  
   Are the nodes (junctions) preserved in roughly the correct grid layout? Do junctions appear where expected versus merged/split/missing?

4. **Noise and Artifacts**  
   Does the result contain noise, speckles, thickened blobs, or other artifacts not present in the clean ground truth?

## Scoring scale (integer 0–5)
- **5:** Identical to ground truth. No missing structures, no false positives.
- **4:** Excellent with very minor differences.
- **3:** Main topology is correct, but noticeable noise or thin struts are missing.
- **2:** Fair, but with significant differences (e.g., large chunks missing).
- **1:** Major structural failure or excessive noise.
- **0:** Blank or unrelated output.

Choose the single score that best matches the overall quality. If criteria conflict, weigh structural integrity and topology highest, then false positives/negatives, then noise/artifacts.

## Output format
Return **only** a JSON object (no markdown fences, no extra commentary) with exactly these keys:

```json
{
  "reasoning": "Concise evaluation covering structural integrity, false positives/negatives, topology, and noise/artifacts.",
  "score": 0
}
```

Requirements:
- `reasoning` must be a string.
- `score` must be an integer from 0 to 5 inclusive.
- Do not include any keys other than `reasoning` and `score`.
