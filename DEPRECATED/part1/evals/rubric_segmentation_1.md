# Segmentation Slice Evaluation Rubric

Evaluate two attached images in this fixed order:

1. The first image is the ground-truth segmentation for slice 380.
2. The second image is the segmentation result for slice 380.

Judge the segmentation content, not presentation differences. Ignore axes,
titles, colorbars, margins, image scaling, and color-map choices. Treat bright
or nonzero regions as foreground and dark or zero regions as background.

## Criteria

Consider all four criteria when assigning one overall score:

1. **Structural integrity:** Determine whether the result preserves the lattice
   struts and their connectivity relative to the ground truth. Note broken,
   shortened, merged, or missing structures.
2. **False positives and false negatives:** Identify over-segmentation, such as
   extra foreground or noise, and under-segmentation, such as missing or overly
   thin struts.
3. **Topology:** Determine whether nodes and junctions are present and connected
   in the same pattern as the ground truth.
4. **Noise and artifacts:** Identify isolated pixels, speckle, holes, edge
   artifacts, or other structures absent from the clean ground truth.

## Overall Score

Assign exactly one integer score from 0 through 5:

- **5 — Identical:** Segmentation content is effectively identical to the
  ground truth, with no missing structures or false positives.
- **4 — Excellent:** The structure and topology match with only very minor
  boundary, thickness, noise, or connectivity differences.
- **3 — Good:** The main topology is correct, but there is noticeable noise or
  some thin struts are missing.
- **2 — Fair:** The broad lattice pattern is recognizable, but significant
  differences exist, such as large missing sections or substantial extra
  foreground.
- **1 — Major failure:** The result has major structural failure, excessive
  noise, or little correspondence with the ground truth.
- **0 — Invalid:** The result is blank, unrelated, or not a segmentation of the
  supplied ground truth.

## Required Output

Return only one valid JSON object with exactly these keys:

```json
{
  "reasoning": "A concise comparison covering structural integrity, false positives and negatives, topology, and noise or artifacts.",
  "score": 0
}
```

`score` must be an integer from 0 through 5. Do not wrap the returned JSON in a
Markdown code fence and do not include text before or after it.
