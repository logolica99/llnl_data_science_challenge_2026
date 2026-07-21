# Segmentation Quality Rubric

You are evaluating an automated CT lattice segmentation result against a ground truth image. You will be shown two images: the ground truth segmentation and the result produced by an automated segmentation pipeline.

## Evaluation Criteria

1. **Structural Integrity**: Does the result capture the connectivity of the lattice struts compared to the ground truth? 

2. **False Positives/Negatives**: Identify over-segmentation (extra noise) or under-segmentation (missing struts).

3. **Topology**: Are the nodes (junctions) preserved?

4. **Noise and Artifacts**: Does the result image contain noise or artifacts not present in the clean ground truth?

## Scoring (0-5)

- **5**: Identical to ground truth. No missing structures, no false positives.
- **4**: Excellent with very minor differences.
- **3**: Main topology is correct, but noticeable noise or thin struts are missing.
- **2**: Fair, but with significant differences (e.g., large chunks missing).
- **1**: Major structural failure or excessive noise.
- **0**: Blank or unrelated output.

## Output Format

Return ONLY a JSON object with exactly these two fields, and nothing else
(no markdown formatting, no explanation outside the JSON):

```json
{
  "reasoning": "<a few sentences explaining your assessment against each of the four criteria above>",
  "score": <integer from 0 to 5>
}
```