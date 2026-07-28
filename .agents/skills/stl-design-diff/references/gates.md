# Stage 1 deterministic gates

- Graph: 10,206 unique junction IDs, 18,468 unique strut IDs, 729 unique cell IDs, valid endpoint and cell references.
- Orientation: either one geometry-search hypothesis or a declaration bound to
  its artifact/self hashes, source/provenance, specimen/design, nominal graph,
  and full-design STL. Verify finite 3-vectors/3×3 rotation, orthonormality,
  right-handedness unless explicitly permitted, the source-backed 2.28
  mm/design-unit centerline pitch, translation, and all-edge correspondence at
  frozen tolerances. Symmetry without a valid declaration is `manual_review`;
  a bad declaration is `halt`.
- Leakage: orientation selection and verification complete before any deletion
  label or answer-bearing variant is opened.
- Deletions: validate the specimens independently; require exactly 18, 93, and 186 IDs for 0.1, 0.5, and 1 percent, with every ID appearing once in nominal topology. Do not require overlap or subset relationships across specimens.
- Mesh evidence: the baseline is a zero-deletion negative control and each variant removes 170–180 triangles per labeled deletion.
- Split: deterministic 30/70 stratification by midpoint X-bin and Z-shell; development and sealed IDs are disjoint and exhaustive over the 0.5 percent labels.
- Isolation: CT and aligned-coordinate access are false and no prior label artifact was read.
