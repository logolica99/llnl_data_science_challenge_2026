# Stage 3 classification decision log

- Specimen: `brian_tran_hackathon`
- Gate: `manual_review`
- Development mode: `true`
- Precedence: `missing > broken > thin > present`
- Missing: primary A-to-B component is disconnected and at most 10% of central (20%-80%) axial slices are material-bearing (foreground fraction ≥ 0.05).
- Broken: missing is false, endpoint material is observed, and either at least 15% of central slices are below 50% of central P90 or the deficient run is at least three slices.
- Connected bite cases may be broken; unresolved disconnections require review.
- Bent is a separate non-competing attribute.
- Deferred specialist implementations: `thin, bent`
- Specialist-review struts: `683`
- Training, evaluation, and intentional-deletion labels accessed: `false`
