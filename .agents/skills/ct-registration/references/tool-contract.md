# Stage 1 MCP contract

Required `segmentation-tools` interfaces:

- `volume_info`: TIFF/NPY metadata, endian and ZYX/XYZ mapping, source hash.
- `replay_exact_otsu`: exact per-scan histogram and Otsu result derived from the
  frozen specimen-manifest segmentation policy; rejection gates; artifacts;
  policy path/hash and `analysis_parameters_sha256` bindings.
- `segment_ct_dataset`: slab-wise canonical uint8 mask and mask contract.
- `compare_segmentation_masks`: bounded aligned-mask statistics and persisted report.
- `verify_canonical_segmentation`: independently replay the frozen exact-Otsu
  recipe and `raw >= threshold`, compare both persisted reports, and atomically
  persist closed specimen/design/path/SHA-bound verification evidence.
- `register_lattice_to_ct`: challenge validation or isolated autonomous-v2 fit.
- `localize_lattice_nodes`: topology-supported deterministic multistart
  mean-shift positions, stable-coarse decisions, CT-support comparisons,
  convergence repeatability, separately counted bounded fallbacks/ambiguity/
  rejection/boundary results, incident-edge quality propagation, and frozen
  aggregate gates without label access.
- `compute_registration_qa`: all-node/all-edge support, localization-report
  derived displacement/repeatability, artifact-backed absolute uncertainty
  when available, hash-bound requested scope/policy, separate ROI/metrology
  gates, authorization lists, reason codes, and status/bias figures. Its MCP
  schema must not accept a caller-supplied scope or uncertainty scalar.

Every call must return `part2-mcp-response/1.0.0`, an explicit gate, repository-relative artifacts with SHA-256, compact counts, warnings, and a structured error on failure.
