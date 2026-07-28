# Stage 2 MCP contract

Required `segmentation-tools` interfaces:

- `volume_info`: TIFF/NPY metadata, endian and ZYX/XYZ mapping, source hash.
- `replay_exact_otsu`: exact per-scan histogram, Otsu result, rejection gates, artifacts.
- `segment_ct_dataset`: slab-wise canonical uint8 mask and mask contract.
- `compare_segmentation_masks`: bounded aligned-mask statistics and persisted report.
- `visualize_slice`: bounded PNG artifact without voxel payloads.
- `register_lattice_to_ct`: challenge validation or isolated autonomous-v2 fit.
- `localize_lattice_nodes`: topology-supported deterministic multistart
  mean-shift positions, stable-coarse decisions, CT-support comparisons,
  convergence repeatability, and bounded fallbacks without label access.
- `compute_registration_qa`: all-node/all-edge support, localization-report
  derived displacement/repeatability, artifact-backed absolute uncertainty
  when available, separate capture/metrology gates, and status/bias figures.
  Its MCP schema must not accept a caller-supplied uncertainty scalar.

Every call must return `part2-mcp-response/1.0.0`, an explicit gate, repository-relative artifacts with SHA-256, compact counts, warnings, and a structured error on failure.
