"""Deterministic Part 2 primitives.

This package has no MCP or agent dependencies.  The MCP server imports these
functions and is responsible only for path policy and structured envelopes.
"""

from .classification import classify_struts
from .evidence import render_strut_evidence
from .graph import GraphNormalizationError, normalize_lattice_graph
from .localization import localize_lattice_nodes
from .otsu import (
    DEFAULT_OTSU_RECIPE,
    OtsuReplayError,
    deterministic_histogram,
    histogram_diagnostics,
    histogram_sha256,
    otsu_from_histogram,
    replay_exact_otsu,
    write_otsu_artifacts,
)
from .qa import compute_registration_qa
from .registration import (
    DEFAULT_REGISTRATION_CONFIG,
    SimilarityTransform,
    coarse_initialization,
    detect_ct_nodes,
    multistart_fit,
    register_lattice_to_ct,
    rotation_difference_deg,
    run_robustness_suite,
    run_synthetic_suite,
    solve_similarity,
    split_candidates,
    trimmed_icp,
)
from .reports import get_strut_report
from .spatial import compute_spatial_stats, render_lattice_3d
from .segmentation import (
    compare_segmentation_masks,
    segment_ct_dataset,
    visualize_slice,
)
from .response import (
    GATES,
    RESPONSE_SCHEMA_VERSION,
    error_response,
    success_response,
)
from .strut_metrics import compute_strut_metrics, read_metrics_csv
from .volume import (
    AXIS_MAPPING,
    VolumeLoadError,
    VolumeView,
    iter_array_chunks,
    load_volume,
    sample_xyz,
    volume_metadata,
    xyz_to_zyx_indices,
)

__all__ = [
    "AXIS_MAPPING",
    "DEFAULT_OTSU_RECIPE",
    "DEFAULT_REGISTRATION_CONFIG",
    "GATES",
    "GraphNormalizationError",
    "OtsuReplayError",
    "RESPONSE_SCHEMA_VERSION",
    "SimilarityTransform",
    "VolumeLoadError",
    "VolumeView",
    "classify_struts",
    "coarse_initialization",
    "compute_registration_qa",
    "compute_spatial_stats",
    "compute_strut_metrics",
    "compare_segmentation_masks",
    "detect_ct_nodes",
    "deterministic_histogram",
    "error_response",
    "get_strut_report",
    "histogram_diagnostics",
    "histogram_sha256",
    "iter_array_chunks",
    "load_volume",
    "localize_lattice_nodes",
    "multistart_fit",
    "normalize_lattice_graph",
    "otsu_from_histogram",
    "read_metrics_csv",
    "replay_exact_otsu",
    "register_lattice_to_ct",
    "render_strut_evidence",
    "render_lattice_3d",
    "rotation_difference_deg",
    "run_robustness_suite",
    "run_synthetic_suite",
    "sample_xyz",
    "segment_ct_dataset",
    "solve_similarity",
    "split_candidates",
    "success_response",
    "trimmed_icp",
    "volume_metadata",
    "visualize_slice",
    "write_otsu_artifacts",
    "xyz_to_zyx_indices",
]
