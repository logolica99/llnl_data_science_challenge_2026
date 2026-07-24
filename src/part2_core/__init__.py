"""Deterministic Part 2 primitives.

This package has no MCP or agent dependencies.  The MCP server imports these
functions and is responsible only for path policy and structured envelopes.
"""

from .graph import GraphNormalizationError, normalize_lattice_graph
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
from .response import (
    GATES,
    RESPONSE_SCHEMA_VERSION,
    error_response,
    success_response,
)
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
    "GATES",
    "GraphNormalizationError",
    "OtsuReplayError",
    "RESPONSE_SCHEMA_VERSION",
    "VolumeLoadError",
    "VolumeView",
    "deterministic_histogram",
    "error_response",
    "histogram_diagnostics",
    "histogram_sha256",
    "iter_array_chunks",
    "load_volume",
    "normalize_lattice_graph",
    "otsu_from_histogram",
    "replay_exact_otsu",
    "sample_xyz",
    "success_response",
    "volume_metadata",
    "write_otsu_artifacts",
    "xyz_to_zyx_indices",
]
