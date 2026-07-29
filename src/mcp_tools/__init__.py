"""Compatibility package for :mod:`llnl_nde.mcp_tools`."""

from importlib import import_module as _import_module
import sys as _sys

_ALIASES = {
    "common": "common",
    "registry": "registry",
    "stage0": "specimen_ingest_stage0",
    "stage1": "data_prep_stage1",
    "stage2": "strut_metrics_stage2",
    "stage3": "defect_analysis_stage3",
    "stage4": "reporting_stage4",
}

for _legacy_name, _canonical_name in _ALIASES.items():
    _module = _import_module(f"llnl_nde.mcp_tools.{_canonical_name}")
    globals()[_legacy_name] = _module
    _sys.modules[f"{__name__}.{_legacy_name}"] = _module

from llnl_nde.mcp_tools import mcp

__all__ = ["mcp"]
