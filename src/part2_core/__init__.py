"""Compatibility package for the canonical :mod:`llnl_nde.core` package."""

from importlib import import_module as _import_module
import sys as _sys

_ALIASES = {
    "artifacts": "artifacts",
    "classification": "classification",
    "evidence": "evidence",
    "graph": "graph",
    "lattice": "lattice",
    "localization": "localization",
    "otsu": "otsu",
    "qa": "qa",
    "registration": "registration",
    "reporting": "reporting",
    "reports": "reporting",
    "response": "response",
    "sampling": "sampling",
    "segmentation": "segmentation",
    "spatial": "spatial",
    "strut_metrics": "strut_metrics",
    "struts": "struts",
    "volume": "volume",
}

for _legacy_name, _canonical_name in _ALIASES.items():
    _module = _import_module(f"llnl_nde.core.{_canonical_name}")
    globals()[_legacy_name] = _module
    _sys.modules[f"{__name__}.{_legacy_name}"] = _module

from llnl_nde.core import *  # noqa: F403,E402
from llnl_nde.core import __all__  # noqa: E402,F401
