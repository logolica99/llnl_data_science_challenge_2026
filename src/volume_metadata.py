"""Compatibility alias for :mod:`llnl_nde.core.volume_inspection`."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("llnl_nde.core.volume_inspection")
_sys.modules[__name__] = _module
