"""Compatibility exports for the former combined strut module.

New code should import Stage 2 metrics from :mod:`strut_metrics` and Stage 3
classification from :mod:`classification` directly.
"""

from .classification import classify_struts
from .strut_metrics import METRIC_FIELDS, compute_strut_metrics, read_metrics_csv

__all__ = [
    "METRIC_FIELDS",
    "classify_struts",
    "compute_strut_metrics",
    "read_metrics_csv",
]
