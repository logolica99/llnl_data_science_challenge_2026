#!/usr/bin/env python3
"""Compatibility entry point for the packaged specimen-ingest CLI."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.cli.specimen_ingest import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
