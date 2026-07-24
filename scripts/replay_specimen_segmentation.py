#!/usr/bin/env python3
"""Replay and verify one specimen's frozen segmentation result."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from segmentation_replay import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
