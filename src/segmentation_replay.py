"""Compatibility entry point for :mod:`llnl_nde.cli.segmentation_replay`."""

from llnl_nde.cli.segmentation_replay import *  # noqa: F403
from llnl_nde.cli.segmentation_replay import main

if __name__ == "__main__":
    raise SystemExit(main())
