"""Compatibility entry point for :mod:`llnl_nde.server`."""

from llnl_nde.server import *  # noqa: F403
from llnl_nde.server import mcp

if __name__ == "__main__":
    mcp.run()
