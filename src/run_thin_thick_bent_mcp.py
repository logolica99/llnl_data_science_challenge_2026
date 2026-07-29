"""Invoke the thin/thick/bent pipeline through a fresh MCP stdio session."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import site
import sys
from datetime import timedelta
from pathlib import Path

# Process project-local dependency .pth files before importing FastMCP.
_REPOSITORY = Path(__file__).resolve().parents[1]
site.addsitedir(str(_REPOSITORY / ".python_packages"))

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport


REQUIRED_TOOLS = {
    "compute_strut_metrics",
    "classify_struts",
    "render_strut_evidence",
    "run_thin_thick_bent_pipeline",
}


async def run(args):
    repository = Path(__file__).resolve().parents[1]
    local_packages = repository / ".python_packages"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(local_packages), str(repository / "src")]
    )
    transport = PythonStdioTransport(
        script_path=repository / "src" / "mcp_server.py",
        python_cmd=sys.executable,
        cwd=str(repository),
        env=environment,
        keep_alive=False,
    )
    client = Client(
        transport,
        timeout=timedelta(hours=args.timeout_hours),
        init_timeout=timedelta(seconds=30),
    )
    async with client:
        available = {tool.name for tool in await client.list_tools()}
        missing = sorted(REQUIRED_TOOLS - available)
        if missing:
            raise RuntimeError(f"Required MCP tools are unavailable: {missing}")
        result = await client.call_tool(
            "run_thin_thick_bent_pipeline",
            {
                "input_tiff": str(args.input_tiff.resolve()),
                "registered_json": str(args.registered_json.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "threshold": args.threshold,
                "thresholds_json": str(args.thresholds_json.resolve()),
                "positions": args.positions,
                "tracking_radius_voxels": args.tracking_radius_voxels,
                "max_struts": args.max_struts,
                "overwrite": args.overwrite,
            },
        )
        if result.is_error:
            raise RuntimeError(str(result))
        print(json.dumps(result.structured_content, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--thresholds-json", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=11)
    parser.add_argument("--tracking-radius-voxels", type=float, default=6.0)
    parser.add_argument("--max-struts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    return parser


if __name__ == "__main__":
    asyncio.run(run(build_parser().parse_args()))
