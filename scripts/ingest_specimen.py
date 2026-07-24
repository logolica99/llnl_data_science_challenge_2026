#!/usr/bin/env python3
"""Create deterministic specimen intake artifacts from explicit inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from specimen_ingest import SpecimenIngestError, ingest_specimen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimen-id", required=True)
    parser.add_argument("--cad", required=True, type=Path)
    parser.add_argument("--design-graph", required=True, type=Path)
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--aligned-graph", type=Path)
    parser.add_argument(
        "--registration-mode",
        required=True,
        choices=("challenge_aligned_json", "autonomous_v2"),
    )
    parser.add_argument(
        "--confirm-association",
        action="store_true",
        help="confirm the scientist explicitly associated these three inputs",
    )
    parser.add_argument("--cad-units", default="unknown")
    parser.add_argument("--cad-units-provenance", default="unknown")
    parser.add_argument("--graph-axes", choices=("xyz", "unknown"), default="xyz")
    parser.add_argument("--array-axes", choices=("zyx", "unknown"), default="unknown")
    parser.add_argument(
        "--aligned-graph-units",
        choices=("voxel", "simulation_voxel", "unknown"),
        default="unknown",
    )
    parser.add_argument(
        "--retention",
        choices=("committed", "external", "regenerable"),
        default="committed",
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--data-root",
        action="append",
        type=Path,
        help="allowed input root; repeat as needed (default: <repository>/data)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = ingest_specimen(
            repository_root=args.repository_root,
            specimen_id=args.specimen_id,
            cad_path=args.cad,
            design_graph_path=args.design_graph,
            ct_path=args.ct,
            aligned_graph_path=args.aligned_graph,
            registration_mode=args.registration_mode,
            association_confirmed=args.confirm_association,
            allowed_data_roots=args.data_root,
            cad_units=args.cad_units,
            cad_units_provenance=args.cad_units_provenance,
            graph_axes=args.graph_axes,
            array_axes=args.array_axes,
            aligned_graph_units=args.aligned_graph_units,
            retention=args.retention,
        )
    except (OSError, TypeError, ValueError, SpecimenIngestError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
