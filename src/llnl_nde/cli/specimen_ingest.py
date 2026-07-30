#!/usr/bin/env python3
"""Create deterministic specimen intake artifacts from explicit inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.specimen_ingest import (  # noqa: E402
    SpecimenIngestError,
    ingest_specimen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specimen-id", required=True)
    parser.add_argument("--design-id", required=True)
    parser.add_argument(
        "--requested-analysis-scope",
        required=True,
        choices=("roi_screening", "direct_metrology"),
    )
    parser.add_argument(
        "--cad",
        type=Path,
        help="Optional research-only CAD input; production intake uses graph + CT only.",
    )
    parser.add_argument("--design-graph", required=True, type=Path)
    parser.add_argument("--normalized-graph", type=Path)
    parser.add_argument("--normalized-graph-sha256")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument(
        "--ct-metadata-response",
        required=True,
        type=Path,
        help=(
            "persisted inspect_volume_metadata MCP response at "
            "analysis/<specimen-id>/config/ct_metadata_response.json"
        ),
    )
    parser.add_argument(
        "--ct-metadata-response-sha256",
        required=True,
        help="exact SHA-256 returned for the persisted MCP response artifact",
    )
    parser.add_argument(
        "--ct-metadata-call-receipt",
        required=True,
        type=Path,
        help=(
            "persisted inspect_volume_metadata call receipt at "
            "analysis/<specimen-id>/config/ct_metadata_mcp_call_receipt.json"
        ),
    )
    parser.add_argument(
        "--ct-metadata-call-receipt-sha256",
        required=True,
        help="exact SHA-256 returned for the persisted MCP call-receipt artifact",
    )
    parser.add_argument("--aligned-graph", type=Path)
    parser.add_argument("--design-transform-declaration", type=Path)
    parser.add_argument(
        "--registration-mode",
        required=True,
        choices=("autonomous_v2",),
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
        default="voxel",
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
            design_id=args.design_id,
            requested_analysis_scope=args.requested_analysis_scope,
            cad_path=args.cad,
            design_graph_path=args.design_graph,
            ct_path=args.ct,
            ct_metadata_response_path=args.ct_metadata_response,
            ct_metadata_response_sha256=args.ct_metadata_response_sha256,
            ct_metadata_call_receipt_path=args.ct_metadata_call_receipt,
            ct_metadata_call_receipt_sha256=(
                args.ct_metadata_call_receipt_sha256
            ),
            aligned_graph_path=args.aligned_graph,
            design_transform_declaration_path=args.design_transform_declaration,
            registration_mode=args.registration_mode,
            association_confirmed=args.confirm_association,
            allowed_data_roots=args.data_root,
            cad_units=args.cad_units,
            cad_units_provenance=args.cad_units_provenance,
            graph_axes=args.graph_axes,
            array_axes=args.array_axes,
            aligned_graph_units=args.aligned_graph_units,
            normalized_graph_path=args.normalized_graph,
            normalized_graph_sha256=args.normalized_graph_sha256,
            retention=args.retention,
        )
    except (OSError, TypeError, ValueError, SpecimenIngestError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
