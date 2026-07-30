"""Create a viewer CSV excluding a known, intentionally sliced lattice edge.

This script does not run CT connectivity and does not modify the complete
``missing_struts.csv`` detection record.  It only removes candidates touching
the explicitly declared registered-lattice boundary plane from a copied CSV.

The source CSV stores nominal endpoints in source-JSON coordinates.  The
registered CT coordinates are recovered with the documented self-inverse
rotation: ``(x, y, z) -> (18-x, 18-z, 18-y)``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/missing_struts.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/true_missing_struts.csv"


def registered_y(source_z: str, cube_max: float) -> float:
    """Convert a source JSON z coordinate to registered CT y."""
    return cube_max - float(source_z)


def is_known_intentional_edge_crop(
    row: dict[str, str], boundary_registered_y: float, cube_max: float, tolerance: float
) -> bool:
    """Return true when either endpoint touches the declared +Y crop plane."""
    endpoint_y_values = (
        registered_y(row["z0_json"], cube_max),
        registered_y(row["z1_json"], cube_max),
    )
    return any(y >= boundary_registered_y - tolerance for y in endpoint_y_values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Complete connectivity-candidate CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Filtered viewer CSV to create.")
    parser.add_argument(
        "--boundary-registered-y",
        type=float,
        default=18.0,
        help="Known intentionally sliced +Y boundary in registered coordinates (default: 18).",
    )
    parser.add_argument("--cube-max", type=float, default=18.0, help="Registered/source cube maximum (default: 18).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Boundary-plane tolerance in lattice-coordinate units (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")

    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {args.input}")
        required = {"z0_json", "z1_json"}
        if missing := required.difference(reader.fieldnames):
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)
        fieldnames = reader.fieldnames

    retained = [
        row
        for row in rows
        if not is_known_intentional_edge_crop(
            row, args.boundary_registered_y, args.cube_max, args.tolerance
        )
    ]
    excluded = len(rows) - len(retained)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)

    print(f"Input candidates: {len(rows)}")
    print(f"Known +Y={args.boundary_registered_y:g} edge-crop candidates excluded: {excluded}")
    print(f"Viewer candidates retained: {len(retained)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
