"""Export strut IDs from one or more result JSON files to review CSV format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OUTPUT_FIELDS = [
    "strut_id",
    "midpoint_surface_gap_mm",
    "x0_json",
    "y0_json",
    "z0_json",
    "x1_json",
    "y1_json",
    "z1_json",
    "x0_stl_mm",
    "y0_stl_mm",
    "z0_stl_mm",
    "x1_stl_mm",
    "y1_stl_mm",
    "z1_stl_mm",
]


def _items_with_ids(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("findings", "struts", "candidates", "results"):
        if isinstance(payload.get(key), list):
            return payload[key]
    numeric_keys = [
        key for key in payload
        if isinstance(key, str) and key.strip().lstrip("-").isdigit()
    ]
    return numeric_keys


def extract_strut_ids(path: str | Path):
    """Return deterministic unique strut IDs from a supported result JSON."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = set()
    for item in _items_with_ids(payload):
        value = item
        if isinstance(item, dict):
            value = item.get("strut_id", item.get("id"))
        if value in (None, ""):
            continue
        try:
            ids.add(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path} contains a non-integer strut ID: {value!r}"
            ) from error
    return sorted(ids)


def export_finding_ids(json_paths, output_csv: str | Path):
    """Combine result IDs and write the requested review-compatible CSV."""
    json_paths = [Path(path) for path in json_paths]
    ids = sorted({
        strut_id
        for path in json_paths
        for strut_id in extract_strut_ids(path)
    })
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for strut_id in ids:
            writer.writerow({"strut_id": strut_id})
    return output_csv, len(ids)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert thin/thick/bent findings JSON files to the existing "
            "review CSV schema. Only strut_id is populated."
        )
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        type=Path,
        help="One or more findings/result JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output CSV. With one input, defaults beside that JSON; with "
            "multiple inputs, defaults to strut_findings.csv beside the first."
        ),
    )
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = (
            args.json_paths[0].with_suffix(".csv")
            if len(args.json_paths) == 1
            else args.json_paths[0].parent / "strut_findings.csv"
        )
    path, count = export_finding_ids(args.json_paths, output)
    print(json.dumps({
        "status": "ready",
        "output_csv": str(path.resolve()),
        "strut_count": count,
    }))


if __name__ == "__main__":
    main()
