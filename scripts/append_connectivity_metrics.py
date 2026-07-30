"""Append endpoint-specific connection metrics to a candidate strut CSV.

The connectivity metrics contain one row for endpoint A and one for endpoint B.
This script keeps both values so no endpoint evidence is discarded. It updates
the requested CSV in place after validating that every listed strut has exactly
two failed-connectivity metric rows.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/missing_struts/analysis/0_5_stl_heatmap/true_missing_struts.csv"
DEFAULT_METRICS_ROOT = REPO_ROOT / "outputs"
ADDED_COLUMNS = (
    "a_seed_foreground_fraction",
    "b_seed_foreground_fraction",
    "a_collar_foreground_fraction",
    "b_collar_foreground_fraction",
    "a_shared_component_voxel_count_in_cuboid",
    "b_shared_component_voxel_count_in_cuboid",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Candidate CSV to update in place.")
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    return parser.parse_args()


def load_metrics(metrics_root: Path, candidate_ids: set[int]) -> dict[int, dict[str, dict[str, str]]]:
    by_strut: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    metric_paths = sorted(metrics_root.glob("strut_node_connectivity_*/connection_metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No connection_metrics.csv files below {metrics_root}")
    for path in metric_paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                strut_id = int(row["strut_id"])
                if strut_id not in candidate_ids:
                    continue
                if row["same_material_component_connects_a_to_b"] != "False":
                    raise ValueError(f"Strut {strut_id} is connected in {path}; it cannot be appended as a failed candidate")
                endpoint = row["endpoint"]
                if endpoint not in {"A", "B"}:
                    raise ValueError(f"Unexpected endpoint {endpoint!r} for strut {strut_id}")
                if endpoint in by_strut[strut_id]:
                    raise ValueError(f"Duplicate endpoint-{endpoint} metric row for strut {strut_id}")
                by_strut[strut_id][endpoint] = row
    return by_strut


def main() -> None:
    args = parse_args()
    with args.input.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "strut_id" not in reader.fieldnames:
            raise ValueError(f"Expected a CSV with a strut_id column: {args.input}")
        source_rows = list(reader)
        retained_columns = [column for column in reader.fieldnames if column not in ADDED_COLUMNS]

    candidate_ids = {int(row["strut_id"]) for row in source_rows}
    metrics = load_metrics(args.metrics_root, candidate_ids)
    missing = sorted(candidate_ids.difference(metrics))
    incomplete = sorted(strut_id for strut_id, endpoints in metrics.items() if set(endpoints) != {"A", "B"})
    if missing or incomplete:
        details = []
        if missing:
            details.append(f"missing metrics for IDs {missing[:10]}")
        if incomplete:
            details.append(f"incomplete endpoint metrics for IDs {incomplete[:10]}")
        raise ValueError("; ".join(details))

    enriched_rows: list[dict[str, str]] = []
    for row in source_rows:
        strut_id = int(row["strut_id"])
        endpoint_a = metrics[strut_id]["A"]
        endpoint_b = metrics[strut_id]["B"]
        enriched_rows.append(
            {
                **{column: row[column] for column in retained_columns},
                "a_seed_foreground_fraction": endpoint_a["seed_foreground_fraction"],
                "b_seed_foreground_fraction": endpoint_b["seed_foreground_fraction"],
                "a_collar_foreground_fraction": endpoint_a["collar_foreground_fraction"],
                "b_collar_foreground_fraction": endpoint_b["collar_foreground_fraction"],
                "a_shared_component_voxel_count_in_cuboid": endpoint_a["shared_component_voxel_count_in_cuboid"],
                "b_shared_component_voxel_count_in_cuboid": endpoint_b["shared_component_voxel_count_in_cuboid"],
            }
        )

    fieldnames = [*retained_columns, *ADDED_COLUMNS]
    temporary = args.input.with_suffix(args.input.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    temporary.replace(args.input)
    print(f"Updated {args.input} with endpoint metrics for {len(enriched_rows)} nonconnected struts.")


if __name__ == "__main__":
    main()
