"""Write viewer CSVs with missing/broken labels and interior material minimum."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "data/missing_struts/analysis/0_5_stl_heatmap"


def main() -> None:
    feature_path = ANALYSIS_DIR / "all_strut_material_loss_features.csv"
    source_path = ANALYSIS_DIR / "true_missing_struts.csv"
    output_path = ANALYSIS_DIR / "true_missing_broken_struts.csv"
    with feature_path.open("r", newline="", encoding="utf-8") as handle:
        feature_by_id = {
            row["strut_id"]: row for row in csv.DictReader(handle)
        }
    with source_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source_path}")
        geometry_fields = [
            column
            for column in reader.fieldnames
            if column
            not in {
                "a_seed_foreground_fraction",
                "b_seed_foreground_fraction",
                "a_collar_foreground_fraction",
                "b_collar_foreground_fraction",
                "a_shared_component_voxel_count_in_cuboid",
                "b_shared_component_voxel_count_in_cuboid",
            }
        ]
        rows = []
        for row in reader:
            feature = feature_by_id.get(row["strut_id"])
            if not feature or feature["classification"] not in {"missing", "broken"}:
                continue
            rows.append(
                {
                    **{column: row[column] for column in geometry_fields},
                    "minimum_foreground_fraction": feature[
                        "central_minimum_smoothed_foreground_fraction"
                    ],
                }
            )
    fieldnames = [*geometry_fields, "minimum_foreground_fraction"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    broken_rows = [
        row
        for row in rows
        if feature_by_id[row["strut_id"]]["classification"] == "broken"
    ]
    broken_output = ANALYSIS_DIR / "broken_strut_candidates.csv"
    with broken_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(broken_rows)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote {len(broken_rows)} rows to {broken_output}")


if __name__ == "__main__":
    main()
