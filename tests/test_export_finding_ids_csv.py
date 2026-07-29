import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_finding_ids_csv import (  # noqa: E402
    OUTPUT_FIELDS,
    export_finding_ids,
)


class ExportFindingIdsCsvTests(unittest.TestCase):
    def test_combines_json_findings_and_leaves_non_id_fields_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "thin.json"
            second = root / "bent.json"
            first.write_text(json.dumps({
                "findings": [{"strut_id": 9}, {"strut_id": 2}],
            }), encoding="utf-8")
            second.write_text(json.dumps({
                "findings": [{"strut_id": 9}, {"strut_id": 12}],
            }), encoding="utf-8")
            output = root / "combined.csv"
            _, count = export_finding_ids([first, second], output)
            with output.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, OUTPUT_FIELDS)
            self.assertEqual(count, 3)
            self.assertEqual(
                [int(row["strut_id"]) for row in rows],
                [2, 9, 12],
            )
            self.assertTrue(all(
                not value
                for row in rows
                for key, value in row.items()
                if key != "strut_id"
            ))


if __name__ == "__main__":
    unittest.main()
