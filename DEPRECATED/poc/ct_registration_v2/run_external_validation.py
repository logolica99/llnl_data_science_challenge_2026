#!/usr/bin/env python3
"""Aggregate independent held-out cases and enforce generalization coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from registration_core import load_json, write_json


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=script_dir / "external_cases.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config.default.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "results/external_validation_summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = args.cases.resolve()
    base = cases_path.parent
    manifest = load_json(cases_path)
    config = load_json(args.config.resolve())
    params = config["external_validation"]
    records = []
    ct_hashes = set()
    geometries = set()
    passed = 0
    for case in manifest.get("cases", []):
        fit_manifest_path = (base / case["fit_manifest"]).resolve()
        validation_path = (base / case["validation_result"]).resolve()
        record = {
            "name": case["name"],
            "geometry_id": case["geometry_id"],
            "fit_manifest": str(fit_manifest_path),
            "validation_result": str(validation_path),
            "available": fit_manifest_path.is_file()
            and validation_path.is_file(),
            "pass": False,
        }
        if record["available"]:
            fit = load_json(fit_manifest_path)
            validation = load_json(validation_path)
            if fit.get("ground_truth_used_for_fit") is not False:
                raise RuntimeError(
                    f"Case {case['name']} does not prove ground-truth isolation"
                )
            ct_hash = fit["input_artifacts"]["ct"]["sha256"]
            ct_hashes.add(ct_hash)
            geometries.add(case["geometry_id"])
            record["ct_sha256"] = ct_hash
            record["validation_overall_pass"] = bool(
                validation.get("overall_pass", False)
            )
            record["pass"] = record["validation_overall_pass"]
            passed += int(record["pass"])
        records.append(record)

    available = [record for record in records if record["available"]]
    pass_fraction = passed / len(available) if available else 0.0
    gates = {
        "enough_independent_scans": bool(
            len(ct_hashes) >= int(params["minimum_independent_scans"])
        ),
        "enough_distinct_geometries": bool(
            len(geometries) >= int(params["minimum_distinct_geometries"])
        ),
        "case_pass_fraction": bool(
            pass_fraction >= float(params["minimum_case_pass_fraction"])
        ),
    }
    payload = {
        "schema_version": 1,
        "available_case_count": len(available),
        "independent_scan_count": len(ct_hashes),
        "distinct_geometry_count": len(geometries),
        "case_pass_fraction": pass_fraction,
        "requirements": params,
        "gates": gates,
        "overall_pass": bool(all(gates.values())),
        "status": (
            "external_generalization_supported"
            if all(gates.values())
            else "insufficient_external_evidence"
        ),
        "cases": records,
    }
    write_json(args.output.resolve(), payload)
    print(
        f"External evidence: scans={len(ct_hashes)}, "
        f"geometries={len(geometries)}, pass_fraction={pass_fraction:.1%}, "
        f"overall={'PASS' if payload['overall_pass'] else 'INCOMPLETE'}"
    )
    print(f"Wrote {args.output.resolve()}")
    return 0 if payload["overall_pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
