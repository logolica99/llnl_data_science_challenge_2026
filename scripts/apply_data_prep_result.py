#!/usr/bin/env python3
"""Advance a specimen manifest using a completed deterministic data-prep result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data_prep_handoff import (  # noqa: E402
    DataPrepHandoffError,
    apply_data_prep_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_prep_result", type=Path)
    parser.add_argument("--completion-receipt", type=Path)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    try:
        result = apply_data_prep_result(
            args.manifest,
            args.data_prep_result,
            repository_root=args.repository_root,
            completion_receipt_path=args.completion_receipt,
        )
    except (OSError, TypeError, ValueError, DataPrepHandoffError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
