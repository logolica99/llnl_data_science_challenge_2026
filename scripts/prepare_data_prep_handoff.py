#!/usr/bin/env python3
"""Verify specimen intake and emit the hash-sealed data-prep hand-off."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from data_prep_handoff import (  # noqa: E402
    DataPrepHandoffError,
    create_data_prep_handoff,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ingest_receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    try:
        result = create_data_prep_handoff(
            args.manifest,
            args.ingest_receipt,
            repository_root=args.repository_root,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, DataPrepHandoffError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
