"""Run bounded all-strut compact-profile checkpoints, then classify material loss."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAST_STRUT_ID = 18_467


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-size", type=int, default=600)
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=REPO_ROOT / "outputs/node_connectivity/strut_node_connectivity_test/analysis_config.json",
    )
    return parser.parse_args()


def checkpoint_complete(directory: Path) -> bool:
    return all(
        (directory / name).is_file()
        for name in ("all_strut_axial_profiles.npz", "connection_metrics.csv", "connection_summary.json")
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint_size < 1:
        raise ValueError("--checkpoint-size must be positive")
    environment = {**os.environ, "MPLCONFIGDIR": str(Path(os.environ["TEMP"]) / "llnl-mpl-cache")}
    test_script = REPO_ROOT / "scripts/test_strut_node_connectivity.py"
    for first_id in range(0, LAST_STRUT_ID + 1, args.checkpoint_size):
        last_id = min(first_id + args.checkpoint_size - 1, LAST_STRUT_ID)
        output_dir = (
            REPO_ROOT
            / "outputs/node_connectivity"
            / f"strut_node_connectivity_profiles_{first_id:05d}_{last_id:05d}"
        )
        if checkpoint_complete(output_dir):
            print(f"Checkpoint {first_id}-{last_id} already complete; skipping.", flush=True)
            continue
        command = [
            sys.executable,
            str(test_script),
            "--strut-id-range",
            str(first_id),
            str(last_id),
            "--failures-only",
            "--write-compact-profiles",
            "--skip-cuboid-artifacts",
            "--skip-overview",
            "--analysis-config",
            str(args.analysis_config),
            "--output-dir",
            str(output_dir),
        ]
        print(f"Running checkpoint {first_id}-{last_id}.", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

    print("All compact-profile checkpoints complete; classifying material loss.", flush=True)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/classify_material_loss_struts.py")],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
