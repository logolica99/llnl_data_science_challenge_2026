#!/usr/bin/env python3
"""Create, validate, advance, and resume the Part 2 pipeline state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.pipeline import (  # noqa: E402
    OrchestrationError,
    build_stage_receipt,
    complete_stage,
    create_pipeline_manifest,
    pipeline_status,
    record_autonomous_registration_freeze,
    resume_manual_review,
    start_stage,
    validate_pipeline_manifest,
)


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _object_file(path: Path) -> dict[str, Any]:
    value = _json_file(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _array_file(path: Path) -> list[dict[str, Any]]:
    value = _json_file(path)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected an array of JSON objects: {path}")
    return value


def _root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root used to resolve every artifact path",
    )


def _timestamp(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timestamp",
        help="explicit audit timestamp; defaults to current UTC time",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create an idempotent pipeline manifest")
    init.add_argument("--specimen-id", required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument(
        "--registration-mode",
        choices=("autonomous_v2",),
        required=True,
    )
    init.add_argument("--manifest", type=Path)
    init.add_argument(
        "--contracts-directory", type=Path, default=Path("analysis/contracts")
    )
    _timestamp(init)
    _root(init)

    validate = subparsers.add_parser("validate", help="validate state and frozen hashes")
    validate.add_argument("manifest", type=Path)
    _root(validate)

    status = subparsers.add_parser("status", help="show compact verified state")
    status.add_argument("manifest", type=Path)
    _root(status)

    start = subparsers.add_parser("start", help="write a sanitized handoff and start a stage")
    start.add_argument("manifest", type=Path)
    start.add_argument("--stage", type=int, required=True)
    start.add_argument("--inputs", type=Path, required=True)
    start.add_argument("--capabilities", type=Path, required=True)
    start.add_argument("--handoff", type=Path)
    _timestamp(start)
    _root(start)

    receipt = subparsers.add_parser(
        "build-receipt", help="build an immutable receipt for a running attempt"
    )
    receipt.add_argument("manifest", type=Path)
    receipt.add_argument("--stage", type=int, required=True)
    receipt.add_argument(
        "--terminal-state", choices=("pass", "manual_review", "halt"), required=True
    )
    receipt.add_argument("--outputs", type=Path, required=True)
    receipt.add_argument("--assertions", type=Path, required=True)
    receipt.add_argument(
        "--stage-policy",
        type=Path,
        help="closed Stage 1/2 policy decision JSON (required by those stages)",
    )
    receipt.add_argument("--failure-kind")
    receipt.add_argument("--error", type=Path)
    receipt.add_argument("--output", type=Path)
    _timestamp(receipt)
    _root(receipt)

    complete = subparsers.add_parser("complete", help="verify and apply a stage receipt")
    complete.add_argument("manifest", type=Path)
    complete.add_argument("receipt", type=Path)
    _root(complete)

    resume = subparsers.add_parser("resume", help="explicitly resolve manual review")
    resume.add_argument("manifest", type=Path)
    resume.add_argument("--stage", type=int, required=True)
    resume.add_argument("--resolution", type=Path, required=True)
    resume.add_argument("--reason", required=True)
    _timestamp(resume)
    _root(resume)

    freeze = subparsers.add_parser(
        "freeze-registration", help="seal autonomous CT-only Stage 1 artifacts"
    )
    freeze.add_argument("manifest", type=Path)
    freeze.add_argument("--artifacts", type=Path, required=True)
    freeze.add_argument("--output", type=Path)
    _timestamp(freeze)
    _root(freeze)

    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    root = args.repository_root
    if args.command == "init":
        return create_pipeline_manifest(
            repository_root=root,
            specimen_id=args.specimen_id,
            config_path=args.config,
            registration_mode=args.registration_mode,
            manifest_path=args.manifest,
            contracts_directory=args.contracts_directory,
            timestamp=args.timestamp,
        )
    if args.command == "validate":
        manifest = validate_pipeline_manifest(
            args.manifest,
            repository_root=root,
            verify_artifacts=True,
        )
        return {
            "status": "ok",
            "manifest_sha256": manifest["manifest_sha256"],
            "pipeline_state": manifest["pipeline_state"],
        }
    if args.command == "status":
        return pipeline_status(args.manifest, repository_root=root)
    if args.command == "start":
        return start_stage(
            args.manifest,
            args.stage,
            input_artifacts=_array_file(args.inputs),
            capability_inventory=_object_file(args.capabilities),
            repository_root=root,
            handoff_path=args.handoff,
            timestamp=args.timestamp,
        )
    if args.command == "build-receipt":
        return build_stage_receipt(
            args.manifest,
            args.stage,
            terminal_state=args.terminal_state,
            output_artifacts=_array_file(args.outputs),
            assertions=_object_file(args.assertions),
            stage_policy=(
                _object_file(args.stage_policy) if args.stage_policy else None
            ),
            failure_kind=args.failure_kind,
            error=_object_file(args.error) if args.error else None,
            repository_root=root,
            output_path=args.output,
            timestamp=args.timestamp,
        )
    if args.command == "complete":
        return complete_stage(
            args.manifest, args.receipt, repository_root=root
        )
    if args.command == "resume":
        return resume_manual_review(
            args.manifest,
            args.stage,
            resolution_artifact=_object_file(args.resolution),
            reason=args.reason,
            repository_root=root,
            timestamp=args.timestamp,
        )
    if args.command == "freeze-registration":
        return record_autonomous_registration_freeze(
            args.manifest,
            frozen_artifacts=_array_file(args.artifacts),
            repository_root=root,
            output_path=args.output,
            timestamp=args.timestamp,
        )
    raise AssertionError(f"Unhandled command: {args.command}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        result = _dispatch(args)
    except (OSError, TypeError, ValueError, OrchestrationError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "fallback_used": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
