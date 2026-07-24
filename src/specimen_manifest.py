"""Validation and provenance helpers for Part 2 specimen manifests.

The manifest is the only production source for specimen-specific paths,
threshold recipes, coordinate conventions, and analysis budgets.  Design notes
may explain those choices, but runtime code must not parse prose for values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator, FormatChecker


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = (
    REPOSITORY_ROOT / "analysis" / "schema" / "specimen_manifest.schema.json"
)
DERIVED_SECTIONS = (
    "graph_summary",
    "voxel_spacing",
    "segmentation_result",
    "registration_result",
)
ANALYSIS_READY = "analysis_ready"


class ManifestValidationError(ValueError):
    """Raised when a specimen manifest fails schema or semantic validation."""


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value deterministically for hashing."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of a canonical JSON representation."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{path} must contain a JSON object")
    return value


def topology_summary(graph_path: Path) -> dict[str, int | str]:
    """Return counts and a coordinate-independent topology hash for a graph.

    Version 1 hashes sorted junction IDs, undirected strut endpoints keyed by
    strut ID, and each unit cell's sorted strut membership.  Coordinates and
    other metrology fields are intentionally excluded so nominal and aligned
    copies of the same topology compare equal.
    """
    graph = load_json(graph_path)
    try:
        junctions = graph["junctions"]
        struts = graph["struts"]
        unit_cells = graph["unit_cells"]
        topology = {
            "junction_ids": sorted(int(item["id"]) for item in junctions),
            "struts": sorted(
                [
                    int(item["id"]),
                    min(int(item["junction0"]), int(item["junction1"])),
                    max(int(item["junction0"]), int(item["junction1"])),
                ]
                for item in struts
            ),
            "unit_cells": sorted(
                [int(item["id"]), sorted(int(value) for value in item["struts"])]
                for item in unit_cells
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestValidationError(
            f"{graph_path} is not a supported lattice graph: {exc}"
        ) from exc
    return {
        "junction_count": len(junctions),
        "strut_count": len(struts),
        "unit_cell_count": len(unit_cells),
        "topology_sha256": canonical_json_sha256(topology),
    }


def _format_schema_error(error: Any) -> str:
    location = ".".join(str(part) for part in error.absolute_path) or "<root>"
    return f"{location}: {error.message}"


def _artifact_items(manifest: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name, artifact in manifest["inputs"].items():
        if name == "ct_metadata":
            continue
        yield name, artifact


def _analysis_readiness_errors(manifest: dict[str, Any]) -> list[str]:
    """Return deterministic reasons a manifest is unsafe for downstream use."""
    errors: list[str] = []
    if manifest["lifecycle_state"] != ANALYSIS_READY:
        errors.append(
            f"lifecycle_state is {manifest['lifecycle_state']!r}, expected "
            f"{ANALYSIS_READY!r}"
        )
        return errors
    if manifest["unresolved_fields"]:
        errors.append("analysis_ready manifest has unresolved_fields")
    aligned_graph = manifest["inputs"].get("aligned_graph")
    if aligned_graph is None:
        errors.append("analysis_ready manifest has no aligned graph")

    required_sections = set(DERIVED_SECTIONS)
    missing_sections = sorted(required_sections - set(manifest["derived"]))
    if missing_sections:
        errors.append("missing derived sections: " + ", ".join(missing_sections))
        return errors

    segmentation = manifest["derived"]["segmentation_result"]["values"]
    if not segmentation["overall_pass"]:
        errors.append("segmentation_result.overall_pass is false")

    registration = manifest["derived"]["registration_result"]["values"]
    failed_registration_gates = [
        field
        for field in (
            "overall_pass",
            "local_recenter_complete",
            "roi_gate_pass",
            "metrology_gate_pass",
        )
        if not registration[field]
    ]
    if failed_registration_gates:
        errors.append(
            "registration_result failed gates: "
            + ", ".join(failed_registration_gates)
        )
    mode = manifest["analysis_parameters"]["registration"]["mode"]
    expected_state = "input" if mode == "challenge_aligned_json" else "derived"
    if registration["aligned_graph_state"] != expected_state:
        errors.append(
            "registration_result.aligned_graph_state is "
            f"{registration['aligned_graph_state']!r}, expected {expected_state!r}"
        )
    return errors


def require_analysis_ready(
    manifest_path: Path,
    *,
    consumer: str,
    schema_path: Path = DEFAULT_SCHEMA,
    repository_root: Path = REPOSITORY_ROOT,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Validate and return a manifest only when downstream analysis is allowed."""
    try:
        validate_manifest(
            manifest_path,
            schema_path=schema_path,
            repository_root=repository_root,
            verify_files=verify_files,
            required_lifecycle=ANALYSIS_READY,
        )
    except ManifestValidationError as exc:
        raise ManifestValidationError(f"{consumer} rejected manifest: {exc}") from exc
    return load_json(manifest_path.resolve())


def validate_manifest(
    manifest_path: Path,
    *,
    schema_path: Path = DEFAULT_SCHEMA,
    repository_root: Path = REPOSITORY_ROOT,
    verify_files: bool = False,
    require_all_files: bool = False,
    required_lifecycle: str | None = None,
) -> list[str]:
    """Validate one manifest and return non-fatal file-availability warnings."""
    manifest_path = manifest_path.resolve()
    schema = load_json(schema_path.resolve())
    manifest = load_json(manifest_path)

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(
        validator.iter_errors(manifest),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if schema_errors:
        details = "\n".join(f"- {_format_schema_error(error)}" for error in schema_errors)
        raise ManifestValidationError(f"{manifest_path} failed JSON Schema:\n{details}")

    errors: list[str] = []
    warnings: list[str] = []
    expected_config_hash = canonical_json_sha256(manifest["analysis_parameters"])
    if manifest["analysis_parameters_sha256"] != expected_config_hash:
        errors.append(
            "analysis_parameters_sha256 does not match canonical analysis_parameters"
        )

    input_hashes = {artifact["sha256"] for _, artifact in _artifact_items(manifest)}
    for section in DERIVED_SECTIONS:
        record = manifest["derived"].get(section)
        if record is None:
            continue
        provenance = record["provenance"]
        if provenance["config_sha256"] != expected_config_hash:
            errors.append(f"derived.{section} uses a stale config_sha256")
        unknown_hashes = sorted(set(provenance["input_sha256"]) - input_hashes)
        if unknown_hashes:
            errors.append(
                f"derived.{section} references unknown input hashes: "
                + ", ".join(unknown_hashes)
            )

    mode = manifest["analysis_parameters"]["registration"]["mode"]
    intake = manifest.get("intake")
    if intake is not None:
        if intake["registration_mode_selection"]["mode"] != mode:
            errors.append(
                "intake registration mode differs from analysis_parameters"
            )
        for input_name, inspection_name in (
            ("cad", "cad_inspection"),
            ("design_graph", "graph_inspection"),
        ):
            artifact = manifest["inputs"][input_name]
            inspection = intake[inspection_name]
            if inspection["path"] != artifact["path"]:
                errors.append(
                    f"intake.{inspection_name}.path differs from inputs.{input_name}"
                )
            if inspection["sha256"] != artifact["sha256"]:
                errors.append(
                    f"intake.{inspection_name}.sha256 differs from inputs.{input_name}"
                )
    aligned_artifact = manifest["inputs"].get("aligned_graph")
    aligned_role = aligned_artifact["role"] if aligned_artifact else None
    if mode == "challenge_aligned_json" and aligned_role != "aligned_graph":
        errors.append(
            "challenge_aligned_json mode requires inputs.aligned_graph.role=aligned_graph"
        )
    if (
        mode == "autonomous_v2"
        and aligned_role is not None
        and aligned_role != "derived_aligned_graph"
    ):
        errors.append(
            "autonomous_v2 mode requires an explicitly derived aligned graph"
        )

    graph_summary = manifest["derived"].get("graph_summary")
    if intake is not None and graph_summary is not None:
        inspection_summary = {
            key: intake["graph_inspection"][key]
            for key in (
                "junction_count",
                "strut_count",
                "unit_cell_count",
                "topology_sha256",
            )
        }
        if inspection_summary != graph_summary["values"]:
            errors.append(
                "intake.graph_inspection differs from derived.graph_summary.values"
            )
    if graph_summary is not None and "aligned_values" in graph_summary:
        nominal = graph_summary["values"]
        aligned = graph_summary["aligned_values"]
        if nominal != aligned:
            errors.append("nominal and aligned graph topology summaries differ")

    if manifest["lifecycle_state"] != "provisional":
        coordinates = manifest["analysis_parameters"]["coordinates"]
        unresolved_coordinates = sorted(
            key for key, value in coordinates.items() if value == "unknown"
        )
        if manifest["inputs"]["ct_metadata"]["array_axes"] == "unknown":
            unresolved_coordinates.append("inputs.ct_metadata.array_axes")
        if unresolved_coordinates:
            errors.append(
                "non-provisional manifest has unresolved coordinate fields: "
                + ", ".join(unresolved_coordinates)
            )

    if (
        required_lifecycle is not None
        and manifest["lifecycle_state"] != required_lifecycle
    ):
        errors.append(
            f"consumer requires lifecycle_state={required_lifecycle}, found "
            f"{manifest['lifecycle_state']}"
        )
    if manifest["lifecycle_state"] == ANALYSIS_READY:
        errors.extend(_analysis_readiness_errors(manifest))

    if verify_files:
        for name, artifact in _artifact_items(manifest):
            path = Path(artifact["path"])
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"inputs.{name}.path must stay within the repository")
                continue
            resolved = repository_root / path
            if not resolved.is_file():
                message = f"inputs.{name} is unavailable locally: {path}"
                if require_all_files or artifact["retention"] == "committed":
                    errors.append(message)
                else:
                    warnings.append(message)
                continue
            actual_hash = sha256_file(resolved)
            if actual_hash != artifact["sha256"]:
                errors.append(
                    f"inputs.{name} SHA-256 mismatch: expected "
                    f"{artifact['sha256']}, found {actual_hash}"
                )

        for graph_name in ("design_graph", "aligned_graph"):
            artifact = manifest["inputs"].get(graph_name)
            if artifact is None:
                continue
            path = repository_root / artifact["path"]
            if not path.is_file():
                continue
            actual = topology_summary(path)
            graph_summary = manifest["derived"].get("graph_summary")
            if graph_summary is None:
                continue
            expected = (
                graph_summary.get("aligned_values", graph_summary["values"])
                if graph_name == "aligned_graph"
                else graph_summary["values"]
            )
            if actual != expected:
                errors.append(
                    f"inputs.{graph_name} topology differs from derived.graph_summary"
                )

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise ManifestValidationError(
            f"{manifest_path} failed semantic validation:\n{details}"
        )
    return warnings


def manifest_paths(repository_root: Path = REPOSITORY_ROOT) -> list[Path]:
    """Discover committed specimen manifests in deterministic order."""
    return sorted(
        repository_root.glob("analysis/*/config/specimen_manifest.json")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="*",
        type=Path,
        help="manifest paths; defaults to analysis/*/config/specimen_manifest.json",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="verify hashes for locally available input artifacts",
    )
    parser.add_argument(
        "--require-all-files",
        action="store_true",
        help="fail when an external or regenerable input is unavailable",
    )
    parser.add_argument(
        "--require-analysis-ready",
        action="store_true",
        help="reject manifests that downstream analysis must not consume",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.manifests or manifest_paths()
    if not paths:
        raise SystemExit("No specimen manifests found")
    failed = False
    for path in paths:
        try:
            warnings = validate_manifest(
                path,
                verify_files=args.verify_files,
                require_all_files=args.require_all_files,
                required_lifecycle=(
                    ANALYSIS_READY if args.require_analysis_ready else None
                ),
            )
            print(f"PASS {path}")
            for warning in warnings:
                print(f"WARN {warning}")
        except (ManifestValidationError, OSError, json.JSONDecodeError) as exc:
            failed = True
            print(f"FAIL {path}\n{exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
