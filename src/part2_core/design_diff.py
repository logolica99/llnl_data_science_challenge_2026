"""Deterministic design-space orientation and STL deletion labeling.

The implementation deliberately uses only the nominal graph and CAD meshes.
It never opens CT, aligned-coordinate, segmentation, or defect-analysis data.
Binary STL triangles are memory mapped and each mesh is released before the
next mesh is opened.
"""

from __future__ import annotations

import gc
import itertools
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from jsonschema import Draft202012Validator
from scipy.spatial import cKDTree

from .artifacts import (
    read_json_object,
    sha256_file,
    sha256_json,
    write_json_atomic,
    write_text_atomic,
)
from .lattice import LatticeGraph, load_lattice_json


ORIENTATION_SCHEMA_VERSION = "part2-cad-graph-orientation/1.0.0"
LABEL_SCHEMA_VERSION = "part2-design-labels/1.0.0"
TRANSFORM_DECLARATION_SCHEMA_VERSION = (
    "part2-graph-to-stl-transform-declaration/1.0.0"
)
TRANSFORM_CONVENTION = (
    "stl_mm = scale * ((design_xyz - design_center) @ rotation.T) + translation_mm"
)
AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT = 2.28
STAGE1_POLICY_SCHEMA_VERSION = "part2-stage1-design-diff-policy/1.0.0"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_POLICY_PATH = (
    REPOSITORY_ROOT / "analysis" / "contracts" / "stage1_design_diff_policy.json"
)
DESIGN_DIFF_CONTRACT_PATH = (
    REPOSITORY_ROOT / "analysis" / "contracts" / "design_diff.json"
)
DEFAULT_STAGE1_POLICY_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "schema"
    / "stage1_design_diff_policy.schema.json"
)
DEFAULT_TRANSFORM_DECLARATION_SCHEMA_PATH = (
    REPOSITORY_ROOT
    / "analysis"
    / "schema"
    / "graph_to_stl_transform.schema.json"
)
# These tolerances are part of the deterministic Stage 1 policy.  They are
# emitted and hashed in every orientation report so a caller cannot silently
# relax transform validation between attempts.
DECLARED_TRANSFORM_TOLERANCES = {
    "scale_absolute_mm_per_design_unit": 1e-6,
    "design_center_absolute_design_unit": 1e-9,
    "rotation_orthonormal_absolute": 1e-8,
    "rotation_determinant_absolute": 1e-8,
    "translation_search_match_absolute_mm": 0.05,
    # CAD tessellation produces small orientation-dependent sampling noise even
    # when the underlying lattice is symmetric.  Hypotheses inside this floor
    # remain equivalent; sort order must never turn that noise into authority.
    "geometry_equivalence_absolute_mm": 0.01,
    "maximum_edge_support_distance_mm": 0.50,
    "maximum_edge_support_p99_minus_p01_mm": 0.05,
}
STL_TRIANGLE_DTYPE = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)
REFERENCE_COUNTS = {"nodes": 10_206, "edges": 18_468, "cells": 729}
REFERENCE_DELETIONS = {"0p1": 18, "0p5": 93, "1p0": 186}


class Stage1PolicyValidationError(ValueError):
    """A hard failure in the hash-bound Stage 1 execution policy."""


class DeclaredTransformValidationError(ValueError):
    """A deterministic hard failure in an immutable transform declaration."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = dict(metadata or {})


def _declaration_error(
    code: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> DeclaredTransformValidationError:
    return DeclaredTransformValidationError(code, message, metadata=metadata)


def _strict_keys(
    value: Any,
    expected: set[str],
    *,
    field: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _declaration_error(
            "declaration_schema_invalid",
            f"{field} must be an object",
            metadata=metadata,
        )
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise _declaration_error(
            "declaration_schema_invalid",
            f"{field} has missing keys {missing} and unexpected keys {extra}",
            metadata=metadata,
        )
    return value


def _nonempty_identifier(
    value: Any,
    *,
    field: str,
    metadata: Mapping[str, Any],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _declaration_error(
            "declaration_schema_invalid",
            f"{field} must be a non-empty string",
            metadata=metadata,
        )
    return value


def _sha256_string(
    value: Any,
    *,
    field: str,
    metadata: Mapping[str, Any],
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _declaration_error(
            "declaration_schema_invalid",
            f"{field} must be a lowercase SHA-256 hex digest",
            metadata=metadata,
        )
    return value


def _policy_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage1PolicyValidationError(
            f"{field} must be a lowercase SHA-256 hex digest"
        )
    return value


def _policy_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_policy_numbers_are_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_policy_numbers_are_finite(item) for item in value.values())
    return False


def load_stage1_policy(
    path: str | Path,
    *,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    """Load a closed Stage 1 policy bound to an exact file digest.

    Production policies additionally have non-overridable scientific values in
    code. Test-fixture policies may change only fixture counts and bounded
    algorithm settings; every accepted value remains frozen by the supplied
    artifact digest.
    """

    resolved = Path(path).expanduser().resolve()
    expected_hash = _policy_sha256(
        expected_artifact_sha256,
        field="stage1_policy_sha256",
    )
    try:
        actual_hash = sha256_file(resolved)
    except OSError as exc:
        raise Stage1PolicyValidationError(
            f"Unable to read Stage 1 policy artifact: {resolved}"
        ) from exc
    if actual_hash != expected_hash:
        raise Stage1PolicyValidationError(
            "Stage 1 policy file SHA-256 does not match its handoff binding"
        )
    try:
        document = read_json_object(resolved)
        schema = read_json_object(DEFAULT_STAGE1_POLICY_SCHEMA_PATH)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Stage1PolicyValidationError(
            "Stage 1 policy or its JSON Schema is unreadable"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in errors
        )
        raise Stage1PolicyValidationError(
            f"Stage 1 policy failed its closed JSON Schema: {details}"
        )
    if not _policy_numbers_are_finite(document):
        raise Stage1PolicyValidationError(
            "Stage 1 policy contains a non-finite or unsupported value"
        )
    orientation = document["orientation_verification"]
    labeling = document["deletion_labeling"]
    if float(orientation["sample_start"]) > float(orientation["sample_end"]):
        raise Stage1PolicyValidationError(
            "Stage 1 orientation sample_start exceeds sample_end"
        )
    if float(labeling["sample_start"]) > float(labeling["sample_end"]):
        raise Stage1PolicyValidationError(
            "Stage 1 labeling sample_start exceeds sample_end"
        )
    if orientation["scale_candidates_mm_per_design_unit"] != [
        AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
    ]:
        raise Stage1PolicyValidationError(
            "Stage 1 policy must freeze the source-backed 2.28 mm/design-unit scale"
        )
    if orientation["verification_tolerances"] != DECLARED_TRANSFORM_TOLERANCES:
        raise Stage1PolicyValidationError(
            "Stage 1 policy verification tolerances differ from production constants"
        )
    if document["intended_use"] == "production":
        expected_orientation = {
            "sample_count": 9,
            "sample_start": 0.40,
            "sample_end": 0.60,
            "maximum_ranked_edges": 2_048,
            "scale_candidates_mm_per_design_unit": [
                AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT
            ],
            "ambiguity_absolute_mm": 1e-4,
            "ambiguity_relative_fraction": 1e-3,
            "reflection_authorized": False,
            "expected_counts": REFERENCE_COUNTS,
            "verification_tolerances": DECLARED_TRANSFORM_TOLERANCES,
        }
        expected_labeling = {
            "sample_count": 9,
            "sample_start": 0.40,
            "sample_end": 0.60,
            "radius_margin_mm": 0.03,
            "radius_rounding_mm": 0.01,
            "split_seed": 20260723,
            "development_fraction": 0.30,
            "x_bins": 5,
            "z_shells": 3,
            "expected_deletions": REFERENCE_DELETIONS,
        }
        if orientation != expected_orientation or labeling != expected_labeling:
            raise Stage1PolicyValidationError(
                "Production Stage 1 policy conflicts with non-overridable constants"
            )
    return {
        "document": document,
        "artifact_path": str(resolved),
        "artifact_sha256": actual_hash,
        "config_sha256": sha256_json(document),
    }


def load_production_stage1_policy() -> dict[str, Any]:
    """Load the one policy path and digest authorized by the Stage 1 contract."""

    try:
        contract = read_json_object(DESIGN_DIFF_CONTRACT_PATH)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise Stage1PolicyValidationError(
            "Stage 1 design-diff contract is unreadable"
        ) from exc
    record = contract.get("stage_policy")
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "schema_version",
        "policy_id",
    }:
        raise Stage1PolicyValidationError(
            "Stage 1 contract has no closed stage_policy binding"
        )
    expected_relative = DEFAULT_STAGE1_POLICY_PATH.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    if record["path"] != expected_relative:
        raise Stage1PolicyValidationError(
            "Stage 1 contract policy path differs from the fixed repository path"
        )
    policy = load_stage1_policy(
        DEFAULT_STAGE1_POLICY_PATH,
        expected_artifact_sha256=_policy_sha256(
            record["sha256"], field="contract.stage_policy.sha256"
        ),
    )
    document = policy["document"]
    if (
        record["schema_version"] != document["schema_version"]
        or record["policy_id"] != document["policy_id"]
        or document["intended_use"] != "production"
    ):
        raise Stage1PolicyValidationError(
            "Stage 1 contract policy identity or production intent mismatch"
        )
    return policy


def _finite_scalar(
    value: Any,
    *,
    field: str,
    metadata: Mapping[str, Any],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _declaration_error(
            "transform_non_finite_or_malformed",
            f"{field} must be a finite number",
            metadata=metadata,
        )
    result = float(value)
    if not math.isfinite(result):
        raise _declaration_error(
            "transform_non_finite_or_malformed",
            f"{field} must be finite",
            metadata=metadata,
        )
    return result


def _finite_vector(
    value: Any,
    *,
    shape: tuple[int, ...],
    field: str,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _declaration_error(
            "transform_non_finite_or_malformed",
            f"{field} must contain only finite numbers",
            metadata=metadata,
        ) from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise _declaration_error(
            "transform_non_finite_or_malformed",
            f"{field} must have shape {shape} with only finite numbers",
            metadata=metadata,
        )
    return array


def _binary_stl_centroids(path: str | Path) -> tuple[np.ndarray, int]:
    """Load triangle centroids without materializing a processed mesh."""

    resolved = Path(path).expanduser().resolve()
    with resolved.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise ValueError(f"Binary STL header is incomplete: {resolved}")
    triangle_count = int(struct.unpack_from("<I", header, 80)[0])
    if triangle_count <= 0:
        raise ValueError(f"Binary STL contains no triangles: {resolved}")
    expected_bytes = 84 + triangle_count * STL_TRIANGLE_DTYPE.itemsize
    if resolved.stat().st_size != expected_bytes:
        raise ValueError(
            f"Binary STL size/header mismatch for {resolved}; ASCII or corrupt STL "
            "input is not accepted by the memory-aware production path"
        )
    triangles = np.memmap(
        resolved,
        dtype=STL_TRIANGLE_DTYPE,
        mode="r",
        offset=84,
        shape=(triangle_count,),
    )
    centroids = np.asarray(triangles["vertices"].mean(axis=1), dtype=np.float32)
    del triangles
    return centroids, triangle_count


def _axis_orientations(*, include_reflections: bool = False) -> list[np.ndarray]:
    hypotheses: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3, dtype=np.float64)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rotation = np.diag(signs) @ base
            if include_reflections or np.linalg.det(rotation) > 0.0:
                hypotheses.append(rotation)
    hypotheses.sort(key=lambda item: tuple(item.ravel().tolist()))
    return hypotheses


def _right_handed_axis_rotations() -> list[np.ndarray]:
    return _axis_orientations(include_reflections=False)


def _edge_core_samples(
    graph: LatticeGraph,
    *,
    sample_count: int,
    sample_start: float,
    sample_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 1 or not 0.0 < sample_start <= sample_end < 1.0:
        raise ValueError("Invalid junction-trimmed centerline sampling configuration")
    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    fractions = np.linspace(sample_start, sample_end, sample_count, dtype=np.float64)
    samples = (
        starts[:, None, :] * (1.0 - fractions[None, :, None])
        + ends[:, None, :] * fractions[None, :, None]
    )
    design_center = (
        graph.node_positions_xyz.min(axis=0)
        + graph.node_positions_xyz.max(axis=0)
    ) / 2.0
    return samples - design_center, design_center


def _scale_hypotheses(
    design_span: np.ndarray,
    centroids: np.ndarray,
    explicit: Sequence[float] | None,
) -> list[float]:
    if explicit is not None:
        values = sorted({float(value) for value in explicit})
    else:
        # Scale proposal only: discard the largest axis ratio because this
        # specimen has known non-lattice tabs/plates along STL Y. Orientation
        # and translation are still ranked exclusively by lattice support, so
        # a whole-mesh box never becomes the registration objective.
        mesh_span = np.ptp(centroids, axis=0)
        valid = np.asarray(design_span, dtype=np.float64) > 0
        ratios = np.sort(mesh_span[valid] / np.asarray(design_span)[valid])
        if ratios.size < 2:
            raise ValueError("Scale proposal requires at least two finite spans")
        center = float(np.median(ratios[:2]))
        values = [
            center * (1.0 + offset)
            for offset in np.linspace(-0.01, 0.01, 21)
        ]
        # The challenge source declares a 4.56 mm unit cell and the graph
        # encodes two design units per cell.  Include that centerline pitch as
        # a candidate.  Do not divide the STL material envelope by the graph
        # centerline span: that folds the finite tube radius into the scale.
        values.append(AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT)
        values = sorted({round(value, 7) for value in values if value > 0})
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Scale hypotheses must be finite and positive")
    return values


def _translation_hypotheses(centroids: np.ndarray) -> list[np.ndarray]:
    low, high = np.quantile(centroids, [0.01, 0.99], axis=0)
    proposals = [
        np.zeros(3, dtype=np.float64),
        np.asarray((low + high) / 2.0, dtype=np.float64),
        np.asarray(np.median(centroids, axis=0), dtype=np.float64),
    ]
    unique: dict[tuple[float, float, float], np.ndarray] = {}
    for value in proposals:
        unique[tuple(np.round(value, 7).tolist())] = value
    return [unique[key] for key in sorted(unique)]


def _declaration_metadata(document: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(document, Mapping):
        return {}
    return {
        key: document.get(key)
        for key in (
            "declaration_id",
            "source_id",
            "provenance_id",
            "specimen_id",
            "design_id",
            "canonical_declaration_sha256",
        )
        if isinstance(document.get(key), str)
    }


def _validate_transform_values(
    transform: Any,
    *,
    expected_design_center: np.ndarray,
    require_authoritative_scale: bool,
    allow_reflection: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    transform = _strict_keys(
        transform,
        {
            "convention",
            "design_center",
            "scale_mm_per_design_unit",
            "rotation_matrix",
            "translation_mm",
            "reflection_permitted",
        },
        field="transform",
        metadata=metadata,
    )
    if transform["convention"] != TRANSFORM_CONVENTION:
        raise _declaration_error(
            "transform_convention_mismatch",
            "Transform convention does not match the frozen Stage 1 convention",
            metadata=metadata,
        )
    center = _finite_vector(
        transform["design_center"],
        shape=(3,),
        field="transform.design_center",
        metadata=metadata,
    )
    if not np.allclose(
        center,
        expected_design_center,
        rtol=0.0,
        atol=DECLARED_TRANSFORM_TOLERANCES[
            "design_center_absolute_design_unit"
        ],
    ):
        raise _declaration_error(
            "design_center_mismatch",
            "Declared design center does not match the nominal graph",
            metadata=metadata,
        )
    scale = _finite_scalar(
        transform["scale_mm_per_design_unit"],
        field="transform.scale_mm_per_design_unit",
        metadata=metadata,
    )
    if scale <= 0.0:
        raise _declaration_error(
            "transform_scale_invalid",
            "Declared transform scale must be positive",
            metadata=metadata,
        )
    if require_authoritative_scale and not math.isclose(
        scale,
        AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
        rel_tol=0.0,
        abs_tol=DECLARED_TRANSFORM_TOLERANCES[
            "scale_absolute_mm_per_design_unit"
        ],
    ):
        raise _declaration_error(
            "authoritative_scale_mismatch",
            "Declared scale disagrees with the source-backed 2.28 mm/design-unit pitch",
            metadata=metadata,
        )
    rotation = _finite_vector(
        transform["rotation_matrix"],
        shape=(3, 3),
        field="transform.rotation_matrix",
        metadata=metadata,
    )
    orthonormal_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
    )
    if orthonormal_error > DECLARED_TRANSFORM_TOLERANCES[
        "rotation_orthonormal_absolute"
    ]:
        raise _declaration_error(
            "rotation_not_orthonormal",
            "Declared rotation/reflection matrix is not orthonormal",
            metadata=metadata,
        )
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(
        abs(determinant),
        1.0,
        rel_tol=0.0,
        abs_tol=DECLARED_TRANSFORM_TOLERANCES[
            "rotation_determinant_absolute"
        ],
    ):
        raise _declaration_error(
            "rotation_determinant_invalid",
            "Declared rotation/reflection determinant must have magnitude one",
            metadata=metadata,
        )
    reflection_permitted = transform["reflection_permitted"]
    if not isinstance(reflection_permitted, bool):
        raise _declaration_error(
            "declaration_schema_invalid",
            "transform.reflection_permitted must be boolean",
            metadata=metadata,
        )
    if reflection_permitted and not allow_reflection:
        raise _declaration_error(
            "reflection_not_authorized",
            "The declaration permits reflection but the Stage 1 contract does not",
            metadata=metadata,
        )
    if determinant < 0.0 and not (allow_reflection and reflection_permitted):
        raise _declaration_error(
            "wrong_handedness",
            "A left-handed transform requires explicit declaration and contract "
            "permission",
            metadata=metadata,
        )
    translation = _finite_vector(
        transform["translation_mm"],
        shape=(3,),
        field="transform.translation_mm",
        metadata=metadata,
    )
    return {
        "convention": TRANSFORM_CONVENTION,
        "design_center": center.tolist(),
        "scale_mm_per_design_unit": scale,
        "rotation_matrix": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "reflection_permitted": reflection_permitted,
        "determinant": determinant,
        "orthonormality_max_absolute_error": orthonormal_error,
    }


def _load_and_validate_declared_transform(
    path: str | Path,
    *,
    expected_artifact_sha256: str | None,
    specimen_id: str | None,
    design_id: str | None,
    nominal_graph_sha256: str,
    full_design_stl_sha256: str,
    design_center: np.ndarray,
    allow_reflection: bool,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    metadata: dict[str, Any] = {"artifact_path": str(resolved)}
    if expected_artifact_sha256 is None:
        raise _declaration_error(
            "declaration_artifact_hash_missing",
            "A declared transform must be bound to its immutable file SHA-256",
            metadata=metadata,
        )
    expected_hash = _sha256_string(
        expected_artifact_sha256,
        field="declared_transform_sha256",
        metadata=metadata,
    )
    try:
        actual_hash = sha256_file(resolved)
    except OSError as exc:
        raise _declaration_error(
            "declaration_unreadable",
            f"Unable to read declared transform artifact: {resolved}",
            metadata=metadata,
        ) from exc
    metadata.update(
        {
            "artifact_sha256": actual_hash,
            "expected_artifact_sha256": expected_hash,
        }
    )
    if actual_hash != expected_hash:
        raise _declaration_error(
            "declaration_artifact_hash_mismatch",
            "Declared transform file SHA-256 does not match the intake artifact hash",
            metadata=metadata,
        )
    try:
        document = read_json_object(resolved)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _declaration_error(
            "declaration_unreadable",
            f"Declared transform is not a readable JSON object: {resolved}",
            metadata=metadata,
        ) from exc
    metadata.update(_declaration_metadata(document))
    document = _strict_keys(
        document,
        {
            "schema_version",
            "declaration_id",
            "source_id",
            "provenance_id",
            "specimen_id",
            "design_id",
            "nominal_graph_sha256",
            "full_design_stl_sha256",
            "transform",
            "canonical_declaration_sha256",
        },
        field="declaration",
        metadata=metadata,
    )
    if document["schema_version"] != TRANSFORM_DECLARATION_SCHEMA_VERSION:
        raise _declaration_error(
            "declaration_schema_incompatible",
            "Declared transform schema version is incompatible",
            metadata=metadata,
        )
    for field in (
        "declaration_id",
        "source_id",
        "provenance_id",
        "specimen_id",
        "design_id",
    ):
        _nonempty_identifier(document[field], field=field, metadata=metadata)
    if not isinstance(specimen_id, str) or not specimen_id:
        raise _declaration_error(
            "expected_specimen_id_missing",
            "The Stage 1 request must bind a specimen_id when using a declaration",
            metadata=metadata,
        )
    if not isinstance(design_id, str) or not design_id:
        raise _declaration_error(
            "expected_design_id_missing",
            "The Stage 1 request must bind a design_id when using a declaration",
            metadata=metadata,
        )
    if document["specimen_id"] != specimen_id:
        raise _declaration_error(
            "specimen_id_mismatch",
            "Declared transform belongs to a different specimen",
            metadata=metadata,
        )
    if document["design_id"] != design_id:
        raise _declaration_error(
            "design_id_mismatch",
            "Declared transform belongs to a different design",
            metadata=metadata,
        )
    declared_graph_hash = _sha256_string(
        document["nominal_graph_sha256"],
        field="nominal_graph_sha256",
        metadata=metadata,
    )
    declared_stl_hash = _sha256_string(
        document["full_design_stl_sha256"],
        field="full_design_stl_sha256",
        metadata=metadata,
    )
    if declared_graph_hash != nominal_graph_sha256:
        raise _declaration_error(
            "nominal_graph_hash_mismatch",
            "Declared transform references a different nominal graph",
            metadata=metadata,
        )
    if declared_stl_hash != full_design_stl_sha256:
        raise _declaration_error(
            "full_design_stl_hash_mismatch",
            "Declared transform references a different full-design STL",
            metadata=metadata,
        )
    declared_self_hash = _sha256_string(
        document["canonical_declaration_sha256"],
        field="canonical_declaration_sha256",
        metadata=metadata,
    )
    # Validate numeric values before canonical serialization so JSON extensions
    # such as NaN/Infinity become a structured hard gate rather than escaping as
    # an incidental serializer exception.
    validated_transform = _validate_transform_values(
        document["transform"],
        expected_design_center=design_center,
        require_authoritative_scale=True,
        allow_reflection=allow_reflection,
        metadata=metadata,
    )
    canonical_payload = {
        key: value
        for key, value in document.items()
        if key != "canonical_declaration_sha256"
    }
    actual_self_hash = sha256_json(canonical_payload)
    metadata["actual_canonical_declaration_sha256"] = actual_self_hash
    if declared_self_hash != actual_self_hash:
        raise _declaration_error(
            "declaration_self_hash_mismatch",
            "Declared transform canonical self-hash is invalid",
            metadata=metadata,
        )
    try:
        declaration_schema = read_json_object(
            DEFAULT_TRANSFORM_DECLARATION_SCHEMA_PATH
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise _declaration_error(
            "declaration_schema_unavailable",
            "Transform declaration JSON Schema is unreadable",
            metadata=metadata,
        ) from exc
    schema_errors = sorted(
        Draft202012Validator(declaration_schema).iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if schema_errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
            f"{error.message}"
            for error in schema_errors
        )
        raise _declaration_error(
            "declaration_schema_invalid",
            f"Transform declaration failed its closed JSON Schema: {details}",
            metadata=metadata,
        )
    return {
        "document": document,
        "transform": validated_transform,
        "metadata": metadata,
        "artifact_path": str(resolved),
        "artifact_sha256": actual_hash,
        "canonical_declaration_sha256": actual_self_hash,
    }


def validate_declared_transform_artifact(
    declaration_path: str | Path,
    *,
    expected_artifact_sha256: str,
    specimen_id: str,
    design_id: str,
    nominal_graph_path: str | Path,
    full_design_stl_path: str | Path,
    stage1_policy_path: str | Path,
    stage1_policy_sha256: str,
) -> dict[str, Any]:
    """Validate an intake declaration without reading answer-bearing meshes.

    This is the Stage 0/1 boundary check: the declaration must satisfy its
    closed schema, file and canonical self hashes, specimen/design identities,
    nominal graph and full-design STL hashes, finite transform semantics, and
    the reflection authorization in the exact Stage 1 policy artifact.
    """

    policy = load_stage1_policy(
        stage1_policy_path,
        expected_artifact_sha256=stage1_policy_sha256,
    )
    graph = load_lattice_json(nominal_graph_path)
    design_center = (
        graph.node_positions_xyz.min(axis=0)
        + graph.node_positions_xyz.max(axis=0)
    ) / 2.0
    declaration = _load_and_validate_declared_transform(
        declaration_path,
        expected_artifact_sha256=expected_artifact_sha256,
        specimen_id=specimen_id,
        design_id=design_id,
        nominal_graph_sha256=graph.source_sha256,
        full_design_stl_sha256=sha256_file(full_design_stl_path),
        design_center=design_center,
        allow_reflection=bool(
            policy["document"]["orientation_verification"][
                "reflection_authorized"
            ]
        ),
    )
    return {
        "schema_valid": True,
        "semantic_validation_pass": True,
        "artifact_sha256": declaration["artifact_sha256"],
        "canonical_declaration_sha256": declaration[
            "canonical_declaration_sha256"
        ],
        "declaration_id": declaration["document"]["declaration_id"],
        "source_id": declaration["document"]["source_id"],
        "provenance_id": declaration["document"]["provenance_id"],
        "stage1_policy_artifact_sha256": policy["artifact_sha256"],
        "stage1_policy_id": policy["document"]["policy_id"],
        "reflection_authorized_by_policy": bool(
            policy["document"]["orientation_verification"][
                "reflection_authorized"
            ]
        ),
    }


def _support_metrics(distances: np.ndarray) -> dict[str, float | int]:
    sample_distances = np.asarray(distances, dtype=np.float64)
    if sample_distances.ndim != 2 or sample_distances.size == 0:
        raise ValueError("Edge support distances must be a non-empty 2-D array")
    all_samples = sample_distances.reshape(-1)
    quantiles = np.quantile(all_samples, [0.01, 0.50, 0.99])
    return {
        "edge_count": int(sample_distances.shape[0]),
        "samples_per_edge": int(sample_distances.shape[1]),
        "sample_support_count": int(all_samples.size),
        "mean_nearest_triangle_centroid_mm": float(np.mean(all_samples)),
        "p01_nearest_triangle_centroid_mm": float(quantiles[0]),
        "median_nearest_triangle_centroid_mm": float(quantiles[1]),
        "p99_nearest_triangle_centroid_mm": float(quantiles[2]),
        "maximum_nearest_triangle_centroid_mm": float(np.max(all_samples)),
        "p99_minus_p01_nearest_triangle_centroid_mm": float(
            quantiles[2] - quantiles[0]
        ),
    }


def _correspondence_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    numeric = [
        float(value)
        for key, value in metrics.items()
        if key not in {"edge_count", "samples_per_edge", "sample_support_count"}
    ]
    return {
        "all_edge_support_finite": all(math.isfinite(value) for value in numeric),
        "maximum_edge_support_distance_within_tolerance": float(
            metrics["maximum_nearest_triangle_centroid_mm"]
        )
        <= DECLARED_TRANSFORM_TOLERANCES["maximum_edge_support_distance_mm"],
        "edge_support_spread_within_tolerance": float(
            metrics["p99_minus_p01_nearest_triangle_centroid_mm"]
        )
        <= DECLARED_TRANSFORM_TOLERANCES[
            "maximum_edge_support_p99_minus_p01_mm"
        ],
    }


def _write_declaration_halt_report(
    *,
    output_path: str | Path,
    graph: LatticeGraph,
    full_design_stl_sha256: str,
    stage1_policy: Mapping[str, Any],
    error: DeclaredTransformValidationError,
    declared_transform_path: str | Path,
    declared_transform_sha256: str | None,
    specimen_id: str | None,
    design_id: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    metadata = dict(error.metadata)
    declared = {
        "present": True,
        "artifact_path": metadata.get(
            "artifact_path", str(Path(declared_transform_path).expanduser().resolve())
        ),
        "artifact_sha256": metadata.get("artifact_sha256"),
        "expected_artifact_sha256": declared_transform_sha256,
        **{
            key: metadata.get(key)
            for key in (
                "declaration_id",
                "source_id",
                "provenance_id",
                "specimen_id",
                "design_id",
                "canonical_declaration_sha256",
            )
            if metadata.get(key) is not None
        },
        "verification": {
            "overall_pass": False,
            "reason_codes": [error.code],
            "message": str(error),
        },
    }
    hashes = {
        "nominal_graph_sha256": graph.source_sha256,
        "full_design_stl_sha256": full_design_stl_sha256,
        "config_sha256": stage1_policy["config_sha256"],
        "stage1_policy_artifact_sha256": stage1_policy["artifact_sha256"],
        "verification_tolerances_sha256": sha256_json(
            DECLARED_TRANSFORM_TOLERANCES
        ),
    }
    if isinstance(metadata.get("artifact_sha256"), str):
        hashes["declared_transform_artifact_sha256"] = metadata["artifact_sha256"]
    if isinstance(declared_transform_sha256, str):
        hashes["intake_declared_transform_artifact_sha256"] = (
            declared_transform_sha256
        )
    if isinstance(metadata.get("actual_canonical_declaration_sha256"), str):
        hashes["canonical_declaration_sha256"] = metadata[
            "actual_canonical_declaration_sha256"
        ]
    report = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "design_id": design_id,
        "gate": "halt",
        "overall_pass": False,
        "resolution_source": "declared_transform",
        "counts": graph.counts,
        "triangle_count": None,
        "stl_coordinate_contract": {
            "units": "millimeter",
            "origin_convention": "origin_centered",
            "extra_y_geometry_is_not_lattice_extent": True,
            "whole_mesh_bounds_used_for_registration": False,
        },
        "transform": None,
        "support": None,
        "ambiguity": {
            "equivalent_hypothesis_count": 0,
            "requires_scientist_review": False,
            "resolved_by_declaration": False,
            "top_hypotheses": [],
        },
        "declared_transform": declared,
        "stage1_policy": {
            "artifact_path": stage1_policy["artifact_path"],
            "artifact_sha256": stage1_policy["artifact_sha256"],
            "policy_id": stage1_policy["document"]["policy_id"],
            "intended_use": stage1_policy["document"]["intended_use"],
        },
        "verification_tolerances": dict(DECLARED_TRANSFORM_TOLERANCES),
        "gates": {
            "declared_transform_valid": False,
            "authoritative_transform_verified": False,
            "orientation_resolved": False,
        },
        "hashes": hashes,
        "provenance": {
            "design_space_only": True,
            "ct_accessed": False,
            "aligned_graph_accessed": False,
            "deleted_edge_labels_accessed": False,
            "geometry_search_performed": False,
            "whole_mesh_bounding_box_used_as_primary_logic": False,
            "mesh_loading": "not_started",
        },
        "warnings": [str(error)],
        "reason_codes": [error.code],
    }
    artifact = write_json_atomic(output_path, report, overwrite=overwrite)
    report["artifact"] = {
        **artifact,
        "role": "cad_graph_orientation",
        "retention": "committed",
    }
    return report


def resolve_cad_graph_orientation(
    nominal_graph_path: str | Path,
    full_design_stl_path: str | Path,
    output_path: str | Path,
    *,
    stage1_policy_path: str | Path,
    stage1_policy_sha256: str,
    declared_transform_path: str | Path | None = None,
    declared_transform_sha256: str | None = None,
    specimen_id: str | None = None,
    design_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve or verify a graph-to-STL transform using baseline geometry only.

    An immutable declared transform may authoritatively select one of several
    geometry-equivalent orientations, but it never bypasses independent
    graph/baseline-STL correspondence checks.  Without a declaration, symmetric
    geometry continues to return ``manual_review`` rather than choosing by sort
    order.  Intentional-deletion meshes and labels are not accepted inputs.
    """

    stage1_policy = load_stage1_policy(
        stage1_policy_path,
        expected_artifact_sha256=stage1_policy_sha256,
    )
    orientation_policy = stage1_policy["document"]["orientation_verification"]
    sample_count = int(orientation_policy["sample_count"])
    sample_start = float(orientation_policy["sample_start"])
    sample_end = float(orientation_policy["sample_end"])
    scale_candidates = list(
        orientation_policy["scale_candidates_mm_per_design_unit"]
    )
    ambiguity_absolute_mm = float(orientation_policy["ambiguity_absolute_mm"])
    ambiguity_relative_fraction = float(
        orientation_policy["ambiguity_relative_fraction"]
    )
    allow_reflection = bool(orientation_policy["reflection_authorized"])
    graph = load_lattice_json(nominal_graph_path)
    samples, design_center = _edge_core_samples(
        graph,
        sample_count=sample_count,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    expected = dict(orientation_policy["expected_counts"])
    full_design_hash = sha256_file(full_design_stl_path)
    resolved_config_hash = stage1_policy["config_sha256"]

    declaration: dict[str, Any] | None = None
    if declared_transform_path is not None:
        try:
            declaration = _load_and_validate_declared_transform(
                declared_transform_path,
                expected_artifact_sha256=declared_transform_sha256,
                specimen_id=specimen_id,
                design_id=design_id,
                nominal_graph_sha256=graph.source_sha256,
                full_design_stl_sha256=full_design_hash,
                design_center=design_center,
                allow_reflection=allow_reflection,
            )
        except DeclaredTransformValidationError as exc:
            return _write_declaration_halt_report(
                output_path=output_path,
                graph=graph,
                full_design_stl_sha256=full_design_hash,
                stage1_policy=stage1_policy,
                error=exc,
                declared_transform_path=declared_transform_path,
                declared_transform_sha256=declared_transform_sha256,
                specimen_id=specimen_id,
                design_id=design_id,
                overwrite=overwrite,
            )
    elif declared_transform_sha256 is not None:
        error = _declaration_error(
            "declaration_path_missing",
            "A declaration SHA-256 was supplied without its artifact path",
        )
        return _write_declaration_halt_report(
            output_path=output_path,
            graph=graph,
            full_design_stl_sha256=full_design_hash,
            stage1_policy=stage1_policy,
            error=error,
            declared_transform_path="<missing>",
            declared_transform_sha256=declared_transform_sha256,
            specimen_id=specimen_id,
            design_id=design_id,
            overwrite=overwrite,
        )

    centroids, triangle_count = _binary_stl_centroids(full_design_stl_path)
    tree: cKDTree | None = None
    try:
        tree = cKDTree(centroids, compact_nodes=False, balanced_tree=False)
        search_candidates = scale_candidates
        if declaration is not None and scale_candidates is not None:
            search_candidates = [
                *scale_candidates,
                AUTHORITATIVE_SCALE_MM_PER_DESIGN_UNIT,
            ]
        scales = _scale_hypotheses(
            np.ptp(graph.node_positions_xyz, axis=0),
            centroids,
            search_candidates,
        )
        translations = _translation_hypotheses(centroids)
        include_reflections = bool(
            declaration is not None
            and allow_reflection
            and declaration["transform"]["reflection_permitted"]
        )
        rotations = _axis_orientations(include_reflections=include_reflections)
        # A stable subset ranks hypotheses; the selected transform is then
        # checked against every edge.  Ranking favors a consistent tube radius,
        # not minimum unsigned centerline-to-surface distance, which otherwise
        # biases the result toward a material-envelope scale.
        edge_rows = np.linspace(
            0,
            len(graph.edge_ids) - 1,
            min(
                len(graph.edge_ids),
                int(orientation_policy["maximum_ranked_edges"]),
            ),
            dtype=int,
        )
        query = samples[edge_rows].reshape(-1, 3)
        ranked: list[dict[str, Any]] = []
        for rotation_index, rotation in enumerate(rotations):
            rotated = query @ rotation.T
            for scale in scales:
                for translation_index, translation in enumerate(translations):
                    distances = tree.query(
                        rotated * scale + translation,
                        k=1,
                        workers=1,
                    )[0].reshape(len(edge_rows), sample_count)
                    all_support = distances.reshape(-1)
                    p01, median, p99 = np.quantile(
                        all_support, [0.01, 0.50, 0.99]
                    )
                    ranked.append(
                        {
                            "rotation_index": rotation_index,
                            "translation_index": translation_index,
                            "scale_mm_per_design_unit": float(scale),
                            "rotation_matrix": rotation.tolist(),
                            "translation_mm": translation.tolist(),
                            "mean_support_distance_mm": float(
                                np.mean(all_support)
                            ),
                            "p01_support_distance_mm": float(p01),
                            "median_support_distance_mm": float(median),
                            "p99_support_distance_mm": float(p99),
                            "support_spread_mm": float(p99 - p01),
                            "maximum_support_distance_mm": float(
                                np.max(all_support)
                            ),
                        }
                    )
        ranked.sort(
            key=lambda item: (
                item["support_spread_mm"],
                item["p99_support_distance_mm"],
                item["maximum_support_distance_mm"],
                item["mean_support_distance_mm"],
                item["rotation_index"],
                item["scale_mm_per_design_unit"],
                item["translation_index"],
            )
        )
        best = ranked[0]
        best_rotation = np.asarray(best["rotation_matrix"], dtype=np.float64)
        best_translation = np.asarray(best["translation_mm"], dtype=np.float64)
        best_points = (
            samples.reshape(-1, 3)
            @ best_rotation.T
            * float(best["scale_mm_per_design_unit"])
            + best_translation
        )
        best_all_distances = tree.query(
            best_points, k=1, workers=1
        )[0].reshape(samples.shape[:2])

        declared_all_distances: np.ndarray | None = None
        if declaration is not None:
            declared_transform = declaration["transform"]
            declared_rotation = np.asarray(
                declared_transform["rotation_matrix"], dtype=np.float64
            )
            declared_translation = np.asarray(
                declared_transform["translation_mm"], dtype=np.float64
            )
            declared_points = (
                samples.reshape(-1, 3)
                @ declared_rotation.T
                * float(declared_transform["scale_mm_per_design_unit"])
                + declared_translation
            )
            declared_all_distances = tree.query(
                declared_points, k=1, workers=1
            )[0].reshape(samples.shape[:2])
    finally:
        del tree, centroids
        gc.collect()

    best_score = float(best["support_spread_mm"])
    ambiguity_limit = max(
        float(ambiguity_absolute_mm),
        DECLARED_TRANSFORM_TOLERANCES["geometry_equivalence_absolute_mm"],
        abs(best_score) * float(ambiguity_relative_fraction),
    )
    equivalent = [
        item
        for item in ranked
        if float(item["support_spread_mm"]) - best_score <= ambiguity_limit
    ]

    selected_transform: dict[str, Any]
    selected_distances: np.ndarray
    declaration_record: dict[str, Any]
    reason_codes: list[str] = []
    if declaration is not None:
        selected_transform = dict(declaration["transform"])
        assert declared_all_distances is not None
        selected_distances = declared_all_distances
        selected_rotation = np.asarray(
            selected_transform["rotation_matrix"], dtype=np.float64
        )
        selected_translation = np.asarray(
            selected_transform["translation_mm"], dtype=np.float64
        )
        matching_search_hypotheses = [
            item
            for item in ranked
            if math.isclose(
                float(item["scale_mm_per_design_unit"]),
                float(selected_transform["scale_mm_per_design_unit"]),
                rel_tol=0.0,
                abs_tol=DECLARED_TRANSFORM_TOLERANCES[
                    "scale_absolute_mm_per_design_unit"
                ],
            )
            and np.allclose(
                np.asarray(item["rotation_matrix"], dtype=np.float64),
                selected_rotation,
                rtol=0.0,
                atol=DECLARED_TRANSFORM_TOLERANCES[
                    "rotation_orthonormal_absolute"
                ],
            )
            and np.linalg.norm(
                np.asarray(item["translation_mm"], dtype=np.float64)
                - selected_translation
            )
            <= DECLARED_TRANSFORM_TOLERANCES[
                "translation_search_match_absolute_mm"
            ]
        ]
        geometry_search_consistent = bool(matching_search_hypotheses)
        declaration_record = {
            "present": True,
            "artifact_path": declaration["artifact_path"],
            "artifact_sha256": declaration["artifact_sha256"],
            "expected_artifact_sha256": declared_transform_sha256,
            "canonical_declaration_sha256": declaration[
                "canonical_declaration_sha256"
            ],
            "declaration_id": declaration["document"]["declaration_id"],
            "source_id": declaration["document"]["source_id"],
            "provenance_id": declaration["document"]["provenance_id"],
            "specimen_id": declaration["document"]["specimen_id"],
            "design_id": declaration["document"]["design_id"],
            "contract_allows_reflection": bool(allow_reflection),
            "verification": {},
        }
    else:
        selected_transform = {
            "convention": TRANSFORM_CONVENTION,
            "design_center": design_center.tolist(),
            "scale_mm_per_design_unit": best["scale_mm_per_design_unit"],
            "rotation_matrix": best["rotation_matrix"],
            "translation_mm": best["translation_mm"],
            "reflection_permitted": False,
            "determinant": float(
                np.linalg.det(np.asarray(best["rotation_matrix"], dtype=np.float64))
            ),
            "orthonormality_max_absolute_error": 0.0,
        }
        selected_distances = best_all_distances
        geometry_search_consistent = True
        declaration_record = {
            "present": False,
            "verification": {
                "overall_pass": False,
                "reason_codes": ["declaration_absent"],
            },
        }
        reason_codes.append("declaration_absent")

    support = _support_metrics(selected_distances)
    correspondence = _correspondence_gates(support)
    declared_valid = bool(
        declaration is not None
        and geometry_search_consistent
        and all(correspondence.values())
    )
    if declaration is not None:
        if not all(correspondence.values()):
            reason_codes.append("declared_transform_geometry_unsupported")
        if not geometry_search_consistent:
            reason_codes.append("declared_transform_contradicts_geometry_search")
        if not reason_codes:
            reason_codes.append("declared_transform_verified")
        declaration_record["verification"] = {
            "overall_pass": declared_valid,
            "reason_codes": reason_codes,
            "geometry_search_consistent": geometry_search_consistent,
            "correspondence_gates": correspondence,
        }

    orientation_unambiguous = len(equivalent) == 1
    orientation_resolved = orientation_unambiguous or declared_valid
    count_gate = graph.counts == expected
    unique_gate = bool(
        len(np.unique(graph.node_ids)) == len(graph.node_ids)
        and len(np.unique(graph.edge_ids)) == len(graph.edge_ids)
        and len(np.unique(graph.cell_ids)) == len(graph.cell_ids)
    )
    scale_gate = bool(
        math.isfinite(float(selected_transform["scale_mm_per_design_unit"]))
        and float(selected_transform["scale_mm_per_design_unit"]) > 0
    )
    gates = {
        "graph_counts_match": count_gate,
        "nominal_ids_unique": unique_gate,
        "orientation_unambiguous": orientation_unambiguous,
        "orientation_resolved": orientation_resolved,
        "scale_preserving_transform": scale_gate,
        **correspondence,
        "geometry_search_consistent": geometry_search_consistent,
        "declared_transform_valid": declaration is None or declared_valid,
        "authoritative_transform_verified": declared_valid,
    }
    hard_gates = (
        count_gate
        and unique_gate
        and scale_gate
        and all(correspondence.values())
        and geometry_search_consistent
        and (declaration is None or declared_valid)
    )
    if not hard_gates:
        gate = "halt"
    elif orientation_resolved:
        gate = "pass"
    else:
        gate = "manual_review"
        reason_codes.append("orientation_symmetry_unresolved")

    hashes = {
        "nominal_graph_sha256": graph.source_sha256,
        "full_design_stl_sha256": full_design_hash,
        "config_sha256": resolved_config_hash,
        "stage1_policy_artifact_sha256": stage1_policy["artifact_sha256"],
        "verification_tolerances_sha256": sha256_json(
            DECLARED_TRANSFORM_TOLERANCES
        ),
    }
    if declaration is not None:
        hashes.update(
            {
                "declared_transform_artifact_sha256": declaration[
                    "artifact_sha256"
                ],
                "intake_declared_transform_artifact_sha256": (
                    declared_transform_sha256
                ),
                "canonical_declaration_sha256": declaration[
                    "canonical_declaration_sha256"
                ],
            }
        )
    warnings: list[str] = []
    if gate == "manual_review":
        warnings.append(
            "Equivalent CAD/graph orientation hypotheses require manual review"
        )
    elif gate == "halt":
        warnings.append("Declared transform failed deterministic verification")
    report = {
        "schema_version": ORIENTATION_SCHEMA_VERSION,
        "specimen_id": specimen_id,
        "design_id": design_id,
        "gate": gate,
        "overall_pass": gate == "pass",
        "resolution_source": (
            "declared_transform" if declaration is not None else "geometry_search"
        ),
        "counts": graph.counts,
        "triangle_count": triangle_count,
        "stl_coordinate_contract": {
            "units": "millimeter",
            "origin_convention": "origin_centered",
            "extra_y_geometry_is_not_lattice_extent": True,
            "whole_mesh_bounds_used_for_registration": False,
        },
        "transform": selected_transform,
        "support": support,
        "ambiguity": {
            "equivalent_hypothesis_count": len(equivalent),
            "absolute_score_tolerance_mm": ambiguity_limit,
            "score": (
                "all_sampled_edge_points_"
                "p99_minus_p01_nearest_triangle_centroid_mm"
            ),
            "requires_scientist_review": not orientation_resolved,
            "resolved_by_declaration": bool(
                declaration is not None and declared_valid
            ),
            "top_hypotheses": ranked[: min(8, len(ranked))],
        },
        "declared_transform": declaration_record,
        "stage1_policy": {
            "artifact_path": stage1_policy["artifact_path"],
            "artifact_sha256": stage1_policy["artifact_sha256"],
            "config_sha256": stage1_policy["config_sha256"],
            "schema_version": stage1_policy["document"]["schema_version"],
            "policy_id": stage1_policy["document"]["policy_id"],
            "intended_use": stage1_policy["document"]["intended_use"],
        },
        "verification_tolerances": dict(DECLARED_TRANSFORM_TOLERANCES),
        "gates": gates,
        "hashes": hashes,
        "provenance": {
            "design_space_only": True,
            "ct_accessed": False,
            "aligned_graph_accessed": False,
            "deleted_edge_labels_accessed": False,
            "geometry_search_performed": True,
            "all_sampled_edge_support_used": True,
            "whole_mesh_bounding_box_used_as_primary_logic": False,
            "mesh_loading": "binary_stl_memmap_equivalent_to_process_false",
        },
        "reason_codes": reason_codes,
        "warnings": warnings,
    }
    artifact = write_json_atomic(output_path, report, overwrite=overwrite)
    report["artifact"] = {
        **artifact,
        "role": "cad_graph_orientation",
        "retention": "committed",
    }
    return report


def _transform_samples(samples: np.ndarray, orientation: Mapping[str, Any]) -> np.ndarray:
    transform = orientation["transform"]
    rotation = np.asarray(transform["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(transform["translation_mm"], dtype=np.float64)
    scale = float(transform["scale_mm_per_design_unit"])
    return samples @ rotation.T * scale + translation


def _analyze_one_mesh(
    path: str | Path,
    transformed_samples: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    centroids, triangle_count = _binary_stl_centroids(path)
    tree: cKDTree | None = None
    try:
        tree = cKDTree(centroids, compact_nodes=False, balanced_tree=False)
        distances = tree.query(
            transformed_samples.reshape(-1, 3), k=1, workers=1
        )[0].reshape(transformed_samples.shape[:2])
        minimum = distances.min(axis=1)
        deleted = minimum > radius_mm
        return deleted, triangle_count, minimum, distances
    finally:
        del tree, centroids
        gc.collect()


def deterministic_stratified_split(
    graph: LatticeGraph,
    deleted_ids: Sequence[int],
    *,
    development_fraction: float = 0.30,
    seed: int = 20260723,
    x_bins: int = 5,
    z_shells: int = 3,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Split positive IDs deterministically by midpoint X bin and Z shell."""

    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between zero and one")
    id_to_row = {int(identifier): row for row, identifier in enumerate(graph.edge_ids)}
    ids = sorted({int(value) for value in deleted_ids})
    if len(ids) != len(deleted_ids) or not set(ids).issubset(id_to_row):
        raise ValueError("Split IDs must be unique nominal strut IDs")
    starts = graph.node_positions_xyz[graph.edge_node_rows[:, 0]]
    ends = graph.node_positions_xyz[graph.edge_node_rows[:, 1]]
    midpoints = (starts + ends) / 2.0
    selected = midpoints[[id_to_row[value] for value in ids]]
    x_min, x_max = graph.node_positions_xyz[:, 0].min(), graph.node_positions_xyz[:, 0].max()
    z_center = (
        graph.node_positions_xyz[:, 2].min() + graph.node_positions_xyz[:, 2].max()
    ) / 2.0
    z_extent = max(float(np.max(np.abs(graph.node_positions_xyz[:, 2] - z_center))), 1e-12)
    x_index = np.minimum(
        x_bins - 1,
        np.floor((selected[:, 0] - x_min) / max(float(x_max - x_min), 1e-12) * x_bins).astype(int),
    )
    z_index = np.minimum(
        z_shells - 1,
        np.floor(np.abs(selected[:, 2] - z_center) / z_extent * z_shells).astype(int),
    )
    strata: dict[tuple[int, int], list[int]] = {}
    for identifier, x_bin, z_shell in zip(ids, x_index, z_index, strict=True):
        strata.setdefault((int(x_bin), int(z_shell)), []).append(identifier)
    target = int(round(len(ids) * development_fraction))
    allocation = {key: int(math.floor(len(values) * development_fraction)) for key, values in strata.items()}
    remaining = target - sum(allocation.values())
    fractional = sorted(
        strata,
        key=lambda key: (
            -(len(strata[key]) * development_fraction - allocation[key]),
            key,
        ),
    )
    for key in fractional[:remaining]:
        allocation[key] += 1

    def stable_key(identifier: int) -> str:
        return sha256_json({"seed": int(seed), "strut_id": identifier})

    development: list[int] = []
    sealed: list[int] = []
    summary: list[dict[str, Any]] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda value: (stable_key(value), value))
        count = allocation[key]
        development.extend(ordered[:count])
        sealed.extend(ordered[count:])
        summary.append(
            {
                "x_bin": key[0],
                "z_shell": key[1],
                "total": len(ordered),
                "development": count,
                "sealed": len(ordered) - count,
            }
        )
    return sorted(development), sorted(sealed), {
        "seed": int(seed),
        "development_fraction": development_fraction,
        "x_bins": x_bins,
        "z_shells": z_shells,
        "strata": summary,
    }


def _validate_orientation_for_labeling(
    *,
    graph: LatticeGraph,
    baseline_stl_path: str | Path,
    orientation: Mapping[str, Any],
    design_center: np.ndarray,
    specimen_id: str | None,
    design_id: str | None,
) -> tuple[str, dict[str, Any]]:
    """Revalidate a frozen orientation before any answer-bearing mesh is read."""

    if orientation.get("schema_version") != ORIENTATION_SCHEMA_VERSION:
        raise ValueError("CAD/graph orientation schema is incompatible")
    for name, expected in (
        ("specimen_id", specimen_id),
        ("design_id", design_id),
    ):
        actual = orientation.get(name)
        if expected is not None and actual != expected:
            raise ValueError(f"CAD/graph orientation {name} mismatch")
    if orientation.get("gate") != "pass" or orientation.get("overall_pass") is not True:
        raise ValueError("CAD/graph orientation did not pass verification")
    source = orientation.get("resolution_source")
    if source not in {"geometry_search", "declared_transform"}:
        raise ValueError("CAD/graph orientation has an unsupported resolution source")
    gates = orientation.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("CAD/graph orientation gates are missing")
    required_gates = (
        "orientation_resolved",
        "scale_preserving_transform",
        "all_edge_support_finite",
        "maximum_edge_support_distance_within_tolerance",
        "edge_support_spread_within_tolerance",
        "geometry_search_consistent",
    )
    if not all(gates.get(name) is True for name in required_gates):
        raise ValueError("CAD/graph orientation verification gates are incomplete")
    if source == "geometry_search" and gates.get("orientation_unambiguous") is not True:
        raise ValueError("Geometry-search orientation is not unambiguously frozen")
    if source == "declared_transform" and gates.get(
        "authoritative_transform_verified"
    ) is not True:
        raise ValueError(
            "Declared CAD/graph transform was not authoritatively verified"
        )

    hashes = orientation.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("CAD/graph orientation hashes are missing")
    baseline_hash = sha256_file(baseline_stl_path)
    if hashes.get("nominal_graph_sha256") != graph.source_sha256:
        raise ValueError("CAD/graph orientation nominal-graph hash mismatch")
    if hashes.get("full_design_stl_sha256") != baseline_hash:
        raise ValueError("CAD/graph orientation baseline-STL hash mismatch")
    policy_record = orientation.get("stage1_policy")
    if not isinstance(policy_record, Mapping):
        raise ValueError("CAD/graph orientation Stage 1 policy binding is missing")
    policy_path = policy_record.get("artifact_path")
    policy_hash = policy_record.get("artifact_sha256")
    if not isinstance(policy_path, str) or not isinstance(policy_hash, str):
        raise ValueError("CAD/graph orientation Stage 1 policy binding is incomplete")
    policy = load_stage1_policy(
        policy_path,
        expected_artifact_sha256=policy_hash,
    )
    if hashes.get("stage1_policy_artifact_sha256") != policy["artifact_sha256"]:
        raise ValueError("CAD/graph orientation Stage 1 policy hash mismatch")
    if hashes.get("config_sha256") != policy["config_sha256"]:
        raise ValueError("CAD/graph orientation Stage 1 config hash mismatch")
    if (
        policy_record.get("policy_id") != policy["document"]["policy_id"]
        or policy_record.get("schema_version")
        != policy["document"]["schema_version"]
        or policy_record.get("intended_use")
        != policy["document"]["intended_use"]
    ):
        raise ValueError("CAD/graph orientation Stage 1 policy identity mismatch")
    if orientation.get("verification_tolerances") != dict(
        DECLARED_TRANSFORM_TOLERANCES
    ):
        raise ValueError("CAD/graph orientation tolerance profile was altered")
    if hashes.get("verification_tolerances_sha256") != sha256_json(
        DECLARED_TRANSFORM_TOLERANCES
    ):
        raise ValueError("CAD/graph orientation tolerance hash mismatch")

    selected = orientation.get("transform")
    if not isinstance(selected, Mapping):
        raise ValueError("CAD/graph orientation transform is missing")
    transform_keys = {
        "convention",
        "design_center",
        "scale_mm_per_design_unit",
        "rotation_matrix",
        "translation_mm",
        "reflection_permitted",
    }
    if not transform_keys.issubset(selected):
        raise ValueError("CAD/graph orientation transform is incomplete")
    declared_record = orientation.get("declared_transform")
    contract_allows_reflection = bool(
        policy["document"]["orientation_verification"][
            "reflection_authorized"
        ]
    )
    if (
        isinstance(declared_record, Mapping)
        and "contract_allows_reflection" in declared_record
        and declared_record.get("contract_allows_reflection")
        is not contract_allows_reflection
    ):
        raise ValueError("Declared-transform reflection policy binding mismatch")
    selected_validated = _validate_transform_values(
        {key: selected[key] for key in transform_keys},
        expected_design_center=design_center,
        require_authoritative_scale=source == "declared_transform",
        allow_reflection=contract_allows_reflection,
        metadata={},
    )

    if source == "declared_transform":
        if not isinstance(declared_record, Mapping):
            raise ValueError("Declared-transform verification record is missing")
        verification = declared_record.get("verification")
        if (
            not isinstance(verification, Mapping)
            or verification.get("overall_pass") is not True
            or "declared_transform_verified"
            not in verification.get("reason_codes", [])
        ):
            raise ValueError("Declared-transform verification evidence is invalid")
        declaration_path = declared_record.get("artifact_path")
        declaration_hash = declared_record.get("artifact_sha256")
        expected_declaration_hash = declared_record.get(
            "expected_artifact_sha256"
        )
        if not isinstance(declaration_path, str) or not isinstance(
            declaration_hash, str
        ):
            raise ValueError("Declared-transform artifact binding is incomplete")
        if expected_declaration_hash != declaration_hash:
            raise ValueError(
                "Declared-transform intake artifact hash differs from verified file"
            )
        if hashes.get(
            "intake_declared_transform_artifact_sha256"
        ) != declaration_hash:
            raise ValueError("Declared-transform intake hash binding is missing")
        replay = _load_and_validate_declared_transform(
            declaration_path,
            expected_artifact_sha256=declaration_hash,
            specimen_id=orientation.get("specimen_id"),
            design_id=orientation.get("design_id"),
            nominal_graph_sha256=graph.source_sha256,
            full_design_stl_sha256=baseline_hash,
            design_center=design_center,
            allow_reflection=contract_allows_reflection,
        )
        if hashes.get("declared_transform_artifact_sha256") != replay[
            "artifact_sha256"
        ]:
            raise ValueError(
                "Declared-transform artifact hash changed after verification"
            )
        if hashes.get("canonical_declaration_sha256") != replay[
            "canonical_declaration_sha256"
        ]:
            raise ValueError("Declared-transform self-hash changed after verification")
        for name in ("declaration_id", "source_id", "provenance_id"):
            if declared_record.get(name) != replay["document"].get(name):
                raise ValueError(f"Declared-transform {name} binding mismatch")
        for name in ("specimen_id", "design_id"):
            if declared_record.get(name) != orientation.get(name):
                raise ValueError(f"Declared-transform {name} binding mismatch")
        replay_transform = replay["transform"]
        if not math.isclose(
            float(selected_validated["scale_mm_per_design_unit"]),
            float(replay_transform["scale_mm_per_design_unit"]),
            rel_tol=0.0,
            abs_tol=DECLARED_TRANSFORM_TOLERANCES[
                "scale_absolute_mm_per_design_unit"
            ],
        ):
            raise ValueError("Selected transform scale differs from its declaration")
        for name in ("design_center", "rotation_matrix", "translation_mm"):
            if not np.allclose(
                np.asarray(selected_validated[name], dtype=np.float64),
                np.asarray(replay_transform[name], dtype=np.float64),
                rtol=0.0,
                atol=DECLARED_TRANSFORM_TOLERANCES[
                    "rotation_orthonormal_absolute"
                ],
            ):
                raise ValueError(
                    f"Selected transform {name} differs from its declaration"
                )
    return str(source), policy


def label_deleted_edges(
    nominal_graph_path: str | Path,
    baseline_stl_path: str | Path,
    variant_stl_paths: Mapping[str, str | Path],
    orientation_path: str | Path,
    output_directory: str | Path,
    *,
    development_split_path: str | Path | None = None,
    sealed_split_path: str | Path | None = None,
    label_report_path: str | Path | None = None,
    specimen_id: str | None = None,
    design_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Label all nominal edges by junction-trimmed tube emptiness."""

    graph = load_lattice_json(nominal_graph_path)
    orientation = read_json_object(orientation_path)
    design_center = (
        graph.node_positions_xyz.min(axis=0)
        + graph.node_positions_xyz.max(axis=0)
    ) / 2.0
    orientation_source, stage1_policy = _validate_orientation_for_labeling(
        graph=graph,
        baseline_stl_path=baseline_stl_path,
        orientation=orientation,
        design_center=design_center,
        specimen_id=specimen_id,
        design_id=design_id,
    )
    labeling_policy = stage1_policy["document"]["deletion_labeling"]
    sample_count = int(labeling_policy["sample_count"])
    sample_start = float(labeling_policy["sample_start"])
    sample_end = float(labeling_policy["sample_end"])
    samples, sampled_design_center = _edge_core_samples(
        graph,
        sample_count=sample_count,
        sample_start=sample_start,
        sample_end=sample_end,
    )
    if not np.array_equal(sampled_design_center, design_center):
        raise ValueError("Stage 1 design-center replay is inconsistent")
    resolved_specimen_id = orientation.get("specimen_id")
    resolved_design_id = orientation.get("design_id")
    transformed = _transform_samples(samples, orientation)
    (
        baseline_deleted,
        baseline_triangles,
        baseline_distances,
        baseline_sample_distances,
    ) = _analyze_one_mesh(baseline_stl_path, transformed, math.inf)
    del baseline_deleted
    baseline_support = _support_metrics(baseline_sample_distances)
    baseline_correspondence = _correspondence_gates(baseline_support)
    if not all(baseline_correspondence.values()):
        raise ValueError(
            "CAD/graph orientation failed independent baseline correspondence "
            "revalidation before deletion labeling"
        )
    radius_margin_mm = float(labeling_policy["radius_margin_mm"])
    radius_rounding_mm = float(labeling_policy["radius_rounding_mm"])
    required_radius = float(np.max(baseline_distances)) + radius_margin_mm
    radius = math.ceil(required_radius / radius_rounding_mm) * radius_rounding_mm
    # Re-check the baseline at the calibrated radius as a strict negative control.
    baseline_empty = baseline_distances > radius
    destination = Path(output_directory).expanduser().resolve()
    expected = dict(labeling_policy["expected_deletions"])
    resolved_config_hash = stage1_policy["config_sha256"]
    labels: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for variant in sorted(variant_stl_paths):
        deleted_mask, triangle_count, distances, _ = _analyze_one_mesh(
            variant_stl_paths[variant], transformed, radius
        )
        deleted_ids = [int(value) for value in graph.edge_ids[deleted_mask]]
        deficit = baseline_triangles - triangle_count
        ratio = deficit / len(deleted_ids) if deleted_ids else None
        payload = {
            "schema_version": LABEL_SCHEMA_VERSION,
            "specimen_id": resolved_specimen_id,
            "design_id": resolved_design_id,
            "variant": variant,
            "deleted_strut_ids": deleted_ids,
            "deleted_count": len(deleted_ids),
            "triangle_count": triangle_count,
            "triangle_deficit_from_baseline": deficit,
            "triangles_per_deleted_strut": ratio,
            "tube_test": {
                "radius_mm": radius,
                "samples_per_edge": sample_count,
                "sample_fraction": [sample_start, sample_end],
                "edge_count": int(len(graph.edge_ids)),
                "minimum_distance_mm": float(np.min(distances)),
                "maximum_distance_mm": float(np.max(distances)),
            },
            "hashes": {
                "nominal_graph_sha256": graph.source_sha256,
                "orientation_sha256": sha256_file(orientation_path),
                "baseline_stl_sha256": sha256_file(baseline_stl_path),
                "variant_stl_sha256": sha256_file(variant_stl_paths[variant]),
                "config_sha256": resolved_config_hash,
                "stage1_policy_artifact_sha256": stage1_policy[
                    "artifact_sha256"
                ],
                **(
                    {
                        "intake_declared_transform_artifact_sha256": orientation[
                            "hashes"
                        ]["intake_declared_transform_artifact_sha256"]
                    }
                    if orientation_source == "declared_transform"
                    else {}
                ),
            },
            "provenance": {
                "design_space_only": True,
                "ct_accessed": False,
                "aligned_graph_accessed": False,
                "primary_logic": "junction_trimmed_tube_emptiness",
                "exact_float_coordinate_differencing": False,
                "clustering_used_as_primary_logic": False,
            },
        }
        path = destination / f"intentional_deletions_{variant}.json"
        artifact = write_json_atomic(path, payload, overwrite=overwrite)
        labels[variant] = payload
        artifacts[f"labels_{variant}"] = {
            **artifact,
            "role": f"intentional_deletions_{variant}",
            "retention": "sealed" if variant == "0p5" else "committed",
        }

    count_gate = all(
        variant in labels and labels[variant]["deleted_count"] == count
        for variant, count in expected.items()
    )
    ratio_gate = all(
        labels[variant]["triangles_per_deleted_strut"] is not None
        and 170.0 <= labels[variant]["triangles_per_deleted_strut"] <= 180.0
        for variant in expected
        if variant in labels
    )
    nominal_ids = set(int(value) for value in graph.edge_ids)
    id_gate = all(
        len(item["deleted_strut_ids"]) == len(set(item["deleted_strut_ids"]))
        and set(item["deleted_strut_ids"]).issubset(nominal_ids)
        for item in labels.values()
    )
    gates = {
        "baseline_negative_control": not bool(np.any(baseline_empty)),
        "orientation_revalidated": True,
        **{
            f"baseline_{name}": value
            for name, value in baseline_correspondence.items()
        },
        "deletion_counts_match": count_gate,
        "triangle_deficit_ratio_between_170_and_180": ratio_gate,
        "label_ids_unique_and_nominal": id_gate,
        "graph_counts_match_policy": graph.counts
        == stage1_policy["document"]["orientation_verification"][
            "expected_counts"
        ],
        "orientation_resolved": True,
    }
    split_summary: dict[str, Any] | None = None
    if "0p5" in labels and development_split_path and sealed_split_path:
        development, sealed, stratification = deterministic_stratified_split(
            graph,
            labels["0p5"]["deleted_strut_ids"],
            development_fraction=float(labeling_policy["development_fraction"]),
            seed=int(labeling_policy["split_seed"]),
            x_bins=int(labeling_policy["x_bins"]),
            z_shells=int(labeling_policy["z_shells"]),
        )
        split_base = {
            "specimen_id": resolved_specimen_id,
            "design_id": resolved_design_id,
            "source_variant": "0p5",
            "source_labels_sha256": artifacts["labels_0p5"]["sha256"],
            "stratification": stratification,
            "config_sha256": resolved_config_hash,
        }
        dev_document = {
            "schema_version": "part2-label-split/1.0.0",
            "role": "development_labels",
            "strut_ids": development,
            **split_base,
        }
        sealed_document = {
            "schema_version": "part2-label-split/1.0.0",
            "role": "sealed_labels",
            "strut_ids": sealed,
            **split_base,
        }
        dev_artifact = write_json_atomic(development_split_path, dev_document, overwrite=overwrite)
        sealed_artifact = write_json_atomic(sealed_split_path, sealed_document, overwrite=overwrite)
        artifacts["development_split"] = {**dev_artifact, "role": "development_labels", "retention": "committed"}
        artifacts["sealed_split"] = {**sealed_artifact, "role": "sealed_labels", "retention": "sealed"}
        split_gates = {
            "disjoint": set(development).isdisjoint(sealed),
            "exhaustive": set(development) | set(sealed) == set(labels["0p5"]["deleted_strut_ids"]),
            "development_count": len(development),
            "sealed_count": len(sealed),
        }
        gates["development_and_sealed_disjoint"] = bool(split_gates["disjoint"])
        gates["development_and_sealed_exhaustive"] = bool(split_gates["exhaustive"])
        split_summary = {**split_gates, "stratification": stratification}

    gate = "pass" if all(bool(value) for value in gates.values()) else "halt"
    report = {
        "schema_version": "part2-design-label-report/1.0.0",
        "specimen_id": resolved_specimen_id,
        "design_id": resolved_design_id,
        "gate": gate,
        "overall_pass": gate == "pass",
        "counts": graph.counts,
        "baseline": {
            "triangle_count": baseline_triangles,
            "calibrated_radius_mm": radius,
            "maximum_support_distance_mm": float(np.max(baseline_distances)),
        },
        "variants": {
            key: {
                "deleted_count": value["deleted_count"],
                "triangle_count": value["triangle_count"],
                "triangles_per_deleted_strut": value["triangles_per_deleted_strut"],
            }
            for key, value in labels.items()
        },
        "split": split_summary,
        "gates": gates,
        "hashes": {
            "nominal_graph_sha256": graph.source_sha256,
            "orientation_sha256": sha256_file(orientation_path),
            "config_sha256": resolved_config_hash,
            "stage1_policy_artifact_sha256": stage1_policy[
                "artifact_sha256"
            ],
            **(
                {
                    "intake_declared_transform_artifact_sha256": orientation[
                        "hashes"
                    ]["intake_declared_transform_artifact_sha256"]
                }
                if orientation_source == "declared_transform"
                else {}
            ),
        },
        "stage1_policy": {
            "artifact_path": stage1_policy["artifact_path"],
            "artifact_sha256": stage1_policy["artifact_sha256"],
            "schema_version": stage1_policy["document"]["schema_version"],
            "policy_id": stage1_policy["document"]["policy_id"],
            "intended_use": stage1_policy["document"]["intended_use"],
        },
        "provenance": {
            "design_space_only": True,
            "ct_accessed": False,
            "aligned_graph_accessed": False,
            "meshes_loaded_sequentially": True,
            "orientation_resolution_source": orientation_source,
            "orientation_revalidated_before_variant_access": True,
            "stage1_policy_revalidated_before_variant_access": True,
            "all_sampled_edge_support_revalidated": True,
            "deleted_edge_labels_used_to_select_orientation": False,
        },
        "warnings": [] if gate == "pass" else ["One or more deterministic design-label gates failed"],
    }
    if label_report_path is not None:
        report_destination = Path(label_report_path)
        if report_destination.suffix.lower() == ".md":
            lines = [
                "# Design-diff label report",
                "",
                f"Gate: `{gate}`",
                "",
                f"Specimen: `{resolved_specimen_id}`",
                f"Design: `{resolved_design_id}`",
                "",
                "| Variant | Deleted struts | Triangles | Triangles/deletion |",
                "|---|---:|---:|---:|",
            ]
            for key, value in report["variants"].items():
                ratio = value["triangles_per_deleted_strut"]
                ratio_text = "n/a" if ratio is None else f"{ratio:.6g}"
                lines.append(
                    f"| {key} | {value['deleted_count']} | {value['triangle_count']} | "
                    f"{ratio_text} |"
                )
            lines.extend(["", "All deterministic gates and hashes are recorded in the MCP response.", ""])
            artifact = write_text_atomic(
                report_destination, "\n".join(lines), overwrite=overwrite
            )
        else:
            artifact = write_json_atomic(report_destination, report, overwrite=overwrite)
        artifacts["label_report"] = {**artifact, "role": "label_report", "retention": "committed"}
    report["artifacts"] = artifacts
    return report
