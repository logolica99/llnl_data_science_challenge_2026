"""Adversarial tests for the Part 2 orchestration control plane.

The fixtures intentionally use tiny opaque files.  They exercise only state,
hash, contract, dependency, and access-control behavior; no scientific code is
imported or executed.
"""

from __future__ import annotations

import ast
import copy
from datetime import datetime, timedelta, timezone
import fnmatch
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from part2_orchestration import (  # noqa: E402
    AccessPolicyError,
    ArtifactVerificationError,
    IllegalTransitionError,
    ManifestValidationError,
    ReceiptValidationError,
    artifact_record,
    authorize_post_freeze_aligned_input,
    build_stage_receipt,
    canonical_json_sha256,
    complete_stage,
    create_pipeline_manifest,
    pipeline_status,
    record_autonomous_registration_freeze,
    resume_manual_review,
    sha256_file,
    start_stage,
    validate_pipeline_manifest,
)
from data_prep_handoff import create_data_prep_handoff  # noqa: E402
from specimen_ingest import ingest_specimen  # noqa: E402
from specimen_manifest import sha256_file as specimen_sha256_file  # noqa: E402
from tests.stage0_metadata_fixture import (  # noqa: E402
    write_ct_metadata_response_fixture,
)


SPECIMEN_ID = "synthetic_part2"
CONTRACT_VERSION = "agent-stage-contract/1.0.0"


class SyntheticPipeline:
    """Create one isolated, deterministic seven-stage control-plane run."""

    def __init__(
        self,
        root: Path,
        registration_mode: str = "autonomous_v2",
        requested_analysis_scope: str = "roi_screening",
    ):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPOSITORY_ROOT / "analysis" / "contracts",
            self.root / "analysis" / "contracts",
        )
        shutil.copytree(
            REPOSITORY_ROOT / "analysis" / "schema",
            self.root / "analysis" / "schema",
        )
        self.specimen_id = SPECIMEN_ID
        self.registration_mode = registration_mode
        self.requested_analysis_scope = requested_analysis_scope
        self.design_id = "synthetic_design"
        self._tick = 0
        self.config_path = self.root / "config" / "frozen_analysis.json"
        self._write_json(
            self.config_path,
            {
                "schema_version": "synthetic-analysis-config/1.0.0",
                "registration_mode": registration_mode,
                "frozen": True,
            },
        )
        self.contracts = self._load_contracts()
        self.inventory = self._capability_inventory()
        created = create_pipeline_manifest(
            repository_root=self.root,
            specimen_id=self.specimen_id,
            config_path=self.config_path,
            registration_mode=registration_mode,
            timestamp=self.timestamp(),
        )
        self.manifest_path = Path(created["path"])

    def timestamp(self) -> str:
        value = datetime(2026, 7, 27, tzinfo=timezone.utc) + timedelta(
            seconds=self._tick
        )
        self._tick += 1
        return value.isoformat().replace("+00:00", "Z")

    def _load_contracts(self) -> dict[int, dict[str, object]]:
        contracts: dict[int, dict[str, object]] = {}
        for path in sorted((self.root / "analysis" / "contracts").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "stage_number" in value:
                contracts[int(value["stage_number"])] = value
        self._assert_contract_order(contracts)
        return contracts

    @staticmethod
    def _assert_contract_order(contracts: dict[int, dict[str, object]]) -> None:
        if set(contracts) != set(range(7)):
            raise AssertionError("Synthetic fixture requires contracts for Stages 0-6")

    def _capability_inventory(self) -> dict[str, object]:
        """Derive a complete compatible inventory from the frozen contracts."""

        inventory: dict[str, object] = {
            "agents": {},
            "mcp_servers": {},
        }
        agents = inventory["agents"]
        servers = inventory["mcp_servers"]
        assert isinstance(agents, dict)
        assert isinstance(servers, dict)
        for contract in self.contracts.values():
            dependencies = contract["required_dependencies"]
            assert isinstance(dependencies, dict)
            for agent in dependencies.get("agents", []):
                agents[agent["name"]] = {
                    "available": True,
                    "contract_version": agent["contract_version"],
                }
            for tool in dependencies.get("mcp_tools", []):
                server_name = tool.get("server", "segmentation-tools")
                server = servers.setdefault(
                    server_name, {"healthy": True, "tools": {}}
                )
                server["tools"][tool["name"]] = {
                    "available": True,
                    "response_schema_version": tool["response_schema_version"],
                }
        return inventory

    def manifest(self, *, verify_artifacts: bool = True) -> dict[str, object]:
        return validate_pipeline_manifest(
            self.manifest_path,
            repository_root=self.root,
            verify_artifacts=verify_artifacts,
        )

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _npy_bytes(
        *, shape: tuple[int, ...] = (2, 3, 4), descr: str = "|u1"
    ) -> bytes:
        header = repr(
            {
                "descr": descr,
                "fortran_order": False,
                "shape": shape,
            }
        ).encode("latin1")
        padding = (16 - ((10 + len(header) + 1) % 16)) % 16
        header = header + (b" " * padding) + b"\n"
        item_size = int(descr[-1])
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        return (
            b"\x93NUMPY"
            + bytes((1, 0))
            + struct.pack("<H", len(header))
            + header
            + bytes(element_count * item_size)
        )

    @staticmethod
    def _graph_bytes() -> bytes:
        return (
            json.dumps(
                {
                    "junctions": [
                        {"id": 0, "position": [0.0, 0.0, 0.0]},
                        {"id": 1, "position": [1.0, 1.0, 1.0]},
                    ],
                    "struts": [{"id": 10, "junction0": 0, "junction1": 1}],
                    "unit_cells": [{"id": 20, "struts": [10]}],
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()

    @staticmethod
    def _stl_bytes() -> bytes:
        return b"""solid synthetic
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid synthetic
"""

    def write(self, relative: str, payload: bytes | None = None, *, overwrite: bool = False) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not path.exists():
            path.write_bytes(payload if payload is not None else f"{relative}\n".encode())
        return path

    @staticmethod
    def _rule_for(
        contract: dict[str, object], direction: str, role: str
    ) -> dict[str, object]:
        policy = contract[f"{direction}_artifacts"]
        assert isinstance(policy, dict)
        for rule in policy["allowed"]:
            if fnmatch.fnmatchcase(role, str(rule.get("role", ""))):
                return rule
        raise AssertionError(
            f"Contract {contract['stage']} has no {direction} rule for {role}"
        )

    def _materialize_path(self, pattern: str, role: str) -> str:
        if pattern.startswith("<") and pattern.endswith(">"):
            suffix = {
                "cad_stl": ".stl",
                "nominal_graph": ".json",
                "ct_volume": ".npy",
                "challenge_aligned_graph": ".json",
                "design_transform_declaration": ".json",
            }.get(role, ".bin")
            return f"inputs/{role}{suffix}"
        value = pattern.replace("<specimen_id>", self.specimen_id)
        value = value.replace("<attempt>", "1")
        value = value.replace("<timestamp>", "20260727T000000Z")
        value = value.replace("<strut_id>", "1")
        value = re.sub(r"<[^>]+>", role, value)
        value = value.replace("**", "manifest.json")
        value = value.replace("*", "synthetic")
        return value

    def record_for_rule(
        self,
        rule: dict[str, object],
        role: str,
        *,
        payload: bytes | None = None,
        overwrite: bool = False,
        consumer: str | None = None,
        phase: str | None = None,
        replaces_sha256: str | None = None,
    ) -> dict[str, object]:
        relative = self._materialize_path(str(rule["path"]), role)
        if payload is None and role == "canonical_segmentation_mask":
            payload = self._npy_bytes()
        elif payload is None and role == "ct_volume":
            payload = self._npy_bytes()
        elif payload is None and role in {"nominal_graph", "challenge_aligned_graph"}:
            payload = self._graph_bytes()
        elif payload is None and role == "cad_stl":
            payload = self._stl_bytes()
        self.write(relative, payload, overwrite=overwrite)
        consumers = rule.get("consumers", [])
        producers = rule.get("producers", [])
        phases = rule.get("phases", [])
        selected_consumer = consumer
        if selected_consumer is None and isinstance(consumers, list) and consumers:
            selected_consumer = str(consumers[0])
        selected_phase = phase
        if selected_phase is None:
            selected_phase = str(phases[0]) if isinstance(phases, list) and phases else "input"
        return artifact_record(
            self.root,
            relative,
            role=role,
            consumer=selected_consumer,
            producer=(
                str(producers[0])
                if isinstance(producers, list) and producers
                else None
            ),
            phase=selected_phase,
            replaces_sha256=replaces_sha256,
        )

    def record(
        self,
        relative: str,
        *,
        role: str,
        consumer: str | None = None,
        producer: str | None = None,
        phase: str = "input",
        payload: bytes | None = None,
        overwrite: bool = False,
    ) -> dict[str, object]:
        self.write(relative, payload, overwrite=overwrite)
        return artifact_record(
            self.root,
            relative,
            role=role,
            consumer=consumer,
            producer=producer,
            phase=phase,
        )

    def manual_review_evidence(
        self, stage_number: int, attempt: int
    ) -> dict[str, object]:
        return self.record(
            f"analysis/{self.specimen_id}/reviews/"
            f"stage_{stage_number}_attempt_{attempt}/evidence.json",
            role="manual_review_evidence",
            phase="manual_review",
        )

    def inputs(self, stage_number: int) -> list[dict[str, object]]:
        contract = self.contracts[stage_number]
        policy = contract["input_artifacts"]
        assert isinstance(policy, dict)
        result: list[dict[str, object]] = []
        for role in policy["required_roles"]:
            if role == "stage_handoff" or (
                stage_number == 0 and role == "scientist_intake_request"
            ):
                continue
            rule = self._rule_for(contract, "input", str(role))
            result.append(self.record_for_rule(rule, str(role)))
        if self.registration_mode == "challenge_aligned_json" and stage_number in {0, 2}:
            rule = self._rule_for(contract, "input", "challenge_aligned_graph")
            result.append(
                self.record_for_rule(
                    rule,
                    "challenge_aligned_graph",
                    phase="challenge_aligned_json",
                )
            )
        if stage_number == 1:
            rule = self._rule_for(
                contract, "input", "design_transform_declaration"
            )
            result.append(
                self.record_for_rule(rule, "design_transform_declaration")
            )
        if stage_number == 0:
            by_role = {str(record["role"]): record for record in result}
            declaration_rule = self._rule_for(
                contract, "input", "design_transform_declaration"
            )
            declaration_base = {
                "schema_version": "part2-graph-to-stl-transform-declaration/1.0.0",
                "declaration_id": "synthetic-declaration",
                "source_id": "synthetic-source",
                "provenance_id": "synthetic-provenance",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "nominal_graph_sha256": by_role["nominal_graph"]["sha256"],
                "full_design_stl_sha256": by_role["cad_stl"]["sha256"],
                "transform": {
                    "convention": "stl_mm = scale * ((design_xyz - design_center) @ rotation.T) + translation_mm",
                    "design_center": [0.5, 0.5, 0.5],
                    "scale_mm_per_design_unit": 2.28,
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "translation_mm": [0.0, 0.0, 0.0],
                    "reflection_permitted": False,
                },
            }
            declaration = {
                **declaration_base,
                "canonical_declaration_sha256": canonical_json_sha256(
                    declaration_base
                ),
            }
            declaration_path = self._materialize_path(
                str(declaration_rule["path"]),
                "design_transform_declaration",
            )
            self._write_json(self.root / declaration_path, declaration)
            declaration_record = self.record_for_rule(
                declaration_rule, "design_transform_declaration"
            )
            result.append(declaration_record)
            by_role["design_transform_declaration"] = declaration_record
            declarations = {
                "cad_units": "millimeter",
                "cad_units_provenance": "synthetic scientist declaration",
                "graph_axes": ["x", "y", "z"],
                "array_axes": ["z", "y", "x"],
                "aligned_graph_units": (
                    "voxel"
                    if self.registration_mode == "challenge_aligned_json"
                    else "simulation_voxel"
                ),
                "retention": "committed",
            }

            def source_binding(role: str, document_role: str) -> dict[str, str]:
                record = by_role[role]
                return {
                    "path": str(record["path"]),
                    "sha256": str(record["sha256"]),
                    "role": document_role,
                    "retention": "committed",
                }

            request_base = {
                "schema_version": "part2-scientist-intake-request/1.0.0",
                "created_at": "2026-07-27T00:00:00Z",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "registration_mode": self.registration_mode,
                "association_confirmed": True,
                "aligned_graph_authorized": (
                    self.registration_mode == "challenge_aligned_json"
                ),
                "declarations": declarations,
                "inputs": {
                    "cad": source_binding("cad_stl", "cad"),
                    "nominal_graph": source_binding(
                        "nominal_graph", "design_graph"
                    ),
                    "ct": source_binding("ct_volume", "ct_volume"),
                    "aligned_graph": (
                        source_binding(
                            "challenge_aligned_graph", "aligned_graph"
                        )
                        if self.registration_mode == "challenge_aligned_json"
                        else None
                    ),
                    "design_transform_declaration": source_binding(
                        "design_transform_declaration",
                        "design_transform_declaration",
                    ),
                },
            }
            request = {
                **request_base,
                "canonical_request_sha256": canonical_json_sha256(request_base),
            }
            rule = self._rule_for(
                contract, "input", "scientist_intake_request"
            )
            request_path = self._materialize_path(
                str(rule["path"]), "scientist_intake_request"
            )
            self._write_json(self.root / request_path, request)
            result.append(
                self.record_for_rule(rule, "scientist_intake_request")
            )
        return result

    def output_record(self, stage_number: int, role: str) -> dict[str, object]:
        contract = self.contracts[stage_number]
        rule = self._rule_for(contract, "output", role)
        return self.record_for_rule(rule, role)

    def outputs(
        self,
        stage_number: int,
        *,
        verifier_updates: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        contract = self.contracts[stage_number]
        policy = contract["output_artifacts"]
        assert isinstance(policy, dict)
        roles = [str(role) for role in policy["required_roles"]]
        if stage_number == 0:
            return self._stage0_outputs(roles)
        if stage_number in {1, 2}:
            return self._bound_stage_outputs(stage_number, roles)
        if stage_number != 4:
            records: list[dict[str, object]] = []
            for role in roles:
                rule = self._rule_for(contract, "output", role)
                payload = None
                if stage_number == 0 and role == "specimen_manifest":
                    attempt = self.manifest(verify_artifacts=False)["stages"]["0"][
                        "attempts"
                    ][-1]
                    intake = {
                        item["role"]: item for item in attempt["input_artifacts"]
                    }
                    payload = (
                        json.dumps(
                            {
                                "schema_version": "2.1.0",
                                "specimen_id": self.specimen_id,
                                "design_id": self.design_id,
                                "lifecycle_state": "ready_for_data_prep",
                                "analysis_parameters": {
                                    "requested_analysis_scope": self.requested_analysis_scope
                                },
                                "inputs": {
                                    "ct_metadata": {
                                        "shape": [2, 3, 4],
                                        "dtype": "uint16",
                                        "array_axes": ["z", "y", "x"],
                                    },
                                    "ct": {
                                        "path": intake["ct_volume"]["path"],
                                        "sha256": intake["ct_volume"]["sha256"],
                                    },
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                if stage_number == 5 and role == "sealed_evaluation_result":
                    attempt = self.manifest()["stages"]["5"]["attempts"][-1]
                    by_input_role = {
                        item["role"]: item for item in attempt["input_artifacts"]
                    }
                    payload = (
                        json.dumps(
                            {
                                "schema_version": "part2-detection-metrics/1.0.0",
                                "gate": "pass",
                                "overall_pass": True,
                                "protocol": "one_shot_reporting_not_pass_fail",
                                "sealed_strut_count": 1,
                                "strict_recall": {
                                    "definition": "predicted missing",
                                    "detected": 0,
                                    "total": 1,
                                    "value": 0.0,
                                    "wilson_95_ci": [0.0, 0.7934506856],
                                },
                                "lenient_recall": {
                                    "definition": "predicted missing or broken",
                                    "detected": 0,
                                    "total": 1,
                                    "value": 0.0,
                                    "wilson_95_ci": [0.0, 0.7934506856],
                                },
                                "confusion_matrix": {
                                    "class_order": [
                                        "missing",
                                        "broken",
                                        "thin",
                                        "present",
                                    ],
                                    "rows_actual_columns_predicted": {
                                        actual: {
                                            predicted: 0
                                            for predicted in (
                                                "missing",
                                                "broken",
                                                "thin",
                                                "present",
                                            )
                                        }
                                        for actual in (
                                            "missing",
                                            "broken",
                                            "thin",
                                            "present",
                                        )
                                    },
                                },
                                "omitted_metrics": {
                                    "precision": (
                                        "undefined because detections outside sealed "
                                        "intentional deletions may be unintentional defects"
                                    ),
                                    "f1": "not computed because precision is undefined",
                                },
                                "artifacts": {},
                                "hashes": {
                                    "classifications_sha256": by_input_role[
                                        "classified_struts"
                                    ]["sha256"],
                                    "sealed_labels_sha256": by_input_role[
                                        "sealed_labels"
                                    ]["sha256"],
                                },
                                "provenance": {
                                    "eval_side": True,
                                    "sealed_labels_read": True,
                                },
                                "warnings": [],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                records.append(self.record_for_rule(rule, role, payload=payload))
            return records

        non_verifier = [
            role
            for role in roles
            if role
            not in {
                "missing_calibration_attestation",
                "classifier_verifier_report",
            }
        ]
        records = [self.output_record(stage_number, role) for role in non_verifier]
        by_role = {str(record["role"]): record for record in records}
        manifest = self.manifest()
        attempt = manifest["stages"]["4"]["attempts"][-1]
        scoped_handoff = next(
            record
            for record in attempt["scoped_handoffs"]
            if record["consumer"] == "missing_strut_agent"
        )
        attestation = {
            "schema_version": "missing-calibration-attestation/1.0.0",
            "owner": "missing_strut_agent",
            "gate": "pass",
            "specimen_id": manifest["specimen_id"],
            "stage_number": 4,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "scoped_handoff_sha256": scoped_handoff["canonical_sha256"],
            "per_strut_metrics_sha256": next(
                item["sha256"]
                for item in attempt["input_artifacts"]
                if item["role"] == "per_strut_metrics"
            ),
            "findings_missing_sha256": by_role["findings_missing"]["sha256"],
            "development_split_accessed": True,
            "raw_development_labels_included": False,
            "calibration_summary": {
                "method": "development_split_calibration",
                "development_sample_count": 1,
                "selected_missing_boundary": 0.5,
            },
        }
        attestation_rule = self._rule_for(
            contract, "output", "missing_calibration_attestation"
        )
        attestation_path = self._materialize_path(
            str(attestation_rule["path"]), "missing_calibration_attestation"
        )
        self._write_json(self.root / attestation_path, attestation)
        attestation_record = self.record_for_rule(
            attestation_rule, "missing_calibration_attestation"
        )
        records.append(attestation_record)
        by_role["missing_calibration_attestation"] = attestation_record
        evidence = sorted(
            (
                {"path": record["path"], "sha256": record["sha256"]}
                for record in records
                if record["role"]
                in {
                    "evidence_packets",
                    "evidence_packet_manifest",
                    "evidence_index",
                }
            ),
            key=lambda record: (record["path"], record["sha256"]),
        )
        verifier = {
            "schema_version": "classifier-verifier-report/1.0.0",
            "owner": "classifier_verifier",
            "gate": "pass",
            "specimen_id": manifest["specimen_id"],
            "stage_number": 4,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": manifest["stages"]["4"]["contract"]["sha256"],
            "predecessor_receipt_sha256": attempt[
                "predecessor_receipt_sha256"
            ],
            "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
            "participated_in_classification": False,
            "label_access": {
                "development_split_read": False,
                "sealed_split_read": False,
            },
            "bindings": {
                "classified_struts_sha256": by_role["classified_struts"]["sha256"],
                "thresholds_sha256": by_role["classification_thresholds"]["sha256"],
                "decision_log_sha256": by_role["classification_decision_log"]["sha256"],
                "evidence_set_sha256": canonical_json_sha256(evidence),
                "per_strut_metrics_sha256": next(
                    item["sha256"]
                    for item in attempt["input_artifacts"]
                    if item["role"] == "per_strut_metrics"
                ),
                "specialist_findings_sha256": {
                    role: by_role[role]["sha256"]
                    for role in (
                        "findings_missing",
                        "missing_calibration_attestation",
                        "findings_thin",
                        "findings_bent",
                        "findings_broken",
                    )
                },
            },
            "self_verification": {
                "every_strut_labeled_once": True,
                "fixed_precedence_respected": True,
                "bent_kept_separate": True,
                "every_adjudication_logged": True,
                "evidence_support_checked": True,
                "cutoffs_audited": True,
                "decision_log_matches_execution": True,
                "development_split_not_accessed": True,
                "sealed_split_not_accessed": True,
            },
        }
        if verifier_updates:
            for key, value in copy.deepcopy(verifier_updates).items():
                if isinstance(value, dict) and isinstance(verifier.get(key), dict):
                    verifier[key].update(value)
                else:
                    verifier[key] = value
        rule = self._rule_for(contract, "output", "classifier_verifier_report")
        verifier_path = self._materialize_path(str(rule["path"]), "classifier_verifier_report")
        self._write_json(self.root / verifier_path, verifier)
        records.append(
            self.record_for_rule(rule, "classifier_verifier_report")
        )
        return records

    def _stage0_outputs(self, roles: list[str]) -> list[dict[str, object]]:
        attempt = self.manifest(verify_artifacts=False)["stages"]["0"][
            "attempts"
        ][-1]
        inputs = {
            str(record["role"]): record for record in attempt["input_artifacts"]
        }
        ct_path = self.root / str(inputs["ct_volume"]["path"])
        metadata_path, metadata_sha256 = write_ct_metadata_response_fixture(
            repository_root=self.root,
            specimen_id=self.specimen_id,
            ct_path=ct_path,
            shape=(2, 3, 4),
            dtype="uint8",
            dtype_string="|u1",
            byte_order="not_applicable",
            volume_format="npy",
            retention="committed",
            array_axes=["z", "y", "x"],
        )
        call_receipt_path = metadata_path.with_name(
            "ct_metadata_mcp_call_receipt.json"
        )
        result = ingest_specimen(
            repository_root=self.root,
            specimen_id=self.specimen_id,
            design_id=self.design_id,
            requested_analysis_scope=self.requested_analysis_scope,
            cad_path=self.root / str(inputs["cad_stl"]["path"]),
            design_graph_path=self.root / str(inputs["nominal_graph"]["path"]),
            ct_path=ct_path,
            ct_metadata_response_path=metadata_path,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=call_receipt_path,
            ct_metadata_call_receipt_sha256=specimen_sha256_file(
                call_receipt_path
            ),
            registration_mode=self.registration_mode,
            association_confirmed=True,
            allowed_data_roots=[self.root / "inputs"],
            aligned_graph_path=(
                self.root / str(inputs["challenge_aligned_graph"]["path"])
                if self.registration_mode == "challenge_aligned_json"
                else None
            ),
            design_transform_declaration_path=self.root
            / str(inputs["design_transform_declaration"]["path"]),
            cad_units="millimeter",
            cad_units_provenance="synthetic scientist declaration",
            graph_axes="xyz",
            array_axes="zyx",
            aligned_graph_units=(
                "voxel"
                if self.registration_mode == "challenge_aligned_json"
                else "simulation_voxel"
            ),
            retention="committed",
            schema_path=self.root
            / "analysis"
            / "schema"
            / "specimen_manifest.schema.json",
        )
        create_data_prep_handoff(
            Path(result["paths"]["specimen_manifest"]),
            Path(result["paths"]["ingest_receipt"]),
            repository_root=self.root,
            schema_path=self.root
            / "analysis"
            / "schema"
            / "specimen_manifest.schema.json",
        )
        return [self.output_record(0, role) for role in roles]

    def _output_binding(self, stage_number: int) -> dict[str, object]:
        manifest = self.manifest(verify_artifacts=False)
        attempt = manifest["stages"][str(stage_number)]["attempts"][-1]
        return {
            "schema_version": "part2-stage-output-binding/1.0.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "stage_number": stage_number,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": manifest["stages"][str(stage_number)]["contract"][
                "sha256"
            ],
        }

    def _bound_stage_outputs(
        self, stage_number: int, roles: list[str]
    ) -> list[dict[str, object]]:
        contract = self.contracts[stage_number]
        manifest = self.manifest(verify_artifacts=False)
        attempt = manifest["stages"][str(stage_number)]["attempts"][-1]
        inputs = {item["role"]: item for item in attempt["input_artifacts"]}
        if stage_number == 1:
            return self._stage1_outputs(contract, roles, inputs, manifest)
        return self._stage2_outputs(contract, roles, inputs, manifest)

    def _json_output(
        self,
        contract: dict[str, object],
        role: str,
        document: dict[str, object],
        *,
        replaces_sha256: str | None = None,
    ) -> dict[str, object]:
        rule = self._rule_for(contract, "output", role)
        return self.record_for_rule(
            rule,
            role,
            payload=(json.dumps(document, sort_keys=True) + "\n").encode(),
            overwrite=True,
            replaces_sha256=replaces_sha256,
        )

    def _stage1_outputs(
        self,
        contract: dict[str, object],
        roles: list[str],
        inputs: dict[str, dict[str, object]],
        manifest: dict[str, object],
    ) -> list[dict[str, object]]:
        by_role: dict[str, dict[str, object]] = {}
        declaration = inputs["design_transform_declaration"]
        declaration_path = str(declaration["path"])
        orientation = {
            "schema_version": "part2-cad-graph-orientation/1.0.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "gate": "pass",
            "overall_pass": True,
            "resolution_source": "declared_transform",
            "declared_transform": {
                "present": True,
                "artifact_path": declaration_path,
                "artifact_sha256": declaration["sha256"],
                "expected_artifact_sha256": declaration["sha256"],
                "verification": {
                    "overall_pass": True,
                    "reason_codes": ["declared_transform_verified"],
                },
            },
            "ambiguity": {
                "equivalent_hypothesis_count": 24,
                "requires_scientist_review": False,
                "resolved_by_declaration": True,
            },
            "gates": {
                "graph_counts_match": True,
                "nominal_ids_unique": True,
                "orientation_unambiguous": False,
                "orientation_resolved": True,
                "scale_preserving_transform": True,
                "declared_transform_valid": True,
                "authoritative_transform_verified": True,
            },
            "hashes": {
                "nominal_graph_sha256": inputs["nominal_graph"]["sha256"],
                "full_design_stl_sha256": inputs["full_design_stl"]["sha256"],
                "declared_transform_artifact_sha256": declaration["sha256"],
                "intake_declared_transform_artifact_sha256": declaration["sha256"],
                "config_sha256": manifest["config"]["sha256"],
            },
            "provenance": {
                "design_space_only": True,
                "ct_accessed": False,
                "aligned_graph_accessed": False,
                "deleted_edge_labels_accessed": False,
            },
        }
        by_role["cad_graph_orientation"] = self._json_output(
            contract, "cad_graph_orientation", orientation
        )
        by_role["normalized_nominal_graph"] = self.output_record(
            1, "normalized_nominal_graph"
        )
        variant_sources = {
            "intentional_deletions_0p1": ("0p1", "intentional_deletion_stl_0p1", 18),
            "intentional_deletions_0p5": ("0p5", "intentional_deletion_stl_0p5", 93),
            "intentional_deletions_1p0": ("1p0", "intentional_deletion_stl_1p0", 186),
        }
        for role, (variant, source_role, count) in variant_sources.items():
            document = {
                "schema_version": "part2-design-labels/1.0.0",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "variant": variant,
                "deleted_count": count,
                "deleted_strut_ids": list(range(count)),
                "hashes": {
                    "nominal_graph_sha256": inputs["nominal_graph"]["sha256"],
                    "orientation_sha256": by_role["cad_graph_orientation"]["sha256"],
                    "baseline_stl_sha256": inputs["full_design_stl"]["sha256"],
                    "variant_stl_sha256": inputs[source_role]["sha256"],
                    "config_sha256": manifest["config"]["sha256"],
                },
                "provenance": {
                    "design_space_only": True,
                    "ct_accessed": False,
                    "aligned_graph_accessed": False,
                },
            }
            by_role[role] = self._json_output(contract, role, document)
        source_hash = by_role["intentional_deletions_0p5"]["sha256"]
        for role, ids in (
            ("development_labels", list(range(47))),
            ("sealed_labels", list(range(47, 93))),
        ):
            by_role[role] = self._json_output(
                contract,
                role,
                {
                    "schema_version": "part2-label-split/1.0.0",
                    "role": role,
                    "specimen_id": self.specimen_id,
                    "design_id": self.design_id,
                    "source_variant": "0p5",
                    "source_labels_sha256": source_hash,
                    "strut_ids": ids,
                    "config_sha256": manifest["config"]["sha256"],
                },
            )
        label_report_rule = self._rule_for(contract, "output", "label_report")
        by_role["label_report"] = self.record_for_rule(
            label_report_rule,
            "label_report",
            payload=b"# Synthetic design-diff label report\n\nGate: `pass`\n",
            overwrite=True,
        )
        return [by_role[role] for role in roles]

    def _analysis_ready_document(
        self,
        *,
        inputs: dict[str, dict[str, object]],
        by_role: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        template_path = (
            REPOSITORY_ROOT
            / "analysis/brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json"
        )
        document = json.loads(template_path.read_text(encoding="utf-8"))
        stage_zero = self.manifest(verify_artifacts=False)["stages"]["0"][
            "attempts"
        ][0]["input_artifacts"]
        intake = {item["role"]: item for item in stage_zero}

        def artifact(source: dict[str, object], role: str) -> dict[str, object]:
            return {
                "path": source["path"],
                "sha256": source["sha256"],
                "role": role,
                "retention": "committed",
            }

        if self.registration_mode == "challenge_aligned_json":
            aligned = artifact(inputs["challenge_aligned_graph"], "aligned_graph")
            aligned_state = "input"
        else:
            aligned = artifact(by_role["localized_graph"], "derived_aligned_graph")
            aligned_state = "derived"
        canonical_mask = {
            **artifact(
                by_role["canonical_segmentation_mask"],
                "canonical_segmentation_mask",
            ),
            "dtype": "uint8",
            "shape": [2, 3, 4],
            "array_axes": ["z", "y", "x"],
        }
        document.update(
            {
                "schema_version": "2.1.0",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "lifecycle_state": "analysis_ready",
                "unresolved_fields": [],
                "inputs": {
                    "ct": artifact(inputs["ct_volume"], "ct_volume"),
                    "ct_metadata": {
                        "format": "npy",
                        "shape": [2, 3, 4],
                        "dtype": "uint16",
                        "byte_order": "little",
                        "array_axes": ["z", "y", "x"],
                    },
                    "design_graph": artifact(inputs["nominal_graph"], "design_graph"),
                    "design_transform_declaration": artifact(
                        intake["design_transform_declaration"],
                        "design_transform_declaration",
                    ),
                    "aligned_graph": aligned,
                    "canonical_mask": canonical_mask,
                    "cad": artifact(intake["cad_stl"], "cad"),
                },
            }
        )
        parameters = document["analysis_parameters"]
        parameters["requested_analysis_scope"] = self.requested_analysis_scope
        parameters["registration"]["mode"] = self.registration_mode
        parameters["artifact_schema_versions"]["node_localization"] = "1.2.0"
        parameters["artifact_schema_versions"]["registration_qa"] = "1.2.0"
        parameters_hash = canonical_json_sha256(parameters)
        document["analysis_parameters_sha256"] = parameters_hash
        input_hashes = {
            name: value["sha256"]
            for name, value in document["inputs"].items()
            if name != "ct_metadata"
        }
        derived = document["derived"]
        provenance_inputs = {
            "graph_summary": sorted(
                {input_hashes["design_graph"], input_hashes["aligned_graph"]}
            ),
            "voxel_spacing": sorted(
                {
                    input_hashes["ct"],
                    input_hashes["cad"],
                    input_hashes["aligned_graph"],
                }
            ),
            "segmentation_result": [input_hashes["ct"]],
            "registration_result": [input_hashes["aligned_graph"]],
        }
        for name, hashes in provenance_inputs.items():
            derived[name]["provenance"]["input_sha256"] = hashes
            derived[name]["provenance"]["config_sha256"] = parameters_hash
        derived["segmentation_result"]["values"].update(
            {
                "threshold": 40054,
                "voxel_count": 24,
                "foreground_voxel_count": 6,
                "foreground_fraction": 0.25,
                "overall_pass": True,
                "histogram_sha256": by_role["exact_histogram"]["sha256"],
            }
        )
        authorized, unauthorized = self._stage2_authorizations()
        registration = derived["registration_result"]["values"]
        registration.update(
            {
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "aligned_graph_state": aligned_state,
                "requested_analysis_scope": self.requested_analysis_scope,
                "overall_pass": True,
                "local_recenter_complete": True,
                "roi_gate_pass": True,
                "metrology_gate_status": (
                    "not_authorized"
                    if self.requested_analysis_scope == "roi_screening"
                    else "pass"
                ),
                "authorized_outputs": authorized,
                "unauthorized_outputs": unauthorized,
                "roi_gate_results": {
                    "image_support": True,
                    "localization_quality": True,
                    "coarse_region_support": True,
                    "padded_roi_in_bounds": True,
                },
                "localization_quality_counts": self._localization_counts(),
                "reason_codes": self._stage2_reason_codes(),
            }
        )
        return document

    @staticmethod
    def _localization_counts() -> dict[str, int]:
        return {
            "primary_nodes": 8714,
            "stable_coarse_nodes": 1098,
            "fallback_nodes": 394,
            "ambiguous_nodes": 0,
            "rejected_or_low_confidence_nodes": 394,
            "boundary_limited_nodes": 0,
            "primary_edges": 0,
            "stable_coarse_edges": 0,
            "fallback_edges": 0,
            "ambiguous_edges": 0,
            "roi_screening_usable_edges": 0,
            "direct_metrology_usable_edges": 0,
        }

    def _stage2_outputs(
        self,
        contract: dict[str, object],
        roles: list[str],
        inputs: dict[str, dict[str, object]],
        manifest: dict[str, object],
    ) -> list[dict[str, object]]:
        prior_manifest_path = self.root / str(inputs["specimen_manifest"]["path"])
        prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
        prior_manifest_canonical_sha256 = canonical_json_sha256(prior_manifest)
        prior_manifest_artifact_sha256 = sha256_file(prior_manifest_path)
        by_role: dict[str, dict[str, object]] = {}
        for role in (
            "exact_histogram",
            "canonical_segmentation_mask",
            "junction_overlay",
            "spatial_bias_figure",
        ):
            by_role[role] = self.output_record(2, role)
        by_role["analysis_config"] = self._json_output(
            contract,
            "analysis_config",
            {
                "schema_version": "part2-analysis-config/1.0.0",
                "registration_mode": self.registration_mode,
                "threshold": 40054.0,
                "input_hashes": {
                    "ct_sha256": inputs["ct_volume"]["sha256"],
                    "nominal_graph_sha256": inputs["nominal_graph"]["sha256"],
                },
                "label_inputs_accessed": False,
            },
        )
        otsu_recipe = {
            "histogram_encoding": "native_uint16",
            "edge_slices_excluded": 0,
            "chunk_voxels": 96,
            "coarse_bins": 1024,
            "peak_smoothing_sigma_bins": 2.0,
            "peak_prominence_fraction": 0.003,
            "minimum_significant_peaks": 2,
            "minimum_foreground_fraction": 0.01,
            "maximum_foreground_fraction": 0.35,
            "minimum_otsu_separability": 0.45,
            "minimum_class_mean_separation_sigma": 0.75,
        }
        by_role["otsu_report"] = self._json_output(
            contract,
            "otsu_report",
            {
                "schema_version": "exact-otsu-replay/1.0.0",
                "method": "exact_histogram_otsu",
                "method_version": "2.0.0",
                "overall_pass": True,
                "threshold": 40054,
                "threshold_histogram_bin": 40054,
                "threshold_comparison": "value >= threshold",
                "histogram_encoding": {
                    "encoding": "native_uint16",
                    "native_dtype": "uint16",
                    "native_min": 0.0,
                    "native_max": 65535.0,
                    "native_units_per_bin": 1.0,
                },
                "recipe": otsu_recipe,
                "voxel_count": 24,
                "foreground_voxel_count": 6,
                "foreground_fraction": 0.25,
                "otsu_separability": 0.814158617485167,
                "background_mean": 32615.08340378344,
                "foreground_mean": 47493.56062968888,
                "class_mean_separation_sigma": 4.321652770590698,
                "significant_modes": [32288.0, 48992.0],
                "histogram_sha256": by_role["exact_histogram"]["sha256"],
                "gates": {
                    "foreground_fraction_plausible": True,
                    "otsu_separability_sufficient": True,
                    "class_mean_separation_sufficient": True,
                    "histogram_not_unimodal": True,
                },
                "source_path": inputs["ct_volume"]["path"],
                "registration_mode": self.registration_mode,
                "hashes": {
                    "input_sha256": inputs["ct_volume"]["sha256"],
                    "config_sha256": canonical_json_sha256(
                        {
                            "recipe": otsu_recipe,
                            "registration_mode": self.registration_mode,
                            "enforce_reference_replay": False,
                        }
                    ),
                },
                "provenance": {
                    "registration_mode": self.registration_mode,
                    "threshold_selected_per_scan": True,
                    "target_foreground_fraction_used": False,
                    "defect_labels_read": False,
                },
            },
        )
        by_role["segmentation_mask_comparison"] = self._json_output(
            contract,
            "segmentation_mask_comparison",
            {
                "status": "ok",
                "raw_path": inputs["ct_volume"]["path"],
                "overall_pass": True,
                "shape": [2, 3, 4],
                "candidates": [
                    {
                        "threshold": 40054.0,
                        "path": by_role["canonical_segmentation_mask"]["path"],
                        "sha256": by_role["canonical_segmentation_mask"]["sha256"],
                        "dtype": "uint8",
                        "foreground_voxels": 6,
                        "expected_foreground_voxels": 6,
                        "total_voxels": 24,
                        "foreground_percent": 25.0,
                        "mismatched_voxels": 0,
                        "false_positive_voxels": 0,
                        "false_negative_voxels": 0,
                        "exact_threshold_match": True,
                    }
                ],
                "registration_mode": self.registration_mode,
                "config_sha256": canonical_json_sha256(
                    {
                        "thresholds": [40054.0],
                        "registration_mode": self.registration_mode,
                        "chunk_depth": 16,
                    }
                ),
            },
        )
        graph_document = {
            "schema_version": "normalized-lattice-graph/1.0.0",
            "junctions": [],
            "struts": [],
            "unit_cells": [],
        }
        by_role["registered_graph"] = self._json_output(
            contract, "registered_graph", graph_document
        )
        by_role["registration_report"] = self._json_output(
            contract,
            "registration_report",
            {
                "schema_version": "part2-registration/1.0.0",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "mode": self.registration_mode,
                "gate": "pass",
                "overall_pass": True,
                "mode_details": (
                    {"threshold": 40054.0}
                    if self.registration_mode == "autonomous_v2"
                    else {}
                ),
                "hashes": {
                    "ct_sha256": inputs["ct_volume"]["sha256"],
                    "nominal_graph_sha256": inputs["nominal_graph"]["sha256"],
                    "registered_graph_sha256": by_role["registered_graph"]["sha256"],
                },
            },
        )
        by_role["localized_graph"] = self._json_output(
            contract, "localized_graph", graph_document
        )
        finalized = self._analysis_ready_document(inputs=inputs, by_role=by_role)
        analysis_parameters_sha256 = finalized["analysis_parameters_sha256"]
        segmentation_policy_sha256 = canonical_json_sha256(
            finalized["analysis_parameters"]["segmentation"]
        )
        otsu_document = json.loads(
            (
                self.root / str(by_role["otsu_report"]["path"])
            ).read_text(encoding="utf-8")
        )
        comparison_document = json.loads(
            (
                self.root / str(by_role["segmentation_mask_comparison"]["path"])
            ).read_text(encoding="utf-8")
        )
        verification_request = {
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "analysis_policy_artifact_filepath": inputs["specimen_manifest"]["path"],
            "exact_otsu_report_filepath": by_role["otsu_report"]["path"],
            "canonical_mask_filepath": by_role["canonical_segmentation_mask"]["path"],
            "mask_comparison_report_filepath": by_role[
                "segmentation_mask_comparison"
            ]["path"],
            "output_filepath": self._materialize_path(
                str(
                    self._rule_for(
                        contract, "output", "segmentation_verification_mcp_response"
                    )["path"]
                ),
                "segmentation_verification_mcp_response",
            ),
            "registration_mode": self.registration_mode,
            "overwrite": False,
        }
        verification_result = {
            "threshold": otsu_document["threshold"],
            "threshold_comparison": otsu_document["threshold_comparison"],
            "shape": [2, 3, 4],
            "dtype": "uint8",
            "voxel_count": otsu_document["voxel_count"],
            "foreground_voxel_count": otsu_document["foreground_voxel_count"],
            "foreground_fraction": otsu_document["foreground_fraction"],
            "otsu_separability": otsu_document["otsu_separability"],
            "background_mean": otsu_document["background_mean"],
            "foreground_mean": otsu_document["foreground_mean"],
            "class_mean_separation_sigma": otsu_document[
                "class_mean_separation_sigma"
            ],
            "significant_modes": otsu_document["significant_modes"],
            "histogram_sha256": otsu_document["histogram_sha256"],
            "mismatched_voxels": comparison_document["candidates"][0][
                "mismatched_voxels"
            ],
            "false_positive_voxels": comparison_document["candidates"][0][
                "false_positive_voxels"
            ],
            "false_negative_voxels": comparison_document["candidates"][0][
                "false_negative_voxels"
            ],
            "exact_threshold_match": comparison_document["candidates"][0][
                "exact_threshold_match"
            ],
            "overall_pass": True,
        }
        verification_bindings = {
            "analysis_policy_artifact": {
                "path": inputs["specimen_manifest"]["path"],
                "sha256": prior_manifest_artifact_sha256,
                "role": "specimen_manifest",
            },
            "ct_volume": {
                "path": inputs["ct_volume"]["path"],
                "sha256": inputs["ct_volume"]["sha256"],
                "role": "ct_volume",
            },
            "exact_otsu_report": {
                "path": by_role["otsu_report"]["path"],
                "sha256": by_role["otsu_report"]["sha256"],
                "role": "otsu_report",
            },
            "canonical_mask": {
                "path": by_role["canonical_segmentation_mask"]["path"],
                "sha256": by_role["canonical_segmentation_mask"]["sha256"],
                "role": "canonical_segmentation_mask",
                "dtype": "uint8",
                "shape": [2, 3, 4],
                "array_axes": ["z", "y", "x"],
            },
            "mask_comparison_report": {
                "path": by_role["segmentation_mask_comparison"]["path"],
                "sha256": by_role["segmentation_mask_comparison"]["sha256"],
                "role": "segmentation_mask_comparison",
            },
        }
        verification_hashes = {
            "request_sha256": canonical_json_sha256(verification_request),
            "analysis_policy_artifact_sha256": prior_manifest_artifact_sha256,
            "analysis_parameters_sha256": finalized[
                "analysis_parameters_sha256"
            ],
            "segmentation_policy_sha256": segmentation_policy_sha256,
            "ct_sha256": inputs["ct_volume"]["sha256"],
            "exact_otsu_report_sha256": by_role["otsu_report"]["sha256"],
            "canonical_mask_sha256": by_role["canonical_segmentation_mask"][
                "sha256"
            ],
            "mask_comparison_report_sha256": by_role[
                "segmentation_mask_comparison"
            ]["sha256"],
        }
        by_role["segmentation_verification_mcp_response"] = self._json_output(
            contract,
            "segmentation_verification_mcp_response",
            {
                "schema_version": "segmentation-verification-mcp-evidence/1.0.0",
                "response_schema_version": "part2-mcp-response/1.0.0",
                "tool": "verify_canonical_segmentation",
                "status": "ok",
                "gate": "pass",
                "summary": "Persisted canonical segmentation verification",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "registration_mode": self.registration_mode,
                "request": verification_request,
                "policy": {
                    "analysis_parameters_sha256": finalized[
                        "analysis_parameters_sha256"
                    ],
                    "segmentation_policy_sha256": segmentation_policy_sha256,
                },
                "result": verification_result,
                "bindings": verification_bindings,
                "hashes": verification_hashes,
                "warnings": [],
                "error": None,
            },
        )
        segmentation_binding = {
            "method": "exact_histogram_otsu",
            "method_version": "2.0.0",
            "threshold": 40054,
            "threshold_comparison": "value >= threshold",
            "ct_sha256": inputs["ct_volume"]["sha256"],
            "segmentation_policy_sha256": segmentation_policy_sha256,
            "overall_pass": True,
        }
        quality_counts = self._localization_counts()
        by_role["localization_report"] = self._json_output(
            contract,
            "localization_report",
            {
                "schema_version": "part2-node-localization/1.2.0",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "registration_mode": self.registration_mode,
                "gate": "pass",
                "overall_pass": True,
                "threshold": 40054.0,
                "segmentation_binding": segmentation_binding,
                "counts": {
                    "nodes": 10206,
                    "edges": 18468,
                    "cells": 729,
                    **quality_counts,
                },
                "hashes": {
                    "ct_sha256": inputs["ct_volume"]["sha256"],
                    "input_registered_graph_sha256": by_role["registered_graph"]["sha256"],
                    "localized_graph_sha256": by_role["localized_graph"]["sha256"],
                    "registration_report_sha256": by_role["registration_report"]["sha256"],
                    "analysis_parameters_sha256": analysis_parameters_sha256,
                },
            },
        )
        authorized, unauthorized = self._stage2_authorizations()
        roi_gates = {
            "image_support": True,
            "localization_quality": True,
            "coarse_region_support": True,
            "padded_roi_in_bounds": True,
        }
        by_role["registration_qa"] = self._json_output(
            contract,
            "registration_qa",
            {
                "schema_version": "part2-registration-qa/1.2.0",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "registration_mode": self.registration_mode,
                "gate": "pass",
                "overall_pass": True,
                "threshold": 40054.0,
                "segmentation_binding": {
                    **segmentation_binding,
                    "gates": {
                        "exact_otsu_replay_passed": True,
                        "threshold_matches_exact_otsu": True,
                    },
                },
                "authorized_outputs": authorized,
                "unauthorized_outputs": unauthorized,
                "reason_codes": self._stage2_reason_codes(),
                "roi_gate_results": roi_gates,
                "localization_quality_counts": quality_counts,
                "counts": {"nodes": 10206, "edges": 18468, "cells": 729},
                "metrology": {
                    "status": (
                        "not_authorized"
                        if self.requested_analysis_scope == "roi_screening"
                        else "pass"
                    )
                },
                "hashes": {
                    "ct_sha256": inputs["ct_volume"]["sha256"],
                    "localized_graph_sha256": by_role["localized_graph"]["sha256"],
                    "localization_report_sha256": by_role["localization_report"]["sha256"],
                    "registration_report_sha256": by_role["registration_report"]["sha256"],
                    "analysis_parameters_sha256": analysis_parameters_sha256,
                },
            },
        )
        canonical_mask = finalized["inputs"]["canonical_mask"]
        aligned_graph = finalized["inputs"]["aligned_graph"]
        result_document = {
            "schema_version": "data-prep-result/1.2.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "input_manifest_sha256": prior_manifest_canonical_sha256,
            "input_manifest_artifact_sha256": prior_manifest_artifact_sha256,
            "analysis_parameters_sha256": finalized["analysis_parameters_sha256"],
            "registration_mode": self.registration_mode,
            "requested_analysis_scope": self.requested_analysis_scope,
            "authorized_outputs": authorized,
            "unauthorized_outputs": unauthorized,
            "roi_gate_results": roi_gates,
            "metrology_gate_status": (
                "not_authorized"
                if self.requested_analysis_scope == "roi_screening"
                else "pass"
            ),
            "localization_quality_counts": quality_counts,
            "reason_codes": self._stage2_reason_codes(),
            "artifact_bindings": {
                role: {
                    "path": by_role[role]["path"],
                    "sha256": by_role[role]["sha256"],
                    "role": role,
                }
                for role in (
                    "segmentation_verification_mcp_response",
                    "localization_report",
                    "registration_qa",
                )
            },
            "aligned_graph": aligned_graph,
            "canonical_mask": canonical_mask,
            "derived": finalized["derived"],
            "self_verification": {
                "exact_otsu_complete": True,
                "registration_complete": True,
                "local_recenter_complete": True,
                "roi_gate_pass": True,
                "scope_bound_to_hashed_intake": True,
                "localization_quality_propagated": True,
                "defect_labels_not_accessed": True,
            },
        }
        by_role["data_prep_result"] = self._json_output(
            contract, "data_prep_result", result_document
        )
        completion_base = {
            "schema_version": "data-prep-completion/1.2.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "requested_analysis_scope": self.requested_analysis_scope,
            "authorized_outputs": authorized,
            "unauthorized_outputs": unauthorized,
            "roi_gate_results": roi_gates,
            "metrology_gate_status": result_document["metrology_gate_status"],
            "localization_quality_counts": quality_counts,
            "reason_codes": self._stage2_reason_codes(),
            "artifact_bindings": result_document["artifact_bindings"],
            "prior_manifest_sha256": prior_manifest_canonical_sha256,
            "prior_manifest_artifact_sha256": prior_manifest_artifact_sha256,
            "analysis_ready_manifest_sha256": canonical_json_sha256(finalized),
            "data_prep_result_sha256": canonical_json_sha256(result_document),
            "analysis_parameters_sha256": finalized["analysis_parameters_sha256"],
            "lifecycle_state": "analysis_ready",
            "registration_mode": self.registration_mode,
            "self_verification": result_document["self_verification"],
            "canonical_mask": canonical_mask,
        }
        by_role["data_prep_completion_receipt"] = self._json_output(
            contract,
            "data_prep_completion_receipt",
            {
                **completion_base,
                "canonical_completion_sha256": canonical_json_sha256(
                    completion_base
                ),
            },
        )
        return [
            by_role[role]
            for role in roles
            if role != "analysis_ready_specimen_manifest"
        ]

    def analysis_ready_manifest_output(self) -> dict[str, object]:
        manifest = self.manifest(verify_artifacts=False)
        attempt = manifest["stages"]["2"]["attempts"][-1]
        prior = next(
            item
            for item in attempt["input_artifacts"]
            if item["role"] == "specimen_manifest"
        )
        rule = self._rule_for(
            self.contracts[2], "output", "analysis_ready_specimen_manifest"
        )
        inputs = {item["role"]: item for item in attempt["input_artifacts"]}
        by_role: dict[str, dict[str, object]] = {}
        for role in (
            "exact_histogram",
            "canonical_segmentation_mask",
            "localized_graph",
        ):
            output_rule = self._rule_for(self.contracts[2], "output", role)
            path = self._materialize_path(str(output_rule["path"]), role)
            by_role[role] = artifact_record(self.root, path, role=role)
        replacement = self._analysis_ready_document(inputs=inputs, by_role=by_role)
        return self.record_for_rule(
            rule,
            "analysis_ready_specimen_manifest",
            payload=(json.dumps(replacement, sort_keys=True) + "\n").encode(),
            overwrite=True,
            replaces_sha256=str(prior["sha256"]),
        )

    def _stage2_authorizations(self) -> tuple[list[str], list[str]]:
        base = sorted(
            {
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
            }
        )
        metrology = sorted({"absolute_metrology", "direct_dimensional_measurement"})
        if self.requested_analysis_scope == "roi_screening":
            return base, metrology
        return sorted(base + metrology), []

    def _stage2_reason_codes(self) -> list[str]:
        return sorted(
            [
                "ROI_GATES_PASS",
                (
                    "METROLOGY_NOT_AUTHORIZED"
                    if self.requested_analysis_scope == "roi_screening"
                    else "METROLOGY_GATES_PASS"
                ),
            ]
        )

    def stage_policy(
        self,
        stage_number: int,
        outputs: list[dict[str, object]],
        *,
        terminal_state: str = "pass",
    ) -> dict[str, object] | None:
        if stage_number not in {1, 2}:
            return None
        manifest = self.manifest(verify_artifacts=False)
        attempt = manifest["stages"][str(stage_number)]["attempts"][-1]
        source_hashes = {
            str(item["role"]): str(item["sha256"])
            for item in attempt["input_artifacts"]
        }
        output_hashes = {
            str(item["role"]): str(item["sha256"])
            for item in outputs
            if item["role"] != "manual_review_evidence"
        }
        if stage_number == 1:
            authorized = sorted(output_hashes) if terminal_state == "pass" else []
            unauthorized = sorted(
                {
                    "segmentation",
                    "registration",
                    "node_localization",
                    "coarse_region_screening",
                    "padded_roi_definition",
                    "absolute_metrology",
                    "direct_dimensional_measurement",
                    "defect_classification",
                    "sealed_evaluation",
                }
            )
            roi = metrology = localization = None
            reasons = [
                "STAGE1_POLICY_PASS"
                if terminal_state == "pass"
                else "STAGE1_REVIEW_OR_HALT"
            ]
        else:
            authorized, unauthorized = self._stage2_authorizations()
            has_science = bool(
                set(output_hashes)
                - {
                    "data_prep_result",
                    "data_prep_completion_receipt",
                    "analysis_ready_specimen_manifest",
                }
            )
            gate = "pass" if terminal_state == "pass" or has_science else "not_run"
            roi = {
                name: gate
                for name in (
                    "segmentation",
                    "registration",
                    "localization",
                    "image_qa",
                    "coarse_region",
                    "padded_roi",
                )
            }
            localization = (
                {
                    "gate": gate,
                    "total_nodes": 10206,
                    "primary_matches": 8714,
                    "stable_coarse_matches": 1098,
                    "fallback_matches": 394,
                    "ambiguous_matches": 0,
                    "rejected_or_low_confidence": 394,
                    "boundary_limited": 0,
                }
                if gate != "not_run"
                else None
            )
            if self.requested_analysis_scope == "roi_screening":
                metrology = {
                    "status": "not_authorized",
                    "absolute_uncertainty_available": False,
                    "uncertainty_within_limit": None,
                    "direct_metrology_authorized": False,
                }
                reasons = self._stage2_reason_codes()
            elif terminal_state == "pass":
                metrology = {
                    "status": "pass",
                    "absolute_uncertainty_available": True,
                    "uncertainty_within_limit": True,
                    "direct_metrology_authorized": True,
                }
                reasons = self._stage2_reason_codes()
            else:
                authorized = sorted(
                    set(authorized)
                    - {"absolute_metrology", "direct_dimensional_measurement"}
                )
                unauthorized = sorted(
                    {"absolute_metrology", "direct_dimensional_measurement"}
                )
                metrology = {
                    "status": "manual_review" if has_science else "not_run",
                    "absolute_uncertainty_available": False,
                    "uncertainty_within_limit": None,
                    "direct_metrology_authorized": False,
                }
                reasons = sorted(
                    ["METROLOGY_EVIDENCE_MISSING", "ROI_GATES_PASS"]
                    if has_science
                    else ["STAGE2_REVIEW_OR_HALT"]
                )
        policy: dict[str, object] = {
            "schema_version": "part2-stage-policy/1.0.0",
            "stage_number": stage_number,
            "terminal_state": terminal_state,
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": manifest["stages"][str(stage_number)]["contract"][
                "sha256"
            ],
            "source_hashes": dict(sorted(source_hashes.items())),
            "requested_analysis_scope": self.requested_analysis_scope,
            "authorized_outputs": authorized,
            "unauthorized_outputs": unauthorized,
            "roi_gate_summary": roi,
            "metrology_summary": metrology,
            "localization_summary": localization,
            "reason_codes": sorted(reasons),
            "mcp_response_bindings": [],
            "output_hashes": dict(sorted(output_hashes.items())),
        }
        tools = self.contracts[stage_number]["required_dependencies"]["mcp_tools"]
        scientific = {
            role: digest
            for role, digest in output_hashes.items()
            if role
            not in {
                "data_prep_result",
                "data_prep_completion_receipt",
                "analysis_ready_specimen_manifest",
            }
        }
        include_tools = terminal_state == "pass" or bool(scientific)
        if include_tools:
            output_records = {
                str(item["role"]): item
                for item in outputs
                if item["role"] != "manual_review_evidence"
            }
            source_records = {
                str(item["role"]): item for item in attempt["input_artifacts"]
            }
            stage1_assignments = {
                "load_lattice_graph": ["normalized_nominal_graph"],
                "resolve_cad_graph_orientation": ["cad_graph_orientation"],
                "label_deleted_edges": [
                    "intentional_deletions_0p1",
                    "intentional_deletions_0p5",
                    "intentional_deletions_1p0",
                    "development_labels",
                    "sealed_labels",
                    "label_report",
                ],
            }
            stage2_assignments = {
                "volume_info": [],
                "replay_exact_otsu": ["exact_histogram", "otsu_report"],
                "segment_ct_dataset": ["canonical_segmentation_mask"],
                "compare_segmentation_masks": ["segmentation_mask_comparison"],
                "verify_canonical_segmentation": [
                    "segmentation_verification_mcp_response"
                ],
                "visualize_slice": ["junction_overlay"],
                "register_lattice_to_ct": [
                    "analysis_config",
                    "registered_graph",
                    "registration_report",
                ],
                "localize_lattice_nodes": [
                    "localized_graph",
                    "localization_report",
                ],
                "compute_registration_qa": [
                    "registration_qa",
                    "spatial_bias_figure",
                ],
            }

            def arguments_for(tool_name: str) -> dict[str, object]:
                if stage_number == 1:
                    if tool_name == "load_lattice_graph":
                        return {
                            "input_filepath": source_records["nominal_graph"]["path"],
                            "output_filepath": output_records["normalized_nominal_graph"]["path"],
                            "overwrite": False,
                        }
                    if tool_name == "resolve_cad_graph_orientation":
                        declaration = source_records["design_transform_declaration"]
                        return {
                            "nominal_graph_filepath": source_records["nominal_graph"]["path"],
                            "full_design_stl_filepath": source_records["full_design_stl"]["path"],
                            "output_filepath": output_records["cad_graph_orientation"]["path"],
                            "specimen_id": self.specimen_id,
                            "design_id": self.design_id,
                            "declared_transform_filepath": declaration["path"],
                            "declared_transform_sha256": declaration["sha256"],
                            "overwrite": False,
                        }
                    return {
                        "nominal_graph_filepath": source_records["nominal_graph"]["path"],
                        "baseline_stl_filepath": source_records["full_design_stl"]["path"],
                        "variant_stl_filepaths": {
                            "0p1": source_records["intentional_deletion_stl_0p1"]["path"],
                            "0p5": source_records["intentional_deletion_stl_0p5"]["path"],
                            "1p0": source_records["intentional_deletion_stl_1p0"]["path"],
                        },
                        "orientation_filepath": output_records["cad_graph_orientation"]["path"],
                        "output_directory": f"analysis/{self.specimen_id}/labels",
                        "specimen_id": self.specimen_id,
                        "design_id": self.design_id,
                        "development_split_filepath": output_records["development_labels"]["path"],
                        "sealed_split_filepath": output_records["sealed_labels"]["path"],
                        "label_report_filepath": output_records["label_report"]["path"],
                        "overwrite": False,
                    }
                ct_path = source_records["ct_volume"]["path"]
                if tool_name == "volume_info":
                    return {
                        "input_filepath": ct_path,
                        "include_sha256": True,
                        "registration_mode": self.registration_mode,
                    }
                if tool_name == "replay_exact_otsu":
                    return {
                        "input_filepath": ct_path,
                        "output_directory": f"analysis/{self.specimen_id}/segmentation",
                        "histogram_encoding": "native_uint16",
                        "edge_slices_excluded": 0,
                        "chunk_voxels": 96,
                        "coarse_bins": 1024,
                        "peak_smoothing_sigma_bins": 2.0,
                        "peak_prominence_fraction": 0.003,
                        "minimum_significant_peaks": 2,
                        "minimum_foreground_fraction": 0.01,
                        "maximum_foreground_fraction": 0.35,
                        "minimum_otsu_separability": 0.45,
                        "minimum_class_mean_separation_sigma": 0.75,
                        "registration_mode": self.registration_mode,
                        "enforce_reference_replay": False,
                        "reference_threshold": 40054,
                        "reference_foreground_voxels": 58653410,
                        "overwrite": False,
                    }
                if tool_name == "segment_ct_dataset":
                    return {
                        "input_filepath": ct_path,
                        "output_filepath": output_records["canonical_segmentation_mask"]["path"],
                        "threshold": 40054.0,
                        "registration_mode": self.registration_mode,
                        "retention": "committed",
                        "overwrite": False,
                    }
                if tool_name == "compare_segmentation_masks":
                    return {
                        "raw_filepath": ct_path,
                        "mask_filepaths": [output_records["canonical_segmentation_mask"]["path"]],
                        "thresholds": [40054.0],
                        "output_report_filepath": output_records["segmentation_mask_comparison"]["path"],
                        "registration_mode": self.registration_mode,
                        "overwrite": False,
                    }
                if tool_name == "verify_canonical_segmentation":
                    return {
                        "specimen_id": self.specimen_id,
                        "design_id": self.design_id,
                        "analysis_policy_artifact_filepath": source_records[
                            "specimen_manifest"
                        ]["path"],
                        "exact_otsu_report_filepath": output_records[
                            "otsu_report"
                        ]["path"],
                        "canonical_mask_filepath": output_records[
                            "canonical_segmentation_mask"
                        ]["path"],
                        "mask_comparison_report_filepath": output_records[
                            "segmentation_mask_comparison"
                        ]["path"],
                        "output_filepath": output_records[
                            "segmentation_verification_mcp_response"
                        ]["path"],
                        "registration_mode": self.registration_mode,
                        "overwrite": False,
                    }
                if tool_name == "visualize_slice":
                    return {
                        "input_filepath": ct_path,
                        "output_filepath": output_records["junction_overlay"]["path"],
                        "slice_index": 1,
                        "axis": 0,
                        "registration_mode": self.registration_mode,
                        "overwrite": False,
                    }
                if tool_name == "register_lattice_to_ct":
                    return {
                        "nominal_graph_filepath": source_records["nominal_graph"]["path"],
                        "output_graph_filepath": output_records["registered_graph"]["path"],
                        "output_report_filepath": output_records["registration_report"]["path"],
                        "registration_mode": self.registration_mode,
                        "ct_filepath": ct_path,
                        "aligned_graph_filepath": (
                            source_records["challenge_aligned_graph"]["path"]
                            if self.registration_mode == "challenge_aligned_json"
                            else None
                        ),
                        "threshold": 40054.0,
                        "analysis_config_filepath": output_records["analysis_config"]["path"],
                        "overwrite": False,
                    }
                if tool_name == "localize_lattice_nodes":
                    return {
                        "ct_filepath": ct_path,
                        "registered_graph_filepath": output_records["registered_graph"]["path"],
                        "output_graph_filepath": output_records["localized_graph"]["path"],
                        "output_report_filepath": output_records["localization_report"]["path"],
                        "threshold": 40054.0,
                        "registration_mode": self.registration_mode,
                        "analysis_policy_artifact_filepath": source_records["specimen_manifest"]["path"],
                        "registration_report_filepath": output_records["registration_report"]["path"],
                        "overwrite": False,
                    }
                return {
                    "ct_filepath": ct_path,
                    "localized_graph_filepath": output_records["localized_graph"]["path"],
                    "output_report_filepath": output_records["registration_qa"]["path"],
                    "threshold": 40054.0,
                    "registration_mode": self.registration_mode,
                    "localization_report_filepath": output_records["localization_report"]["path"],
                    "analysis_scope_artifact_filepath": source_records["specimen_manifest"]["path"],
                    "slice_output_filepath": None,
                    "bias_output_filepath": output_records["spatial_bias_figure"]["path"],
                    "slice_index": 1,
                    "overwrite": False,
                }

            assignments = (
                stage1_assignments if stage_number == 1 else stage2_assignments
            )

            def output_document(role: str) -> dict[str, object]:
                return json.loads(
                    (self.root / str(output_records[role]["path"])).read_text(
                        encoding="utf-8"
                    )
                )

            def response_result_for(tool_name: str) -> dict[str, object]:
                if stage_number == 1:
                    return {"overall_pass": True}
                if tool_name == "replay_exact_otsu":
                    return output_document("otsu_report")
                if tool_name == "segment_ct_dataset":
                    return {
                        "threshold": 40054.0,
                        "threshold_comparison": "value >= threshold",
                        "shape": [2, 3, 4],
                        "dtype": "uint8",
                        "foreground_voxels": 6,
                        "total_voxels": 24,
                        "foreground_fraction": 0.25,
                        "registration_mode": self.registration_mode,
                        "changed": True,
                        "message": "Saved 6 foreground voxels out of 24 total voxels",
                    }
                if tool_name == "compare_segmentation_masks":
                    return output_document("segmentation_mask_comparison")
                if tool_name == "verify_canonical_segmentation":
                    evidence = output_document(
                        "segmentation_verification_mcp_response"
                    )
                    return {
                        "schema_version": evidence["schema_version"],
                        "specimen_id": evidence["specimen_id"],
                        "design_id": evidence["design_id"],
                        "requested_analysis_scope": evidence[
                            "requested_analysis_scope"
                        ],
                        "registration_mode": evidence["registration_mode"],
                        **evidence["result"],
                    }
                result_role = {
                    "register_lattice_to_ct": "registration_report",
                    "localize_lattice_nodes": "localization_report",
                    "compute_registration_qa": "registration_qa",
                }.get(tool_name)
                if result_role is None:
                    return {"overall_pass": True}
                document = output_document(result_role)
                return {
                    key: value
                    for key, value in document.items()
                    if key not in {"artifacts", "hashes", "warnings"}
                }

            for tool in tools:
                tool_name = str(tool["name"])
                request_arguments = arguments_for(tool_name)
                request = {
                    "schema_version": "part2-mcp-request-binding/1.0.0",
                    "tool_name": tool_name,
                    "specimen_id": policy["specimen_id"],
                    "design_id": policy["design_id"],
                    "stage_number": policy["stage_number"],
                    "attempt": policy["attempt"],
                    "run_token": policy["run_token"],
                    "input_handoff_sha256": policy["input_handoff_sha256"],
                    "config_sha256": policy["config_sha256"],
                    "contract_sha256": policy["contract_sha256"],
                    "source_hashes": policy["source_hashes"],
                    "arguments": request_arguments,
                }
                request_sha = canonical_json_sha256(request)
                bound = sorted(
                    [
                        {
                            "role": role,
                            "path": output_records[role]["path"],
                            "sha256": output_records[role]["sha256"],
                        }
                        for role in assignments[tool_name]
                        if role in output_records
                    ],
                    key=lambda item: (item["role"], item["path"], item["sha256"]),
                )
                response_hashes = {
                    f"{item['role']}_sha256": item["sha256"]
                    for item in bound
                }
                if stage_number == 2 and tool_name == "replay_exact_otsu":
                    otsu_result = output_document("otsu_report")
                    response_hashes.update(
                        {
                            "input_sha256": source_records["ct_volume"]["sha256"],
                            "histogram_sha256": otsu_result["histogram_sha256"],
                            "histogram_artifact_sha256": output_records[
                                "exact_histogram"
                            ]["sha256"],
                            "report_artifact_sha256": output_records[
                                "otsu_report"
                            ]["sha256"],
                        }
                    )
                if (
                    stage_number == 2
                    and tool_name == "verify_canonical_segmentation"
                ):
                    evidence = output_document(
                        "segmentation_verification_mcp_response"
                    )
                    response_hashes = {
                        **evidence["hashes"],
                        "segmentation_verification_sha256": output_records[
                            "segmentation_verification_mcp_response"
                        ]["sha256"],
                    }
                response_artifacts = {
                    item["role"]: {
                        "path": item["path"],
                        "sha256": item["sha256"],
                        "role": item["role"],
                        "retention": "committed",
                    }
                    for item in bound
                }
                if (
                    stage_number == 2
                    and tool_name == "verify_canonical_segmentation"
                ):
                    response_artifacts = {
                        "segmentation_verification": next(
                            {
                                "path": item["path"],
                                "sha256": item["sha256"],
                                "changed": True,
                                "role": item["role"],
                                "retention": "committed",
                            }
                            for item in bound
                        )
                    }
                response = {
                    "response_schema_version": tool["response_schema_version"],
                    "tool": tool_name,
                    "status": "ok",
                    "gate": "pass",
                    "summary": (
                        "Persisted canonical segmentation verification"
                        if tool_name == "verify_canonical_segmentation"
                        else f"Synthetic structured response for {tool_name}"
                    ),
                    "result": response_result_for(tool_name),
                    "artifacts": response_artifacts,
                    "hashes": response_hashes,
                    "warnings": [],
                    "error": None,
                }
                policy["mcp_response_bindings"].append(
                    {
                        "tool_name": tool_name,
                        "request_arguments": request_arguments,
                        "request_sha256": request_sha,
                        "response": response,
                        "response_sha256": canonical_json_sha256(response),
                        "response_schema_version": tool[
                            "response_schema_version"
                        ],
                        "output_artifacts": bound,
                    }
                )
        return policy

    def assertions(self, stage_number: int) -> dict[str, bool]:
        assertions = {
            str(name): True
            for name in self.contracts[stage_number].get(
                "required_receipt_assertions", []
            )
        }
        if stage_number == 5:
            assertions["optimization_performed"] = False
        return assertions

    def start(
        self,
        stage_number: int,
        *,
        inputs: list[dict[str, object]] | None = None,
        inventory: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return start_stage(
            self.manifest_path,
            stage_number,
            input_artifacts=self.inputs(stage_number) if inputs is None else inputs,
            capability_inventory=self.inventory if inventory is None else inventory,
            repository_root=self.root,
            timestamp=self.timestamp(),
        )

    def aligned_reference(self) -> dict[str, object]:
        return self.record(
            "inputs/authorized_aligned_graph.json",
            role="autonomous_validation_reference",
            consumer="data_prep",
            phase="autonomous_v2_post_freeze_validation",
        )

    def freeze_stage_2(
        self, outputs: list[dict[str, object]]
    ) -> dict[str, object]:
        frozen = [
            record
            for record in outputs
            if record["role"] in {"registered_graph", "registration_report"}
        ]
        return record_autonomous_registration_freeze(
            self.manifest_path,
            frozen_artifacts=frozen,
            repository_root=self.root,
            timestamp=self.timestamp(),
        )

    def build_receipt(
        self,
        stage_number: int,
        *,
        terminal_state: str = "pass",
        outputs: list[dict[str, object]] | None = None,
        failure_kind: str | None = None,
        assertions: dict[str, bool] | None = None,
        stage_policy: dict[str, object] | None = None,
    ) -> dict[str, object]:
        selected_outputs = self.outputs(stage_number) if outputs is None else outputs
        if (
            stage_number == 2
            and terminal_state == "pass"
            and not any(
                item["role"] == "analysis_ready_specimen_manifest"
                for item in selected_outputs
            )
        ):
            selected_outputs = [
                *selected_outputs,
                self.analysis_ready_manifest_output(),
            ]
        selected_policy = (
            self.stage_policy(
                stage_number,
                selected_outputs,
                terminal_state=terminal_state,
            )
            if stage_policy is None
            else stage_policy
        )
        return build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state=terminal_state,
            output_artifacts=selected_outputs,
            assertions=self.assertions(stage_number) if assertions is None else assertions,
            stage_policy=selected_policy,
            repository_root=self.root,
            failure_kind=failure_kind,
            timestamp=self.timestamp(),
        )

    def pass_stage(self, stage_number: int) -> dict[str, object]:
        self.start(stage_number)
        outputs = self.outputs(stage_number)
        if stage_number == 2 and self.registration_mode == "autonomous_v2":
            self.freeze_stage_2(outputs)
            authorize_post_freeze_aligned_input(
                self.manifest_path,
                aligned_artifact=self.aligned_reference(),
                repository_root=self.root,
                timestamp=self.timestamp(),
            )
        receipt = self.build_receipt(stage_number, outputs=outputs)
        return complete_stage(
            self.manifest_path,
            receipt["path"],
            repository_root=self.root,
        )

    def advance_to(self, stage_number: int) -> None:
        while self.manifest()["current_stage"] != stage_number:
            current = self.manifest()["current_stage"]
            if current is None or int(current) > stage_number:
                raise AssertionError(f"Cannot advance backward to Stage {stage_number}")
            self.pass_stage(int(current))

    def run_all(self) -> None:
        for stage_number in range(7):
            self.pass_stage(stage_number)


class Part2OrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".part2-orchestration-tests-", dir=REPOSITORY_ROOT
        )
        self.scratch = Path(self.temporary.name)
        self.pipeline = SyntheticPipeline(self.scratch / "case")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def new_pipeline(
        self,
        name: str,
        registration_mode: str = "autonomous_v2",
        requested_analysis_scope: str = "roi_screening",
    ) -> SyntheticPipeline:
        return SyntheticPipeline(
            self.scratch / name,
            registration_mode,
            requested_analysis_scope,
        )

    def test_control_plane_imports_no_scientific_or_pipeline_algorithms(self) -> None:
        tree = ast.parse(
            (REPOSITORY_ROOT / "src/part2_orchestration.py").read_text()
        )
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(
            imported.isdisjoint(
                {
                    "numpy",
                    "scipy",
                    "pandas",
                    "skimage",
                    "matplotlib",
                    "pyvista",
                    "part2_core",
                    "mcp_server",
                }
            )
        )

    def test_complete_synthetic_autonomous_stage_0_through_6_happy_path(self) -> None:
        self.pipeline.run_all()
        manifest = self.pipeline.manifest()

        self.assertEqual("pass", manifest["pipeline_state"])
        self.assertIsNone(manifest["current_stage"])
        self.assertEqual(list(range(7)), manifest["stage_order"])
        self.assertTrue(manifest["sealed_evaluation"]["consumed"])
        for number in range(7):
            stage = manifest["stages"][str(number)]
            self.assertEqual("pass", stage["state"])
            self.assertEqual(1, stage["attempt_count"])
            self.assertIsNotNone(stage["completion_receipt"])
        stage_2_attempt = manifest["stages"]["2"]["attempts"][0]
        self.assertIsNotNone(stage_2_attempt["registration_freeze"])
        self.assertEqual(1, len(stage_2_attempt["supplemental_handoffs"]))
        evaluation = json.loads(
            (self.pipeline.root / "evals/results/20260727T000000Z.json").read_text()
        )
        self.assertEqual(0.0, evaluation["strict_recall"]["value"])
        self.assertEqual(
            "one_shot_reporting_not_pass_fail", evaluation["protocol"]
        )

    def test_stage_0_metadata_artifacts_are_outputs_not_prestart_inputs(self) -> None:
        contract = self.pipeline.contracts[0]
        input_roles = set(contract["input_artifacts"]["required_roles"])
        output_roles = set(contract["output_artifacts"]["required_roles"])
        self.assertNotIn("ct_metadata_mcp_response", input_roles)
        self.assertNotIn("ct_metadata_mcp_call_receipt", input_roles)
        self.assertIn("ct_metadata_mcp_response", output_roles)
        self.assertIn("ct_metadata_mcp_call_receipt", output_roles)

        started = self.pipeline.start(0)
        self.assertEqual("running", started["state"])

    def test_stage_0_pass_rejects_opaque_contract_shaped_outputs(self) -> None:
        self.pipeline.start(0)
        roles = self.pipeline.contracts[0]["output_artifacts"]["required_roles"]
        opaque_outputs = [
            self.pipeline.output_record(0, str(role)) for role in roles
        ]
        with self.assertRaisesRegex(
            ReceiptValidationError, "semantic validation|unreadable"
        ):
            self.pipeline.build_receipt(0, outputs=opaque_outputs)

    def test_stage_0_outputs_are_anchored_to_scientist_design_and_scope(self) -> None:
        for field, replacement in (
            ("design_id", "different_design"),
            ("requested_analysis_scope", "direct_metrology"),
        ):
            with self.subTest(field=field):
                pipeline = self.new_pipeline(f"scientist_{field}")
                inputs = pipeline.inputs(0)
                request_record = next(
                    record
                    for record in inputs
                    if record["role"] == "scientist_intake_request"
                )
                request_path = pipeline.root / str(request_record["path"])
                request = json.loads(request_path.read_text(encoding="utf-8"))
                request[field] = replacement
                request["canonical_request_sha256"] = canonical_json_sha256(
                    {
                        key: value
                        for key, value in request.items()
                        if key != "canonical_request_sha256"
                    }
                )
                pipeline._write_json(request_path, request)
                inputs = [
                    (
                        artifact_record(
                            pipeline.root,
                            request_record["path"],
                            role="scientist_intake_request",
                            consumer="specimen_ingest",
                            phase="input",
                        )
                        if record is request_record
                        else record
                    )
                    for record in inputs
                ]
                pipeline.start(0, inputs=inputs)
                with self.assertRaisesRegex(
                    ReceiptValidationError,
                    "frozen pipeline identity|frozen scientist",
                ):
                    pipeline.build_receipt(0, outputs=pipeline.outputs(0))

    def test_stage_0_start_rejects_open_or_badly_hashed_scientist_request(self) -> None:
        for mutation, expected in (
            ("open", "open or schema-incompatible"),
            ("self_hash", "canonical hash is invalid"),
        ):
            with self.subTest(mutation=mutation):
                pipeline = self.new_pipeline(f"request_{mutation}")
                inputs = pipeline.inputs(0)
                old = next(
                    record
                    for record in inputs
                    if record["role"] == "scientist_intake_request"
                )
                path = pipeline.root / str(old["path"])
                request = json.loads(path.read_text(encoding="utf-8"))
                if mutation == "open":
                    request["undeclared"] = True
                    request["canonical_request_sha256"] = canonical_json_sha256(
                        {
                            key: value
                            for key, value in request.items()
                            if key != "canonical_request_sha256"
                        }
                    )
                else:
                    request["canonical_request_sha256"] = "0" * 64
                pipeline._write_json(path, request)
                replacement = artifact_record(
                    pipeline.root,
                    old["path"],
                    role="scientist_intake_request",
                    consumer="specimen_ingest",
                    phase="input",
                )
                updated = [replacement if record is old else record for record in inputs]
                with self.assertRaisesRegex(ReceiptValidationError, expected):
                    pipeline.start(0, inputs=updated)

    def test_stage_1_policy_is_closed_and_run_bound(self) -> None:
        self.pipeline.pass_stage(0)
        self.pipeline.start(1)
        outputs = self.pipeline.outputs(1)
        policy = self.pipeline.stage_policy(1, outputs)
        assert policy is not None
        open_policy = copy.deepcopy(policy)
        open_policy["undeclared"] = True
        with self.assertRaisesRegex(ReceiptValidationError, "closed stage_policy"):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=open_policy
            )
        stale_policy = copy.deepcopy(policy)
        stale_policy["run_token"] = "0" * 64
        with self.assertRaisesRegex(ReceiptValidationError, "stale or misbound"):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=stale_policy
            )

    def test_mcp_bindings_reject_argument_response_and_artifact_tampering(self) -> None:
        self.pipeline.pass_stage(0)
        self.pipeline.start(1)
        outputs = self.pipeline.outputs(1)
        base_policy = self.pipeline.stage_policy(1, outputs)
        assert base_policy is not None

        def binding(policy: dict[str, object], tool_name: str) -> dict[str, object]:
            return next(
                item
                for item in policy["mcp_response_bindings"]
                if item["tool_name"] == tool_name
            )

        request_tamper = copy.deepcopy(base_policy)
        binding(request_tamper, "load_lattice_graph")["request_arguments"][
            "input_filepath"
        ] = "inputs/not-the-frozen-graph.json"
        with self.assertRaisesRegex(ReceiptValidationError, "request is not bound"):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=request_tamper
            )

        response_tamper = copy.deepcopy(base_policy)
        binding(response_tamper, "load_lattice_graph")["response"][
            "summary"
        ] = "tampered response"
        with self.assertRaisesRegex(ReceiptValidationError, "response hash is stale"):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=response_tamper
            )

        artifact_tamper = copy.deepcopy(base_policy)
        resolve = binding(artifact_tamper, "resolve_cad_graph_orientation")
        resolve["response"]["artifacts"]["cad_graph_orientation"][
            "sha256"
        ] = "0" * 64
        resolve["response_sha256"] = canonical_json_sha256(resolve["response"])
        with self.assertRaisesRegex(
            ReceiptValidationError, "do not contain the bound output"
        ):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=artifact_tamper
            )

        for name, mutate in (
            (
                "extra",
                lambda response: response.update({"undeclared": True}),
            ),
            ("missing", lambda response: response.pop("warnings")),
        ):
            with self.subTest(response_fields=name):
                envelope_tamper = copy.deepcopy(base_policy)
                selected = binding(envelope_tamper, "load_lattice_graph")
                mutate(selected["response"])
                selected["response_sha256"] = canonical_json_sha256(
                    selected["response"]
                )
                with self.assertRaisesRegex(
                    ReceiptValidationError, "closed structured response"
                ):
                    self.pipeline.build_receipt(
                        1, outputs=outputs, stage_policy=envelope_tamper
                    )

    def test_stage_1_declaration_hash_must_match_intake_source(self) -> None:
        self.pipeline.pass_stage(0)
        self.pipeline.start(1)
        outputs = self.pipeline.outputs(1)
        orientation = next(
            item for item in outputs if item["role"] == "cad_graph_orientation"
        )
        path = self.pipeline.root / orientation["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        document["declared_transform"]["artifact_sha256"] = "0" * 64
        self.pipeline._write_json(path, document)
        refreshed = artifact_record(
            self.pipeline.root,
            orientation["path"],
            role="cad_graph_orientation",
            phase=orientation["phase"],
        )
        outputs = [
            refreshed if item["role"] == "cad_graph_orientation" else item
            for item in outputs
        ]
        with self.assertRaisesRegex(
            ReceiptValidationError, "declaration is stale or unverified"
        ):
            self.pipeline.build_receipt(1, outputs=outputs)

    def test_stage_1_accepts_response_bound_output_without_embedded_binding(self) -> None:
        self.pipeline.pass_stage(0)
        self.pipeline.start(1)
        outputs = self.pipeline.outputs(1)
        orientation = next(
            item for item in outputs if item["role"] == "cad_graph_orientation"
        )
        path = self.pipeline.root / orientation["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("orchestration_binding", document)
        built = self.pipeline.build_receipt(1, outputs=outputs)
        self.assertTrue(Path(built["path"]).is_file())

        policy = self.pipeline.stage_policy(1, outputs)
        assert policy is not None
        resolve = next(
            item
            for item in policy["mcp_response_bindings"]
            if item["tool_name"] == "resolve_cad_graph_orientation"
        )
        resolve["output_artifacts"] = []
        resolve["response"]["artifacts"] = {}
        resolve["response"]["hashes"] = {}
        resolve["response_sha256"] = canonical_json_sha256(resolve["response"])
        with self.assertRaisesRegex(
            ReceiptValidationError, "do not cover every scientific output"
        ):
            self.pipeline.build_receipt(
                1, outputs=outputs, stage_policy=policy
            )

    def test_stage_1_policy_receipt_replay_is_byte_idempotent(self) -> None:
        self.pipeline.pass_stage(0)
        self.pipeline.start(1)
        receipt = self.pipeline.build_receipt(1)
        first = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        before = self.pipeline.manifest_path.read_bytes()
        replay = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertTrue(first["changed"])
        self.assertFalse(replay["changed"])
        self.assertTrue(replay["manifest_bytes_unchanged"])
        self.assertEqual(before, self.pipeline.manifest_path.read_bytes())

    def test_stage_2_accepts_response_bound_outputs_and_rejects_cross_run_binding(self) -> None:
        pipeline = self.new_pipeline("stage2_response_binding")
        pipeline.advance_to(2)
        pipeline.start(2)
        outputs = pipeline.outputs(2)
        pipeline.freeze_stage_2(outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        qa = next(item for item in outputs if item["role"] == "registration_qa")
        path = pipeline.root / qa["path"]
        document = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotIn("orchestration_binding", document)
        built = pipeline.build_receipt(2, outputs=outputs)
        self.assertTrue(Path(built["path"]).is_file())

        document["orchestration_binding"] = pipeline._output_binding(2)
        document["orchestration_binding"]["specimen_id"] = "other_specimen"
        pipeline._write_json(path, document)
        refreshed = artifact_record(
            pipeline.root,
            qa["path"],
            role="registration_qa",
            phase=qa["phase"],
        )
        outputs = [
            refreshed if item["role"] == "registration_qa" else item
            for item in outputs
        ]
        with self.assertRaisesRegex(
            ReceiptValidationError, "standalone or cross-run"
        ):
            pipeline.build_receipt(2, outputs=outputs)

    def test_stage_2_rejects_threshold_123_mixed_mcp_receipt(self) -> None:
        pipeline = self.new_pipeline("stage2_threshold_123_mixed_receipt")
        pipeline.advance_to(2)
        pipeline.start(2)
        stage_outputs = pipeline.outputs(2)
        pipeline.freeze_stage_2(stage_outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        outputs = [*stage_outputs, pipeline.analysis_ready_manifest_output()]
        policy = pipeline.stage_policy(2, outputs)
        assert policy is not None

        def binding(tool_name: str) -> dict[str, object]:
            return next(
                item
                for item in policy["mcp_response_bindings"]
                if item["tool_name"] == tool_name
            )

        def rehash_request(selected: dict[str, object]) -> None:
            request = {
                "schema_version": "part2-mcp-request-binding/1.0.0",
                "tool_name": selected["tool_name"],
                "specimen_id": policy["specimen_id"],
                "design_id": policy["design_id"],
                "stage_number": policy["stage_number"],
                "attempt": policy["attempt"],
                "run_token": policy["run_token"],
                "input_handoff_sha256": policy["input_handoff_sha256"],
                "config_sha256": policy["config_sha256"],
                "contract_sha256": policy["contract_sha256"],
                "source_hashes": policy["source_hashes"],
                "arguments": selected["request_arguments"],
            }
            selected["request_sha256"] = canonical_json_sha256(request)

        for tool_name in (
            "segment_ct_dataset",
            "register_lattice_to_ct",
            "localize_lattice_nodes",
            "compute_registration_qa",
        ):
            selected = binding(tool_name)
            selected["request_arguments"]["threshold"] = 123.0
            selected["response"]["result"]["threshold"] = 123.0
            if tool_name in {"localize_lattice_nodes", "compute_registration_qa"}:
                selected["response"]["result"]["segmentation_binding"][
                    "threshold"
                ] = 123.0
            if tool_name == "register_lattice_to_ct":
                selected["response"]["result"]["mode_details"][
                    "threshold"
                ] = 123.0
            rehash_request(selected)
            selected["response_sha256"] = canonical_json_sha256(
                selected["response"]
            )
        comparison = binding("compare_segmentation_masks")
        comparison["request_arguments"]["thresholds"] = [123.0]
        comparison["response"]["result"]["candidates"][0]["threshold"] = 123.0
        rehash_request(comparison)
        comparison["response_sha256"] = canonical_json_sha256(
            comparison["response"]
        )

        with self.assertRaisesRegex(
            ReceiptValidationError,
            "downstream MCP requests are not bound to the exact Otsu threshold",
        ):
            pipeline.build_receipt(2, outputs=outputs, stage_policy=policy)

    def test_stage_2_rejects_open_persisted_segmentation_verification(self) -> None:
        pipeline = self.new_pipeline("stage2_open_segmentation_verification")
        pipeline.advance_to(2)
        pipeline.start(2)
        stage_outputs = pipeline.outputs(2)
        pipeline.freeze_stage_2(stage_outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        outputs = [*stage_outputs, pipeline.analysis_ready_manifest_output()]
        policy = pipeline.stage_policy(2, outputs)
        assert policy is not None
        evidence = next(
            item
            for item in outputs
            if item["role"] == "segmentation_verification_mcp_response"
        )
        evidence_path = pipeline.root / str(evidence["path"])
        document = json.loads(evidence_path.read_text(encoding="utf-8"))
        document["unexpected_probe"] = True
        pipeline._write_json(evidence_path, document)
        refreshed = artifact_record(
            pipeline.root,
            evidence["path"],
            role="segmentation_verification_mcp_response",
            phase=evidence["phase"],
        )
        outputs = [
            refreshed
            if item["role"] == "segmentation_verification_mcp_response"
            else item
            for item in outputs
        ]
        policy["output_hashes"][
            "segmentation_verification_mcp_response"
        ] = refreshed["sha256"]
        verifier = next(
            item
            for item in policy["mcp_response_bindings"]
            if item["tool_name"] == "verify_canonical_segmentation"
        )
        verifier["output_artifacts"][0]["sha256"] = refreshed["sha256"]
        verifier["response"]["artifacts"]["segmentation_verification"][
            "sha256"
        ] = refreshed["sha256"]
        verifier["response"]["hashes"][
            "segmentation_verification_sha256"
        ] = refreshed["sha256"]
        verifier["response_sha256"] = canonical_json_sha256(
            verifier["response"]
        )
        with self.assertRaisesRegex(
            ReceiptValidationError,
            "segmentation verification evidence must be a closed object",
        ):
            pipeline.build_receipt(
                2,
                outputs=outputs,
                stage_policy=policy,
            )

    def test_stage_2_rejects_each_threshold_bearing_mcp_result_tamper(self) -> None:
        pipeline = self.new_pipeline("stage2_threshold_result_binding")
        pipeline.advance_to(2)
        pipeline.start(2)
        stage_outputs = pipeline.outputs(2)
        pipeline.freeze_stage_2(stage_outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        outputs = [*stage_outputs, pipeline.analysis_ready_manifest_output()]
        base_policy = pipeline.stage_policy(2, outputs)
        assert base_policy is not None

        def mutate_result(
            tool_name: str, policy: dict[str, object]
        ) -> dict[str, object]:
            selected = next(
                item
                for item in policy["mcp_response_bindings"]
                if item["tool_name"] == tool_name
            )
            result = selected["response"]["result"]
            if tool_name == "compare_segmentation_masks":
                result["candidates"][0]["threshold"] = 123.0
            elif tool_name == "register_lattice_to_ct":
                result["mode_details"]["threshold"] = 123.0
            elif tool_name in {
                "localize_lattice_nodes",
                "compute_registration_qa",
            }:
                result["threshold"] = 123.0
                result["segmentation_binding"]["threshold"] = 123.0
            else:
                result["threshold"] = 123.0
            selected["response_sha256"] = canonical_json_sha256(
                selected["response"]
            )
            return policy

        for tool_name in (
            "replay_exact_otsu",
            "segment_ct_dataset",
            "compare_segmentation_masks",
            "register_lattice_to_ct",
            "localize_lattice_nodes",
            "compute_registration_qa",
        ):
            with self.subTest(tool_name=tool_name):
                tampered = mutate_result(tool_name, copy.deepcopy(base_policy))
                with self.assertRaisesRegex(
                    ReceiptValidationError, "(?:exact|canonical) Otsu"
                ):
                    pipeline.build_receipt(
                        2,
                        outputs=outputs,
                        stage_policy=tampered,
                    )

    def test_stage_2_rejects_registration_threshold_result_artifact_tamper(self) -> None:
        pipeline = self.new_pipeline("stage2_registration_threshold_artifact")
        pipeline.advance_to(2)
        pipeline.start(2)
        stage_outputs = pipeline.outputs(2)
        pipeline.freeze_stage_2(stage_outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        analysis_config = next(
            item for item in stage_outputs if item["role"] == "analysis_config"
        )
        path = pipeline.root / str(analysis_config["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["threshold"] = 123.0
        pipeline._write_json(path, document)
        refreshed = artifact_record(
            pipeline.root,
            analysis_config["path"],
            role="analysis_config",
            phase=analysis_config["phase"],
        )
        stage_outputs = [
            refreshed if item["role"] == "analysis_config" else item
            for item in stage_outputs
        ]
        outputs = [*stage_outputs, pipeline.analysis_ready_manifest_output()]

        with self.assertRaisesRegex(
            ReceiptValidationError,
            "registration response/result artifacts are not bound",
        ):
            pipeline.build_receipt(2, outputs=outputs)

    def test_stage_2_validates_analysis_ready_manifest_and_predecessor(self) -> None:
        malformed = self.new_pipeline("malformed_analysis_ready")
        malformed.advance_to(2)
        malformed.start(2)
        outputs = malformed.outputs(2)
        replacement = malformed.analysis_ready_manifest_output()
        path = malformed.root / str(replacement["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        del document["derived"]["segmentation_result"]
        malformed._write_json(path, document)
        replacement = artifact_record(
            malformed.root,
            replacement["path"],
            role="analysis_ready_specimen_manifest",
            phase=replacement["phase"],
            replaces_sha256=replacement["replaces_sha256"],
        )
        with self.assertRaisesRegex(
            ReceiptValidationError, "failed full validation"
        ):
            malformed.build_receipt(2, outputs=[*outputs, replacement])

        stale = self.new_pipeline("stale_manifest_predecessor")
        stale.advance_to(2)
        stale.start(2)
        outputs = stale.outputs(2)
        replacement = stale.analysis_ready_manifest_output()
        replacement["replaces_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            ReceiptValidationError, "stale or inconsistent with Stage 2"
        ):
            stale.build_receipt(2, outputs=[*outputs, replacement])

    def test_stage_2_rejects_binary_dtype_and_shape_mismatch(self) -> None:
        for name, payload, expected in (
            (
                "dtype",
                SyntheticPipeline._npy_bytes(descr="<u2"),
                "dtype is uint16",
            ),
            (
                "shape",
                SyntheticPipeline._npy_bytes(shape=(2, 3, 5)),
                r"shape is \[2, 3, 5\]",
            ),
        ):
            with self.subTest(metadata=name):
                pipeline = self.new_pipeline(f"binary_{name}")
                pipeline.advance_to(2)
                pipeline.start(2)
                outputs = pipeline.outputs(2)
                mask = next(
                    item
                    for item in outputs
                    if item["role"] == "canonical_segmentation_mask"
                )
                (pipeline.root / str(mask["path"])).write_bytes(payload)
                refreshed = artifact_record(
                    pipeline.root,
                    mask["path"],
                    role="canonical_segmentation_mask",
                    phase=mask["phase"],
                )
                outputs = [
                    refreshed
                    if item["role"] == "canonical_segmentation_mask"
                    else item
                    for item in outputs
                ]
                with self.assertRaisesRegex(ArtifactVerificationError, expected):
                    pipeline.build_receipt(2, outputs=outputs)

    def test_stage_2_localization_partition_allows_overlapping_rejection_bound(self) -> None:
        self.pipeline.advance_to(2)
        self.pipeline.start(2)
        outputs = self.pipeline.outputs(2)
        outputs = [*outputs, self.pipeline.analysis_ready_manifest_output()]
        policy = self.pipeline.stage_policy(2, outputs)
        assert policy is not None
        localization = policy["localization_summary"]
        self.assertEqual(10206, localization["total_nodes"])
        self.assertEqual(
            localization["total_nodes"],
            localization["primary_matches"]
            + localization["stable_coarse_matches"]
            + localization["fallback_matches"]
            + localization["ambiguous_matches"],
        )
        self.assertEqual(394, localization["rejected_or_low_confidence"])

        invalid = copy.deepcopy(policy)
        invalid["localization_summary"]["rejected_or_low_confidence"] = 10207
        with self.assertRaisesRegex(
            ReceiptValidationError, "localization summary is malformed"
        ):
            self.pipeline.build_receipt(
                2,
                outputs=outputs,
                stage_policy=invalid,
            )

    def test_direct_metrology_manual_review_preserves_stage_2_evidence(self) -> None:
        pipeline = self.new_pipeline(
            "direct_metrology_review",
            requested_analysis_scope="direct_metrology",
        )
        pipeline.advance_to(2)
        pipeline.start(2)
        outputs = [
            item
            for item in pipeline.outputs(2)
            if item["role"]
            not in {
                "data_prep_result",
                "data_prep_completion_receipt",
                "analysis_ready_specimen_manifest",
            }
        ]
        pipeline.freeze_stage_2(outputs)
        authorize_post_freeze_aligned_input(
            pipeline.manifest_path,
            aligned_artifact=pipeline.aligned_reference(),
            repository_root=pipeline.root,
            timestamp=pipeline.timestamp(),
        )
        receipt = pipeline.build_receipt(
            2,
            terminal_state="manual_review",
            outputs=outputs,
        )
        completed = complete_stage(
            pipeline.manifest_path,
            receipt["path"],
            repository_root=pipeline.root,
        )
        self.assertEqual("manual_review", completed["state"])
        manifest = pipeline.manifest()
        self.assertEqual("locked", manifest["stages"]["3"]["state"])
        self.assertTrue(
            any(
                item["role"] == "registration_qa"
                for item in manifest["stages"]["2"]["attempts"][-1][
                    "output_artifacts"
                ]
            )
        )
    def test_legal_transitions_and_stage_skipping_are_enforced(self) -> None:
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(1)

        inputs = self.pipeline.inputs(0)
        first = self.pipeline.start(0, inputs=inputs)
        replay = self.pipeline.start(0, inputs=inputs)
        self.assertTrue(first["changed"])
        self.assertFalse(replay["changed"])
        self.assertEqual(first["run_token"], replay["run_token"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(1)

        receipt = self.pipeline.build_receipt(0)
        complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual(1, self.pipeline.manifest()["current_stage"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(0)
        # The legacy Stage-0 contract used to point directly to data prep.  The
        # numeric control plane must still require Stage 1 first.
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(2)
        started = self.pipeline.start(1)
        self.assertEqual("running", started["state"])

    def test_receipt_and_artifact_tampering_and_staleness_are_rejected(self) -> None:
        tampered_receipt = self.new_pipeline("tampered_receipt")
        tampered_receipt.start(0)
        built = tampered_receipt.build_receipt(0)
        path = Path(built["path"])
        receipt = json.loads(path.read_text())
        receipt["owner"] = "attacker"
        tampered_receipt._write_json(path, receipt)
        with self.assertRaisesRegex(ReceiptValidationError, "canonical hash"):
            complete_stage(
                tampered_receipt.manifest_path,
                path,
                repository_root=tampered_receipt.root,
            )

        stale_receipt = self.new_pipeline("stale_receipt")
        stale_receipt.start(0)
        built = stale_receipt.build_receipt(0)
        path = Path(built["path"])
        receipt = json.loads(path.read_text())
        receipt["run_token"] = "0" * 64
        receipt["canonical_receipt_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in receipt.items()
                if key != "canonical_receipt_sha256"
            }
        )
        stale_receipt._write_json(path, receipt)
        with self.assertRaisesRegex(ReceiptValidationError, "stale or misbound"):
            complete_stage(
                stale_receipt.manifest_path,
                path,
                repository_root=stale_receipt.root,
            )

        tampered_output = self.new_pipeline("tampered_output")
        tampered_output.start(0)
        outputs = tampered_output.outputs(0)
        built = tampered_output.build_receipt(0, outputs=outputs)
        (tampered_output.root / str(outputs[0]["path"])).write_bytes(b"changed\n")
        with self.assertRaises(ArtifactVerificationError):
            complete_stage(
                tampered_output.manifest_path,
                built["path"],
                repository_root=tampered_output.root,
            )

        tampered_input = self.new_pipeline("tampered_input")
        inputs = tampered_input.inputs(0)
        tampered_input.start(0, inputs=inputs)
        built = tampered_input.build_receipt(0)
        (tampered_input.root / str(inputs[0]["path"])).write_bytes(b"changed\n")
        with self.assertRaises(ArtifactVerificationError):
            complete_stage(
                tampered_input.manifest_path,
                built["path"],
                repository_root=tampered_input.root,
            )

    def test_config_hash_change_invalidates_pipeline(self) -> None:
        frozen_hash = self.pipeline.manifest()["config"]["sha256"]
        self.pipeline.config_path.write_text('{"frozen": false}\n', encoding="utf-8")
        with self.assertRaisesRegex(ManifestValidationError, "Frozen config SHA-256"):
            self.pipeline.manifest()
        with self.assertRaisesRegex(ManifestValidationError, "Frozen config SHA-256"):
            self.pipeline.start(0)
        raw = json.loads(self.pipeline.manifest_path.read_text())
        self.assertEqual(frozen_hash, raw["config"]["sha256"])
        self.assertEqual(0, raw["stages"]["0"]["attempt_count"])

    def test_frozen_contract_hash_change_invalidates_pipeline(self) -> None:
        contract_path = self.pipeline.root / "analysis/contracts/specimen_ingest.json"
        contract_path.write_bytes(contract_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ManifestValidationError, "contract hash changed"):
            self.pipeline.manifest()

    def test_two_attempt_exhaustion_halts_and_preserves_attempt_history(self) -> None:
        first_inputs = self.pipeline.inputs(0)
        self.pipeline.start(0, inputs=first_inputs)
        evidence = [self.pipeline.manual_review_evidence(0, 1)]
        first_receipt = self.pipeline.build_receipt(
            0,
            terminal_state="manual_review",
            outputs=evidence,
            assertions={},
        )
        first_result = complete_stage(
            self.pipeline.manifest_path,
            first_receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("manual_review", first_result["state"])
        resolution = self.pipeline.record(
            f"analysis/{SPECIMEN_ID}/reviews/attempt_1.json",
            role="manual_review_resolution",
        )
        resume_manual_review(
            self.pipeline.manifest_path,
            0,
            resolution_artifact=resolution,
            reason="Scientist supplied the missing declaration",
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        self.pipeline.start(0, inputs=first_inputs)
        second_evidence = [self.pipeline.manual_review_evidence(0, 2)]
        second_receipt = self.pipeline.build_receipt(
            0,
            terminal_state="manual_review",
            outputs=second_evidence,
            assertions={},
        )
        second_result = complete_stage(
            self.pipeline.manifest_path,
            second_receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("halt", second_result["state"])
        manifest = self.pipeline.manifest()
        self.assertEqual(2, manifest["stages"]["0"]["attempt_count"])
        self.assertEqual(2, len(manifest["stages"]["0"]["attempts"]))
        self.assertEqual("attempts_exhausted", manifest["stages"]["0"]["control_halt"]["code"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(0, inputs=first_inputs)

    def test_deterministic_gate_failure_is_non_retryable(self) -> None:
        self.pipeline.advance_to(3)
        self.pipeline.start(3)
        receipt = self.pipeline.build_receipt(
            3,
            terminal_state="halt",
            outputs=[],
            assertions={},
            failure_kind="deterministic_gate",
        )
        result = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("halt", result["state"])
        self.assertEqual(1, self.pipeline.manifest()["stages"]["3"]["attempt_count"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(3)

    def test_manual_review_stops_downstream_and_explicit_resume_preserves_evidence(self) -> None:
        self.pipeline.start(0)
        evidence = [self.pipeline.manual_review_evidence(0, 1)]
        evidence_bytes = (self.pipeline.root / str(evidence[0]["path"])).read_bytes()
        receipt = self.pipeline.build_receipt(
            0,
            terminal_state="manual_review",
            outputs=evidence,
            assertions={},
        )
        receipt_bytes = Path(receipt["path"]).read_bytes()
        complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("manual_review", pipeline_status(
            self.pipeline.manifest_path, repository_root=self.pipeline.root
        )["pipeline_state"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(1)
        resolution = self.pipeline.record(
            f"analysis/{SPECIMEN_ID}/reviews/intake_resolution.json",
            role="manual_review_resolution",
        )
        resumed = resume_manual_review(
            self.pipeline.manifest_path,
            0,
            resolution_artifact=resolution,
            reason="Association confirmed in the attached review record",
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        self.assertEqual("ready", resumed["state"])
        self.assertEqual(receipt_bytes, Path(receipt["path"]).read_bytes())
        self.assertEqual(
            evidence_bytes,
            (self.pipeline.root / str(evidence[0]["path"])).read_bytes(),
        )
        retry = self.pipeline.start(0)
        self.assertEqual("running", retry["state"])
        self.assertEqual(2, self.pipeline.manifest()["stages"]["0"]["attempt_count"])

    def test_missing_and_incompatible_dependencies_create_structured_halts(self) -> None:
        missing = self.new_pipeline("missing_dependency")
        inventory = copy.deepcopy(missing.inventory)
        del inventory["mcp_servers"]["segmentation-tools"]["tools"][
            "inspect_volume_metadata"
        ]
        result = missing.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        self.assertEqual("missing_or_incompatible_dependency", result["error"]["code"])
        self.assertFalse(result["error"]["fallback_used"])
        manifest = missing.manifest()
        self.assertEqual(0, manifest["stages"]["0"]["attempt_count"])
        self.assertEqual("halt", manifest["pipeline_state"])

        incompatible = self.new_pipeline("incompatible_dependency")
        inventory = copy.deepcopy(incompatible.inventory)
        inventory["agents"]["specimen_ingest"]["contract_version"] = "wrong/9.9.9"
        result = incompatible.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        failures = result["error"]["failures"]
        self.assertTrue(any("schema_incompatible" in item["reason"] for item in failures))
        receipt = json.loads(
            (incompatible.root / result["receipt"]["path"]).read_text()
        )
        self.assertEqual("halt", receipt["terminal_state"])
        self.assertEqual(0, receipt["attempt"])

        missing_agent = self.new_pipeline("missing_agent")
        inventory = copy.deepcopy(missing_agent.inventory)
        del inventory["agents"]["specimen_ingest"]
        result = missing_agent.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        self.assertTrue(
            any(
                item["kind"] == "agent" and item["reason"] == "missing"
                for item in result["error"]["failures"]
            )
        )

    def test_challenge_and_autonomous_aligned_json_boundaries(self) -> None:
        autonomous = self.new_pipeline("autonomous_boundary")
        stage_0_inputs = autonomous.inputs(0)
        aligned_rule = autonomous._rule_for(
            autonomous.contracts[0], "input", "challenge_aligned_graph"
        )
        stage_0_inputs.append(
            autonomous.record_for_rule(
                aligned_rule,
                "challenge_aligned_graph",
                phase="challenge_aligned_json",
            )
        )
        with self.assertRaisesRegex(AccessPolicyError, "Autonomous-v2 intake"):
            autonomous.start(0, inputs=stage_0_inputs)

        challenge = self.new_pipeline(
            "challenge_boundary", "challenge_aligned_json"
        )
        started = challenge.start(0)
        self.assertEqual("running", started["state"])
        handoff = json.loads((challenge.root / started["handoff"]["path"]).read_text())
        self.assertTrue(
            any(item["role"] == "challenge_aligned_graph" for item in handoff["input_artifacts"])
        )
        challenge.pass_stage(0)
        challenge.pass_stage(1)
        started = challenge.start(2)
        handoff = json.loads((challenge.root / started["handoff"]["path"]).read_text())
        self.assertTrue(
            any(item["role"] == "challenge_aligned_graph" for item in handoff["input_artifacts"])
        )

    def test_autonomous_aligned_input_requires_registration_freeze(self) -> None:
        self.pipeline.advance_to(2)
        self.pipeline.start(2)
        reference = self.pipeline.aligned_reference()
        with self.assertRaisesRegex(AccessPolicyError, "until CT-only registration is frozen"):
            authorize_post_freeze_aligned_input(
                self.pipeline.manifest_path,
                aligned_artifact=reference,
                repository_root=self.pipeline.root,
                timestamp=self.pipeline.timestamp(),
            )
        outputs = self.pipeline.outputs(2)
        frozen = self.pipeline.freeze_stage_2(outputs)
        authorized = authorize_post_freeze_aligned_input(
            self.pipeline.manifest_path,
            aligned_artifact=reference,
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        replayed_freeze = self.pipeline.freeze_stage_2(outputs)
        replayed_authorization = authorize_post_freeze_aligned_input(
            self.pipeline.manifest_path,
            aligned_artifact=reference,
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        self.assertTrue(frozen["changed"])
        self.assertTrue(authorized["changed"])
        self.assertFalse(replayed_freeze["changed"])
        self.assertFalse(replayed_authorization["changed"])
        manifest = self.pipeline.manifest()
        attempt = manifest["stages"]["2"]["attempts"][0]
        freeze = json.loads((self.pipeline.root / attempt["registration_freeze"]["path"]).read_text())
        self.assertFalse(freeze["aligned_graph_accessed"])
        self.assertEqual(
            {item["sha256"] for item in outputs if item["role"] in {"registered_graph", "registration_report"}},
            {item["sha256"] for item in freeze["frozen_artifacts"]},
        )

    def test_autonomous_stage_start_cannot_smuggle_post_freeze_aligned_input(self) -> None:
        self.pipeline.advance_to(2)
        inputs = self.pipeline.inputs(2)
        rule = self.pipeline._rule_for(
            self.pipeline.contracts[2], "input", "autonomous_validation_reference"
        )
        inputs.append(
            self.pipeline.record_for_rule(
                rule,
                "autonomous_validation_reference",
                consumer="data_prep",
                phase="autonomous_v2_post_freeze_validation",
            )
        )
        with self.assertRaisesRegex(AccessPolicyError, "frozen CT-only fit"):
            self.pipeline.start(2, inputs=inputs)

    def test_dev_and_sealed_hashes_cannot_be_disguised_as_allowed_inputs(self) -> None:
        self.pipeline.advance_to(2)
        dev = self.pipeline.root / f"analysis/{SPECIMEN_ID}/labels/dev_split.json"
        sealed = self.pipeline.root / "evals/labels/sealed_split.json"

        for name, source, message in (
            ("renamed_dev.bin", dev, "Development labels"),
            ("renamed_sealed.bin", sealed, "sealed split"),
        ):
            disguised_path = self.pipeline.root / "inputs" / name
            disguised_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, disguised_path)
            disguised = artifact_record(
                self.pipeline.root,
                disguised_path,
                role="ct_volume",
                consumer="data_prep",
                phase="input",
            )
            with self.assertRaisesRegex(AccessPolicyError, message):
                self.pipeline.start(2, inputs=[*self.pipeline.inputs(2), disguised])

    def test_development_split_is_scoped_only_to_missing_specialist(self) -> None:
        self.pipeline.advance_to(4)
        inputs = self.pipeline.inputs(4)
        dev = next(item for item in inputs if item["role"] == "development_labels")
        self.assertEqual("missing_strut_agent", dev["consumer"])
        wrong = [dict(item) for item in inputs]
        next(item for item in wrong if item["role"] == "development_labels")[
            "consumer"
        ] = "classifier_verifier"
        with self.assertRaises((AccessPolicyError, ReceiptValidationError)):
            self.pipeline.start(4, inputs=wrong)
        self.assertEqual("running", self.pipeline.start(4, inputs=inputs)["state"])

    def test_stage_4_requires_fresh_independent_dev_blind_verifier(self) -> None:
        missing = self.new_pipeline("missing_verifier")
        missing.advance_to(4)
        missing.start(4)
        without_verifier = [
            item
            for item in missing.outputs(4)
            if item["role"] != "classifier_verifier_report"
        ]
        with self.assertRaisesRegex(ReceiptValidationError, "missing required output roles"):
            missing.build_receipt(4, outputs=without_verifier)

        stale = self.new_pipeline("stale_verifier")
        stale.advance_to(4)
        stale.start(4)
        outputs = stale.outputs(
            4,
            verifier_updates={
                "bindings": {
                    "classified_struts_sha256": "0" * 64,
                    "thresholds_sha256": "0" * 64,
                    "decision_log_sha256": "0" * 64,
                    "evidence_set_sha256": "0" * 64,
                }
            },
        )
        receipt = stale.build_receipt(4, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "stale or missing"):
            complete_stage(
                stale.manifest_path,
                receipt["path"],
                repository_root=stale.root,
            )

        dev_reader = self.new_pipeline("dev_reading_verifier")
        dev_reader.advance_to(4)
        dev_reader.start(4)
        outputs = dev_reader.outputs(
            4,
            verifier_updates={
                "label_access": {
                    "development_split_read": True,
                    "sealed_split_read": False,
                }
            },
        )
        receipt = dev_reader.build_receipt(4, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "must not read the dev split"):
            complete_stage(
                dev_reader.manifest_path,
                receipt["path"],
                repository_root=dev_reader.root,
            )

    def test_stage_5_is_consumed_on_start_one_shot_and_zero_recall_still_passes(self) -> None:
        self.pipeline.advance_to(5)
        inputs = self.pipeline.inputs(5)
        first = self.pipeline.start(5, inputs=inputs)
        manifest = self.pipeline.manifest()
        self.assertTrue(manifest["sealed_evaluation"]["consumed"])
        self.assertEqual(1, manifest["sealed_evaluation"]["stage_attempt"])
        with self.assertRaisesRegex(IllegalTransitionError, "already reserved"):
            self.pipeline.start(5, inputs=inputs)
        self.assertEqual(1, self.pipeline.manifest()["stages"]["5"]["attempt_count"])

        outputs = self.pipeline.outputs(5)
        receipt = self.pipeline.build_receipt(5, outputs=outputs)
        completed = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("pass", completed["state"])
        self.assertEqual(6, completed["next_stage"])
        with self.assertRaises(IllegalTransitionError):
            self.pipeline.start(5, inputs=inputs)

    def test_exact_receipt_replay_is_byte_idempotent_and_rechecks_artifacts(self) -> None:
        self.pipeline.start(0)
        outputs = self.pipeline.outputs(0)
        receipt = self.pipeline.build_receipt(0, outputs=outputs)
        first = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        before = self.pipeline.manifest_path.read_bytes()
        replay = complete_stage(
            self.pipeline.manifest_path,
            receipt["path"],
            repository_root=self.pipeline.root,
        )
        self.assertTrue(first["changed"])
        self.assertFalse(replay["changed"])
        self.assertTrue(replay["manifest_bytes_unchanged"])
        self.assertEqual(before, self.pipeline.manifest_path.read_bytes())
        self.assertEqual(1, self.pipeline.manifest()["stages"]["0"]["attempt_count"])

        (self.pipeline.root / str(outputs[0]["path"])).write_bytes(b"tampered-after-pass\n")
        with self.assertRaises(ArtifactVerificationError):
            complete_stage(
                self.pipeline.manifest_path,
                receipt["path"],
                repository_root=self.pipeline.root,
            )

    def test_pipeline_manifest_is_deterministic_for_identical_event_sequences(self) -> None:
        first = self.new_pipeline("determinism_a")
        second = self.new_pipeline("determinism_b")
        first.run_all()
        second.run_all()
        self.assertEqual(
            first.manifest_path.read_bytes(), second.manifest_path.read_bytes()
        )
        again = create_pipeline_manifest(
            repository_root=first.root,
            specimen_id=first.specimen_id,
            config_path=first.config_path,
            registration_mode=first.registration_mode,
            timestamp="2099-01-01T00:00:00Z",
        )
        self.assertFalse(again["changed"])
        self.assertEqual(
            first.manifest_path.read_bytes(), second.manifest_path.read_bytes()
        )

    def test_control_artifact_destinations_reject_traversal_and_collisions(self) -> None:
        escaped = self.pipeline.root.parent / "escaped-handoff.json"
        escaped_manifest = self.pipeline.root.parent / "outside-manifest.json"
        escaped_lock = escaped_manifest.parent / f".{escaped_manifest.name}.lock"
        with self.assertRaises(ArtifactVerificationError):
            validate_pipeline_manifest(
                escaped_manifest,
                repository_root=self.pipeline.root,
            )
        self.assertFalse(escaped_lock.exists())
        with self.assertRaises(ArtifactVerificationError):
            start_stage(
                self.pipeline.manifest_path,
                0,
                input_artifacts=self.pipeline.inputs(0),
                capability_inventory=self.pipeline.inventory,
                repository_root=self.pipeline.root,
                handoff_path="../escaped-handoff.json",
                timestamp=self.pipeline.timestamp(),
            )
        self.assertFalse(escaped.exists())

        self.pipeline.start(0)
        with self.assertRaises(ArtifactVerificationError):
            build_stage_receipt(
                self.pipeline.manifest_path,
                0,
                terminal_state="pass",
                output_artifacts=self.pipeline.outputs(0),
                assertions=self.pipeline.assertions(0),
                repository_root=self.pipeline.root,
                output_path=(
                    f"analysis/{self.pipeline.specimen_id}/receipts/alternate.json"
                ),
                timestamp=self.pipeline.timestamp(),
            )

        freeze_case = self.new_pipeline("freeze_collision")
        freeze_case.advance_to(2)
        freeze_case.start(2)
        outputs = freeze_case.outputs(2)
        config_before = freeze_case.config_path.read_bytes()
        with self.assertRaises(ArtifactVerificationError):
            record_autonomous_registration_freeze(
                freeze_case.manifest_path,
                frozen_artifacts=[
                    item
                    for item in outputs
                    if item["role"] in {"registered_graph", "registration_report"}
                ],
                repository_root=freeze_case.root,
                output_path=freeze_case.config_path,
                timestamp=freeze_case.timestamp(),
            )
        self.assertEqual(config_before, freeze_case.config_path.read_bytes())

    def test_verified_status_rechecks_receipts_and_state_topology(self) -> None:
        missing_receipt = self.new_pipeline("deleted_pass_receipt")
        missing_receipt.pass_stage(0)
        manifest = missing_receipt.manifest()
        receipt_path = (
            missing_receipt.root
            / manifest["stages"]["0"]["completion_receipt"]["path"]
        )
        receipt_path.unlink()
        with self.assertRaises((ArtifactVerificationError, ReceiptValidationError)):
            pipeline_status(
                missing_receipt.manifest_path,
                repository_root=missing_receipt.root,
            )

        impossible = self.new_pipeline("impossible_state")
        raw = json.loads(impossible.manifest_path.read_text())
        raw["pipeline_state"] = "pass"
        raw["current_stage"] = None
        raw["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "manifest_sha256"}
        )
        impossible._write_json(impossible.manifest_path, raw)
        with self.assertRaisesRegex(ManifestValidationError, "skips|inconsistent"):
            impossible.manifest()

    def test_autonomous_freeze_is_exact_ct_only_and_reverified(self) -> None:
        self.pipeline.advance_to(2)
        self.pipeline.start(2)
        outputs = self.pipeline.outputs(2)
        extra = self.pipeline.aligned_reference()
        with self.assertRaises((AccessPolicyError, ReceiptValidationError)):
            record_autonomous_registration_freeze(
                self.pipeline.manifest_path,
                frozen_artifacts=[
                    item
                    for item in outputs
                    if item["role"] in {"registered_graph", "registration_report"}
                ]
                + [extra],
                repository_root=self.pipeline.root,
                timestamp=self.pipeline.timestamp(),
            )
        self.pipeline.freeze_stage_2(outputs)
        manifest = self.pipeline.manifest()
        freeze_path = (
            self.pipeline.root
            / manifest["stages"]["2"]["attempts"][0]["registration_freeze"][
                "path"
            ]
        )
        freeze_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaises(ReceiptValidationError):
            pipeline_status(
                self.pipeline.manifest_path,
                repository_root=self.pipeline.root,
            )
        with self.assertRaises(ReceiptValidationError):
            self.pipeline.build_receipt(2, outputs=outputs)

    def test_stage_4_dev_split_is_absent_from_shared_handoff(self) -> None:
        self.pipeline.advance_to(4)
        started = self.pipeline.start(4)
        shared = json.loads(
            (self.pipeline.root / started["handoff"]["path"]).read_text()
        )
        dev = next(
            item
            for item in self.pipeline.manifest()["stages"]["4"]["attempts"][0][
                "input_artifacts"
            ]
            if item["role"] == "development_labels"
        )
        shared_bytes = json.dumps(shared, sort_keys=True)
        self.assertNotIn(dev["path"], shared_bytes)
        self.assertNotIn(dev["sha256"], shared_bytes)
        self.assertEqual(1, len(started["scoped_handoffs"]))
        scoped = json.loads(
            (
                self.pipeline.root
                / started["scoped_handoffs"][0]["path"]
            ).read_text()
        )
        self.assertEqual("missing_strut_agent", scoped["owner"])
        self.assertEqual([dev], scoped["input_artifacts"])

    def test_manifest_declared_source_path_and_hash_are_exact(self) -> None:
        self.pipeline.advance_to(2)
        inputs = self.pipeline.inputs(2)
        replacement = self.pipeline.record(
            "other-specimen/not-declared.bin",
            role="ct_volume",
            consumer="data_prep",
        )
        inputs = [
            replacement if item["role"] == "ct_volume" else item for item in inputs
        ]
        with self.assertRaisesRegex(AccessPolicyError, "frozen by Stage 0"):
            self.pipeline.start(2, inputs=inputs)

    def test_declared_specimen_manifest_replacement_preserves_history(self) -> None:
        pipeline = self.new_pipeline(
            "manifest_replacement", "challenge_aligned_json"
        )
        pipeline.advance_to(2)
        pipeline.start(2)
        outputs = pipeline.outputs(2)
        manifest = pipeline.manifest()
        prior = next(
            item
            for item in manifest["artifact_index"]
            if item["role"] == "specimen_manifest" and item["state"] == "active"
        )
        replacement = pipeline.analysis_ready_manifest_output()
        receipt = pipeline.build_receipt(2, outputs=[*outputs, replacement])
        completed = complete_stage(
            pipeline.manifest_path,
            receipt["path"],
            repository_root=pipeline.root,
        )
        self.assertEqual("pass", completed["state"])
        verified = pipeline.manifest()
        records = [
            item
            for item in verified["artifact_index"]
            if item["path"] == replacement["path"]
        ]
        self.assertEqual(
            {"active", "superseded"}, {item["state"] for item in records}
        )

    def test_manual_review_retry_can_publish_corrected_canonical_outputs(self) -> None:
        inputs = self.pipeline.inputs(0)
        self.pipeline.start(0, inputs=inputs)
        review = self.pipeline.build_receipt(
            0,
            terminal_state="manual_review",
            outputs=[self.pipeline.manual_review_evidence(0, 1)],
            assertions={},
        )
        complete_stage(
            self.pipeline.manifest_path,
            review["path"],
            repository_root=self.pipeline.root,
        )
        resolution = self.pipeline.record(
            f"analysis/{SPECIMEN_ID}/reviews/corrected_intake.json",
            role="manual_review_resolution",
        )
        resume_manual_review(
            self.pipeline.manifest_path,
            0,
            resolution_artifact=resolution,
            reason="Corrected intake evidence is available",
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        self.pipeline.start(0, inputs=inputs)
        passed = self.pipeline.build_receipt(0, outputs=self.pipeline.outputs(0))
        result = complete_stage(
            self.pipeline.manifest_path,
            passed["path"],
            repository_root=self.pipeline.root,
        )
        self.assertEqual("pass", result["state"])
        self.assertEqual(2, self.pipeline.manifest()["stages"]["0"]["attempt_count"])

    def test_flat_mcp_inventory_cannot_bypass_required_server_health(self) -> None:
        inventory = copy.deepcopy(self.pipeline.inventory)
        tool = inventory["mcp_servers"]["segmentation-tools"]["tools"][
            "inspect_volume_metadata"
        ]
        inventory["mcp_servers"] = {}
        inventory["mcp_tools"] = {"inspect_volume_metadata": tool}
        result = self.pipeline.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        self.assertTrue(
            any(
                item["kind"] == "mcp_server"
                for item in result["error"]["failures"]
            )
        )

    def test_stage_5_rejects_second_receipt_and_label_bearing_result(self) -> None:
        self.pipeline.advance_to(5)
        self.pipeline.start(5)
        outputs = self.pipeline.outputs(5)
        first = self.pipeline.build_receipt(5, outputs=outputs)
        self.assertTrue(first["changed"])
        with self.assertRaises(ReceiptValidationError):
            self.pipeline.build_receipt(5, outputs=outputs)

        leaking = self.new_pipeline("leaking_stage_5")
        leaking.advance_to(5)
        leaking.start(5)
        outputs = leaking.outputs(5)
        result_record = next(
            item for item in outputs if item["role"] == "sealed_evaluation_result"
        )
        result_path = leaking.root / result_record["path"]
        payload = json.loads(result_path.read_text())
        payload["sealed_strut_ids"] = [123]
        leaking._write_json(result_path, payload)
        outputs = [
            artifact_record(
                leaking.root,
                item["path"],
                role=item["role"],
                consumer=item.get("consumer"),
                phase=item.get("phase", "input"),
            )
            if item["role"] == "sealed_evaluation_result"
            else item
            for item in outputs
        ]
        receipt = leaking.build_receipt(5, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "leak labels"):
            complete_stage(
                leaking.manifest_path,
                receipt["path"],
                repository_root=leaking.root,
            )

    def test_reviewed_retry_cannot_repoint_completion_to_older_receipt(self) -> None:
        inputs = self.pipeline.inputs(0)
        self.pipeline.start(0, inputs=inputs)
        review = self.pipeline.build_receipt(
            0,
            terminal_state="manual_review",
            outputs=[self.pipeline.manual_review_evidence(0, 1)],
            assertions={},
        )
        complete_stage(
            self.pipeline.manifest_path,
            review["path"],
            repository_root=self.pipeline.root,
        )
        resolution = self.pipeline.record(
            f"analysis/{SPECIMEN_ID}/reviews/retry-resolution.json",
            role="manual_review_resolution",
        )
        resume_manual_review(
            self.pipeline.manifest_path,
            0,
            resolution_artifact=resolution,
            reason="Review resolved",
            repository_root=self.pipeline.root,
            timestamp=self.pipeline.timestamp(),
        )
        self.pipeline.start(0, inputs=inputs)
        passed = self.pipeline.build_receipt(0, outputs=self.pipeline.outputs(0))
        complete_stage(
            self.pipeline.manifest_path,
            passed["path"],
            repository_root=self.pipeline.root,
        )

        raw = json.loads(self.pipeline.manifest_path.read_text())
        older = raw["stages"]["0"]["attempts"][0]["receipt"]
        raw["stages"]["0"]["completion_receipt"] = older
        raw["predecessor_receipt_sha256"] = older["canonical_sha256"]
        raw["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "manifest_sha256"}
        )
        self.pipeline._write_json(self.pipeline.manifest_path, raw)
        with self.assertRaisesRegex(ReceiptValidationError, "current attempt evidence"):
            self.pipeline.manifest()

    def test_dependency_inventory_requires_real_booleans(self) -> None:
        agent_case = self.new_pipeline("string_agent_capability")
        inventory = copy.deepcopy(agent_case.inventory)
        agent_name = agent_case.contracts[0]["required_dependencies"]["agents"][0][
            "name"
        ]
        inventory["agents"][agent_name]["available"] = "false"
        result = agent_case.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        self.assertTrue(
            any(item["kind"] == "agent" for item in result["error"]["failures"])
        )

        server_case = self.new_pipeline("string_server_health")
        inventory = copy.deepcopy(server_case.inventory)
        inventory["mcp_servers"]["segmentation-tools"]["healthy"] = "true"
        result = server_case.start(0, inventory=inventory)
        self.assertEqual("halt", result["state"])
        self.assertTrue(
            any(
                item["kind"] == "mcp_server"
                for item in result["error"]["failures"]
            )
        )

    def test_manifest_rejects_unknown_registration_mode_after_rehash(self) -> None:
        raw = json.loads(self.pipeline.manifest_path.read_text())
        raw["registration_mode"] = "rogue_mode"
        raw["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "manifest_sha256"}
        )
        self.pipeline._write_json(self.pipeline.manifest_path, raw)
        with self.assertRaisesRegex(ManifestValidationError, "registration_mode"):
            self.pipeline.manifest()

    def test_handoffs_are_closed_and_revalidated_before_receipt_build(self) -> None:
        started = self.pipeline.start(0)
        handoff_path = self.pipeline.root / started["handoff"]["path"]
        handoff = json.loads(handoff_path.read_text())
        handoff["owner"] = "eval_agent"
        handoff["forbidden_operations"] = []
        handoff["canonical_handoff_sha256"] = canonical_json_sha256(
            {
                key: value
                for key, value in handoff.items()
                if key != "canonical_handoff_sha256"
            }
        )
        self.pipeline._write_json(handoff_path, handoff)
        raw = json.loads(self.pipeline.manifest_path.read_text())
        record = raw["stages"]["0"]["attempts"][0]["handoff"]
        record["sha256"] = sha256_file(handoff_path)
        record["canonical_sha256"] = handoff["canonical_handoff_sha256"]
        raw["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "manifest_sha256"}
        )
        self.pipeline._write_json(self.pipeline.manifest_path, raw)
        with self.assertRaisesRegex(ReceiptValidationError, "open-ended|misbound"):
            self.pipeline.build_receipt(0, outputs=self.pipeline.outputs(0))

    def test_completion_requires_canonical_receipt_path(self) -> None:
        self.pipeline.start(0)
        receipt = self.pipeline.build_receipt(0, outputs=self.pipeline.outputs(0))
        alternate = (
            self.pipeline.root
            / f"analysis/{SPECIMEN_ID}/receipts/alternate.json"
        )
        alternate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(receipt["path"], alternate)
        with self.assertRaisesRegex(ReceiptValidationError, "canonical attempt path"):
            complete_stage(
                self.pipeline.manifest_path,
                alternate,
                repository_root=self.pipeline.root,
            )

    def test_stage_4_attestation_and_every_evidence_packet_are_bound(self) -> None:
        evidence_case = self.new_pipeline("unbound_evidence")
        evidence_case.advance_to(4)
        evidence_case.start(4)
        outputs = evidence_case.outputs(4)
        outputs.append(
            evidence_case.record(
                f"analysis/{SPECIMEN_ID}/evidence/strut_2/manifest.json",
                role="evidence_packets",
                phase="output",
            )
        )
        receipt = evidence_case.build_receipt(4, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "evidence-set"):
            complete_stage(
                evidence_case.manifest_path,
                receipt["path"],
                repository_root=evidence_case.root,
            )

        leak_case = self.new_pipeline("calibration_attestation_leak")
        leak_case.advance_to(4)
        leak_case.start(4)
        outputs = leak_case.outputs(4)
        attestation = next(
            item
            for item in outputs
            if item["role"] == "missing_calibration_attestation"
        )
        attestation_path = leak_case.root / attestation["path"]
        payload = json.loads(attestation_path.read_text())
        payload["raw_development_ids"] = [1, 2, 3]
        leak_case._write_json(attestation_path, payload)
        replacement = artifact_record(
            leak_case.root,
            attestation_path,
            role="missing_calibration_attestation",
            phase=attestation["phase"],
        )
        outputs = [
            replacement
            if item["role"] == "missing_calibration_attestation"
            else item
            for item in outputs
        ]
        receipt = leak_case.build_receipt(4, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "undeclared|missing fields"):
            complete_stage(
                leak_case.manifest_path,
                receipt["path"],
                repository_root=leak_case.root,
            )

    def test_stage_5_rejects_nested_aggregate_and_receipt_channels(self) -> None:
        aggregate_case = self.new_pipeline("nested_sealed_count")
        aggregate_case.advance_to(5)
        aggregate_case.start(5)
        outputs = aggregate_case.outputs(5)
        result = next(
            item for item in outputs if item["role"] == "sealed_evaluation_result"
        )
        result_path = aggregate_case.root / result["path"]
        payload = json.loads(result_path.read_text())
        payload["sealed_strut_count"] = {"raw_sealed_ids": [1, 2, 3]}
        aggregate_case._write_json(result_path, payload)
        updated = artifact_record(
            aggregate_case.root,
            result_path,
            role="sealed_evaluation_result",
            phase=result["phase"],
        )
        outputs = [
            updated if item["role"] == "sealed_evaluation_result" else item
            for item in outputs
        ]
        receipt = aggregate_case.build_receipt(5, outputs=outputs)
        with self.assertRaisesRegex(ReceiptValidationError, "sealed_strut_count"):
            complete_stage(
                aggregate_case.manifest_path,
                receipt["path"],
                repository_root=aggregate_case.root,
            )

        assertion_case = self.new_pipeline("receipt_assertion_leak")
        assertion_case.advance_to(5)
        assertion_case.start(5)
        assertions = assertion_case.assertions(5)
        assertions["raw_sealed_ids"] = [1, 2, 3]  # type: ignore[assignment]
        with self.assertRaisesRegex(ReceiptValidationError, "assertions|assertion set"):
            assertion_case.build_receipt(
                5,
                outputs=assertion_case.outputs(5),
                assertions=assertions,
            )

    def test_stage_6_accepts_public_label_report_but_not_label_hashes(self) -> None:
        self.pipeline.advance_to(6)
        label_path = f"analysis/{SPECIMEN_ID}/labels/label_report.md"
        inputs = [
            *self.pipeline.inputs(6),
            self.pipeline.record(
                label_path,
                role="label_report",
                consumer="report_agent",
            ),
        ]
        started = self.pipeline.start(6, inputs=inputs)
        self.assertEqual("running", started["state"])

    def test_sensitive_registry_is_rederived_before_renamed_label_access(self) -> None:
        self.pipeline.advance_to(4)
        inputs = self.pipeline.inputs(4)
        sealed_path = self.pipeline.root / "evals/labels/sealed_split.json"
        disguised_path = (
            self.pipeline.root
            / f"analysis/{SPECIMEN_ID}/struts/thresholds.json"
        )
        disguised_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sealed_path, disguised_path)
        inputs.append(
            artifact_record(
                self.pipeline.root,
                disguised_path,
                role="classification_thresholds",
                consumer="classifier_verifier",
                phase="independent_verification",
            )
        )

        raw = json.loads(self.pipeline.manifest_path.read_text())
        raw["sensitive_artifact_hashes"]["sealed_labels"] = []
        sealed_record = next(
            record
            for record in raw["artifact_index"]
            if record["role"] == "sealed_labels"
        )
        sealed_record["sensitivity"] = None
        raw["manifest_sha256"] = canonical_json_sha256(
            {key: value for key, value in raw.items() if key != "manifest_sha256"}
        )
        self.pipeline._write_json(self.pipeline.manifest_path, raw)
        with self.assertRaisesRegex(
            ManifestValidationError, "artifact_index|Sensitive artifact registry"
        ):
            self.pipeline.start(4, inputs=inputs)

    def test_invalid_public_stage_numbers_fail_closed(self) -> None:
        with self.assertRaisesRegex(IllegalTransitionError, "Unknown stage"):
            build_stage_receipt(
                self.pipeline.manifest_path,
                99,
                terminal_state="halt",
                output_artifacts=[],
                assertions={},
                repository_root=self.pipeline.root,
            )
        with self.assertRaisesRegex(IllegalTransitionError, "Unknown stage"):
            resume_manual_review(
                self.pipeline.manifest_path,
                99,
                resolution_artifact={},
                reason="invalid",
                repository_root=self.pipeline.root,
            )


if __name__ == "__main__":
    unittest.main()
