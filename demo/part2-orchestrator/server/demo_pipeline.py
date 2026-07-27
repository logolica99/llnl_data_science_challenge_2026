"""Synthetic specialist fixtures that exercise the real Part 2 control plane.

This module deliberately creates tiny opaque artifacts.  It calls the public
orchestration API for every state change, hash, handoff, receipt, retry, and
access-policy decision.  It never executes scientific or numerical code and
its outputs must never be presented as specimen results.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import fnmatch
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from part2_orchestration import (  # noqa: E402
    artifact_record,
    authorize_post_freeze_aligned_input,
    build_stage_receipt,
    canonical_json_sha256,
    complete_stage,
    create_pipeline_manifest,
    record_autonomous_registration_freeze,
    resume_manual_review,
    start_stage,
    validate_pipeline_manifest,
)


SPECIMEN_ID = "demo_missing_strut_specimen"


class SyntheticFixtureStageRunner:
    """Drive one isolated Stage 0-6 run with transparent fixture outputs."""

    def __init__(self, root: Path, registration_mode: str) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPOSITORY_ROOT / "analysis" / "contracts",
            self.root / "analysis" / "contracts",
        )
        self.specimen_id = SPECIMEN_ID
        self.registration_mode = registration_mode
        self._tick = 0
        self.config_path = self.root / "config" / "frozen_analysis.json"
        self._write_json(
            self.config_path,
            {
                "schema_version": "part2-demo-analysis-config/1.0.0",
                "registration_mode": registration_mode,
                "frozen": True,
                "fixture_specialists": True,
                "scientific_algorithms_executed": False,
            },
        )
        self.contracts = self._load_contracts()
        self.inventory = self._fixture_capability_inventory()
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

    def manifest(self, *, verify_artifacts: bool = True) -> dict[str, Any]:
        return validate_pipeline_manifest(
            self.manifest_path,
            repository_root=self.root,
            verify_artifacts=verify_artifacts,
        )

    def _load_contracts(self) -> dict[int, dict[str, Any]]:
        contracts: dict[int, dict[str, Any]] = {}
        for path in sorted((self.root / "analysis" / "contracts").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if "stage_number" in value:
                self._validate_contract_paths(value)
                contracts[int(value["stage_number"])] = value
        if set(contracts) != set(range(7)):
            raise RuntimeError("The demo requires Stage 0-6 contracts")
        return contracts

    @staticmethod
    def _validate_contract_paths(contract: Mapping[str, Any]) -> None:
        """Reject artifact patterns that could address files outside the demo root."""

        for direction in ("input", "output"):
            policy = contract.get(f"{direction}_artifacts", {})
            for rule in policy.get("allowed", []):
                pattern = Path(str(rule.get("path", "")))
                if pattern.is_absolute() or any(
                    part in {".", ".."} for part in pattern.parts
                ):
                    raise RuntimeError(
                        f"Demo contract {contract.get('stage', 'unknown')} has an "
                        f"unsafe {direction} artifact path"
                    )

    def _fixture_capability_inventory(self) -> dict[str, Any]:
        """Declare fixture providers matching the frozen dependency schemas."""

        inventory: dict[str, Any] = {"agents": {}, "mcp_servers": {}}
        for contract in self.contracts.values():
            dependencies = contract["required_dependencies"]
            for agent in dependencies.get("agents", []):
                inventory["agents"][agent["name"]] = {
                    "available": True,
                    "contract_version": agent["contract_version"],
                    "fixture": True,
                }
            for tool in dependencies.get("mcp_tools", []):
                server_name = tool.get("server", "segmentation-tools")
                server = inventory["mcp_servers"].setdefault(
                    server_name,
                    {"healthy": True, "fixture": True, "tools": {}},
                )
                server["tools"][tool["name"]] = {
                    "available": True,
                    "response_schema_version": tool["response_schema_version"],
                    "fixture": True,
                }
        return inventory

    def inventory_with_missing_dependency(self) -> dict[str, Any]:
        inventory = copy.deepcopy(self.inventory)
        inventory["mcp_servers"]["segmentation-tools"]["tools"].pop(
            "inspect_volume_metadata",
            None,
        )
        return inventory

    def _confined_path(self, supplied: str | Path) -> Path:
        candidate = Path(supplied)
        lexical = candidate if candidate.is_absolute() else self.root / candidate
        path = lexical.resolve()
        if not path.is_relative_to(self.root):
            raise RuntimeError("Demo fixture path escaped its isolated root")
        return path

    def _write_json(self, path: str | Path, value: object) -> None:
        path = self._confined_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_fixture(
        self,
        relative: str,
        payload: bytes | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        path = self._confined_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not path.exists():
            body = payload if payload is not None else f"fixture:{relative}\n".encode()
            path.write_bytes(body)
        return path

    @staticmethod
    def _rule_for(
        contract: Mapping[str, Any], direction: str, role: str
    ) -> dict[str, Any]:
        policy = contract[f"{direction}_artifacts"]
        for rule in policy["allowed"]:
            if fnmatch.fnmatchcase(role, str(rule.get("role", ""))):
                return rule
        raise RuntimeError(
            f"Contract {contract['stage']} has no {direction} rule for {role}"
        )

    def _current_attempt(self) -> int:
        manifest = self.manifest(verify_artifacts=False)
        current = manifest["current_stage"]
        if current is None:
            return 1
        return max(1, manifest["stages"][str(current)]["attempt_count"])

    def _materialize_path(self, pattern: str, role: str) -> str:
        if pattern.startswith("<") and pattern.endswith(">"):
            return f"inputs/{role}.bin"
        value = pattern.replace("<specimen_id>", self.specimen_id)
        value = value.replace("<attempt>", str(self._current_attempt()))
        value = value.replace("<timestamp>", "20260727T000000Z")
        value = value.replace("<strut_id>", "1")
        value = re.sub(r"<[^>]+>", role, value)
        value = value.replace("**", "manifest.json")
        return value.replace("*", "fixture")

    def _record_for_rule(
        self,
        rule: Mapping[str, Any],
        role: str,
        *,
        payload: bytes | None = None,
        consumer: str | None = None,
        phase: str | None = None,
    ) -> dict[str, Any]:
        relative = self._materialize_path(str(rule["path"]), role)
        self._write_fixture(relative, payload)
        consumers = rule.get("consumers", [])
        producers = rule.get("producers", [])
        phases = rule.get("phases", [])
        selected_consumer = consumer
        if selected_consumer is None and consumers:
            selected_consumer = str(consumers[0])
        selected_phase = phase
        if selected_phase is None:
            selected_phase = str(phases[0]) if phases else "input"
        return artifact_record(
            self.root,
            relative,
            role=role,
            consumer=selected_consumer,
            producer=str(producers[0]) if producers else None,
            phase=selected_phase,
        )

    def _record(
        self,
        relative: str,
        *,
        role: str,
        consumer: str | None = None,
        producer: str | None = None,
        phase: str = "input",
    ) -> dict[str, Any]:
        self._write_fixture(relative)
        return artifact_record(
            self.root,
            relative,
            role=role,
            consumer=consumer,
            producer=producer,
            phase=phase,
        )

    def inputs(self, stage_number: int) -> list[dict[str, Any]]:
        contract = self.contracts[stage_number]
        policy = contract["input_artifacts"]
        result: list[dict[str, Any]] = []
        for role in policy["required_roles"]:
            if role == "stage_handoff":
                continue
            rule = self._rule_for(contract, "input", str(role))
            result.append(self._record_for_rule(rule, str(role)))
        if self.registration_mode == "challenge_aligned_json" and stage_number in {
            0,
            2,
        }:
            rule = self._rule_for(contract, "input", "challenge_aligned_graph")
            result.append(
                self._record_for_rule(
                    rule,
                    "challenge_aligned_graph",
                    phase="challenge_aligned_json",
                )
            )
        return result

    def _output_record(self, stage_number: int, role: str) -> dict[str, Any]:
        rule = self._rule_for(self.contracts[stage_number], "output", role)
        return self._record_for_rule(rule, role)

    def outputs(self, stage_number: int) -> list[dict[str, Any]]:
        contract = self.contracts[stage_number]
        roles = [str(role) for role in contract["output_artifacts"]["required_roles"]]
        if stage_number == 4:
            return self._stage4_outputs(roles)

        records: list[dict[str, Any]] = []
        for role in roles:
            rule = self._rule_for(contract, "output", role)
            payload = None
            if stage_number == 5 and role == "sealed_evaluation_result":
                payload = self._stage5_evaluation_payload()
            records.append(self._record_for_rule(rule, role, payload=payload))
        return records

    def _stage4_outputs(self, roles: list[str]) -> list[dict[str, Any]]:
        contract = self.contracts[4]
        non_verifier = [
            role
            for role in roles
            if role
            not in {
                "missing_calibration_attestation",
                "classifier_verifier_report",
            }
        ]
        records = [self._output_record(4, role) for role in non_verifier]
        by_role = {str(record["role"]): record for record in records}
        manifest = self.manifest()
        attempt = manifest["stages"]["4"]["attempts"][-1]
        scoped_handoff = next(
            item
            for item in attempt["scoped_handoffs"]
            if item["consumer"] == "missing_strut_agent"
        )
        metrics_hash = next(
            item["sha256"]
            for item in attempt["input_artifacts"]
            if item["role"] == "per_strut_metrics"
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
            "per_strut_metrics_sha256": metrics_hash,
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
            contract,
            "output",
            "missing_calibration_attestation",
        )
        attestation_path = self._materialize_path(
            str(attestation_rule["path"]),
            "missing_calibration_attestation",
        )
        self._write_json(self.root / attestation_path, attestation)
        attestation_record = self._record_for_rule(
            attestation_rule,
            "missing_calibration_attestation",
        )
        records.append(attestation_record)
        by_role["missing_calibration_attestation"] = attestation_record

        evidence = sorted(
            (
                {"path": item["path"], "sha256": item["sha256"]}
                for item in records
                if item["role"]
                in {"evidence_packets", "evidence_packet_manifest", "evidence_index"}
            ),
            key=lambda item: (item["path"], item["sha256"]),
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
            "predecessor_receipt_sha256": attempt["predecessor_receipt_sha256"],
            "input_handoff_sha256": attempt["handoff"]["canonical_sha256"],
            "participated_in_classification": False,
            "label_access": {
                "development_split_read": False,
                "sealed_split_read": False,
            },
            "bindings": {
                "classified_struts_sha256": by_role["classified_struts"]["sha256"],
                "thresholds_sha256": by_role["classification_thresholds"]["sha256"],
                "decision_log_sha256": by_role[
                    "classification_decision_log"
                ]["sha256"],
                "evidence_set_sha256": canonical_json_sha256(evidence),
                "per_strut_metrics_sha256": metrics_hash,
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
        verifier_rule = self._rule_for(
            contract,
            "output",
            "classifier_verifier_report",
        )
        verifier_path = self._materialize_path(
            str(verifier_rule["path"]),
            "classifier_verifier_report",
        )
        self._write_json(self.root / verifier_path, verifier)
        records.append(
            self._record_for_rule(verifier_rule, "classifier_verifier_report")
        )
        return records

    def _stage5_evaluation_payload(self) -> bytes:
        manifest = self.manifest()
        attempt = manifest["stages"]["5"]["attempts"][-1]
        by_role = {item["role"]: item for item in attempt["input_artifacts"]}
        classes = ("missing", "broken", "thin", "present")
        value = {
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
                "class_order": list(classes),
                "rows_actual_columns_predicted": {
                    actual: {predicted: 0 for predicted in classes}
                    for actual in classes
                },
            },
            "omitted_metrics": {
                "precision": (
                    "undefined because detections outside sealed intentional "
                    "deletions may be unintentional defects"
                ),
                "f1": "not computed because precision is undefined",
            },
            "artifacts": {},
            "hashes": {
                "classifications_sha256": by_role["classified_struts"]["sha256"],
                "sealed_labels_sha256": by_role["sealed_labels"]["sha256"],
            },
            "provenance": {"eval_side": True, "sealed_labels_read": True},
            "warnings": [],
        }
        value["confusion_matrix"]["rows_actual_columns_predicted"]["missing"][
            "present"
        ] = 1
        return (json.dumps(value, sort_keys=True) + "\n").encode()

    def assertions(self, stage_number: int) -> dict[str, bool]:
        assertions = {
            str(name): True
            for name in self.contracts[stage_number].get(
                "required_receipt_assertions",
                [],
            )
        }
        if stage_number == 5:
            assertions["optimization_performed"] = False
        return assertions

    def start(
        self,
        stage_number: int,
        *,
        missing_dependency: bool = False,
    ) -> dict[str, Any]:
        inventory = (
            self.inventory_with_missing_dependency()
            if missing_dependency
            else self.inventory
        )
        return start_stage(
            self.manifest_path,
            stage_number,
            input_artifacts=self.inputs(stage_number),
            capability_inventory=inventory,
            repository_root=self.root,
            timestamp=self.timestamp(),
        )

    def complete_pass(self, stage_number: int) -> dict[str, Any]:
        outputs = self.outputs(stage_number)
        freeze = None
        authorization = None
        if stage_number == 2 and self.registration_mode == "autonomous_v2":
            freeze = record_autonomous_registration_freeze(
                self.manifest_path,
                frozen_artifacts=[
                    item
                    for item in outputs
                    if item["role"] in {"registered_graph", "registration_report"}
                ],
                repository_root=self.root,
                timestamp=self.timestamp(),
            )
            reference = self._record(
                "inputs/authorized_aligned_graph.json",
                role="autonomous_validation_reference",
                consumer="data_prep",
                phase="autonomous_v2_post_freeze_validation",
            )
            authorization = authorize_post_freeze_aligned_input(
                self.manifest_path,
                aligned_artifact=reference,
                repository_root=self.root,
                timestamp=self.timestamp(),
            )
        receipt = build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state="pass",
            output_artifacts=outputs,
            assertions=self.assertions(stage_number),
            repository_root=self.root,
            timestamp=self.timestamp(),
        )
        completed = complete_stage(
            self.manifest_path,
            receipt["path"],
            repository_root=self.root,
        )
        return {
            "output_count": len(outputs),
            "receipt": receipt,
            "completed": completed,
            "freeze": freeze,
            "authorization": authorization,
        }

    def complete_manual_review(self, stage_number: int) -> dict[str, Any]:
        attempt = self.manifest()["stages"][str(stage_number)]["attempt_count"]
        evidence = self._record(
            (
                f"analysis/{self.specimen_id}/reviews/"
                f"stage_{stage_number}_attempt_{attempt}/evidence.json"
            ),
            role="manual_review_evidence",
            phase="manual_review",
        )
        receipt = build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state="manual_review",
            output_artifacts=[evidence],
            assertions={},
            repository_root=self.root,
            timestamp=self.timestamp(),
        )
        return complete_stage(
            self.manifest_path,
            receipt["path"],
            repository_root=self.root,
        )

    def resume_review(self, stage_number: int) -> dict[str, Any]:
        resolution = self._record(
            (
                f"analysis/{self.specimen_id}/reviews/"
                f"stage_{stage_number}_resolution.json"
            ),
            role="manual_review_resolution",
            phase="manual_review_resolution",
        )
        return resume_manual_review(
            self.manifest_path,
            stage_number,
            resolution_artifact=resolution,
            reason="Scientist accepted the fixture review evidence for this demo.",
            repository_root=self.root,
            timestamp=self.timestamp(),
        )

    def build_tampered_receipt(self, stage_number: int) -> Path:
        outputs = self.outputs(stage_number)
        receipt = build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state="pass",
            output_artifacts=outputs,
            assertions=self.assertions(stage_number),
            repository_root=self.root,
            timestamp=self.timestamp(),
        )
        path = Path(receipt["path"])
        document = json.loads(path.read_text(encoding="utf-8"))
        document["canonical_receipt_sha256"] = "0" * 64
        self._write_json(path, document)
        return path
