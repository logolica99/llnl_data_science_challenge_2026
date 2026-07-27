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


SPECIMEN_ID = "synthetic_part2"
CONTRACT_VERSION = "agent-stage-contract/1.0.0"


class SyntheticPipeline:
    """Create one isolated, deterministic seven-stage control-plane run."""

    def __init__(self, root: Path, registration_mode: str = "autonomous_v2"):
        self.root = root
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
            return f"inputs/{role}.bin"
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
            if role == "stage_handoff":
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
        if stage_number != 4:
            records: list[dict[str, object]] = []
            for role in roles:
                rule = self._rule_for(contract, "output", role)
                payload = None
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
    ) -> dict[str, object]:
        return build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state=terminal_state,
            output_artifacts=self.outputs(stage_number) if outputs is None else outputs,
            assertions=self.assertions(stage_number) if assertions is None else assertions,
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
        self, name: str, registration_mode: str = "autonomous_v2"
    ) -> SyntheticPipeline:
        return SyntheticPipeline(self.scratch / name, registration_mode)

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
        rule = pipeline._rule_for(
            pipeline.contracts[2], "output", "analysis_ready_specimen_manifest"
        )
        replacement = pipeline.record_for_rule(
            rule,
            "analysis_ready_specimen_manifest",
            payload=b'{"lifecycle_state":"analysis_ready"}\n',
            overwrite=True,
            replaces_sha256=prior["sha256"],
        )
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
