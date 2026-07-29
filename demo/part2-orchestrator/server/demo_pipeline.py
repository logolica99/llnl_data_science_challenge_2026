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
import struct
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.orchestration.pipeline import (  # noqa: E402
    artifact_record,
    build_stage_receipt,
    canonical_json_sha256,
    complete_stage,
    create_pipeline_manifest,
    record_autonomous_registration_freeze,
    resume_manual_review,
    sha256_file,
    start_stage,
    validate_pipeline_manifest,
)
from llnl_nde.orchestration.specimen_ingest import ingest_specimen  # noqa: E402
from llnl_nde.orchestration.receipts import create_data_prep_handoff  # noqa: E402


SPECIMEN_ID = "demo_missing_strut_specimen"


class SyntheticFixtureStageRunner:
    """Drive one isolated five-stage run with transparent fixture outputs."""

    def __init__(self, root: Path, registration_mode: str) -> None:
        self.root = root.resolve()
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
        self.design_id = "demo_fixture_design"
        self.requested_analysis_scope = "roi_screening"
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
        if set(contracts) != set(range(5)):
            raise RuntimeError("The demo requires contiguous Stage 0-4 contracts")
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
    def _npy_bytes() -> bytes:
        shape = (2, 3, 4)
        header = repr(
            {"descr": "|u1", "fortran_order": False, "shape": shape}
        ).encode("latin1")
        padding = (16 - ((10 + len(header) + 1) % 16)) % 16
        header = header + (b" " * padding) + b"\n"
        return (
            b"\x93NUMPY"
            + bytes((1, 0))
            + struct.pack("<H", len(header))
            + header
            + bytes(24)
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
        return b"""solid fixture
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid fixture
"""

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
            suffix = {
                "cad_stl": ".stl",
                "nominal_graph": ".json",
                "ct_volume": ".npy",
                "challenge_aligned_graph": ".json",
                "design_transform_declaration": ".json",
            }.get(role, ".bin")
            return f"inputs/{role}{suffix}"
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
        overwrite: bool = False,
        consumer: str | None = None,
        phase: str | None = None,
        replaces_sha256: str | None = None,
    ) -> dict[str, Any]:
        relative = self._materialize_path(str(rule["path"]), role)
        if payload is None and role in {"canonical_segmentation_mask", "ct_volume"}:
            payload = self._npy_bytes()
        elif payload is None and role in {"nominal_graph", "challenge_aligned_graph"}:
            payload = self._graph_bytes()
        elif payload is None and role == "cad_stl":
            payload = self._stl_bytes()
        self._write_fixture(relative, payload, overwrite=overwrite)
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
            replaces_sha256=replaces_sha256,
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
            if role == "stage_handoff" or (
                stage_number == 0 and role == "scientist_intake_request"
            ):
                continue
            rule = self._rule_for(contract, "input", str(role))
            result.append(self._record_for_rule(rule, str(role)))
        if stage_number == 0:
            by_role = {str(record["role"]): record for record in result}
            def binding(role: str, document_role: str) -> dict[str, str]:
                record = by_role[role]
                return {
                    "path": str(record["path"]),
                    "sha256": str(record["sha256"]),
                    "role": document_role,
                    "retention": "committed",
                }

            request_base = {
                "schema_version": "part2-scientist-intake-request/2.0.0",
                "created_at": "2026-07-27T00:00:00Z",
                "specimen_id": self.specimen_id,
                "design_id": self.design_id,
                "requested_analysis_scope": self.requested_analysis_scope,
                "registration_mode": self.registration_mode,
                "association_confirmed": True,
                "graph_axes": ["x", "y", "z"],
                "array_axes": ["z", "y", "x"],
                "inputs": {
                    "nominal_graph": binding("nominal_graph", "design_graph"),
                    "ct": binding("ct_volume", "ct_volume"),
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
            self._write_json(request_path, request)
            result.append(
                self._record_for_rule(rule, "scientist_intake_request")
            )
        return result

    def _output_record(self, stage_number: int, role: str) -> dict[str, Any]:
        rule = self._rule_for(self.contracts[stage_number], "output", role)
        return self._record_for_rule(rule, role)

    def outputs(self, stage_number: int) -> list[dict[str, Any]]:
        contract = self.contracts[stage_number]
        roles = [str(role) for role in contract["output_artifacts"]["required_roles"]]
        if stage_number == 0:
            return self._production_stage0_outputs(roles)
        if stage_number == 1:
            return self._bound_stage_outputs(stage_number, roles)
        if stage_number == 3:
            return self._production_stage3_outputs(roles)

        records: list[dict[str, Any]] = []
        for role in roles:
            rule = self._rule_for(contract, "output", role)
            payload = None
            records.append(self._record_for_rule(rule, role, payload=payload))
        return records

    def _production_stage0_outputs(self, roles: list[str]) -> list[dict[str, Any]]:
        """Run the real intake validator over tiny graph/CT fixture inputs."""

        attempt = self.manifest(verify_artifacts=False)["stages"]["0"][
            "attempts"
        ][-1]
        sources = {
            str(record["role"]): record for record in attempt["input_artifacts"]
        }
        ct_record = sources["ct_volume"]
        graph_record = sources["nominal_graph"]
        config_relative = Path("analysis") / self.specimen_id / "config"
        metadata_relative = config_relative / "ct_metadata_response.json"
        call_relative = config_relative / "ct_metadata_mcp_call_receipt.json"
        normalized_relative = (
            Path("analysis")
            / self.specimen_id
            / "design"
            / "normalized_nominal_graph.npz"
        )
        normalized_path = self._write_fixture(
            normalized_relative.as_posix(), b"fixture-normalized-graph\n"
        )
        ct_path = self.root / str(ct_record["path"])
        spacing = {
            axis: {
                "value": "unknown",
                "unit": "unknown",
                "provenance": {
                    "source": "unknown",
                    "field": "unknown",
                    "raw_value": "unknown",
                },
            }
            for axis in ("z", "y", "x")
        }
        request_binding = {
            "input_filepath": ct_record["path"],
            "output_filepath": metadata_relative.as_posix(),
            "call_receipt_filepath": call_relative.as_posix(),
            "header_only": True,
            "include_sha256": True,
            "retention": "committed",
        }
        header_facts = {
            "file_bytes": ct_path.stat().st_size,
            "format": "npy",
            "shape": [2, 3, 4],
            "ndim": 3,
            "dtype": "uint8",
            "dtype_string": "|u1",
            "byte_order": "not_applicable",
            "axes": "unknown",
            "voxel_count": 24,
            "array_bytes": 24,
        }
        metadata_result = {
            "status": "ok",
            "authoritative": True,
            "inspection_mode": "header_only",
            "method": "volume_metadata",
            "method_version": "1.0.0",
            "output_schema_version": "volume-metadata/1.0.0",
            "path": ct_record["path"],
            "sha256": ct_record["sha256"],
            **header_facts,
            "voxel_spacing": spacing,
            "statistics": {
                "status": "not_computed",
                "minimum": "unknown",
                "maximum": "unknown",
                "mean": "unknown",
                "finite_count": "unknown",
                "nonfinite_count": "unknown",
            },
            "manifest_fragment": {
                "ct_volume": {
                    "path": ct_record["path"],
                    "sha256": ct_record["sha256"],
                    "role": "ct_volume",
                    "retention": "committed",
                },
                "ct_metadata": {
                    "format": "npy",
                    "shape": [2, 3, 4],
                    "dtype": "uint8",
                    "byte_order": "not_applicable",
                    "array_axes": "unknown",
                    "voxel_spacing": spacing,
                },
            },
        }
        summary = "Persisted authoritative header-only CT metadata response"
        metadata_evidence = {
            "schema_version": "volume-metadata-mcp-evidence/1.1.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": "pass",
            "summary": summary,
            "request": request_binding,
            "result": metadata_result,
            "warnings": [],
            "error": None,
        }
        self._write_json(metadata_relative, metadata_evidence)
        metadata_sha256 = sha256_file(self.root / metadata_relative)
        call_base = {
            "schema_version": "volume-metadata-mcp-call-receipt/1.0.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": "pass",
            "summary": summary,
            "request": request_binding,
            "header_facts": header_facts,
            "artifacts": {
                "metadata_response": {
                    "path": metadata_relative.as_posix(),
                    "sha256": metadata_sha256,
                    "role": "ct_metadata_mcp_response",
                    "retention": "committed",
                }
            },
            "hashes": {
                "input_sha256": ct_record["sha256"],
                "request_sha256": canonical_json_sha256(request_binding),
                "result_sha256": canonical_json_sha256(metadata_result),
                "header_facts_sha256": canonical_json_sha256(header_facts),
                "metadata_response_sha256": metadata_sha256,
            },
            "warnings": [],
            "error": None,
        }
        call_receipt = {
            **call_base,
            "canonical_call_receipt_sha256": canonical_json_sha256(call_base),
        }
        self._write_json(call_relative, call_receipt)
        call_sha256 = sha256_file(self.root / call_relative)
        ingest_specimen(
            repository_root=self.root,
            specimen_id=self.specimen_id,
            design_id=self.design_id,
            requested_analysis_scope=self.requested_analysis_scope,
            design_graph_path=self.root / str(graph_record["path"]),
            ct_path=ct_path,
            ct_metadata_response_path=self.root / metadata_relative,
            ct_metadata_response_sha256=metadata_sha256,
            ct_metadata_call_receipt_path=self.root / call_relative,
            ct_metadata_call_receipt_sha256=call_sha256,
            registration_mode="autonomous_v2",
            association_confirmed=True,
            allowed_data_roots=[self.root],
            graph_axes="xyz",
            array_axes="zyx",
            retention="committed",
            schema_path=self.root / "analysis/schema/specimen_manifest.schema.json",
            normalized_graph_path=normalized_path,
            normalized_graph_sha256=sha256_file(normalized_path),
        )
        create_data_prep_handoff(
            self.root / config_relative / "specimen_manifest.json",
            self.root / config_relative / "ingest_receipt.json",
            repository_root=self.root,
            output_path=self.root / config_relative / "data_prep_handoff.json",
            schema_path=self.root / "analysis/schema/specimen_manifest.schema.json",
        )
        return [self._output_record(0, role) for role in roles]

    def _production_stage3_outputs(
        self, roles: list[str]
    ) -> list[dict[str, Any]]:
        """Create label-free classifier outputs with a bound verifier report."""

        contract = self.contracts[3]
        records = [
            self._output_record(3, role)
            for role in roles
            if role != "classifier_verifier_report"
        ]
        by_role = {str(record["role"]): record for record in records}
        manifest = self.manifest()
        attempt = manifest["stages"]["3"]["attempts"][-1]
        metrics_hash = next(
            item["sha256"]
            for item in attempt["input_artifacts"]
            if item["role"] == "per_strut_metrics"
        )
        evidence = sorted(
            (
                {"path": item["path"], "sha256": item["sha256"]}
                for item in records
                if item["role"] == "evidence_packets"
            ),
            key=lambda item: (item["path"], item["sha256"]),
        )
        verifier = {
            "schema_version": "classifier-verifier-report/1.0.0",
            "owner": "classifier_verifier",
            "gate": "pass",
            "specimen_id": manifest["specimen_id"],
            "stage_number": 3,
            "attempt": attempt["attempt"],
            "run_token": attempt["run_token"],
            "config_sha256": manifest["config"]["sha256"],
            "contract_sha256": manifest["stages"]["3"]["contract"]["sha256"],
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
                "decision_log_sha256": by_role[
                    "classification_decision_log"
                ]["sha256"],
                "evidence_set_sha256": canonical_json_sha256(evidence),
                "per_strut_metrics_sha256": metrics_hash,
                "specialist_findings_sha256": {
                    role: by_role[role]["sha256"]
                    for role in (
                        "findings_missing",
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
        rule = self._rule_for(contract, "output", "classifier_verifier_report")
        records.append(
            self._record_for_rule(
                rule,
                "classifier_verifier_report",
                payload=(json.dumps(verifier, sort_keys=True) + "\n").encode(),
                overwrite=True,
            )
        )
        return records

    def _stage0_outputs(self, roles: list[str]) -> list[dict[str, Any]]:
        """Materialize a valid tiny Stage 0 chain using fixture facts only."""

        attempt = self.manifest(verify_artifacts=False)["stages"]["0"][
            "attempts"
        ][-1]
        sources = {
            str(record["role"]): record for record in attempt["input_artifacts"]
        }
        config_relative = Path("analysis") / self.specimen_id / "config"
        metadata_relative = config_relative / "ct_metadata_response.json"
        call_relative = config_relative / "ct_metadata_mcp_call_receipt.json"
        manifest_relative = config_relative / "specimen_manifest.json"
        request_relative = config_relative / "ingest_request.json"
        receipt_relative = config_relative / "ingest_receipt.json"
        handoff_relative = config_relative / "data_prep_handoff.json"
        ct_record = sources["ct_volume"]
        graph_record = sources["nominal_graph"]
        cad_record = sources["cad_stl"]
        declaration_record = sources["design_transform_declaration"]
        aligned_record = sources.get("challenge_aligned_graph")
        ct_path = self.root / str(ct_record["path"])
        spacing = {
            axis: {
                "value": "unknown",
                "unit": "unknown",
                "provenance": {
                    "source": "unknown",
                    "field": "unknown",
                    "raw_value": "unknown",
                },
            }
            for axis in ("z", "y", "x")
        }
        request_binding = {
            "input_filepath": ct_record["path"],
            "output_filepath": metadata_relative.as_posix(),
            "call_receipt_filepath": call_relative.as_posix(),
            "header_only": True,
            "include_sha256": True,
            "retention": "committed",
        }
        header_facts = {
            "file_bytes": ct_path.stat().st_size,
            "format": "npy",
            "shape": [2, 3, 4],
            "ndim": 3,
            "dtype": "uint8",
            "dtype_string": "|u1",
            "byte_order": "not_applicable",
            "axes": "unknown",
            "voxel_count": 24,
            "array_bytes": 24,
        }
        metadata_result = {
            "status": "ok",
            "authoritative": True,
            "inspection_mode": "header_only",
            "method": "volume_metadata",
            "method_version": "1.0.0",
            "output_schema_version": "volume-metadata/1.0.0",
            "path": ct_record["path"],
            "sha256": ct_record["sha256"],
            **header_facts,
            "voxel_spacing": spacing,
            "statistics": {
                "status": "not_computed",
                "minimum": "unknown",
                "maximum": "unknown",
                "mean": "unknown",
                "finite_count": "unknown",
                "nonfinite_count": "unknown",
            },
            "manifest_fragment": {
                "ct_volume": {
                    "path": ct_record["path"],
                    "sha256": ct_record["sha256"],
                    "role": "ct_volume",
                    "retention": "committed",
                },
                "ct_metadata": {
                    "format": "npy",
                    "shape": [2, 3, 4],
                    "dtype": "uint8",
                    "byte_order": "not_applicable",
                    "array_axes": "unknown",
                    "voxel_spacing": spacing,
                },
            },
        }
        summary = "Persisted authoritative header-only CT metadata response"
        metadata_evidence = {
            "schema_version": "volume-metadata-mcp-evidence/1.1.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": "pass",
            "summary": summary,
            "request": request_binding,
            "result": metadata_result,
            "warnings": [],
            "error": None,
        }
        self._write_json(metadata_relative, metadata_evidence)
        metadata_sha256 = sha256_file(self.root / metadata_relative)
        call_base = {
            "schema_version": "volume-metadata-mcp-call-receipt/1.0.0",
            "response_schema_version": "part2-mcp-response/1.0.0",
            "tool": "inspect_volume_metadata",
            "status": "ok",
            "gate": "pass",
            "summary": summary,
            "request": request_binding,
            "header_facts": header_facts,
            "artifacts": {
                "metadata_response": {
                    "path": metadata_relative.as_posix(),
                    "sha256": metadata_sha256,
                    "role": "ct_metadata_mcp_response",
                    "retention": "committed",
                }
            },
            "hashes": {
                "input_sha256": ct_record["sha256"],
                "request_sha256": canonical_json_sha256(request_binding),
                "result_sha256": canonical_json_sha256(metadata_result),
                "header_facts_sha256": canonical_json_sha256(header_facts),
                "metadata_response_sha256": metadata_sha256,
            },
            "warnings": [],
            "error": None,
        }
        call_receipt = {
            **call_base,
            "canonical_call_receipt_sha256": canonical_json_sha256(call_base),
        }
        self._write_json(call_relative, call_receipt)
        call_sha256 = sha256_file(self.root / call_relative)

        parameters = copy.deepcopy(
            json.loads(
                (
                    REPOSITORY_ROOT
                    / "analysis/brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json"
                ).read_text(encoding="utf-8")
            )["analysis_parameters"]
        )
        parameters["requested_analysis_scope"] = self.requested_analysis_scope
        parameters["registration"]["mode"] = self.registration_mode
        parameters["coordinates"] = {
            "graph_axes": ["x", "y", "z"],
            "array_axes": ["z", "y", "x"],
            "numpy_index_expression": "volume[round(z), round(y), round(x)]",
            "aligned_graph_units": (
                "voxel"
                if False
                else "simulation_voxel"
            ),
        }
        parameters["segmentation"][
            "histogram_encoding"
        ] = "full_volume_affine_uint16"
        parameters_sha256 = canonical_json_sha256(parameters)

        def manifest_artifact(
            source: Mapping[str, Any], role: str
        ) -> dict[str, str]:
            return {
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
                "role": role,
                "retention": "committed",
            }

        topology = {
            "junction_ids": [0, 1],
            "struts": [[10, 0, 1]],
            "unit_cells": [[20, [10]]],
        }
        graph_inspection = {
            "method": "canonical_lattice_topology",
            "method_version": "1.0.0",
            "path": graph_record["path"],
            "sha256": graph_record["sha256"],
            "junction_count": 2,
            "strut_count": 1,
            "unit_cell_count": 1,
            "topology_sha256": canonical_json_sha256(topology),
            "id_reference_integrity": True,
            "extra_top_level_keys": [],
        }
        cad_inspection = {
            "method": "trimesh_ascii_stl",
            "method_version": "1.0.0",
            "path": cad_record["path"],
            "sha256": cad_record["sha256"],
            "format": "stl",
            "vertex_count": 3,
            "face_count": 1,
            "bounds": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [1.0, 1.0, 0.0],
            },
            "units": "millimeter",
            "units_provenance": "synthetic fixture declaration",
            "readable": True,
        }
        manifest_inputs: dict[str, Any] = {
            "ct": manifest_artifact(ct_record, "ct_volume"),
            "ct_metadata": {
                "format": "npy",
                "shape": [2, 3, 4],
                "dtype": "uint8",
                "byte_order": "not_applicable",
                "array_axes": ["z", "y", "x"],
                "voxel_spacing": spacing,
            },
            "design_graph": manifest_artifact(graph_record, "design_graph"),
            "design_transform_declaration": manifest_artifact(
                declaration_record, "design_transform_declaration"
            ),
            "cad": manifest_artifact(cad_record, "cad"),
        }
        graph_input_hashes = [graph_record["sha256"]]
        if aligned_record is not None:
            manifest_inputs["aligned_graph"] = manifest_artifact(
                aligned_record, "aligned_graph"
            )
            graph_input_hashes = sorted(
                set(graph_input_hashes) | {aligned_record["sha256"]}
            )
        graph_values = {
            key: graph_inspection[key]
            for key in (
                "junction_count",
                "strut_count",
                "unit_cell_count",
                "topology_sha256",
            )
        }
        graph_summary: dict[str, Any] = {
            "method": graph_inspection["method"],
            "method_version": graph_inspection["method_version"],
            "provenance": {
                "source": "scientist-supplied graph schema and reference inspection",
                "input_sha256": graph_input_hashes,
                "config_sha256": parameters_sha256,
            },
            "values": graph_values,
        }
        if aligned_record is not None:
            graph_summary["aligned_values"] = graph_values
        specimen_manifest = {
            "schema_version": "2.1.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "lifecycle_state": "ready_for_data_prep",
            "unresolved_fields": [],
            "inputs": manifest_inputs,
            "intake": {
                "association": {
                    "source": "scientist_explicit",
                    "confirmed": True,
                    "design_graph_to_cad": True,
                    "ct_to_specimen": True,
                },
                "registration_mode_selection": {
                    "mode": self.registration_mode,
                    "source": "scientist_explicit",
                },
                "cad_inspection": cad_inspection,
                "graph_inspection": graph_inspection,
                "volume_metadata": {
                    "method": "volume_metadata",
                    "method_version": "1.0.0",
                    "output_schema_version": "volume-metadata/1.0.0",
                },
            },
            "analysis_parameters": parameters,
            "analysis_parameters_sha256": parameters_sha256,
            "derived": {"graph_summary": graph_summary},
        }
        self._write_json(manifest_relative, specimen_manifest)
        manifest_sha256 = canonical_json_sha256(specimen_manifest)

        declaration_document = json.loads(
            (self.root / str(declaration_record["path"])).read_text(
                encoding="utf-8"
            )
        )
        stage1_policy_path = (
            self.root / "analysis/contracts/removed_research_policy.json"
        )
        stage1_policy = json.loads(stage1_policy_path.read_text(encoding="utf-8"))
        declaration_verification = {
            "schema_valid": True,
            "semantic_validation_pass": True,
            "artifact_sha256": declaration_record["sha256"],
            "canonical_declaration_sha256": declaration_document[
                "canonical_declaration_sha256"
            ],
            "declaration_id": declaration_document["declaration_id"],
            "source_id": declaration_document["source_id"],
            "provenance_id": declaration_document["provenance_id"],
            "stage1_policy_artifact_sha256": sha256_file(stage1_policy_path),
            "stage1_policy_id": stage1_policy["policy_id"],
            "reflection_authorized_by_policy": stage1_policy[
                "orientation_verification"
            ]["reflection_authorized"],
        }

        ingest_request = {
            "schema_version": "ingest-request/1.3.0",
            "method": "specimen_ingest",
            "method_version": "1.3.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "requested_analysis_scope": self.requested_analysis_scope,
            "paths": {
                "cad": cad_record["path"],
                "design_graph": graph_record["path"],
                "ct": ct_record["path"],
                "ct_metadata_response": metadata_relative.as_posix(),
                "ct_metadata_mcp_call_receipt": call_relative.as_posix(),
                "aligned_graph": (
                    aligned_record["path"] if aligned_record is not None else None
                ),
                "design_transform_declaration": declaration_record["path"],
            },
            "registration_mode": self.registration_mode,
            "association_confirmed": True,
            "mcp_response_binding": {
                "tool": "inspect_volume_metadata",
                "response_schema_version": "part2-mcp-response/1.0.0",
                "artifact_path": metadata_relative.as_posix(),
                "artifact_sha256": metadata_sha256,
                "call_receipt_path": call_relative.as_posix(),
                "call_receipt_sha256": call_sha256,
                "canonical_call_receipt_sha256": call_receipt[
                    "canonical_call_receipt_sha256"
                ],
                "ct_sha256": ct_record["sha256"],
                "authoritative": True,
                "header_only": True,
            },
            "declared": {
                "cad_units": "millimeter",
                "cad_units_provenance": "synthetic fixture declaration",
                "graph_axes": ["x", "y", "z"],
                "array_axes": ["z", "y", "x"],
                "aligned_graph_units": parameters["coordinates"][
                    "aligned_graph_units"
                ],
                "retention": "committed",
                "design_transform_declaration_verification": (
                    declaration_verification
                ),
            },
        }
        self._write_json(request_relative, ingest_request)
        input_sha256 = {
            "cad": cad_record["sha256"],
            "design_graph": graph_record["sha256"],
            "design_transform_declaration": declaration_record["sha256"],
            "ct": ct_record["sha256"],
            "ct_metadata_response": metadata_sha256,
            "ct_metadata_mcp_call_receipt": call_sha256,
        }
        if aligned_record is not None:
            input_sha256["aligned_graph"] = aligned_record["sha256"]
        receipt_base = {
            "schema_version": "ingest-receipt/1.3.0",
            "method": "specimen_ingest",
            "method_version": "1.3.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "requested_analysis_scope": self.requested_analysis_scope,
            "lifecycle_state": "ready_for_data_prep",
            "input_sha256": input_sha256,
            "request_sha256": canonical_json_sha256(ingest_request),
            "manifest_sha256": manifest_sha256,
            "design_transform_declaration_verification": declaration_verification,
            "warnings": ["CT voxel spacing is unavailable from file metadata"],
            "unresolved_fields": [],
            "self_verification": {
                "association_explicit": True,
                "all_paths_repository_relative": True,
                "all_inputs_hashed": True,
                "ct_metadata_mcp_integrity_chain_valid": True,
                "ct_metadata_response_schema_closed": True,
                "ct_metadata_response_hash_bound": True,
                "ct_metadata_response_header_only": True,
                "ct_metadata_response_path_and_ct_hash_match": True,
                "ct_metadata_mcp_call_receipt_closed": True,
                "ct_metadata_mcp_call_receipt_hash_bound": True,
                "ct_metadata_header_facts_bound_to_call_receipt": True,
                "cad_readable": True,
                "graph_id_reference_integrity": True,
                "manifest_schema_valid": True,
                "design_transform_declaration_valid": True,
                "design_transform_declaration_hash_bound": True,
                "segmentation_not_run": True,
                "registration_not_run": True,
                "defect_labels_not_derived": True,
            },
        }
        ingest_receipt = {
            **receipt_base,
            "canonical_receipt_sha256": canonical_json_sha256(receipt_base),
        }
        self._write_json(receipt_relative, ingest_receipt)

        allowlisted_inputs = {
            name: {
                "path": artifact["path"],
                "sha256": artifact["sha256"],
                "role": artifact["role"],
            }
            for name, artifact in manifest_inputs.items()
            if name != "ct_metadata"
        }
        handoff_base = {
            "schema_version": "data-prep-handoff/1.1.0",
            "specimen_id": self.specimen_id,
            "design_id": self.design_id,
            "requested_analysis_scope": self.requested_analysis_scope,
            "status": "ready",
            "action": "run_data_prep",
            "lifecycle_state": "ready_for_data_prep",
            "manifest_path": manifest_relative.as_posix(),
            "manifest_sha256": manifest_sha256,
            "ingest_receipt_sha256": ingest_receipt[
                "canonical_receipt_sha256"
            ],
            "analysis_parameters_sha256": parameters_sha256,
            "registration_mode": self.registration_mode,
            "localization_policy": parameters["localization_policy"],
            "qa_policy": parameters["qa_policy"],
            "authorized_outputs": [
                "segmentation",
                "registration",
                "node_localization",
                "coarse_region_screening",
                "padded_roi_definition",
            ],
            "unauthorized_outputs": [
                "absolute_metrology",
                "direct_dimensional_measurement",
            ],
            "allowlisted_inputs": allowlisted_inputs,
            "unresolved_fields": [],
            "forbidden_inputs": [
                "defect labels",
                "dev split",
                "sealed split",
                "ground-truth segmentation",
            ],
            "required_outputs": [
                "aligned graph",
                "exact-histogram Otsu result",
                "canonical uint8 ZYX mask contract",
                "bounded segmentation-mask comparison",
                "registration QA",
                "local node recentering",
                "ROI capture gate",
                "metrology gate",
                "data-prep completion receipt",
            ],
            "maximum_agent_retries": parameters["budgets"][
                "maximum_agent_retries"
            ],
        }
        self._write_json(
            handoff_relative,
            {
                **handoff_base,
                "canonical_handoff_sha256": canonical_json_sha256(handoff_base),
            },
        )
        return [self._output_record(0, role) for role in roles]

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
            return self._stage2_outputs(contract, roles, inputs, manifest)
        raise RuntimeError("Only Stage 1 uses bound data-preparation fixtures")

    def _json_output(
        self,
        contract: dict[str, object],
        role: str,
        document: dict[str, object],
        *,
        replaces_sha256: str | None = None,
    ) -> dict[str, object]:
        rule = self._rule_for(contract, "output", role)
        return self._record_for_rule(
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
        by_role["normalized_nominal_graph"] = self._output_record(
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
        by_role["label_report"] = self._record_for_rule(
            label_report_rule,
            "label_report",
            payload=b"# Removed research-only label fixture\n\nGate: `halt`\n",
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
        def artifact(source: dict[str, object], role: str) -> dict[str, object]:
            return {
                "path": source["path"],
                "sha256": source["sha256"],
                "role": role,
                "retention": "committed",
            }

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
                    "normalized_nominal_graph": artifact(
                        inputs["normalized_nominal_graph"],
                        "normalized_nominal_graph",
                    ),
                    "aligned_graph": aligned,
                    "canonical_mask": canonical_mask,
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
            by_role[role] = self._output_record(1, role)
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
        attempt = manifest["stages"]["1"]["attempts"][-1]
        prior = next(
            item
            for item in attempt["input_artifacts"]
            if item["role"] == "specimen_manifest"
        )
        rule = self._rule_for(
            self.contracts[1], "output", "analysis_ready_specimen_manifest"
        )
        inputs = {item["role"]: item for item in attempt["input_artifacts"]}
        by_role: dict[str, dict[str, object]] = {}
        for role in (
            "exact_histogram",
            "canonical_segmentation_mask",
            "localized_graph",
        ):
            output_rule = self._rule_for(self.contracts[1], "output", role)
            path = self._materialize_path(str(output_rule["path"]), role)
            by_role[role] = artifact_record(self.root, path, role=role)
        replacement = self._analysis_ready_document(inputs=inputs, by_role=by_role)
        return self._record_for_rule(
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
        if stage_number != 1:
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
                else ["STAGE1_REVIEW_OR_HALT"]
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
                            if False
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

            assignments = stage2_assignments

            def output_document(role: str) -> dict[str, object]:
                return json.loads(
                    (self.root / str(output_records[role]["path"])).read_text(
                        encoding="utf-8"
                    )
                )

            def response_result_for(tool_name: str) -> dict[str, object]:
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
                if stage_number == 1 and tool_name == "replay_exact_otsu":
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
                    stage_number == 1
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
                    stage_number == 1
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
        if stage_number == 1 and self.registration_mode == "autonomous_v2":
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
        if stage_number == 1:
            outputs = [*outputs, self.analysis_ready_manifest_output()]
        stage_policy = self.stage_policy(stage_number, outputs)
        receipt = build_stage_receipt(
            self.manifest_path,
            stage_number,
            terminal_state="pass",
            output_artifacts=outputs,
            assertions=self.assertions(stage_number),
            stage_policy=stage_policy,
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
            stage_policy=self.stage_policy(
                stage_number,
                [evidence],
                terminal_state="manual_review",
            ),
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
