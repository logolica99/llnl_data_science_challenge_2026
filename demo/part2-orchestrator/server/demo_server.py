#!/usr/bin/env python3
"""Localhost API for the Part 2 orchestration demonstrator.

The API exposes a redacted projection of the real control-plane state.  Every
run lives in an isolated temporary repository and uses fixture specialist
outputs; browser requests never receive filesystem authority or raw label
handoffs.
"""

from __future__ import annotations

import atexit
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any
from urllib.parse import urlparse
import uuid

from demo_pipeline import SyntheticFixtureStageRunner

from part2_orchestration import (  # type: ignore  # imported via demo_pipeline
    OrchestrationError,
    complete_stage,
    pipeline_status,
)


HOST = "127.0.0.1"
PORT = 8765
MAX_BODY_BYTES = 4096
MAX_ACTIVE_RUNS = 32
SESSION_MARKER = ".part2-demo-session"
SESSION_MARKER_SCHEMA = "part2-demo-session/1.0.0"
REGISTRATION_MODES = {"autonomous_v2"}
SCENARIOS = {
    "verified_walkthrough",
    "manual_review",
    "tampered_receipt",
    "missing_dependency",
}
ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
}

STAGE_COPY = (
    {
        "title": "Confirm specimen",
        "shortTitle": "Intake",
        "description": "Bind the scientist-confirmed specimen to its source files and provenance.",
        "activity": "Checking the specimen association, metadata receipt, and immutable intake handoff.",
    },
    {
        "title": "Register & validate",
        "shortTitle": "Data prep",
        "description": "Run exact Otsu, autonomous graph-to-CT registration, localization, and registration QA.",
        "activity": "Verifying the frozen autonomous registration evidence and analysis-ready manifest.",
    },
    {
        "title": "Measure each strut",
        "shortTitle": "ROI metrics",
        "description": "Produce blind, per-strut ROI measurements without access to defect labels.",
        "activity": "Checking that only registered geometry and CT-derived measurements enter this stage.",
    },
    {
        "title": "Classify & verify",
        "shortTitle": "Classification",
        "description": "Merge missing, broken, and thin findings without labels, then require an independent classifier verifier.",
        "activity": "Binding label-free specialist findings and independent verification.",
    },
    {
        "title": "Assemble NDE report",
        "shortTitle": "Report",
        "description": "Package spatial statistics, 3D evidence, number crosscheck, and a recompute-free report.",
        "activity": "Verifying committed presentation artifacts and cited values without recomputing science.",
    },
)


class StaleStateError(RuntimeError):
    """Raised when a browser tries to mutate an older manifest revision."""


class DemoRun:
    """One thread-safe isolated demo session."""

    def __init__(
        self,
        *,
        run_id: str,
        base_directory: Path,
        scenario: str,
        registration_mode: str,
        emit_terminal_stdout: bool = False,
    ) -> None:
        self.run_id = run_id
        self.scenario = scenario
        self.registration_mode = registration_mode
        self.lock = threading.RLock()
        self.root = (base_directory / run_id).resolve()
        if not self.root.is_relative_to(base_directory.resolve()):
            raise RuntimeError("Demo session escaped the server-owned base directory")
        self.root.mkdir(parents=True)
        (self.root / SESSION_MARKER).write_text(
            json.dumps(self._marker_document(), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.runner = SyntheticFixtureStageRunner(
            self.root / "repository",
            registration_mode,
        )
        self.events: list[dict[str, Any]] = []
        self.terminal_lines: list[dict[str, Any]] = []
        self.emit_terminal_stdout = emit_terminal_stdout
        self.verification_block: dict[str, Any] | None = None
        self.review_triggered = False
        self._add_event(
            source="manifest",
            kind="pipeline_created",
            stage=0,
            tone="info",
            title="Frozen run created",
            detail=(
                "The real control plane hashed the configuration and Stage 0-4 "
                "contracts. Stage 0 is the only unlocked stage."
            ),
        )

    def _marker_document(self) -> dict[str, str]:
        return {
            "schema_version": SESSION_MARKER_SCHEMA,
            "run_id": self.run_id,
        }

    def _add_event(
        self,
        *,
        source: str,
        kind: str,
        stage: int | None,
        tone: str,
        title: str,
        detail: str,
        proof: str | None = None,
    ) -> None:
        sequence = len(self.events) + 1
        event = {
            "sequence": sequence,
            "source": source,
            "kind": kind,
            "stage": stage,
            "tone": tone,
            "title": title,
            "detail": detail,
            "proof": proof,
        }
        self.events.append(event)

        stage_text = "pipeline" if stage is None else f"stage={stage}"
        proof_text = f" sha256={proof[:12]}…" if proof else ""
        rendered = (
            f"[part2-run] {sequence:04d} {source.upper()} {stage_text} {kind}"
            f" :: {title}{proof_text}"
        )
        self.terminal_lines.append(
            {
                "sequence": sequence,
                "source": source,
                "line": rendered,
            }
        )
        if self.emit_terminal_stdout:
            print(rendered, flush=True)

    def _status(self) -> dict[str, Any]:
        return pipeline_status(
            self.runner.manifest_path,
            repository_root=self.runner.root,
        )

    def _require_fresh(self, expected_manifest_sha256: str) -> None:
        actual = self._status()["manifest_sha256"]
        if expected_manifest_sha256 != actual:
            raise StaleStateError(
                "The demo advanced in another request; refresh the verified state."
            )

    def advance(self, expected_manifest_sha256: str) -> dict[str, Any]:
        with self.lock:
            self._require_fresh(expected_manifest_sha256)
            if self.verification_block is not None:
                return self.projection()
            status = self._status()
            if status["pipeline_state"] in {"pass", "halt", "manual_review"}:
                return self.projection()

            stage_number = int(status["current_stage"])
            stage = status["stages"][str(stage_number)]
            if stage["state"] == "ready":
                result = self.runner.start(
                    stage_number,
                    missing_dependency=(
                        self.scenario == "missing_dependency" and stage_number == 0
                    ),
                )
                if result["state"] == "halt":
                    failures = result["error"]["failures"]
                    missing_names = ", ".join(
                        str(item.get("name", item.get("kind", "dependency")))
                        for item in failures
                    )
                    self._add_event(
                        source="manifest",
                        kind="dependency_halt",
                        stage=stage_number,
                        tone="halt",
                        title="Dependency preflight halted safely",
                        detail=(
                            f"Unavailable or incompatible: {missing_names}. No "
                            "fallback was used and every downstream stage stayed locked."
                        ),
                        proof=result["receipt"]["canonical_sha256"],
                    )
                else:
                    self._add_event(
                        source="manifest",
                        kind="stage_started",
                        stage=stage_number,
                        tone="running",
                        title=f"Stage {stage_number} handoff sealed",
                        detail=(
                            "Dependencies, allowlisted inputs, contract version, and "
                            "predecessor receipt were verified before dispatch."
                        ),
                        proof=result["handoff"]["canonical_sha256"],
                    )
                    self._add_event(
                        source="demo_adapter",
                        kind="fixture_dispatched",
                        stage=stage_number,
                        tone="fixture",
                        title="Fixture specialist dispatched",
                        detail=(
                            "The bounded specialist is represented by deterministic "
                            "opaque fixture files; no scientific algorithm is running."
                        ),
                    )
                return self.projection()

            if stage["state"] != "running":
                return self.projection()

            if (
                self.scenario == "manual_review"
                and stage_number == 1
                and not self.review_triggered
            ):
                self.runner.complete_manual_review(stage_number)
                self.review_triggered = True
                manifest = self.runner.manifest()
                receipt_hash = manifest["stages"][str(stage_number)][
                    "completion_receipt"
                ]["canonical_sha256"]
                self._add_event(
                    source="manifest",
                    kind="manual_review",
                    stage=stage_number,
                    tone="review",
                    title="Automation paused for scientist review",
                    detail=(
                        "Attempt 1 evidence and its receipt are preserved. Stage 2 "
                        "remains locked until an explicit hashed resolution is recorded."
                    ),
                    proof=receipt_hash,
                )
                return self.projection()

            if self.scenario == "tampered_receipt" and stage_number == 0:
                receipt_path = self.runner.build_tampered_receipt(stage_number)
                try:
                    complete_stage(
                        self.runner.manifest_path,
                        receipt_path,
                        repository_root=self.runner.root,
                    )
                except OrchestrationError:
                    self.verification_block = {
                        "code": "receipt_integrity_rejected",
                        "stage": stage_number,
                        "message": (
                            "The completion receipt failed integrity verification. "
                            "The artifact-backed manifest remains running, while this "
                            "demo run and every downstream stage remain blocked."
                        ),
                    }
                    self._add_event(
                        source="demo_adapter",
                        kind="receipt_rejected",
                        stage=stage_number,
                        tone="rejected",
                        title="Tampered receipt rejected",
                        detail=(
                            "The receipt self-hash no longer matches. The transition "
                            "was rejected and Stage 1 remains locked."
                        ),
                    )
                return self.projection()

            result = self.runner.complete_pass(stage_number)
            self._add_event(
                source="demo_adapter",
                kind="fixture_outputs",
                stage=stage_number,
                tone="fixture",
                title=f"{result['output_count']} fixture artifacts returned",
                detail=(
                    "The artifacts are representative opaque files. Their paths, roles, "
                    "and SHA-256 values are checked by the production control plane."
                ),
            )
            if result["freeze"] is not None:
                self._add_event(
                    source="manifest",
                    kind="registration_frozen",
                    stage=stage_number,
                    tone="sealed",
                    title="CT-only registration frozen",
                    detail=(
                        "Registration artifacts were hashed before Stage 1 completion."
                    ),
                    proof=result["freeze"]["freeze"]["canonical_sha256"],
                )
            receipt_hash = result["receipt"]["receipt"][
                "canonical_receipt_sha256"
            ]
            completed = result["completed"]
            next_text = (
                "The pipeline is complete."
                if completed.get("next_stage") is None
                else f"Stage {completed['next_stage']} is now the only unlocked stage."
            )
            self._add_event(
                source="manifest",
                kind="stage_completed",
                stage=stage_number,
                tone="pass",
                title=f"Stage {stage_number} receipt verified",
                detail=f"Every declared output hash and receipt binding passed. {next_text}",
                proof=receipt_hash,
            )
            return self.projection()

    def resume(self, expected_manifest_sha256: str) -> dict[str, Any]:
        with self.lock:
            self._require_fresh(expected_manifest_sha256)
            status = self._status()
            if status["pipeline_state"] != "manual_review":
                raise OrchestrationError("This run is not waiting for manual review")
            stage_number = int(status["current_stage"])
            self.runner.resume_review(stage_number)
            self._add_event(
                source="manifest",
                kind="manual_review_resumed",
                stage=stage_number,
                tone="info",
                title="Hashed scientist resolution recorded",
                detail=(
                    "The original evidence remains immutable. The same stage is ready "
                    "for its second and final judgment attempt."
                ),
            )
            return self.projection()

    def projection(self) -> dict[str, Any]:
        with self.lock:
            status = self._status()
            manifest = self.runner.manifest()
            stages: list[dict[str, Any]] = []
            for stage_number in range(5):
                raw = manifest["stages"][str(stage_number)]
                attempt = raw["attempts"][-1] if raw["attempts"] else None
                handoff_hash = None
                output_count = 0
                if attempt is not None:
                    handoff_hash = attempt["handoff"]["canonical_sha256"]
                    output_count = len(attempt.get("output_artifacts", []))
                receipt = raw.get("completion_receipt")
                copy_item = STAGE_COPY[stage_number]
                stages.append(
                    {
                        "number": stage_number,
                        "name": raw["name"],
                        "owner": raw["owner"],
                        "executionKind": raw["execution_kind"],
                        "state": raw["state"],
                        "attemptCount": raw["attempt_count"],
                        "maximumAttempts": raw["maximum_attempts"],
                        "oneShot": raw["one_shot"],
                        "title": copy_item["title"],
                        "shortTitle": copy_item["shortTitle"],
                        "description": copy_item["description"],
                        "activity": copy_item["activity"],
                        "proof": {
                            "handoffSha256": handoff_hash,
                            "receiptSha256": (
                                receipt["canonical_sha256"] if receipt else None
                            ),
                            "contractSha256": raw["contract"]["sha256"],
                            "contractVersion": raw["contract"]["version"],
                            "outputCount": output_count,
                        },
                    }
                )

            pipeline_state = status["pipeline_state"]
            allowed_action = None
            if self.verification_block is None:
                if pipeline_state == "manual_review":
                    allowed_action = "resume"
                elif pipeline_state not in {"pass", "halt"}:
                    allowed_action = "advance"
            active_stage = (
                int(status["current_stage"])
                if status["current_stage"] is not None
                else 4
            )
            return {
                "schemaVersion": "part2-orchestrator-demo/1.0.0",
                "runId": self.run_id,
                "scenario": self.scenario,
                "registrationMode": self.registration_mode,
                "specimenId": status["specimen_id"],
                "pipelineState": pipeline_state,
                "currentStage": status["current_stage"],
                "activeStage": active_stage,
                "manifestSha256": status["manifest_sha256"],
                "configSha256": status["config_sha256"],
                "predecessorReceiptSha256": manifest[
                    "predecessor_receipt_sha256"
                ],
                "allowedAction": allowed_action,
                "verificationState": (
                    "blocked" if self.verification_block is not None else "clear"
                ),
                "verificationBlock": (
                    dict(self.verification_block)
                    if self.verification_block is not None
                    else None
                ),
                "blockedReason": (
                    self.verification_block["message"]
                    if self.verification_block is not None
                    else None
                ),
                "disclosure": {
                    "controlPlane": "production code",
                    "specialists": "deterministic fixtures",
                    "scientificAlgorithmsExecuted": False,
                    "rawLabelMetadataExposed": False,
                },
                "stages": stages,
                "events": list(self.events),
                "terminalLines": list(self.terminal_lines),
            }

    def close(self, base_directory: Path) -> None:
        with self.lock:
            resolved_base = base_directory.resolve()
            expected_root = resolved_base / self.run_id
            marker = self.root / SESSION_MARKER
            if (
                self.root != expected_root
                or self.root.is_symlink()
                or not self.root.is_dir()
                or marker.is_symlink()
                or not marker.is_file()
            ):
                raise RuntimeError("Refusing to remove an unmarked demo directory")
            try:
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError("Refusing to remove an invalid demo directory") from exc
            if marker_value != self._marker_document():
                raise RuntimeError("Refusing to remove a mismatched demo directory")
            shutil.rmtree(self.root)


class DemoRunStore:
    """Own demo sessions and their temporary filesystem roots."""

    def __init__(
        self,
        base_directory: Path | None = None,
        *,
        emit_terminal_stdout: bool = False,
    ) -> None:
        if base_directory is None:
            self.base_directory = Path(
                tempfile.mkdtemp(prefix="llnl-part2-orchestrator-demo-")
            ).resolve()
            self._owns_base = True
        else:
            self.base_directory = base_directory.resolve()
            self.base_directory.mkdir(parents=True, exist_ok=True)
            self._owns_base = False
        self.lock = threading.RLock()
        self.runs: dict[str, DemoRun] = {}
        self.emit_terminal_stdout = emit_terminal_stdout

    def create(self, *, scenario: str, registration_mode: str) -> DemoRun:
        if scenario not in SCENARIOS:
            raise ValueError("Unsupported demo scenario")
        if registration_mode not in REGISTRATION_MODES:
            raise ValueError("Unsupported registration mode")
        with self.lock:
            if len(self.runs) >= MAX_ACTIVE_RUNS:
                raise ValueError("Too many active demo runs; reset an existing run")
            run_id = uuid.uuid4().hex
            run = DemoRun(
                run_id=run_id,
                base_directory=self.base_directory,
                scenario=scenario,
                registration_mode=registration_mode,
                emit_terminal_stdout=self.emit_terminal_stdout,
            )
            self.runs[run_id] = run
            return run

    def get(self, run_id: str) -> DemoRun:
        if len(run_id) != 32 or any(character not in "0123456789abcdef" for character in run_id):
            raise KeyError(run_id)
        with self.lock:
            return self.runs[run_id]

    def delete(self, run_id: str) -> None:
        with self.lock:
            run = self.runs[run_id]
            run.close(self.base_directory)
            del self.runs[run_id]

    def close(self) -> None:
        with self.lock:
            run_ids = list(self.runs)
        for run_id in run_ids:
            try:
                self.delete(run_id)
            except (KeyError, OSError, RuntimeError):
                pass
        if self._owns_base and self.base_directory.is_dir():
            try:
                self.base_directory.rmdir()
            except OSError:
                pass


STORE = DemoRunStore(emit_terminal_stdout=True)
atexit.register(STORE.close)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server_version = "Part2OrchestratorDemo/1.0"

    def log_message(self, format_string: str, *args: object) -> None:
        print(
            f"[demo-api] {self.address_string()} {format_string % args}",
            flush=True,
        )

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in ALLOWED_ORIGINS

    def _headers(self, status: HTTPStatus, *, origin: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()

    def _send(self, status: HTTPStatus, value: object) -> None:
        origin = self.headers.get("Origin")
        payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        value = json.loads(raw or b"{}")
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object")
        return value

    @staticmethod
    def _route(path: str) -> tuple[str | None, str | None]:
        parts = [part for part in path.split("/") if part]
        if parts[:3] != ["api", "v1", "demo-runs"]:
            return None, None
        if len(parts) == 3:
            return "collection", None
        if len(parts) == 4:
            return "run", parts[3]
        if len(parts) == 5 and parts[4] in {"steps", "resume"}:
            return parts[4], parts[3]
        return None, None

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "origin_not_allowed"})
            return
        origin = self.headers.get("Origin")
        self.send_response(HTTPStatus.NO_CONTENT)
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "origin_not_allowed"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/health":
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "bind": "localhost-only",
                    "scientificAlgorithmsExecuted": False,
                },
            )
            return
        route, run_id = self._route(parsed.path)
        if route != "run" or run_id is None:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            self._send(HTTPStatus.OK, STORE.get(run_id).projection())
        except KeyError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown_demo_run"})
        except OrchestrationError:
            self._send(
                HTTPStatus.CONFLICT,
                {
                    "error": "verification_failed",
                    "message": "Verified demo state is unavailable; start a fresh run.",
                },
            )

    def do_POST(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "origin_not_allowed"})
            return
        route, run_id = self._route(urlparse(self.path).path)
        try:
            body = self._body()
            if route == "collection":
                if set(body) - {"scenario", "registrationMode"}:
                    raise ValueError("Unexpected create-run field")
                run = STORE.create(
                    scenario=str(body.get("scenario", "verified_walkthrough")),
                    registration_mode=str(
                        body.get("registrationMode", "autonomous_v2")
                    ),
                )
                self._send(HTTPStatus.CREATED, run.projection())
                return
            if route not in {"steps", "resume"} or run_id is None:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            if set(body) != {"expectedManifestSha256"}:
                raise ValueError("Mutation requires only expectedManifestSha256")
            expected = body["expectedManifestSha256"]
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
            ):
                raise ValueError("expectedManifestSha256 must be a SHA-256 string")
            run = STORE.get(run_id)
            projection = run.advance(expected) if route == "steps" else run.resume(expected)
            self._send(HTTPStatus.OK, projection)
        except KeyError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown_demo_run"})
        except StaleStateError as exc:
            self._send(
                HTTPStatus.CONFLICT,
                {"error": "stale_manifest", "message": str(exc)},
            )
        except OrchestrationError:
            self._send(
                HTTPStatus.CONFLICT,
                {
                    "error": "orchestration_rejected",
                    "message": (
                        "The production control plane rejected this action and "
                        "kept downstream stages locked."
                    ),
                },
            )
        except ValueError as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "message": str(exc)},
            )

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._origin_allowed():
            self._send(HTTPStatus.FORBIDDEN, {"error": "origin_not_allowed"})
            return
        route, run_id = self._route(urlparse(self.path).path)
        if route != "run" or run_id is None:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            STORE.delete(run_id)
            self._send(HTTPStatus.OK, {"deleted": True})
        except KeyError:
            self._send(HTTPStatus.NOT_FOUND, {"error": "unknown_demo_run"})
        except (OSError, RuntimeError):
            self._send(
                HTTPStatus.CONFLICT,
                {
                    "error": "cleanup_rejected",
                    "message": "The demo run could not be removed safely.",
                },
            )


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), DemoRequestHandler)
    print(f"Part 2 demo API listening on http://{HOST}:{PORT}", flush=True)
    print(
        "Real control plane; deterministic fixture specialists; no CT algorithms.",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        STORE.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
