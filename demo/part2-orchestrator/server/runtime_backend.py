#!/usr/bin/env python3
"""Localhost bridge from the proof page to a real Codex app-server thread."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import tomllib
from typing import Any, Callable
import uuid


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPOSITORY_ROOT / "demo" / "part2-orchestrator"
SOURCE_CONFIG = (
    REPOSITORY_ROOT
    / "analysis"
    / "brian_tran_9x9x9_0point5dash1"
    / "config"
    / "specimen_manifest.json"
)
ORCHESTRATOR_CONFIG = REPOSITORY_ROOT / ".codex" / "agents" / "orchestrator.toml"
HTML_PATH = PROJECT_ROOT / "runtime.html"
HOST = "127.0.0.1"
PORT = 3000
MAX_BODY_BYTES = 4096
TURN_TIMEOUT_SECONDS = 30 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compact(value: object, limit: int = 900) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    rendered = " ".join(rendered.split())
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


class CodexProtocolError(RuntimeError):
    """Raised when app-server closes or rejects a protocol request."""


class CodexAppServer:
    """Small JSONL client for one local Codex app-server process."""

    def __init__(self, on_notification: Callable[[str, dict[str, Any]], None]):
        self.on_notification = on_notification
        self.process: subprocess.Popen[str] | None = None
        self.pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self.pending_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.turn_completed = threading.Event()
        self.turn_result: dict[str, Any] | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            ["codex", "app-server", "--listen", "stdio://"],
            cwd=REPOSITORY_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "llnl-part2-runtime-proof",
                    "title": "LLNL Part 2 Runtime Proof",
                    "version": "1.0.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                    "requestAttestation": False,
                    "optOutNotificationMethods": [
                        "command/exec/outputDelta",
                        "item/agentMessage/delta",
                        "item/plan/delta",
                        "item/fileChange/outputDelta",
                        "item/reasoning/summaryTextDelta",
                        "item/reasoning/textDelta",
                    ],
                },
            },
            timeout=30,
        )
        self.notify("initialized", {})

    def request(
        self, method: str, params: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        result_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self.pending_lock:
            self.pending[request_id] = result_queue
        self._write({"id": request_id, "method": method, "params": params})
        try:
            message = result_queue.get(timeout=timeout)
        except queue.Empty as error:
            with self.pending_lock:
                self.pending.pop(request_id, None)
            raise CodexProtocolError(f"Timed out waiting for {method}") from error
        if "error" in message:
            raise CodexProtocolError(f"{method} failed: {compact(message['error'])}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise CodexProtocolError(f"{method} returned a non-object result")
        return result

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    def prepare_turn(self) -> None:
        """Reset completion tracking before dispatching another turn."""

        self.turn_result = None
        self.turn_completed.clear()

    def _write(self, message: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise CodexProtocolError("Codex app-server is not running")
        line = json.dumps(message, separators=(",", ":")) + "\n"
        with self.write_lock:
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise CodexProtocolError("Codex app-server input closed") from error

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(f"[codex-app-server:stdout] {line.rstrip()}", flush=True)
                continue
            request_id = message.get("id")
            if request_id is not None and ("result" in message or "error" in message):
                with self.pending_lock:
                    destination = self.pending.pop(str(request_id), None)
                if destination is not None:
                    destination.put(message)
                continue
            method = message.get("method")
            params = message.get("params", {})
            if isinstance(method, str) and isinstance(params, dict):
                if request_id is not None:
                    self.on_notification(
                        "client/request",
                        {"method": method, "disposition": "rejected_fail_closed"},
                    )
                    self._write(
                        {
                            "id": request_id,
                            "error": {
                                "code": -32601,
                                "message": "This proof backend does not grant interactive approvals",
                            },
                        }
                    )
                else:
                    self.on_notification(method, params)
                    if method == "turn/completed":
                        self.turn_result = params
                        self.turn_completed.set()
        self.turn_completed.set()

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            print(f"[codex-app-server] {line.rstrip()}", flush=True)

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        self.process = None


class RuntimeRun:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.state = "idle"
        self.run_id: str | None = None
        self.thread_id: str | None = None
        self.turn_id: str | None = None
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.events: list[dict[str, Any]] = []
        self.terminal: list[str] = []
        self.final_message: str | None = None
        self.error: str | None = None
        self.manifest: dict[str, Any] | None = None
        self.client: CodexAppServer | None = None
        self.evidence_directory: Path | None = None

    def begin(self) -> None:
        with self.lock:
            if self.state == "running":
                raise RuntimeError("A Codex run is already active")
            suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()
            self.run_id = f"brian_tran_9x9x9_runtime_{suffix}"
            self.thread_id = None
            self.turn_id = None
            self.started_at = utc_now()
            self.completed_at = None
            self.events = []
            self.terminal = []
            self.final_message = None
            self.error = None
            self.manifest = None
            self.evidence_directory = (
                PROJECT_ROOT / "runtime-evidence" / self.run_id
            )
            self.state = "running"
            self._emit("backend", "run_started", "Scientist confirmation accepted")
        threading.Thread(target=self._execute, daemon=True).start()

    def _emit(self, source: str, kind: str, detail: str) -> None:
        with self.lock:
            sequence = len(self.events) + 1
            event = {
                "sequence": sequence,
                "timestamp": utc_now(),
                "source": source,
                "kind": kind,
                "detail": detail,
            }
            self.events.append(event)
            line = f"[runtime-proof] {sequence:04d} {source} {kind} :: {detail}"
            self.terminal.append(line)
            print(line, flush=True)
            self._flush_evidence()

    def _flush_evidence(self) -> None:
        if self.evidence_directory is None:
            return
        self.evidence_directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self.evidence_directory / "events.json", self.events)
        terminal_path = self.evidence_directory / "terminal.log"
        temporary = terminal_path.with_name(".terminal.log.tmp")
        temporary.write_text("\n".join(self.terminal) + "\n", encoding="utf-8")
        temporary.replace(terminal_path)

    def _notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "mcpServer/startupStatus/updated":
            name = str(params.get("name", "unknown"))
            status = str(params.get("status", "unknown"))
            if name == "segmentation-tools" or status == "failed":
                self._emit("codex", "mcp_status", f"{name}={status}")
            return
        if method == "thread/status/changed":
            status = params.get("status", {})
            self._emit("codex", "thread_status", compact(status, 250))
            return
        if method in {"turn/started", "turn/completed"}:
            turn = params.get("turn", {})
            self._emit("codex", method.replace("/", "_"), compact(turn, 350))
            return
        if method in {"item/started", "item/completed"}:
            item = params.get("item", {})
            if not isinstance(item, dict):
                return
            item_type = str(item.get("type", "unknown"))
            detail = self._item_detail(item)
            self._emit("codex", f"{method.split('/')[1]}_{item_type}", detail)
            if method == "item/completed" and item_type == "agentMessage":
                text = item.get("text")
                if isinstance(text, str) and text:
                    with self.lock:
                        self.final_message = text
            return
        if method == "error":
            self._emit("codex", "runtime_error", compact(params.get("error"), 700))
            return
        if method == "client/request":
            self._emit("codex", "approval_rejected", compact(params, 350))

    @staticmethod
    def _item_detail(item: dict[str, Any]) -> str:
        item_type = item.get("type")
        if item_type == "agentMessage":
            return compact(item.get("text", "agent response"), 1200)
        if item_type == "commandExecution":
            return compact(
                {
                    "command": item.get("command"),
                    "status": item.get("status"),
                    "exitCode": item.get("exitCode"),
                },
                900,
            )
        if item_type == "mcpToolCall":
            return compact(
                {
                    "server": item.get("server"),
                    "tool": item.get("tool"),
                    "status": item.get("status"),
                    "error": item.get("error"),
                },
                700,
            )
        if item_type == "collabAgentToolCall":
            return compact(
                {
                    "tool": item.get("tool"),
                    "status": item.get("status"),
                    "receiverThreadIds": item.get("receiverThreadIds"),
                },
                700,
            )
        if item_type == "subAgentActivity":
            return compact(
                {
                    "agentPath": item.get("agentPath"),
                    "agentThreadId": item.get("agentThreadId"),
                    "kind": item.get("kind"),
                },
                700,
            )
        if item_type == "fileChange":
            return compact({"status": item.get("status"), "changes": item.get("changes")}, 700)
        return compact({key: item.get(key) for key in ("type", "id", "status")}, 400)

    def _execute(self) -> None:
        try:
            request = self._write_confirmed_request()
            orchestrator = tomllib.loads(
                ORCHESTRATOR_CONFIG.read_text(encoding="utf-8")
            )
            developer_instructions = str(orchestrator["developer_instructions"])
            client = CodexAppServer(self._notification)
            with self.lock:
                self.client = client
            self._emit("backend", "runtime_starting", "Starting codex app-server over stdio")
            client.start()
            self._emit("backend", "runtime_ready", "Codex app-server initialized")
            thread_result = client.request(
                "thread/start",
                {
                    "model": orchestrator.get("model"),
                    "cwd": str(REPOSITORY_ROOT),
                    "runtimeWorkspaceRoots": [str(REPOSITORY_ROOT)],
                    "approvalPolicy": "never",
                    "sandbox": "workspace-write",
                    "developerInstructions": developer_instructions,
                    "ephemeral": False,
                    "config": {"features": {"multi_agent": True}},
                },
                timeout=60,
            )
            thread = thread_result["thread"]
            with self.lock:
                self.thread_id = str(thread["id"])
            self._emit("backend", "thread_created", f"thread_id={self.thread_id}")
            self._run_turn(
                client,
                self._prompt(request),
                effort=str(orchestrator.get("model_reasoning_effort", "high")),
            )
            self._load_manifest_summary()
            if self._manifest_is_running():
                self._emit(
                    "backend",
                    "manifest_reconciliation",
                    "Completed Codex turn left a running stage; dispatching one bounded receipt-reconciliation turn",
                )
                self._run_turn(
                    client,
                    self._reconciliation_prompt(),
                    effort=str(orchestrator.get("model_reasoning_effort", "high")),
                )
                self._load_manifest_summary()
            if self._manifest_is_running():
                raise CodexProtocolError(
                    "Orchestrator ended twice without applying a terminal Stage 0 receipt"
                )
            with self.lock:
                self.state = "completed"
                self.completed_at = utc_now()
            self._emit("backend", "run_completed", "Codex turn completed; evidence preserved")
        except Exception as error:  # fail closed at the HTTP adapter boundary
            with self.lock:
                self.state = "error"
                self.error = f"{type(error).__name__}: {error}"
                self.completed_at = utc_now()
            self._emit("backend", "run_error", self.error)
        finally:
            client = self.client
            if client is not None:
                client.close()
            with self.lock:
                self.client = None
            self._flush_evidence()

    def _run_turn(
        self,
        client: CodexAppServer,
        prompt: str,
        *,
        effort: str,
    ) -> None:
        if self.thread_id is None:
            raise CodexProtocolError("Cannot dispatch a turn before thread creation")
        client.prepare_turn()
        turn_result = client.request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": prompt, "text_elements": []}],
                "cwd": str(REPOSITORY_ROOT),
                "effort": effort,
                "approvalPolicy": "never",
            },
            timeout=60,
        )
        with self.lock:
            self.turn_id = str(turn_result["turn"]["id"])
        self._emit("backend", "turn_dispatched", f"turn_id={self.turn_id}")
        if not client.turn_completed.wait(TURN_TIMEOUT_SECONDS):
            raise CodexProtocolError("Stage 0 Codex turn exceeded the 30-minute limit")
        if client.turn_result is None:
            raise CodexProtocolError("Codex app-server closed before turn completion")
        turn = client.turn_result.get("turn", {})
        if isinstance(turn, dict) and turn.get("error") is not None:
            raise CodexProtocolError(
                f"Codex turn failed: {compact(turn['error'], 700)}"
            )

    def _manifest_is_running(self) -> bool:
        with self.lock:
            return bool(
                self.manifest is not None
                and self.manifest.get("pipeline_state") == "running"
            )

    def _reconciliation_prompt(self) -> str:
        return f"""Reconcile the unfinished Stage 0 attempt for `{self.run_id}`.

The preceding turn ended after the bounded `specimen_ingest` agent returned,
but the deterministic manifest still says `running`. Do not dispatch the agent
again and do not call any scientific or MCP tool again. Use the agent result
already present in this thread to build the exact attempt-bound `pass`,
`manual_review`, or `halt` receipt through `scripts/manage_part2_pipeline.py`,
then apply it with the CLI and validate the manifest. A dependency failure must
use the contract's bounded aggregate-only dependency error schema. Do not
return until the manifest is no longer `running`; keep Stages 1-6 locked.
"""

    def _write_confirmed_request(self) -> dict[str, Any]:
        source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
        assert self.run_id is not None
        request = {
            "schema_version": "part2-runtime-intake-request/1.0.0",
            "created_at": self.started_at,
            "specimen_id": self.run_id,
            "association_confirmed": True,
            "registration_mode": "challenge_aligned_json",
            "aligned_graph_authorized": True,
            "cad_units": "unknown",
            "cad_provenance": "committed LLNL challenge specimen input",
            "graph_axes": source["analysis_parameters"]["coordinates"]["graph_axes"],
            "ct_array_axes": source["analysis_parameters"]["coordinates"]["array_axes"],
            "aligned_graph_units": source["analysis_parameters"]["coordinates"]["aligned_graph_units"],
            "inputs": {
                "cad": source["inputs"]["cad"],
                "nominal_graph": source["inputs"]["design_graph"],
                "ct": source["inputs"]["ct"],
                "aligned_graph": source["inputs"]["aligned_graph"],
            },
        }
        request_path = (
            REPOSITORY_ROOT / "analysis" / self.run_id / "config" / "runtime_request.json"
        )
        atomic_json(request_path, request)
        if self.evidence_directory is not None:
            atomic_json(self.evidence_directory / "runtime_request.json", request)
        self._emit(
            "backend",
            "request_frozen",
            f"path={request_path.relative_to(REPOSITORY_ROOT).as_posix()}",
        )
        return request

    def _prompt(self, request: dict[str, Any]) -> str:
        request_path = f"analysis/{self.run_id}/config/runtime_request.json"
        return f"""Run one real Stage 0 orchestration attempt only.

The scientist confirmed the exact association recorded at `{request_path}` by
pressing the runtime demonstrator's confirmation control. The aligned graph is
explicitly authorized for challenge mode. CAD units remain `unknown`; preserve
that uncertainty and do not invent it.

Use the production runbook and deterministic state CLI. Initialize the pipeline
for specimen `{self.run_id}` with `{request_path}` as the frozen control config.
Perform dependency preflight using the live runtime evidence available in this
thread. Then invoke the project custom agent named `specimen_ingest` through
the Codex subagent runtime and give it only the Stage 0 contract-scoped handoff.
Include `{request_path}` in that handoff as the required
`scientist_intake_request`; it contains the scientist-confirmed identity,
coordinate, units, provenance, registration-mode, and authorization fields.
Do not perform specimen intake yourself. Do not continue to Stage 1 even if
Stage 0 passes. If any dependency, input, contract, dispatch, receipt, or gate
fails, record the required structured halt or manual-review state without a
fallback. Never return while the manifest or Stage 0 remains `running`. Return
the compact control-plane summary required by your developer instructions.
"""

    def _load_manifest_summary(self) -> None:
        if self.run_id is None:
            return
        path = REPOSITORY_ROOT / "analysis" / self.run_id / "manifest.json"
        if not path.is_file():
            self._emit("backend", "manifest_absent", "No pipeline manifest was written")
            return
        document = json.loads(path.read_text(encoding="utf-8"))
        stages = document.get("stages", {})
        with self.lock:
            self.manifest = {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "manifest_sha256": document.get("manifest_sha256"),
                "pipeline_state": document.get("pipeline_state"),
                "current_stage": document.get("current_stage"),
                "stages": {
                    number: {
                        "name": stage.get("name"),
                        "state": stage.get("state"),
                        "attempt_count": stage.get("attempt_count"),
                    }
                    for number, stage in stages.items()
                    if isinstance(stage, dict)
                },
            }
        self._emit(
            "backend",
            "manifest_found",
            f"state={self.manifest['pipeline_state']} sha256={self.manifest['manifest_sha256']}",
        )

    def projection(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "runId": self.run_id,
                "threadId": self.thread_id,
                "turnId": self.turn_id,
                "startedAt": self.started_at,
                "completedAt": self.completed_at,
                "events": list(self.events),
                "terminal": list(self.terminal),
                "finalMessage": self.final_message,
                "error": self.error,
                "manifest": self.manifest,
                "association": association_projection(),
            }

    def stop(self) -> None:
        client = self.client
        if client is not None:
            client.close()


def association_projection() -> dict[str, Any]:
    source = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    return {
        "sourceSpecimenId": source["specimen_id"],
        "registrationMode": "challenge_aligned_json",
        "cad": source["inputs"]["cad"]["path"],
        "nominalGraph": source["inputs"]["design_graph"]["path"],
        "ct": source["inputs"]["ct"]["path"],
        "alignedGraph": source["inputs"]["aligned_graph"]["path"],
        "cadUnits": "unknown",
        "graphAxes": source["analysis_parameters"]["coordinates"]["graph_axes"],
        "ctAxes": source["analysis_parameters"]["coordinates"]["array_axes"],
        "alignedGraphUnits": source["analysis_parameters"]["coordinates"]["aligned_graph_units"],
    }


RUN = RuntimeRun()
atexit.register(RUN.stop)


class Handler(BaseHTTPRequestHandler):
    server_version = "Part2RuntimeProof/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/runtime.html"}:
            self._send_bytes(HTTPStatus.OK, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path == "/api/runtime":
            self._send_json(HTTPStatus.OK, RUN.projection())
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/runtime/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_length"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_body_size"})
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return
        expected = {"confirmAssociation": True, "authorizeAlignedGraph": True}
        if body != expected:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "explicit_confirmation_required"},
            )
            return
        try:
            RUN.begin()
        except RuntimeError as error:
            self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            return
        self._send_json(HTTPStatus.ACCEPTED, RUN.projection())

    def log_message(self, format: str, *args: object) -> None:
        if self.command == "GET" and self.path == "/api/runtime":
            return
        print(f"[runtime-http] {self.address_string()} {format % args}", flush=True)

    def _send_json(self, status: HTTPStatus, value: object) -> None:
        self._send_bytes(
            status,
            (json.dumps(value, separators=(",", ":")) + "\n").encode(),
            "application/json; charset=utf-8",
        )

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--check-runtime",
        action="store_true",
        help="initialize and close app-server without starting a model turn",
    )
    args = parser.parse_args()
    if args.check_runtime:
        client = CodexAppServer(lambda method, params: None)
        client.start()
        client.close()
        print("[runtime-proof] Codex app-server initialize: pass")
        return 0
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[runtime-proof] Open http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        RUN.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
