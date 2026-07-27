"""Integration and isolation tests for the local Part 2 demonstrator."""

from __future__ import annotations

import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIRECTORY = REPOSITORY_ROOT / "demo" / "part2-orchestrator" / "server"
sys.path.insert(0, str(SERVER_DIRECTORY))

import demo_server  # noqa: E402
from demo_server import DemoRunStore, StaleStateError  # noqa: E402


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result


class Part2DemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="part2-demo-tests-")
        self.base = Path(self.temporary.name) / "sessions"
        self.store = DemoRunStore(self.base)
        self.original_store = demo_server.STORE
        demo_server.STORE = self.store
        self.http_server = demo_server.ThreadingHTTPServer(
            ("127.0.0.1", 0),
            demo_server.DemoRequestHandler,
        )
        self.http_thread = threading.Thread(
            target=self.http_server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.http_thread.start()

    def tearDown(self) -> None:
        self.http_server.shutdown()
        self.http_server.server_close()
        self.http_thread.join(timeout=2)
        demo_server.STORE = self.original_store
        self.store.close()
        self.temporary.cleanup()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], dict[str, object]]:
        connection = http.client.HTTPConnection(
            "127.0.0.1",
            int(self.http_server.server_address[1]),
            timeout=3,
        )
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            response_headers = {
                key.lower(): value for key, value in response.getheaders()
            }
            payload = json.loads(response.read() or b"{}")
            return response.status, response_headers, payload
        finally:
            connection.close()

    def run_to_stop(self, run, *, resume_review: bool = False) -> dict[str, object]:
        for _ in range(24):
            state = run.projection()
            if state["allowedAction"] == "resume":
                if not resume_review:
                    return state
                run.resume(str(state["manifestSha256"]))
                continue
            if state["allowedAction"] != "advance":
                return state
            run.advance(str(state["manifestSha256"]))
        self.fail("Demo did not reach a stop state")

    def test_verified_autonomous_walkthrough_runs_real_control_plane(self) -> None:
        analysis_before = _tree_hashes(REPOSITORY_ROOT / "analysis")
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        final = self.run_to_stop(run)

        self.assertEqual("pass", final["pipelineState"])
        self.assertTrue(final["sealedEvaluationConsumed"])
        self.assertTrue(all(stage["state"] == "pass" for stage in final["stages"]))
        self.assertTrue(
            any(event["kind"] == "registration_frozen" for event in final["events"])
        )
        self.assertTrue(
            run.runner.manifest_path.resolve().is_relative_to(self.base.resolve())
        )
        self.assertEqual(analysis_before, _tree_hashes(REPOSITORY_ROOT / "analysis"))

    def test_challenge_walkthrough_uses_declared_branch_without_freeze(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="challenge_aligned_json",
        )
        final = self.run_to_stop(run)
        self.assertEqual("pass", final["pipelineState"])
        self.assertFalse(
            any(event["kind"] == "registration_frozen" for event in final["events"])
        )

    def test_manual_review_stops_and_requires_explicit_resume(self) -> None:
        run = self.store.create(
            scenario="manual_review",
            registration_mode="autonomous_v2",
        )
        paused = self.run_to_stop(run)
        self.assertEqual("manual_review", paused["pipelineState"])
        self.assertEqual(2, paused["currentStage"])
        self.assertEqual("locked", paused["stages"][3]["state"])
        self.assertEqual("resume", paused["allowedAction"])

        run.resume(str(paused["manifestSha256"]))
        final = self.run_to_stop(run)
        self.assertEqual("pass", final["pipelineState"])
        self.assertEqual(2, final["stages"][2]["attemptCount"])

    def test_missing_dependency_records_halt_without_consuming_attempt(self) -> None:
        run = self.store.create(
            scenario="missing_dependency",
            registration_mode="autonomous_v2",
        )
        initial = run.projection()
        halted = run.advance(str(initial["manifestSha256"]))

        self.assertEqual("halt", halted["pipelineState"])
        self.assertEqual(0, halted["stages"][0]["attemptCount"])
        self.assertTrue(all(stage["state"] == "locked" for stage in halted["stages"][1:]))
        event = halted["events"][-1]
        self.assertEqual("dependency_halt", event["kind"])
        self.assertIn("No fallback was used", event["detail"])

    def test_tampered_receipt_is_rejected_and_downstream_remains_locked(self) -> None:
        run = self.store.create(
            scenario="tampered_receipt",
            registration_mode="autonomous_v2",
        )
        ready = run.projection()
        running = run.advance(str(ready["manifestSha256"]))
        rejected = run.advance(str(running["manifestSha256"]))

        self.assertEqual("running", rejected["pipelineState"])
        self.assertEqual("blocked", rejected["verificationState"])
        self.assertEqual(
            {
                "code": "receipt_integrity_rejected",
                "stage": 0,
                "message": rejected["blockedReason"],
            },
            rejected["verificationBlock"],
        )
        self.assertIn("manifest remains running", rejected["blockedReason"])
        self.assertNotIn("canonical hash", rejected["blockedReason"])
        self.assertEqual("locked", rejected["stages"][1]["state"])
        self.assertIsNone(rejected["allowedAction"])
        self.assertEqual("receipt_rejected", rejected["events"][-1]["kind"])
        self.assertEqual("demo_adapter", rejected["events"][-1]["source"])
        self.assertEqual("rejected", rejected["events"][-1]["tone"])

        replayed = run.advance(str(rejected["manifestSha256"]))
        self.assertEqual(rejected["verificationBlock"], replayed["verificationBlock"])
        self.assertEqual(len(rejected["events"]), len(replayed["events"]))

    def test_stale_browser_state_cannot_advance_again(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        stale_hash = str(run.projection()["manifestSha256"])
        run.advance(stale_hash)
        with self.assertRaises(StaleStateError):
            run.advance(stale_hash)
        self.assertEqual(1, run.projection()["stages"][0]["attemptCount"])

    def test_concurrent_double_advance_has_one_logical_winner(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        expected = str(run.projection()["manifestSha256"])

        def advance() -> str:
            try:
                return str(run.advance(expected)["pipelineState"])
            except StaleStateError:
                return "stale"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: advance(), range(2)))
        self.assertEqual(1, outcomes.count("stale"))
        state = run.projection()
        self.assertEqual(1, state["stages"][0]["attemptCount"])
        self.assertEqual("running", state["stages"][0]["state"])

    def test_browser_projection_redacts_sensitive_paths_and_absolute_roots(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        final = self.run_to_stop(run)
        serialized = json.dumps(final, sort_keys=True)
        self.assertNotIn(str(self.base.resolve()), serialized)
        self.assertNotIn("evals/labels", serialized)
        self.assertNotIn("development_labels", serialized)
        self.assertNotIn("sealed_labels_sha256", serialized)
        self.assertNotIn("scoped_handoffs", serialized)

    def test_cleanup_refuses_unmarked_directory(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        marker = run.root / demo_server.SESSION_MARKER
        marker.unlink()
        with self.assertRaises(RuntimeError):
            run.close(self.base)
        self.assertTrue(run.root.is_dir())
        marker.write_text(
            json.dumps(run._marker_document(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_cleanup_refuses_symlink_substitution_and_preserves_both_runs(self) -> None:
        first = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        second = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        shutil.rmtree(first.root)
        first.root.symlink_to(second.root, target_is_directory=True)

        with self.assertRaises(RuntimeError):
            self.store.delete(first.run_id)

        self.assertIn(first.run_id, self.store.runs)
        self.assertIn(second.run_id, self.store.runs)
        self.assertTrue(second.root.is_dir())

        first.root.unlink()
        first.root.mkdir()
        (first.root / demo_server.SESSION_MARKER).write_text(
            json.dumps(first._marker_document(), sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_fixture_writes_and_contract_paths_cannot_escape_session_root(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        outside = Path(self.temporary.name) / "outside.json"
        with self.assertRaises(RuntimeError):
            run.runner._write_json(outside, {"unsafe": True})
        self.assertFalse(outside.exists())

        for unsafe in ("../outside.json", str(outside.resolve())):
            contract = copy.deepcopy(run.runner.contracts[4])
            target = next(
                rule
                for rule in contract["output_artifacts"]["allowed"]
                if rule["role"] == "classifier_verifier_report"
            )
            target["path"] = unsafe
            with self.subTest(path=unsafe), self.assertRaises(RuntimeError):
                run.runner._validate_contract_paths(contract)

    def test_corrupt_sensitive_artifact_errors_are_redacted_over_http(self) -> None:
        for role in ("development_labels", "sealed_labels"):
            with self.subTest(role=role):
                run = self.store.create(
                    scenario="verified_walkthrough",
                    registration_mode="autonomous_v2",
                )
                for _ in range(4):
                    state = run.projection()
                    run.advance(str(state["manifestSha256"]))
                expected_manifest_sha256 = str(run.projection()["manifestSha256"])
                manifest = run.runner.manifest()
                artifact = next(
                    item
                    for item in manifest["stages"]["1"]["attempts"][-1][
                        "output_artifacts"
                    ]
                    if item["role"] == role
                )
                (run.runner.root / artifact["path"]).write_text(
                    "tampered fixture\n",
                    encoding="utf-8",
                )

                method = "GET" if role == "development_labels" else "POST"
                path = f"/api/v1/demo-runs/{run.run_id}"
                request_body = None
                headers = {"Origin": "http://localhost:3000"}
                if method == "POST":
                    path += "/steps"
                    request_body = json.dumps(
                        {"expectedManifestSha256": expected_manifest_sha256}
                    ).encode()
                    headers["Content-Type"] = "application/json"
                status, _, payload = self.request(
                    method,
                    path,
                    body=request_body,
                    headers=headers,
                )
                serialized = json.dumps(payload, sort_keys=True)
                self.assertEqual(409, status)
                self.assertEqual(
                    (
                        "verification_failed"
                        if method == "GET"
                        else "orchestration_rejected"
                    ),
                    payload["error"],
                )
                self.assertNotIn(str(self.base.resolve()), serialized)
                self.assertNotIn(str(artifact["path"]), serialized)
                self.assertNotIn(str(artifact["sha256"]), serialized)
                self.assertNotIn(role, serialized)

    def test_http_origin_and_cors_policy(self) -> None:
        status, headers, payload = self.request(
            "GET",
            "/api/v1/health",
            headers={"Origin": "https://attacker.invalid"},
        )
        self.assertEqual(403, status)
        self.assertEqual("origin_not_allowed", payload["error"])
        self.assertNotIn("access-control-allow-origin", headers)

        status, headers, _ = self.request(
            "OPTIONS",
            "/api/v1/demo-runs",
            headers={"Origin": "http://localhost:3000"},
        )
        self.assertEqual(204, status)
        self.assertEqual(
            "http://localhost:3000",
            headers["access-control-allow-origin"],
        )
        self.assertIn("POST", headers["access-control-allow-methods"])

    def test_http_rejects_malformed_and_oversized_bodies(self) -> None:
        for body in (b"{", b"[]"):
            with self.subTest(body=body):
                status, _, payload = self.request(
                    "POST",
                    "/api/v1/demo-runs",
                    body=body,
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(400, status)
                self.assertEqual("invalid_request", payload["error"])

        status, _, payload = self.request(
            "POST",
            "/api/v1/demo-runs",
            body=b"x" * (demo_server.MAX_BODY_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(400, status)
        self.assertEqual("invalid_request", payload["error"])
        self.assertEqual("Request body is too large", payload["message"])

    def test_stage5_fixture_aggregate_is_internally_coherent(self) -> None:
        run = self.store.create(
            scenario="verified_walkthrough",
            registration_mode="autonomous_v2",
        )
        self.run_to_stop(run)
        manifest = run.runner.manifest()
        artifact = next(
            item
            for item in manifest["stages"]["5"]["attempts"][-1][
                "output_artifacts"
            ]
            if item["role"] == "sealed_evaluation_result"
        )
        aggregate = json.loads(
            (run.runner.root / artifact["path"]).read_text(encoding="utf-8")
        )
        matrix = aggregate["confusion_matrix"]["rows_actual_columns_predicted"]
        self.assertEqual(
            aggregate["sealed_strut_count"],
            sum(sum(row.values()) for row in matrix.values()),
        )
        self.assertEqual(
            aggregate["strict_recall"]["detected"],
            matrix["missing"]["missing"],
        )
        self.assertEqual(
            aggregate["lenient_recall"]["detected"],
            matrix["missing"]["missing"] + matrix["missing"]["broken"],
        )

    def test_demo_adapter_imports_no_scientific_implementation(self) -> None:
        imported: set[str] = set()
        for filename in ("demo_pipeline.py", "demo_server.py"):
            tree = ast.parse((SERVER_DIRECTORY / filename).read_text(encoding="utf-8"))
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
        self.assertIn("part2_orchestration", imported)


if __name__ == "__main__":
    unittest.main()
