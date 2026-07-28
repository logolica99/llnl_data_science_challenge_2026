#!/usr/bin/env python3
"""Run an evidence-only Part 2 control-plane proof and render a static page.

This runner never creates specialist outputs. It copies the repository's real
contracts and frozen specimen configuration into an isolated directory, asks
the production control plane to initialize and preflight Stage 0, and renders
the resulting verified halt/pass state. Configuration is not treated as proof
that an agent or MCP interface is callable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from part2_orchestration import (  # noqa: E402
    create_pipeline_manifest,
    pipeline_status,
    start_stage,
    validate_pipeline_manifest,
)


SOURCE_CONFIG = Path(
    "analysis/brian_tran_9x9x9_0point5dash1/config/specimen_manifest.json"
)
PROJECT_ROOT = REPOSITORY_ROOT / "demo" / "part2-orchestrator"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def discover_agents(repository_root: Path) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for path in sorted((repository_root / ".codex" / "agents").glob("*.toml")):
        try:
            document = tomllib.loads(path.read_text(encoding="utf-8"))
            name = document.get("name")
            valid = (
                isinstance(name, str)
                and bool(name)
                and isinstance(document.get("developer_instructions"), str)
            )
        except (OSError, tomllib.TOMLDecodeError):
            name = path.stem
            valid = False
        discovered[str(name)] = {
            "path": path.relative_to(repository_root).as_posix(),
            "configured": valid,
            "callable_verified": False,
        }
    return discovered


def discover_mcp_configuration(repository_root: Path) -> dict[str, Any]:
    command = ["codex", "mcp", "list"]
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "configured": False,
            "enabled": False,
            "live_schema_verified": False,
            "discovery": f"unavailable:{type(error).__name__}",
        }
    matching = next(
        (
            line
            for line in completed.stdout.splitlines()
            if line.strip().startswith("segmentation-tools ")
        ),
        "",
    )
    return {
        "configured": bool(matching),
        "enabled": bool(matching and " enabled " in f" {matching} "),
        "live_schema_verified": False,
        "discovery": "codex mcp list",
    }


def load_contracts(repository_root: Path) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for path in sorted((repository_root / "analysis" / "contracts").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if "stage_number" not in document:
            continue
        contracts.append(
            {
                "stage_number": int(document["stage_number"]),
                "stage": document["stage"],
                "owner": document["owner"],
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
                "required_dependencies": document["required_dependencies"],
            }
        )
    contracts.sort(key=lambda item: item["stage_number"])
    if [item["stage_number"] for item in contracts] != list(range(5)):
        raise RuntimeError("The production Stage 0-4 contract set is incomplete")
    return contracts


def capability_audit(
    contracts: list[dict[str, Any]],
    agents: dict[str, dict[str, Any]],
    mcp: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        required_agents = [
            item["name"]
            for item in contract["required_dependencies"].get("agents", [])
        ]
        required_tools = [
            f"{item.get('server', 'segmentation-tools')}.{item['name']}"
            for item in contract["required_dependencies"].get("mcp_tools", [])
        ]
        rows.append(
            {
                "stage_number": contract["stage_number"],
                "stage": contract["stage"],
                "required_agents": required_agents,
                "configured_agents": [
                    name
                    for name in required_agents
                    if agents.get(name, {}).get("configured") is True
                ],
                "missing_agents": [
                    name
                    for name in required_agents
                    if agents.get(name, {}).get("configured") is not True
                ],
                "required_tools": required_tools,
                "mcp_configured": mcp["configured"],
                "mcp_live_schema_verified": mcp["live_schema_verified"],
            }
        )
    return rows


def log_line(lines: list[str], message: str) -> None:
    rendered = f"[part2-orchestrator] {message}"
    lines.append(rendered)
    print(rendered, flush=True)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def render_html(report: dict[str, Any]) -> str:
    status = report["status"]
    stages = status["stages"]
    audit = report["capability_audit"]
    halt = report["halt"]
    stage_markup = "".join(
        f"""
        <li class="stage stage-{html.escape(stage['state'])}">
          <span class="stage-number">{number}</span>
          <span><strong>{html.escape(stage['name'])}</strong><small>{html.escape(stage['state'])} · attempts {stage['attempt_count']}/{stage['maximum_attempts']}</small></span>
        </li>"""
        for number, stage in stages.items()
    )
    agent_rows = "".join(
        f"""
        <tr>
          <td>{row['stage_number']}</td>
          <td>{html.escape(row['stage'])}</td>
          <td>{html.escape(', '.join(row['configured_agents']) or 'none')}</td>
          <td>{html.escape(', '.join(row['missing_agents']) or 'none')}</td>
        </tr>"""
        for row in audit
    )
    terminal = "\n".join(html.escape(line) for line in report["terminal"])
    failures = "".join(
        f"<li><code>{html.escape(item['kind'])}</code> {html.escape(item['name'])}: {html.escape(item['reason'])}</li>"
        for item in halt["failures"]
    )
    configured_agents = ", ".join(sorted(report["agents"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Part 2 orchestrator — real fail-closed proof</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1018; --panel:#121a25; --line:#2a3748; --text:#edf4ff; --muted:#9babc0; --good:#63d7a1; --bad:#ff7b72; --lock:#77869a; --accent:#69a7ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    main {{ width:min(1100px,calc(100% - 32px)); margin:32px auto 56px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(24px,4vw,42px); font-weight:650; letter-spacing:-.03em; }}
    h2 {{ font-size:18px; margin:0 0 14px; }}
    p {{ color:var(--muted); margin:0; }}
    .status {{ display:inline-flex; gap:8px; align-items:center; color:var(--bad); margin:18px 0 26px; font-weight:650; }}
    .status::before {{ content:""; width:10px; height:10px; border-radius:50%; background:var(--bad); box-shadow:0 0 18px var(--bad); }}
    .grid {{ display:grid; grid-template-columns:minmax(260px,.8fr) minmax(0,1.6fr); gap:16px; }}
    section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; }}
    .stages {{ list-style:none; padding:0; margin:0; display:grid; gap:8px; }}
    .stage {{ display:flex; gap:12px; align-items:center; border-left:3px solid var(--lock); padding:9px 10px; background:#0d141e; }}
    .stage-halt {{ border-color:var(--bad); }}
    .stage-number {{ width:25px; height:25px; border:1px solid var(--line); border-radius:50%; display:grid; place-items:center; color:var(--muted); font-size:12px; }}
    .stage small {{ display:block; color:var(--muted); margin-top:2px; }}
    .proof {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:16px; }}
    .proof div {{ background:#0d141e; padding:12px; border-radius:8px; }}
    .proof span {{ display:block; color:var(--muted); font-size:12px; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; overflow-wrap:anywhere; }}
    .halt {{ border-left:3px solid var(--bad); padding-left:14px; margin:14px 0 18px; }}
    .halt strong {{ color:var(--bad); }}
    .halt ul {{ margin:8px 0 0; padding-left:20px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th,td {{ padding:8px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
    th {{ color:var(--muted); font-weight:500; }}
    .wide {{ grid-column:1/-1; }}
    pre {{ margin:0; background:#070b11; border:1px solid var(--line); border-radius:8px; padding:16px; color:#c8f7d8; white-space:pre-wrap; overflow-wrap:anywhere; font:12px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .foot {{ margin-top:14px; font-size:12px; }}
    @media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; }} .proof {{ grid-template-columns:1fr; }} .wide {{ grid-column:auto; }} }}
  </style>
</head>
<body>
<main>
  <h1>Real fail-closed run</h1>
  <p>No specialist outputs, receipts, labels, CT results, or success states were simulated.</p>
  <div class="status">HALT at Stage {status['current_stage']} · downstream locked</div>
  <div class="grid">
    <section>
      <h2>Pipeline</h2>
      <ol class="stages">{stage_markup}</ol>
    </section>
    <section>
      <h2>What the orchestrator proved</h2>
      <div class="proof">
        <div><span>Manifest SHA-256</span><code>{html.escape(status['manifest_sha256'])}</code></div>
        <div><span>Frozen config SHA-256</span><code>{html.escape(status['config_sha256'])}</code></div>
        <div><span>Stage contracts</span><strong>{len(report['contracts'])} hashed and validated</strong></div>
      </div>
      <div class="halt">
        <strong>Structured dependency halt</strong>
        <ul>{failures}</ul>
      </div>
      <p>Configured agents found: {html.escape(configured_agents)}. The MCP server is configured={str(report['mcp']['configured']).lower()} and enabled={str(report['mcp']['enabled']).lower()}, but a configuration listing is not a live response-schema attestation. No fallback was used.</p>
    </section>
    <section class="wide">
      <h2>Declared agent coverage</h2>
      <table>
        <thead><tr><th>Stage</th><th>Contract</th><th>Configured files</th><th>Missing files</th></tr></thead>
        <tbody>{agent_rows}</tbody>
      </table>
    </section>
    <section class="wide">
      <h2>Terminal transcript</h2>
      <pre>{terminal}</pre>
      <p class="foot">Generated {html.escape(report['generated_at'])} from production control-plane state and copied evidence.</p>
    </section>
  </div>
</main>
</body>
</html>
"""


def run(output_directory: Path) -> dict[str, Any]:
    source_config = REPOSITORY_ROOT / SOURCE_CONFIG
    if not source_config.is_file():
        raise RuntimeError(f"Real frozen specimen configuration is missing: {SOURCE_CONFIG}")

    contracts = load_contracts(REPOSITORY_ROOT)
    agents = discover_agents(REPOSITORY_ROOT)
    mcp = discover_mcp_configuration(REPOSITORY_ROOT)
    audit = capability_audit(contracts, agents, mcp)
    terminal: list[str] = []
    log_line(terminal, f"loaded and hashed {len(contracts)} production stage contracts")
    log_line(terminal, f"source config {SOURCE_CONFIG} sha256={sha256_file(source_config)}")
    log_line(terminal, "configured agents: " + ", ".join(sorted(agents)))
    log_line(
        terminal,
        "agent configuration found; callable agent runtime is not attested to this standalone runner",
    )
    log_line(
        terminal,
        "segmentation-tools configuration: "
        f"configured={mcp['configured']} enabled={mcp['enabled']} live_schema_verified=False",
    )

    with tempfile.TemporaryDirectory(prefix="llnl-part2-real-proof-") as temporary:
        isolated_root = Path(temporary) / "repository"
        shutil.copytree(
            REPOSITORY_ROOT / "analysis" / "contracts",
            isolated_root / "analysis" / "contracts",
        )
        isolated_config = isolated_root / "config" / "frozen_specimen_manifest.json"
        isolated_config.parent.mkdir(parents=True)
        shutil.copy2(source_config, isolated_config)
        config_document = json.loads(isolated_config.read_text(encoding="utf-8"))
        specimen_id = str(config_document["specimen_id"])
        mode = "autonomous_v2"

        created = create_pipeline_manifest(
            repository_root=isolated_root,
            specimen_id=specimen_id,
            config_path=isolated_config,
            registration_mode=mode,
        )
        manifest_path = Path(created["path"])
        initial = pipeline_status(manifest_path, repository_root=isolated_root)
        log_line(
            terminal,
            f"manifest created sha256={initial['manifest_sha256']} stage=0 state=ready",
        )
        log_line(terminal, "preflight Stage 0 before creating any attempt or handoff")

        stage_zero_agent = audit[0]["required_agents"][0]
        inventory = {
            "agents": {
                stage_zero_agent: {
                    "available": agents.get(stage_zero_agent, {}).get("callable_verified") is True,
                    "contract_version": contracts[0]["required_dependencies"]["agents"][0]["contract_version"],
                }
            },
            "mcp_servers": {
                "segmentation-tools": {
                    "healthy": False,
                    "tools": {},
                }
            },
        }
        result = start_stage(
            manifest_path,
            0,
            input_artifacts=[],
            capability_inventory=inventory,
            repository_root=isolated_root,
        )
        verified_manifest = validate_pipeline_manifest(
            manifest_path,
            repository_root=isolated_root,
            verify_artifacts=True,
        )
        status = pipeline_status(manifest_path, repository_root=isolated_root)
        halt = dict(result["error"])
        log_line(
            terminal,
            "dependency_halt code="
            f"{halt['code']} attempt_count={status['stages']['0']['attempt_count']}",
        )
        log_line(terminal, "fallback_used=False; Stages 1-4 remain locked")
        log_line(terminal, f"halt receipt sha256={result['receipt']['sha256']}")
        log_line(terminal, f"final manifest sha256={status['manifest_sha256']}")

        receipt_path = isolated_root / result["receipt"]["path"]
        evidence_directory = output_directory / "proof-evidence" / "latest"
        evidence_directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, evidence_directory / "manifest.json")
        shutil.copy2(receipt_path, evidence_directory / "dependency-halt-receipt.json")

        report = {
            "schema_version": "part2-orchestrator-real-proof/1.0.0",
            "generated_at": utc_now(),
            "source_config": {
                "path": SOURCE_CONFIG.as_posix(),
                "sha256": sha256_file(source_config),
            },
            "contracts": contracts,
            "agents": agents,
            "mcp": mcp,
            "capability_audit": audit,
            "status": status,
            "halt": halt,
            "receipt": result["receipt"],
            "terminal": terminal,
            "manifest_revision": verified_manifest["revision"],
            "synthetic_specialist_artifacts": False,
            "fallback_used": False,
        }
        atomic_write(
            evidence_directory / "proof-report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        atomic_write(evidence_directory / "terminal.log", "\n".join(terminal) + "\n")
        atomic_write(output_directory / "real-proof.html", render_html(report))
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    output_directory = args.output_directory.resolve()
    report = run(output_directory)
    print(f"[part2-orchestrator] proof page: {output_directory / 'real-proof.html'}")
    print(
        "[part2-orchestrator] evidence: "
        f"{output_directory / 'proof-evidence' / 'latest' / 'proof-report.json'}"
    )
    return 0 if report["status"]["pipeline_state"] == "halt" else 1


if __name__ == "__main__":
    raise SystemExit(main())
