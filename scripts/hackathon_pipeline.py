#!/usr/bin/env python3
"""Hackathon Part 2 runner: metadata → registration → defects → NDE report.

Offline human fallback only. Agents must use `$hackathon-nde-pipeline` and
`segmentation-tools` MCP subagents instead of this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from llnl_nde.core.defect_analysis import (  # noqa: E402
    DEFAULT_STAGE3_CONFIG,
    DEFECT_KINDS,
    analyze_strut_specialist,
    export_stage3_validation_csvs,
    merge_strut_classifications,
    prepare_hackathon_report_classifications,
)
from llnl_nde.core.localization import localize_lattice_nodes  # noqa: E402
from llnl_nde.core.otsu import replay_exact_otsu  # noqa: E402
from llnl_nde.core.registration import (  # noqa: E402
    DEFAULT_REGISTRATION_CONFIG,
    register_lattice_to_ct,
)
from llnl_nde.core.reporting import get_strut_report  # noqa: E402
from llnl_nde.core.spatial import compute_spatial_stats, render_lattice_3d  # noqa: E402
from llnl_nde.core.strut_metrics import compute_strut_metrics  # noqa: E402
from llnl_nde.core.volume_inspection import inspect_volume  # noqa: E402


# Extensible registry: add new kinds here when teammate agents land.
AVAILABLE_DEFECT_AGENTS: tuple[str, ...] = DEFECT_KINDS

# Lenient registration gates for demo runs (science still runs; soft gates).
HACKATHON_REGISTRATION_CONFIG: dict[str, Any] = copy.deepcopy(DEFAULT_REGISTRATION_CONFIG)
HACKATHON_REGISTRATION_CONFIG["fitting"]["maximum_multistart_p95_spread_voxels"] = 10.0
HACKATHON_REGISTRATION_CONFIG["robustness"]["maximum_p95_prediction_spread_voxels"] = 15.0
HACKATHON_REGISTRATION_CONFIG["robustness"]["minimum_successful_cases"] = 3

HACKATHON_LOCALIZATION_CONFIG: dict[str, Any] = {
    "minimum_primary_or_stable_coarse_fraction": 0.0,
    "maximum_fallback_fraction": 1.0,
    "maximum_ambiguous_fraction": 1.0,
    "maximum_rejected_fraction": 1.0,
    "maximum_boundary_limited_fraction": 1.0,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "hackathon-pipeline-status/1.0.0",
            "stages": {},
            "updated_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_status(path: Path, status: dict[str, Any], *, overwrite: bool = True) -> None:
    status = dict(status)
    status["updated_at"] = _utc_now()
    _write_json(path, status, overwrite=overwrite or path.exists())


def stage0_metadata(
    *,
    ct_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    stage_dir = output_dir / "stage0"
    stage_dir.mkdir(parents=True, exist_ok=True)
    metadata = inspect_volume(
        ct_path,
        repository_root=REPOSITORY_ROOT,
        header_only=True,
        include_sha256=True,
    )
    out = stage_dir / "ct_metadata.json"
    _write_json(out, metadata, overwrite=overwrite)
    return {
        "gate": "pass",
        "artifacts": {"ct_metadata": _rel(out)},
        "summary": {
            "shape": metadata.get("shape"),
            "dtype": metadata.get("dtype"),
            "format": metadata.get("format"),
        },
    }


def stage1_registration(
    *,
    ct_path: Path,
    nominal_path: Path,
    output_dir: Path,
    overwrite: bool,
    continue_on_halt: bool,
) -> dict[str, Any]:
    stage_dir = output_dir / "stage1"
    stage_dir.mkdir(parents=True, exist_ok=True)
    registered = stage_dir / "registered_graph.json"
    report = stage_dir / "registration_report.json"
    otsu_report, _ = replay_exact_otsu(ct_path)
    if not otsu_report.get("overall_pass", False):
        failed = sorted(
            name for name, passed in otsu_report.get("gates", {}).items() if not passed
        )
        raise RuntimeError(
            "Otsu rejected the CT volume before registration: " + ", ".join(failed)
        )
    threshold = float(otsu_report["threshold"])
    result = register_lattice_to_ct(
        nominal_path,
        registered,
        report,
        mode="autonomous_v2",
        ct_path=ct_path,
        threshold=threshold,
        config=HACKATHON_REGISTRATION_CONFIG,
        overwrite=overwrite,
    )
    gate = str(result.get("gate", "halt"))
    if gate == "halt" and not continue_on_halt:
        raise RuntimeError(
            "Registration gate halted; re-run with --continue-on-halt to keep going"
        )
    return {
        "gate": gate if gate != "halt" else "continued_after_halt",
        "threshold": threshold,
        "artifacts": {
            "registered_graph": _rel(registered),
            "registration_report": _rel(report),
        },
        "summary": {
            "registration_gate": gate,
            "overall_pass": bool(result.get("overall_pass")),
            "failed_gates": sorted(
                name
                for name, passed in (result.get("gates") or {}).items()
                if not passed
            ),
        },
    }


def _report_ready_classifications(
    classifications_path: Path,
    output_path: Path,
    *,
    nominal_path: Path,
    metrics_path: Path,
    overwrite: bool,
) -> Path:
    """Remap deferred + crop-plane + connected-bite broken for report tools."""

    prepare_hackathon_report_classifications(
        classifications_path,
        output_path,
        nominal_graph_path=nominal_path,
        metrics_path=metrics_path,
        excluded_nominal_axis="y",
        excluded_nominal_value=18.0,
        require_disconnected_for_broken=True,
        overwrite=overwrite,
    )
    return output_path


def stage2_defects(
    *,
    ct_path: Path,
    nominal_path: Path,
    output_dir: Path,
    specimen_id: str,
    threshold: float,
    overwrite: bool,
    defect_kinds: Sequence[str],
) -> dict[str, Any]:
    stage1 = output_dir / "stage1"
    stage2 = output_dir / "stage2"
    stage2.mkdir(parents=True, exist_ok=True)
    registered = stage1 / "registered_graph.json"
    if not registered.is_file():
        raise FileNotFoundError("Stage 1 registered_graph.json is missing; run stage 1 first")

    localized = stage2 / "localized_graph.json"
    localization_report = stage2 / "localization_report.json"
    localization = localize_lattice_nodes(
        ct_path,
        registered,
        localized,
        localization_report,
        threshold=threshold,
        registration_mode="autonomous_v2",
        config=HACKATHON_LOCALIZATION_CONFIG,
        overwrite=overwrite,
    )

    metrics_path = stage2 / "per_strut_metrics.csv"
    profiles_path = stage2 / "per_strut_profiles.json"
    metrics_report = stage2 / "metrics_report.json"
    metrics = compute_strut_metrics(
        ct_path,
        localized,
        metrics_path,
        profiles_path,
        metrics_report,
        threshold=threshold,
        registration_mode="autonomous_v2",
        overwrite=overwrite,
    )

    stage3_config = {
        "stage_3_defect_analysis": copy.deepcopy(DEFAULT_STAGE3_CONFIG),
    }
    config_path = stage2 / "defect_analysis_config.json"
    _write_json(config_path, stage3_config, overwrite=overwrite)

    findings_paths: dict[str, Path] = {}
    agent_summaries: dict[str, Any] = {}
    for kind in DEFECT_KINDS:
        if kind not in defect_kinds:
            raise ValueError(
                f"Merge requires all defect kinds; missing agent selection for '{kind}'"
            )
        findings_path = stage2 / f"findings_{kind}.json"
        result = analyze_strut_specialist(
            metrics_path,
            profiles_path,
            stage3_config,
            findings_path,
            specimen_id=specimen_id,
            defect_kind=kind,
            overwrite=overwrite,
        )
        findings_paths[kind] = findings_path
        agent_summaries[kind] = {
            "status": result.get("status"),
            "gate": result.get("gate"),
            "artifact": _rel(findings_path),
            "warnings": result.get("warnings") or [],
        }

    classifications_path = stage2 / "classified_struts.json"
    thresholds_path = stage2 / "thresholds.json"
    decision_log_path = stage2 / "decision_log.md"
    merge = merge_strut_classifications(
        metrics_path,
        profiles_path,
        stage3_config,
        findings_paths,
        classifications_path,
        thresholds_path,
        decision_log_path,
        specimen_id=specimen_id,
        overwrite=overwrite,
    )

    csv_dir = stage2 / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_export = export_stage3_validation_csvs(
        classifications_path,
        findings_paths["missing"],
        findings_paths["broken"],
        metrics_path,
        nominal_path,
        csv_dir,
        overwrite=overwrite,
    )

    report_classifications = _report_ready_classifications(
        classifications_path,
        stage2 / "classified_struts_report.json",
        nominal_path=nominal_path,
        metrics_path=metrics_path,
        overwrite=overwrite,
    )

    return {
        "gate": str(merge.get("gate", "manual_review")),
        "threshold": threshold,
        "artifacts": {
            "localized_graph": _rel(localized),
            "localization_report": _rel(localization_report),
            "per_strut_metrics": _rel(metrics_path),
            "per_strut_profiles": _rel(profiles_path),
            "metrics_report": _rel(metrics_report),
            "classified_struts": _rel(classifications_path),
            "classified_struts_report": _rel(report_classifications),
            "thresholds": _rel(thresholds_path),
            "decision_log": _rel(decision_log_path),
            "csv_directory": _rel(csv_dir),
            "defect_agents": agent_summaries,
        },
        "summary": {
            "localization_gate": localization.get("gate"),
            "metrics_rows": (metrics.get("counts") or {}).get("metric_rows"),
            "classification_counts": merge.get("counts"),
            "report_counts": json.loads(
                report_classifications.read_text(encoding="utf-8")
            ).get("counts"),
            "csv_export_counts": csv_export.get("counts"),
            "csv_files": [
                _rel(csv_dir / "missing_struts.csv"),
                _rel(csv_dir / "broken_struts.csv"),
                _rel(csv_dir / "missing_struts_viewer_filtered.csv"),
                _rel(csv_dir / "broken_struts_viewer_filtered.csv"),
            ],
            "available_defect_agents": list(defect_kinds),
        },
    }


def _write_nde_report(
    *,
    specimen_id: str,
    output_path: Path,
    metadata_path: Path,
    registration_summary: Mapping[str, Any],
    defect_summary: Mapping[str, Any],
    spatial_path: Path,
    lattice_render_path: Path,
    classifications_path: Path,
    metrics_path: Path,
    thresholds_path: Path,
    flagged_reports: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> Path:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    spatial = json.loads(spatial_path.read_text(encoding="utf-8"))
    classifications = json.loads(classifications_path.read_text(encoding="utf-8"))
    counts = classifications.get("counts") or Counter(
        row["class"] for row in classifications.get("classifications", [])
    )
    class_counts = spatial.get("class_counts") or counts
    cluster_payload = spatial.get("defect_clusters") or {}
    cluster_ids = cluster_payload.get("cluster_strut_ids") or []

    lines = [
        f"# Non-Destructive Evaluation Report: `{specimen_id}`",
        "",
        f"_Generated {_utc_now()} (hackathon pipeline, materials-science orientation)_",
        "",
        "## 1. Specimen and measurement context",
        "",
        "This report evaluates an additively manufactured lattice specimen from CT "
        "relative to its nominal design graph. Defect labels describe observed "
        "material presence and connectivity in the as-built volume; they do not "
        "assert design intent or manufacturing root cause.",
        "",
        f"- Specimen ID: `{specimen_id}`",
        f"- CT shape / dtype: `{metadata.get('shape')}` / `{metadata.get('dtype')}`",
        f"- CT path: `{metadata.get('relative_path') or metadata.get('path')}`",
        f"- Registration gate: `{registration_summary.get('registration_gate')}`",
        f"- Defect classification gate: `{defect_summary.get('gate')}`",
        "",
        "## 2. Process and materials interpretation",
        "",
        "Lattice struts are load-bearing ligaments. Missing struts remove a "
        "nominal load path entirely. Broken struts retain endpoint material but "
        "show a central discontinuity or material-loss bite. Present struts "
        "satisfy the missing/broken specialist rules under the frozen Stage-3 "
        "policy. Thin and bent specialists may still be deferred; deferred "
        "findings are treated as present for this report until those agents ship. "
        "Missing/broken struts touching the specimen high-Y crop plane "
        "(nominal `y=18`) are excluded from these report totals as known "
        "crop-face artifacts; full scientific counts remain in "
        "`classified_struts.json` / unfiltered CSVs.",
        "",
        "## 3. Population summary (crop-plane filtered)",
        "",
        "| Class | Count |",
        "|---|---:|",
    ]
    for label in ("missing", "broken", "thin", "present"):
        lines.append(f"| {label} | {int(class_counts.get(label, 0))} |")
    lines.extend(
        [
            "",
            f"- Total classified struts: **{sum(int(class_counts.get(label, 0)) for label in ('missing', 'broken', 'thin', 'present'))}**",
            f"- Spatial defect clusters: **{int(cluster_payload.get('count', len(cluster_ids)))}**",
            "",
            "## 4. Spatial pattern notes",
            "",
        ]
    )
    if not cluster_ids:
        lines.append(
            "No multi-strut defect clusters were detected from the classified graph."
        )
    else:
        lines.append(
            "Defect clusters suggest localized process excursions (for example "
            "build-direction bands, powder starvation pockets, or scan-path "
            "anomalies) rather than uniformly random strut failure."
        )
        for index, member_ids in enumerate(cluster_ids[:10], start=1):
            preview = member_ids[:12]
            lines.append(
                f"- Cluster {index}: size={len(member_ids)}, strut_ids={preview}"
                + (" ..." if len(member_ids) > 12 else "")
            )
        if len(cluster_ids) > 10:
            lines.append(f"- … {len(cluster_ids) - 10} additional clusters omitted")

    lines.extend(["", "## 5. Flagged strut evidence", ""])
    if not flagged_reports:
        lines.append("No missing/broken/thin struts were flagged for citation.")
    else:
        for record in flagged_reports[:40]:
            lines.extend(
                [
                    f"### Strut `{record['strut_id']}` — `{record['class']}`",
                    "",
                    f"- Reasons: {', '.join(record.get('reasons') or []) or 'n/a'}",
                    f"- Corridor foreground fraction: "
                    f"`{(record.get('metrics') or {}).get('corridor_foreground_fraction')}`",
                    f"- Max axial gap fraction: "
                    f"`{(record.get('metrics') or {}).get('maximum_axial_gap_fraction')}`",
                    f"- Median EDT radius (vx): "
                    f"`{(record.get('metrics') or {}).get('edt_radius_median_voxels')}`",
                    "",
                ]
            )
        if len(flagged_reports) > 40:
            lines.append(f"_… {len(flagged_reports) - 40} additional flagged struts omitted_")

    lines.extend(
        [
            "",
            "## 6. Figures",
            "",
            f"- Spatial statistics: `{_rel(spatial_path)}`",
            f"- Classified lattice render: `{_rel(lattice_render_path)}`",
            "",
            "![Spatial statistics]("
            + Path(spatial_path).with_suffix(".png").name
            + ")",
            "",
            f"![Classified lattice]({Path(lattice_render_path).name})",
            "",
            "## 7. Limitations",
            "",
            "- Registration may continue after soft gate failures in hackathon mode.",
            "- Thin/bent defect agents may be deferred; report remaps deferred → present.",
            "- No sealed ground-truth labels were consulted.",
            "- Hash-sealed production orchestration is not used by this runner.",
            "- Report totals exclude nominal `y=18` crop-face missing/broken.",
            "",
            "## 8. Artifact index",
            "",
            f"- Metrics: `{_rel(Path(metrics_path))}`",
            f"- Classifications (report-ready): `{_rel(Path(classifications_path))}`",
            f"- Thresholds / policy: `{_rel(Path(thresholds_path))}`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def stage3_report(
    *,
    output_dir: Path,
    specimen_id: str,
    overwrite: bool,
    stage_status: Mapping[str, Any],
) -> dict[str, Any]:
    stage0 = output_dir / "stage0"
    stage2 = output_dir / "stage2"
    stage3 = output_dir / "stage3"
    stage3.mkdir(parents=True, exist_ok=True)

    metadata_path = stage0 / "ct_metadata.json"
    localized = stage2 / "localized_graph.json"
    metrics_path = stage2 / "per_strut_metrics.csv"
    thresholds_path = stage2 / "thresholds.json"
    report_classifications = stage2 / "classified_struts_report.json"
    if not report_classifications.is_file():
        report_classifications = stage2 / "classified_struts.json"
    for required in (metadata_path, localized, metrics_path, report_classifications, thresholds_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing Stage 2/0 artifact for report: {required}")

    spatial_json = stage3 / "spatial_statistics.json"
    spatial_png = stage3 / "spatial_statistics.png"
    lattice_png = stage3 / "lattice_3d.png"
    spatial = compute_spatial_stats(
        localized,
        report_classifications,
        metrics_path,
        spatial_json,
        spatial_png,
        overwrite=overwrite,
    )
    render = render_lattice_3d(
        localized,
        report_classifications,
        lattice_png,
        overwrite=overwrite,
    )

    classifications = json.loads(report_classifications.read_text(encoding="utf-8"))
    flagged_reports: list[dict[str, Any]] = []
    for row in classifications.get("classifications", []):
        if row.get("class") not in {"missing", "broken", "thin"}:
            continue
        flagged_reports.append(
            get_strut_report(
                int(row["strut_id"]),
                metrics_path,
                report_classifications,
                thresholds_path,
            )
        )

    registration_summary = (stage_status.get("1") or {}).get("summary") or {}
    defect_summary = {
        "gate": (stage_status.get("2") or {}).get("gate"),
        **((stage_status.get("2") or {}).get("summary") or {}),
    }
    report_path = stage3 / "nde_report.md"
    _write_nde_report(
        specimen_id=specimen_id,
        output_path=report_path,
        metadata_path=metadata_path,
        registration_summary=registration_summary,
        defect_summary=defect_summary,
        spatial_path=spatial_json,
        lattice_render_path=lattice_png,
        classifications_path=report_classifications,
        metrics_path=metrics_path,
        thresholds_path=thresholds_path,
        flagged_reports=flagged_reports,
        overwrite=overwrite,
    )

    # Keep markdown figure links local to stage3/
    return {
        "gate": "pass",
        "artifacts": {
            "spatial_statistics": _rel(spatial_json),
            "spatial_figure": _rel(spatial_png),
            "lattice_3d": _rel(lattice_png),
            "nde_report": _rel(report_path),
        },
        "summary": {
            "spatial_gate": spatial.get("gate"),
            "render_gate": render.get("gate"),
            "flagged_struts": len(flagged_reports),
            "class_counts": spatial.get("class_counts"),
        },
    }


def _parse_stages(raw: str) -> list[int]:
    stages = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in {0, 1, 2, 3}:
            raise argparse.ArgumentTypeError(f"Unsupported stage: {value}")
        stages.append(value)
    if not stages:
        raise argparse.ArgumentTypeError("At least one stage is required")
    return stages


def _resolve_threshold(output_dir: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    status = _load_status(output_dir / "pipeline_status.json")
    stage1 = status.get("stages", {}).get("1") or {}
    if stage1.get("threshold") is not None:
        return float(stage1["threshold"])
    report = output_dir / "stage1" / "registration_report.json"
    if report.is_file():
        payload = json.loads(report.read_text(encoding="utf-8"))
        details = payload.get("mode_details") or {}
        thr = details.get("threshold")
        if thr is not None:
            return float(thr)
    raise RuntimeError(
        "Could not resolve Otsu threshold; re-run stage 1 or pass --threshold"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True, help="CT .tif/.tiff/.npy path")
    parser.add_argument("--nominal", type=Path, required=True, help="Nominal lattice JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run directory under analysis/ (created if needed)",
    )
    parser.add_argument("--specimen-id", required=True)
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=[0, 1, 2, 3],
        help="Comma-separated stages to run (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--defect-agents",
        default=",".join(AVAILABLE_DEFECT_AGENTS),
        help="Comma-separated defect agents (default: all known kinds)",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="Stop if registration gate is halt (default: continue)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ct_path = args.ct.expanduser()
    if not ct_path.is_absolute():
        ct_path = (REPOSITORY_ROOT / ct_path).resolve()
    nominal_path = args.nominal.expanduser()
    if not nominal_path.is_absolute():
        nominal_path = (REPOSITORY_ROOT / nominal_path).resolve()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (REPOSITORY_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_kinds = tuple(
        part.strip() for part in str(args.defect_agents).split(",") if part.strip()
    )
    unknown = sorted(set(defect_kinds) - set(AVAILABLE_DEFECT_AGENTS))
    if unknown:
        raise SystemExit(f"Unknown defect agents: {unknown}")
    # Merge still requires all four finding docs; always run the full registry.
    defect_kinds = AVAILABLE_DEFECT_AGENTS

    status_path = output_dir / "pipeline_status.json"
    status = _load_status(status_path)
    status.setdefault("schema_version", "hackathon-pipeline-status/1.0.0")
    status["specimen_id"] = args.specimen_id
    status["ct"] = _rel(ct_path)
    status["nominal"] = _rel(nominal_path)
    status.setdefault("stages", {})
    continue_on_halt = not bool(args.strict_gates)

    runners: dict[int, Callable[[], dict[str, Any]]] = {
        0: lambda: stage0_metadata(
            ct_path=ct_path, output_dir=output_dir, overwrite=args.overwrite
        ),
        1: lambda: stage1_registration(
            ct_path=ct_path,
            nominal_path=nominal_path,
            output_dir=output_dir,
            overwrite=args.overwrite,
            continue_on_halt=continue_on_halt,
        ),
        2: lambda: stage2_defects(
            ct_path=ct_path,
            nominal_path=nominal_path,
            output_dir=output_dir,
            specimen_id=args.specimen_id,
            threshold=_resolve_threshold(output_dir, args.threshold),
            overwrite=args.overwrite,
            defect_kinds=defect_kinds,
        ),
        3: lambda: stage3_report(
            output_dir=output_dir,
            specimen_id=args.specimen_id,
            overwrite=args.overwrite,
            stage_status=status["stages"],
        ),
    }

    for stage in args.stages:
        print(f"[hackathon] running stage {stage} …", flush=True)
        result = runners[stage]()
        status["stages"][str(stage)] = {
            "completed_at": _utc_now(),
            **result,
        }
        _save_status(status_path, status, overwrite=True)
        print(
            f"[hackathon] stage {stage} done gate={result.get('gate')} "
            f"artifacts={list((result.get('artifacts') or {}).keys())}",
            flush=True,
        )

    print(f"[hackathon] status written to {_rel(status_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
