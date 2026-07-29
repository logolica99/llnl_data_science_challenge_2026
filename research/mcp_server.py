"""Disabled-by-default MCP surface for isolated research and legacy exercises."""

from __future__ import annotations

from contextlib import redirect_stdout
import math
from pathlib import Path
import sys
from typing import Any

from fastmcp import FastMCP
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llnl_nde.core.otsu import replay_exact_otsu, write_otsu_artifacts  # noqa: E402
from llnl_nde.core.defect_analysis import (  # noqa: E402
    export_stage3_validation_csvs as _export_stage3_validation_csvs,
)
from llnl_nde.core.segmentation import (  # noqa: E402
    compare_segmentation_masks,
    segment_ct_dataset,
    visualize_slice,
)
from research.evaluation import compute_detection_metrics as _compute_detection_metrics  # noqa: E402
from research.skeletonization import skeletonize_mask  # noqa: E402
from research.volume_artifacts import (  # noqa: E402
    render_volume_3d as _render_volume_3d,
    summarize_nde_artifacts as _summarize_nde_artifacts,
)


mcp = FastMCP("LLNL Research Tools")
RESEARCH_ROOT = REPOSITORY_ROOT / "research"
RESEARCH_RUNS_ROOT = RESEARCH_ROOT / "runs"
RESEARCH_RESPONSE_SCHEMA_VERSION = "research-mcp-response/1.0.0"


def _repository_input(
    filepath: str,
    *,
    suffixes: set[str],
    research_copy: bool = False,
) -> Path:
    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {resolved}") from exc
    if research_copy:
        try:
            resolved.relative_to(RESEARCH_ROOT)
        except ValueError as exc:
            raise ValueError(
                "Labeled evaluation inputs must be copied under research/ before use"
            ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Research input does not exist: {resolved}")
    if resolved.suffix.lower() not in suffixes:
        raise ValueError(f"Unexpected research input suffix: {resolved.suffix}")
    return resolved


def _research_output(filepath: str, *, suffix: str | None = None) -> Path:
    candidate = Path(filepath).expanduser()
    resolved = (
        (REPOSITORY_ROOT / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        resolved.relative_to(RESEARCH_RUNS_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Research outputs must remain under {RESEARCH_RUNS_ROOT.relative_to(REPOSITORY_ROOT)}"
        ) from exc
    if suffix and resolved.suffix.lower() != suffix:
        raise ValueError(f"Research output must use {suffix}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(REPOSITORY_ROOT).as_posix()


def _research_response(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_schema_version": RESEARCH_RESPONSE_SCHEMA_VERSION,
        "tool": tool,
        "status": "ok",
        "research_only": True,
        "production_manifest_mutation_allowed": False,
        "result": result,
    }


@mcp.tool()
def compute_detection_metrics(
    classifications_filepath: str,
    sealed_labels_filepath: str,
    output_filepath: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Score copied frozen outputs against copied labels in research/runs only."""

    classifications = _repository_input(
        classifications_filepath, suffixes={".json"}, research_copy=True
    )
    labels = _repository_input(
        sealed_labels_filepath, suffixes={".json"}, research_copy=True
    )
    output = _research_output(output_filepath, suffix=".json")
    result = _compute_detection_metrics(
        classifications, labels, output, overwrite=overwrite
    )
    result["artifacts"]["detection_metrics"]["path"] = _relative(output)
    return _research_response("compute_detection_metrics", result)


@mcp.tool()
def export_stage3_validation_csvs(
    classifications_filepath: str,
    missing_findings_filepath: str,
    broken_findings_filepath: str,
    metrics_filepath: str,
    nominal_graph_filepath: str,
    output_directory: str,
    excluded_nominal_axis: str = "y",
    excluded_nominal_value: float = 18.0,
    coordinate_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Export non-authoritative Stage 3 validation CSVs under research/runs."""

    classifications = _repository_input(
        classifications_filepath, suffixes={".json"}, research_copy=True
    )
    missing = _repository_input(
        missing_findings_filepath, suffixes={".json"}, research_copy=True
    )
    broken = _repository_input(
        broken_findings_filepath, suffixes={".json"}, research_copy=True
    )
    metrics = _repository_input(
        metrics_filepath, suffixes={".csv"}, research_copy=True
    )
    nominal_graph = _repository_input(
        nominal_graph_filepath, suffixes={".json"}, research_copy=True
    )
    output = _research_output(output_directory)
    result = _export_stage3_validation_csvs(
        classifications,
        missing,
        broken,
        metrics,
        nominal_graph,
        output,
        excluded_nominal_axis=excluded_nominal_axis,
        excluded_nominal_value=excluded_nominal_value,
        coordinate_tolerance=coordinate_tolerance,
    )
    for artifact in result["artifacts"].values():
        artifact["path"] = _relative(artifact["path"])
    return _research_response("export_stage3_validation_csvs", result)


@mcp.tool()
def explore_ct_thresholds(
    input_filepath: str,
    output_directory: str,
    threshold_offsets: list[float],
    slice_index: int | None = None,
) -> dict[str, Any]:
    """Run a bounded Otsu-centered threshold comparison outside production."""

    source = _repository_input(
        input_filepath, suffixes={".npy", ".tif", ".tiff"}
    )
    output = _research_output(output_directory)
    if not 1 <= len(threshold_offsets) <= 9:
        raise ValueError("Provide between one and nine explicit threshold offsets")
    offsets = [float(value) for value in threshold_offsets]
    if any(not math.isfinite(value) for value in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("Threshold offsets must be finite and unique")
    otsu, histogram = replay_exact_otsu(source)
    otsu_artifacts = write_otsu_artifacts(output / "otsu", otsu, histogram)
    base = float(otsu["threshold"])
    thresholds = [base + offset for offset in offsets]
    masks: list[Path] = []
    slices: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        mask = output / "masks" / f"candidate_{index:02d}.npy"
        segment_ct_dataset(
            source,
            mask,
            threshold=threshold,
            registration_mode="autonomous_v2",
            retention="regenerable",
        )
        masks.append(mask)
        loaded = np.load(mask, mmap_mode="r", allow_pickle=False)
        selected_slice = int(slice_index) if slice_index is not None else loaded.shape[0] // 2
        del loaded
        rendered = visualize_slice(
            mask,
            output / "slices" / f"candidate_{index:02d}.png",
            slice_index=selected_slice,
            axis=0,
            registration_mode="autonomous_v2",
        )
        slices.append(
            {
                "threshold": threshold,
                "path": _relative(rendered["artifacts"]["slice"]["path"]),
                "sha256": rendered["artifacts"]["slice"]["sha256"],
            }
        )
    comparison = compare_segmentation_masks(
        source,
        masks,
        thresholds,
        registration_mode="autonomous_v2",
        output_report_path=output / "threshold_comparison.json",
        repository_root=REPOSITORY_ROOT,
    )
    return _research_response(
        "explore_ct_thresholds",
        {
            "otsu_threshold": base,
            "thresholds": thresholds,
            "candidate_count": len(thresholds),
            "comparison": comparison["result"],
            "slices": slices,
            "artifacts": {
                "otsu_report": {
                    **otsu_artifacts["report"],
                    "path": _relative(otsu_artifacts["report"]["path"]),
                },
                "comparison_report": {
                    **comparison["artifacts"]["comparison_report"],
                    "path": _relative(
                        comparison["artifacts"]["comparison_report"]["path"]
                    ),
                },
            },
        },
    )


@mcp.tool()
def summarize_nde_artifacts(
    raw_filepath: str,
    mask_filepath: str,
    skeleton_filepath: str | None = None,
) -> dict[str, Any]:
    """Run the legacy voxel/mask/skeleton summary on research inputs."""

    raw = _repository_input(raw_filepath, suffixes={".npy"})
    mask = _repository_input(mask_filepath, suffixes={".npy"})
    skeleton = (
        _repository_input(skeleton_filepath, suffixes={".npy"})
        if skeleton_filepath
        else None
    )
    result = _summarize_nde_artifacts(
        str(raw), str(mask), str(skeleton) if skeleton else None
    )
    return _research_response("summarize_nde_artifacts", result)


@mcp.tool()
def render_volume_3d(
    input_filepath: str,
    output_filepath: str,
    surface_level: float = 0.5,
    downsample_factor: int = 2,
    elevation: float = 30.0,
    azimuth: float = 45.0,
    skeleton_filepath: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the legacy volume isosurface renderer into research/runs only."""

    source = _repository_input(input_filepath, suffixes={".npy"})
    skeleton = (
        _repository_input(skeleton_filepath, suffixes={".npy"})
        if skeleton_filepath
        else None
    )
    output = _research_output(output_filepath, suffix=".png")
    result = _render_volume_3d(
        input_filepath=str(source),
        output_filepath=str(output),
        surface_level=surface_level,
        downsample_factor=downsample_factor,
        elevation=elevation,
        azimuth=azimuth,
        skeleton_filepath=str(skeleton) if skeleton else None,
        overwrite=overwrite,
    )
    result["output_path"] = _relative(output)
    return _research_response("render_volume_3d", result)


@mcp.tool()
def skeletonize(
    input_filepath: str,
    output_filepath: str,
) -> dict[str, Any]:
    """Run the legacy skeletonizer into research/runs only."""

    source = _repository_input(input_filepath, suffixes={".npy"})
    output = _research_output(output_filepath, suffix=".npy")
    if output.exists():
        raise FileExistsError(f"Research artifact already exists: {output}")
    with redirect_stdout(sys.stderr):
        skeleton = skeletonize_mask(str(source), str(output))
    if skeleton is None or not output.is_file():
        raise RuntimeError("Skeletonization did not produce an output")
    return _research_response(
        "skeletonize",
        {
            "output_path": _relative(output),
            "skeleton_voxels": int(np.count_nonzero(skeleton)),
        },
    )


if __name__ == "__main__":
    RESEARCH_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    mcp.run()
