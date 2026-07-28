#!/usr/bin/env python3
"""Memory-aware, bounded segmentation of one x-ray CT TIFF volume."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


REQUIRED_SLICE = 380
MAX_ITERATIONS = 10


@dataclass
class Iteration:
    iteration: int
    method: str
    threshold: int
    sampled_foreground_fraction: float
    sampled_3d_components: int
    largest_3d_component_fraction: float
    small_3d_component_fraction: float
    slice_380_components: int
    largest_slice_component_fraction: float
    continuity_noise_score: float
    decision: str
    feedback_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tiff", type=Path, help="exactly one input .tif/.tiff")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def open_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
        raise ValueError("input must be one existing .tif or .tiff")
    volume = tifffile.memmap(path, mode="r")
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {volume.shape}")
    if volume.shape[0] <= REQUIRED_SLICE:
        raise ValueError(f"axis 0 has no slice {REQUIRED_SLICE}: shape={volume.shape}")
    return volume


def component_metrics(mask: np.ndarray, ndim: int) -> tuple[int, float, float]:
    structure = ndi.generate_binary_structure(ndim, ndim)
    labels, count = ndi.label(mask, structure=structure)
    foreground = int(mask.sum())
    if foreground == 0:
        return count, 0.0, 1.0
    sizes = np.bincount(labels.ravel())[1:]
    largest = float(sizes.max() / foreground) if sizes.size else 0.0
    # On the 1/8-resolution 3-D evaluation grid, <8 voxels is a tiny island.
    small = float(sizes[sizes < 8].sum() / foreground) if sizes.size else 0.0
    return int(count), largest, small


def save_feedback(
    path: Path,
    sample_values: np.ndarray,
    source_slice: np.ndarray,
    threshold: int,
    metrics: dict[str, float],
) -> None:
    preview = source_slice >= threshold
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(sample_values.ravel(), bins=256, color="0.35")
    axes[0].axvline(threshold, color="crimson", linewidth=2, label=f"T={threshold}")
    axes[0].set(title="Sampled intensity histogram", xlabel="uint16 intensity", ylabel="count")
    axes[0].legend()
    axes[1].imshow(source_slice, cmap="gray")
    axes[1].set_title("Source slice 380")
    axes[1].axis("off")
    axes[2].imshow(preview, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Candidate mask, slice 380")
    axes[2].axis("off")
    fig.suptitle(
        "fg={fg:.4f} | 3-D comps={n3:d} | largest={largest:.4f} | small={small:.4f}".format(
            **metrics
        )
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_candidate(
    iteration: int,
    threshold: int,
    sample_values: np.ndarray,
    connectivity_values: np.ndarray,
    source_slice: np.ndarray,
    feedback_dir: Path,
) -> tuple[Iteration, np.ndarray]:
    mask3 = connectivity_values >= threshold
    n3, largest3, small3 = component_metrics(mask3, 3)
    mask2 = source_slice >= threshold
    n2, largest2, _ = component_metrics(mask2, 2)
    fg = float(mask3.mean())
    score = largest3 - small3
    feedback = feedback_dir / f"iteration_{iteration:02d}.png"
    save_feedback(
        feedback,
        sample_values,
        source_slice,
        threshold,
        {"fg": fg, "n3": n3, "largest": largest3, "small": small3},
    )
    result = Iteration(
        iteration=iteration,
        method="global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement",
        threshold=int(threshold),
        sampled_foreground_fraction=fg,
        sampled_3d_components=n3,
        largest_3d_component_fraction=largest3,
        small_3d_component_fraction=small3,
        slice_380_components=n2,
        largest_slice_component_fraction=largest2,
        continuity_noise_score=score,
        decision="pending",
        feedback_path=str(feedback.resolve()),
    )
    return result, mask3


def optimize(volume: np.ndarray, out_dir: Path) -> tuple[int, list[Iteration], str, dict[str, float]]:
    # Sampling strides are based on volume dimensions, not dataset-specific answers.
    z_stride = max(1, int(np.ceil(volume.shape[0] / 32)))
    y_stride = max(1, int(np.ceil(volume.shape[1] / 210)))
    x_stride = max(1, int(np.ceil(volume.shape[2] / 210)))
    sample = np.asarray(volume[::z_stride, ::y_stride, ::x_stride])
    connectivity = np.asarray(volume[::8, ::8, ::8])
    source_slice = np.asarray(volume[REQUIRED_SLICE])

    initial = int(threshold_otsu(sample))
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample.astype(np.float32) - median)))
    step = max(1, int(round(0.25 * mad)))
    feedback_dir = out_dir / "iterations"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    history: list[Iteration] = []
    best: Iteration | None = None
    baseline_fg: float | None = None
    failed = 0
    stopping_reason = ""

    # Refinement proceeds downward only while continuity improves without adding
    # more than 5% foreground relative to Otsu. This guards against swallowing
    # the background while allowing small gaps in struts to reconnect.
    for index in range(1, MAX_ITERATIONS + 1):
        threshold = initial - (index - 1) * step
        candidate, _ = evaluate_candidate(
            index, threshold, sample, connectivity, source_slice, feedback_dir
        )
        if baseline_fg is None:
            baseline_fg = candidate.sampled_foreground_fraction
            candidate.decision = "accepted as Otsu baseline"
            best = candidate
            failed = 0
        else:
            assert best is not None
            foreground_guard = candidate.sampled_foreground_fraction <= baseline_fg * 1.05
            meaningful_gain = candidate.continuity_noise_score > best.continuity_noise_score + 1e-4
            if foreground_guard and meaningful_gain:
                candidate.decision = "accepted: continuity/noise score improved within 5% foreground guard"
                best = candidate
                failed = 0
            else:
                reasons = []
                if not foreground_guard:
                    reasons.append("rejected: foreground increase exceeded 5% guard")
                if not meaningful_gain:
                    reasons.append("rejected: no meaningful continuity/noise improvement")
                candidate.decision = "; ".join(reasons)
                failed += 1
        history.append(candidate)

        if failed >= 3:
            stopping_reason = "three consecutive failed attempts without improvement"
            break

    if not stopping_reason:
        stopping_reason = f"maximum iteration limit ({MAX_ITERATIONS}) reached"
    assert best is not None
    params = {
        "sampled_otsu_threshold": initial,
        "sample_median": median,
        "sample_mad": mad,
        "refinement_step": step,
        "final_threshold": best.threshold,
        "foreground_guard_relative_to_otsu": 1.05,
        "histogram_sample_strides": [z_stride, y_stride, x_stride],
        "connectivity_sample_strides": [8, 8, 8],
    }
    return best.threshold, history, stopping_reason, params


def write_mask(volume: np.ndarray, threshold: int, mask_path: Path) -> tuple[int, int]:
    total = int(np.prod(volume.shape, dtype=np.int64))
    bigtiff = total > (4 * 1024**3 - 64 * 1024**2)
    output = tifffile.memmap(
        mask_path,
        shape=volume.shape,
        dtype=np.uint8,
        photometric="minisblack",
        bigtiff=bigtiff,
    )
    foreground = 0
    for start in range(0, volume.shape[0], 8):
        stop = min(start + 8, volume.shape[0])
        block = np.asarray(volume[start:stop]) >= threshold
        output[start:stop] = block
        foreground += int(block.sum())
    output.flush()
    del output
    return foreground, total - foreground


def report_text(
    input_path: Path,
    volume: np.ndarray,
    threshold: int,
    foreground: int,
    background: int,
    history: list[Iteration],
    stopping_reason: str,
    params: dict[str, float],
    out_dir: Path,
) -> str:
    total = foreground + background
    lines = [
        "# CT lattice segmentation report",
        "",
        "## Input and metadata",
        "",
        f"- Input path: `{input_path}`",
        f"- Shape (axis 0, 1, 2): `{tuple(volume.shape)}`",
        f"- Input dtype: `{volume.dtype}`",
        f"- Total voxels: {total}",
        "- Ground truth: not inspected or used.",
        "",
        "## Final method and parameters",
        "",
        "A global high-density mask was initialized by Otsu thresholding on a regular 3-D intensity sample. "
        "Lower, MAD-scaled candidate thresholds were admitted only if a downsampled 3-D continuity/noise "
        "score improved while foreground stayed within 5% of the Otsu baseline. No morphology was applied, "
        "avoiding erosion or artificial thickening of thin struts.",
        "",
        f"- Final threshold rule: `input >= {threshold}`",
        f"- Parameters: `{json.dumps(params, sort_keys=True)}`",
        "",
        "## Final exact voxel statistics",
        "",
        f"- Foreground voxels: {foreground} ({100.0 * foreground / total:.6f}%)",
        f"- Background voxels: {background} ({100.0 * background / total:.6f}%)",
        "",
        "## Iteration history",
        "",
    ]
    for item in history:
        lines.extend(
            [
                f"### Iteration {item.iteration}",
                "",
                f"- Method: {item.method}",
                f"- Threshold: {item.threshold}",
                f"- Sampled foreground: {item.sampled_foreground_fraction:.6f}",
                f"- Sampled 3-D components: {item.sampled_3d_components}",
                f"- Largest sampled 3-D component / sampled foreground: {item.largest_3d_component_fraction:.6f}",
                f"- Small sampled 3-D component burden: {item.small_3d_component_fraction:.6f}",
                f"- Slice 380 components: {item.slice_380_components}",
                f"- Largest slice component / slice foreground: {item.largest_slice_component_fraction:.6f}",
                f"- Continuity/noise score: {item.continuity_noise_score:.6f}",
                f"- Decision/failure: {item.decision}",
                f"- Feedback: `{item.feedback_path}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Stopping reason",
            "",
            stopping_reason,
            "",
            "## Limitations",
            "",
            "The result is unsupervised and is not an accuracy estimate. Component statistics use an "
            "8-voxel regular subsample and can undercount diagonal or sub-resolution connections. Global "
            "thresholding may miss severe local attenuation changes; visual review of the saved slice and "
            "iteration feedback remains appropriate.",
            "",
            "## Artifacts",
            "",
            f"- Program: `{(out_dir / 'segment_ct.py').resolve()}`",
            f"- Binary mask: `{(out_dir / 'mask.tif').resolve()}`",
            f"- Mask slice 380: `{(out_dir / 'slice_380.png').resolve()}`",
            f"- Report: `{(out_dir / 'report.md').resolve()}`",
            f"- Iteration feedback directory: `{(out_dir / 'iterations').resolve()}`",
            f"- Verification record: `{(out_dir / 'verification.json').resolve()}`",
            "",
        ]
    )
    return "\n".join(lines)


def verify(input_path: Path, out_dir: Path) -> dict[str, object]:
    paths = {name: out_dir / name for name in ("segment_ct.py", "mask.tif", "slice_380.png", "report.md")}
    existence = {name: p.is_file() for name, p in paths.items()}
    if not all(existence.values()):
        raise AssertionError(f"missing required artifact: {existence}")
    source = open_volume(input_path)
    mask = tifffile.memmap(paths["mask.tif"], mode="r")
    unique: set[int] = set()
    for start in range(0, mask.shape[0], 16):
        unique.update(int(v) for v in np.unique(mask[start : start + 16]))
    mask_valid = mask.dtype == np.uint8 and mask.shape == source.shape and unique.issubset({0, 1})
    image = mpimg.imread(paths["slice_380.png"])
    nonempty = paths["slice_380.png"].stat().st_size > 0 and image.size > 0
    gray = image[..., 0] if image.ndim == 3 else image
    provenance = gray.shape == mask[REQUIRED_SLICE].shape and np.array_equal(
        gray >= 0.5, np.asarray(mask[REQUIRED_SLICE], dtype=bool)
    )
    report = paths["report.md"].read_text()
    fg_match = re.search(r"Foreground voxels: (\d+)", report)
    bg_match = re.search(r"Background voxels: (\d+)", report)
    if not fg_match or not bg_match:
        raise AssertionError("report voxel counts could not be parsed")
    reported_sum = int(fg_match.group(1)) + int(bg_match.group(1))
    count_sum_valid = reported_sum == int(np.prod(source.shape, dtype=np.int64))
    result = {
        "required_artifacts_exist": existence,
        "mask_dtype": str(mask.dtype),
        "mask_shape": list(mask.shape),
        "input_shape": list(source.shape),
        "mask_unique_values": sorted(unique),
        "mask_uint8_binary_shape_compatible": bool(mask_valid),
        "visualization_nonempty": bool(nonempty),
        "visualization_matches_mask_slice_380": bool(provenance),
        "reported_voxel_counts_sum_to_input_total": bool(count_sum_valid),
        "reported_voxel_sum": reported_sum,
    }
    if not (mask_valid and nonempty and provenance and count_sum_valid):
        raise AssertionError(json.dumps(result, indent=2))
    return result


def main() -> None:
    args = parse_args()
    input_path = args.input_tiff.resolve(strict=True)
    out_dir = input_path.parent / "segmentation"
    out_dir.mkdir(exist_ok=True)
    if args.verify_only:
        result = verify(input_path, out_dir)
        (out_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    volume = open_volume(input_path)
    threshold, history, stopping_reason, params = optimize(volume, out_dir)
    foreground, background = write_mask(volume, threshold, out_dir / "mask.tif")
    final_mask = tifffile.memmap(out_dir / "mask.tif", mode="r")
    plt.imsave(
        out_dir / "slice_380.png",
        np.asarray(final_mask[REQUIRED_SLICE]),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    (out_dir / "report.md").write_text(
        report_text(
            input_path,
            volume,
            threshold,
            foreground,
            background,
            history,
            stopping_reason,
            params,
            out_dir,
        )
    )
    result = verify(input_path, out_dir)
    (out_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "iterations": len(history),
                "stopping_reason": stopping_reason,
                "final_threshold": threshold,
                "foreground": foreground,
                "background": background,
                "verification": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
