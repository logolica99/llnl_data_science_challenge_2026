"""Local CT viewer for JSON-described thin, thick, and bent strut findings.

The uploaded TIFF is streamed to temporary storage and memory-mapped. A
registered graph supplies geometry; one or more analysis JSON files supply the
strut catalog and classification metadata. Radius/deviation profiles are
computed only for selected struts.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import tifffile
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
MAX_METADATA_BYTES = 128 * 1024 * 1024


def json_bytes(payload):
    return json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")


def finite_or_none(value):
    if value in (None, ""):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def make_basis(start, end):
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError("The selected strut has zero length.")
    direction /= length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    v /= np.linalg.norm(v)
    return direction, u, v, length


def estimate_threshold(volume, maximum_samples=500_000):
    shape = np.asarray(volume.shape, dtype=int)
    stride = max(1, int(math.ceil((np.prod(shape) / maximum_samples) ** (1 / 3))))
    sample = np.asarray(volume[::stride, ::stride, ::stride], dtype=np.float64).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("The TIFF contains no finite intensity values.")
    if sample.size > maximum_samples:
        sample = sample[::int(math.ceil(sample.size / maximum_samples))]
    low, high = np.percentile(sample, [0.5, 99.8])
    if high <= low:
        return float(low)
    hist, edges = np.histogram(sample, bins=512, range=(low, high))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight0 = np.cumsum(hist)
    weight1 = hist.sum() - weight0
    mean0 = np.cumsum(hist * centers) / np.maximum(weight0, 1)
    reverse_sum = np.cumsum((hist * centers)[::-1])[::-1]
    mean1 = reverse_sum / np.maximum(weight1, 1)
    score = weight0 * weight1 * (mean0 - mean1) ** 2
    score[(weight0 <= 0) | (weight1 <= 0)] = -1
    return float(centers[int(np.argmax(score))])


def sample_plane(volume, center, offsets):
    xyz = center + offsets
    coordinates = np.vstack((
        xyz[..., 2].ravel(),
        xyz[..., 1].ravel(),
        xyz[..., 0].ravel(),
    ))
    values = ndimage.map_coordinates(
        volume,
        coordinates,
        order=1,
        mode="constant",
        cval=float(np.asarray(volume[0, 0, 0])),
    )
    return values.reshape(xyz.shape[:2])


def component_candidates(mask, axis, search_radius, spacing, maximum=4):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    yy, xx = np.indices(mask.shape)
    coord_u = axis[xx]
    coord_v = axis[yy]
    distance = np.hypot(coord_u, coord_v)
    candidates = []
    for label in range(1, count + 1):
        selected = labels == label
        if not selected.any() or float(distance[selected].min()) > search_radius:
            continue
        area_pixels = int(selected.sum())
        centroid_u = float(coord_u[selected].mean())
        centroid_v = float(coord_v[selected].mean())
        registered_distance = float(math.hypot(centroid_u, centroid_v))
        if registered_distance > search_radius + 4.0:
            continue
        boundary = selected & ~ndimage.binary_erosion(
            selected, structure=np.ones((3, 3), dtype=bool)
        )
        perimeter = max(float(boundary.sum()) * spacing, spacing)
        area = float(area_pixels) * spacing * spacing
        circularity = float(np.clip(
            4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0
        ))
        candidates.append({
            "mask": selected,
            "area_pixels": area_pixels,
            "u": centroid_u,
            "v": centroid_v,
            "offset": registered_distance,
            "circularity": circularity,
        })
    candidates.sort(key=lambda item: (
        item["offset"], -item["circularity"], -item["area_pixels"]
    ))
    return candidates[:maximum]


def consensus_centerline(candidate_planes, ignore_edges=1, maximum_offset=14.0):
    edge = min(max(int(ignore_edges), 0), max((len(candidate_planes) - 3) // 2, 0))
    stop = len(candidate_planes) - edge if edge else len(candidate_planes)
    interior = candidate_planes[edge:stop]
    required = max(3, int(math.ceil(0.70 * len(interior))))
    if sum(bool(items) for items in interior) < required:
        return None
    if any(len(items) > 1 for items in interior):
        return None
    points = [
        (index + edge, items[0])
        for index, items in enumerate(interior)
        if items
    ]
    if len(points) < required:
        return None
    offsets = [item["offset"] for _, item in points]
    if float(np.median(offsets)) > maximum_offset:
        return None
    steps = [
        math.hypot(b["u"] - a["u"], b["v"] - a["v"])
        for (_, a), (_, b) in zip(points[:-1], points[1:])
    ]
    if steps and float(np.percentile(steps, 90)) > 4.0:
        return None
    indices = np.asarray([index for index, _ in points], dtype=float)
    u = np.asarray([item["u"] for _, item in points], dtype=float)
    v = np.asarray([item["v"] for _, item in points], dtype=float)
    target = np.arange(len(candidate_planes), dtype=float)
    return np.column_stack((
        ndimage.median_filter(np.interp(target, indices, u), size=3, mode="nearest"),
        ndimage.median_filter(np.interp(target, indices, v), size=3, mode="nearest"),
    ))


def select_path(candidate_planes, expected_area, centerline=None):
    expected_area = max(float(expected_area or 1.0), 1.0)
    states_by_plane = [list(items) + [None] for items in candidate_planes]
    costs = []
    backs = []
    for plane_index, states in enumerate(states_by_plane):
        current = np.full(len(states), np.inf)
        back = np.full(len(states), -1, dtype=int)
        for state_index, state in enumerate(states):
            if state is None:
                emission = 4.5
            else:
                area_penalty = abs(math.log(max(state["area_pixels"], 1) / expected_area))
                continuity = 0.0
                if centerline is not None:
                    continuity = math.hypot(
                        state["u"] - centerline[plane_index, 0],
                        state["v"] - centerline[plane_index, 1],
                    )
                state["continuity"] = continuity
                registration_weight = 0.25 if centerline is not None else 1.0
                continuity_weight = 0.75 if centerline is not None else 0.0
                emission = (
                    registration_weight * (state["offset"] / 3.0) ** 2
                    + continuity_weight * (continuity / 2.5) ** 2
                    + 0.40 * area_penalty
                    + 0.45 * (1.0 - state["circularity"])
                )
                state["emission"] = float(emission)
            if plane_index == 0:
                current[state_index] = emission
                continue
            for prior_index, prior in enumerate(states_by_plane[plane_index - 1]):
                if state is None and prior is None:
                    transition = 0.35
                elif state is None or prior is None:
                    transition = 1.15
                else:
                    step = math.hypot(state["u"] - prior["u"], state["v"] - prior["v"])
                    area_change = abs(math.log(
                        max(state["area_pixels"], 1) / max(prior["area_pixels"], 1)
                    ))
                    transition = 0.50 * (step / 2.0) ** 2 + 0.30 * area_change
                value = costs[-1][prior_index] + transition + emission
                if value < current[state_index]:
                    current[state_index] = value
                    back[state_index] = prior_index
        costs.append(current)
        backs.append(back)
    indices = [int(np.argmin(costs[-1]))]
    for plane_index in range(len(states_by_plane) - 1, 0, -1):
        indices.append(int(backs[plane_index][indices[-1]]))
    indices.reverse()
    result = []
    for plane_index, state_index in enumerate(indices):
        state = states_by_plane[plane_index][state_index]
        if state is None:
            result.append((None, 0.0))
            continue
        distance = (
            state.get("continuity", state["offset"])
            if centerline is not None else state["offset"]
        )
        confidence = float(np.clip(math.exp(-0.015 * distance * distance), 0.0, 1.0))
        result.append((state, confidence))
    return result


def extract_profile(volume, start, end, threshold, positions=None,
                    extent=18.0, grid_size=49, search_radius=12.0):
    positions = np.asarray(
        positions if positions is not None else np.linspace(0.10, 0.90, 11),
        dtype=float,
    )
    _, u, v, length = make_basis(start, end)
    axis = np.linspace(-extent, extent, grid_size)
    spacing = float(axis[1] - axis[0])
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    offsets = uu[..., None] * u + vv[..., None] * v
    planes = []
    candidates = []
    for fraction in positions:
        center = start + fraction * (end - start)
        intensities = sample_plane(volume, center, offsets)
        planes.append(intensities)
        candidates.append(component_candidates(
            intensities >= threshold, axis, search_radius, spacing
        ))
    central_areas = [
        items[0]["area_pixels"]
        for fraction, items in zip(positions, candidates)
        if 0.25 <= fraction <= 0.75 and items
    ]
    expected_area = float(np.median(central_areas)) if central_areas else 1.0
    original = select_path(candidates, expected_area)
    interior = original[1:-1]
    required = max(3, int(math.ceil(0.70 * len(interior))))
    centerline = None
    if sum(state is not None for state, _ in interior) < required:
        centerline = consensus_centerline(candidates, ignore_edges=1)
    path = (
        select_path(candidates, expected_area, centerline)
        if centerline is not None else original
    )
    profile = []
    for fraction, state_info, items in zip(positions, path, candidates):
        state, confidence = state_info
        if state is None:
            area = radius = center_u = center_v = float("nan")
        else:
            area = float(state["area_pixels"]) * spacing * spacing
            radius = math.sqrt(area / math.pi)
            center_u = float(state["u"])
            center_v = float(state["v"])
        profile.append({
            "fraction": float(fraction),
            "distance_voxels": float(fraction * length),
            "radius_voxels": finite_or_none(radius),
            "area_voxels_squared": finite_or_none(area),
            "center_u_voxels": finite_or_none(center_u),
            "center_v_voxels": finite_or_none(center_v),
            "confidence": float(confidence),
            "candidate_count": len(items),
        })
    distances = np.asarray([item["distance_voxels"] for item in profile], dtype=float)
    centers_u = np.asarray([
        item["center_u_voxels"] if item["center_u_voxels"] is not None else np.nan
        for item in profile
    ], dtype=float)
    centers_v = np.asarray([
        item["center_v_voxels"] if item["center_v_voxels"] is not None else np.nan
        for item in profile
    ], dtype=float)
    confidence = np.asarray([item["confidence"] for item in profile], dtype=float)
    valid_center = (
        np.isfinite(centers_u) & np.isfinite(centers_v) & (confidence >= 0.45)
    )
    deviation = np.full(len(profile), np.nan, dtype=float)
    curvature = np.full(len(profile), np.nan, dtype=float)
    if valid_center.sum() >= 2:
        indices = np.flatnonzero(valid_center)
        fit_u = np.polyfit(distances[indices], centers_u[indices], 1)
        fit_v = np.polyfit(distances[indices], centers_v[indices], 1)
        deviation[indices] = np.hypot(
            centers_u[indices] - np.polyval(fit_u, distances[indices]),
            centers_v[indices] - np.polyval(fit_v, distances[indices]),
        )
    if valid_center.sum() >= 3:
        indices = np.flatnonzero(valid_center)
        degree = min(3, indices.size - 1)
        smooth_u = np.polyfit(distances[indices], centers_u[indices], degree)
        smooth_v = np.polyfit(distances[indices], centers_v[indices], degree)
        curvature[indices] = np.hypot(
            np.polyval(np.polyder(smooth_u, 2), distances[indices]),
            np.polyval(np.polyder(smooth_v, 2), distances[indices]),
        )
    for item, residual, curve in zip(profile, deviation, curvature):
        item["deviation_voxels"] = finite_or_none(residual)
        item["curvature_inverse_voxels"] = finite_or_none(curve)
    finite_radii = [
        item["radius_voxels"] for item in profile
        if item["radius_voxels"] is not None
    ]
    return {
        "length_voxels": length,
        "threshold": float(threshold),
        "extent_voxels": float(extent),
        "profile": profile,
        "coverage": float(len(finite_radii) / len(profile)),
        "median_radius_voxels": (
            float(np.median(finite_radii)) if finite_radii else None
        ),
        "centerline_deviation_rms_voxels": (
            float(np.sqrt(np.mean(deviation[np.isfinite(deviation)] ** 2)))
            if np.any(np.isfinite(deviation)) else None
        ),
        "centerline_deviation_max_voxels": (
            float(np.max(deviation[np.isfinite(deviation)]))
            if np.any(np.isfinite(deviation)) else None
        ),
        "curvature_rms_inverse_voxels": (
            float(np.sqrt(np.mean(curvature[np.isfinite(curvature)] ** 2)))
            if np.any(np.isfinite(curvature)) else None
        ),
    }


DEFECT_CLASSES = ("thin", "thick", "bent", "normal", "uncertain")


def _flatten_fields(payload, prefix=""):
    fields = {}
    for key, value in payload.items():
        if key in {
            "strut_id", "id", "profile", "sections", "measurement_profile"
        }:
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            fields.update(_flatten_fields(value, prefix=f"{name}_"))
        elif not isinstance(value, (list, tuple)):
            fields[name] = value
    return fields


def _embedded_profile(payload):
    if not isinstance(payload, dict):
        return None
    raw_samples = payload.get("samples", payload.get("profile"))
    if not isinstance(raw_samples, list):
        return None
    samples = []
    for raw in raw_samples:
        if not isinstance(raw, dict):
            continue
        samples.append({
            "sample_index": raw.get("sample_index"),
            "fraction": finite_or_none(
                raw.get("fraction", raw.get("axis_fraction"))
            ),
            "distance_voxels": finite_or_none(raw.get("distance_voxels")),
            "radius_voxels": finite_or_none(raw.get("radius_voxels")),
            "area_voxels_squared": finite_or_none(
                raw.get("area_voxels_squared")
            ),
            "center_x_voxels": finite_or_none(raw.get("center_x_voxels")),
            "center_y_voxels": finite_or_none(raw.get("center_y_voxels")),
            "center_z_voxels": finite_or_none(raw.get("center_z_voxels")),
            "center_u_voxels": finite_or_none(
                raw.get("center_u_voxels",
                        raw.get("tracked_center_u_voxels"))
            ),
            "center_v_voxels": finite_or_none(
                raw.get("center_v_voxels",
                        raw.get("tracked_center_v_voxels"))
            ),
            "deviation_voxels": finite_or_none(
                raw.get("deviation_voxels",
                        raw.get("centerline_deviation_voxels"))
            ),
            "curvature_inverse_voxels": finite_or_none(
                raw.get("curvature_inverse_voxels")
            ),
            "confidence": finite_or_none(
                raw.get("confidence", raw.get("tracking_confidence"))
            ),
            "tracking_recovered": bool(raw.get("tracking_recovered", False)),
            "valid": bool(raw.get("valid", False)),
            "exclusion_reason": str(raw.get("exclusion_reason", "")),
        })
    if not samples:
        return None
    threshold = finite_or_none(
        payload.get("ct_threshold", payload.get("threshold"))
    )
    return {
        "profile_source": "embedded_pipeline",
        "source": str(payload.get("source", "thin_thick_bent_pipeline")),
        "schema_version": payload.get("schema_version", 1),
        "threshold": threshold,
        "section_measurements_sha256": payload.get(
            "section_measurements_sha256"
        ),
        "length_voxels": finite_or_none(payload.get("length_voxels")),
        "coverage": finite_or_none(
            payload.get("tracking_coverage", payload.get("coverage"))
        ),
        "median_radius_voxels": finite_or_none(
            payload.get("median_radius_voxels")
        ),
        "centerline_deviation_rms_voxels": finite_or_none(
            payload.get("centerline_deviation_rms_voxels")
        ),
        "centerline_deviation_max_voxels": finite_or_none(
            payload.get("centerline_deviation_max_voxels")
        ),
        "curvature_rms_inverse_voxels": finite_or_none(
            payload.get("curvature_rms_inverse_voxels")
        ),
        "profile": samples,
    }


def _classification_tokens(item, hint=None):
    text = " ".join(str(item.get(key, "")) for key in (
        "classification", "defect_class", "class", "label", "decision_status"
    )).lower()
    tokens = set()
    if hint:
        text += f" {hint}"
    for defect_class in DEFECT_CLASSES:
        if defect_class in text or bool(item.get(f"is_{defect_class}", False)):
            tokens.add(defect_class)
    if any(value in tokens for value in ("thin", "thick", "bent")):
        tokens.discard("normal")
    return tokens or {"uncertain"}


def extract_result_rows(payload):
    """Return normalized strut findings plus the JSON document role."""
    if isinstance(payload, list):
        return payload, None, "findings"
    if not isinstance(payload, dict):
        raise ValueError("Analysis JSON must contain an object or array.")
    if payload.get("junctions") and payload.get("struts"):
        return [], None, "registered_graph"
    hint = payload.get("defect_class")
    for key in ("findings", "classified_struts", "entries", "results"):
        if isinstance(payload.get(key), list):
            return payload[key], hint, key
    if payload.get("stage") == "strut_measurement" and isinstance(
        payload.get("config"), dict
    ):
        return [], None, "measurement_manifest"
    if "artifacts" in payload and "status" in payload:
        return [], None, "handoff"
    if any(key in payload for key in (
        "thin_radius_ratio_max",
        "bent_adjacent_deviation_voxels",
        "radius_robust_z_threshold",
    )):
        return [], None, "thresholds"
    numeric_rows = []
    if payload and all(isinstance(value, dict) for value in payload.values()):
        for key, value in payload.items():
            if re.fullmatch(r"\d+", str(key)):
                numeric_rows.append({"strut_id": int(key), **value})
    if numeric_rows:
        return numeric_rows, hint, "id_map"
    return [], None, "metadata"


class AppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.temp_dir = None
        self.tiff_path = None
        self.volume = None
        self.threshold = None
        self.analysis_threshold = None
        self.thresholds = {}
        self.junctions = {}
        self.struts = {}
        self.result_rows = {}
        self.result_documents = []
        self.profile_cache = OrderedDict()

    def clear_tiff(self):
        volume = self.volume
        self.volume = None
        if volume is not None:
            current = volume
            seen = set()
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                mmap = getattr(current, "_mmap", None)
                if mmap is not None:
                    try:
                        mmap.close()
                    except Exception:
                        pass
                current = getattr(current, "base", None)
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = None
        self.tiff_path = None
        self.threshold = None
        self.profile_cache.clear()

    def clear(self):
        with self.lock:
            self.clear_tiff()
            self.junctions = {}
            self.struts = {}
            self.result_rows = {}
            self.result_documents = []
            self.analysis_threshold = None
            self.thresholds = {}

    def set_tiff(self, temp_dir, path, volume, threshold):
        with self.lock:
            self.clear_tiff()
            self.temp_dir = temp_dir
            self.tiff_path = path
            self.volume = volume
            self.threshold = float(threshold)

    def set_registration(self, payload):
        junctions = {
            int(item["id"]): np.asarray(item["position"], dtype=float)
            for item in payload.get("junctions", [])
        }
        struts = {int(item["id"]): dict(item) for item in payload.get("struts", [])}
        if not junctions or not struts:
            raise ValueError(
                "Registered JSON must contain non-empty junctions and struts arrays."
            )
        for strut_id, item in struts.items():
            j0 = int(item["junction0"])
            j1 = int(item["junction1"])
            if j0 not in junctions or j1 not in junctions:
                raise ValueError(
                    f"Strut {strut_id} references a missing junction."
                )
        with self.lock:
            self.junctions = junctions
            self.struts = struts
            self.profile_cache.clear()

    def set_result_json(self, payload, source_name="analysis.json"):
        rows, hint, role = extract_result_rows(payload)
        source_name = Path(str(source_name or "analysis.json")).name
        with self.lock:
            if role == "measurement_manifest":
                configured = payload.get("config", {}).get("threshold")
                if configured is not None:
                    self.analysis_threshold = float(configured)
            elif role == "thresholds":
                for key, value in payload.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        self.thresholds[str(key)] = value
            document_provenance = (
                payload.get("measurement_provenance", {})
                if isinstance(payload, dict) else {}
            )
            document_threshold = finite_or_none(
                document_provenance.get(
                    "ct_threshold", document_provenance.get("threshold")
                )
            )
            if document_threshold is not None:
                self.analysis_threshold = document_threshold

            added = 0
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                value = raw.get("strut_id", raw.get("id"))
                try:
                    strut_id = int(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{source_name} contains an invalid strut_id: {value!r}"
                    ) from exc
                classifications = _classification_tokens(raw, hint)
                fields = _flatten_fields(raw)
                embedded = _embedded_profile(raw.get("measurement_profile"))
                row = self.result_rows.setdefault(strut_id, {
                    "strut_id": strut_id,
                    "classifications": set(),
                    "fields": {},
                    "sources": [],
                    "measurement_profile": None,
                })
                existing = row.get("measurement_profile")
                if embedded is not None and existing is not None:
                    existing_hash = existing.get("section_measurements_sha256")
                    incoming_hash = embedded.get("section_measurements_sha256")
                    if (
                        existing_hash and incoming_hash
                        and existing_hash != incoming_hash
                    ):
                        raise ValueError(
                            f"Conflicting embedded measurement profiles for "
                            f"strut {strut_id}; upload findings from one "
                            f"pipeline run."
                        )
                if embedded is not None:
                    row["measurement_profile"] = embedded
                row["classifications"].update(classifications)
                row["fields"].update(fields)
                if source_name not in row["sources"]:
                    row["sources"].append(source_name)
                added += 1
            self.result_documents.append({
                "name": source_name,
                "role": role,
                "rows": added,
            })
            return {"role": role, "rows": added}

    def catalog(self):
        with self.lock:
            entries = []
            unmatched = []
            class_counts = {name: 0 for name in DEFECT_CLASSES}
            class_order = {name: index for index, name in enumerate(DEFECT_CLASSES)}
            for row in self.result_rows.values():
                strut_id = row["strut_id"]
                strut = self.struts.get(strut_id)
                if strut is None:
                    unmatched.append(strut_id)
                    continue
                start = self.junctions[int(strut["junction0"])]
                end = self.junctions[int(strut["junction1"])]
                classifications = sorted(
                    row["classifications"], key=lambda name: class_order.get(name, 99)
                )
                for name in classifications:
                    if name in class_counts:
                        class_counts[name] += 1
                confidence = row["fields"].get(
                    "confidence", row["fields"].get("classification_confidence")
                )
                entries.append({
                    "strut_id": strut_id,
                    "start_xyz": [float(value) for value in start],
                    "end_xyz": [float(value) for value in end],
                    "length_voxels": float(np.linalg.norm(end - start)),
                    "fields": row["fields"],
                    "classifications": classifications,
                    "classification": classifications[0],
                    "confidence": confidence,
                    "sources": row["sources"],
                    "has_embedded_measurements": bool(
                        row.get("measurement_profile")
                    ),
                    "measurement_source": (
                        row["measurement_profile"].get("source")
                        if row.get("measurement_profile") else None
                    ),
                })
            effective_threshold = (
                self.analysis_threshold
                if self.analysis_threshold is not None else self.threshold
            )
            return {
                "ready": bool(self.volume is not None and entries),
                "volume_shape_zyx": (
                    list(map(int, self.volume.shape))
                    if self.volume is not None else None
                ),
                "volume_dtype": (
                    str(self.volume.dtype) if self.volume is not None else None
                ),
                "threshold": effective_threshold,
                "auto_threshold": self.threshold,
                "class_counts": class_counts,
                "thresholds": self.thresholds,
                "documents": self.result_documents,
                "entries": entries,
                "unmatched_ids": sorted(unmatched),
            }

    def profile(self, strut_id, threshold):
        key = (int(strut_id), round(float(threshold), 6))
        with self.lock:
            result_row = self.result_rows.get(int(strut_id))
            if (
                result_row is not None
                and result_row.get("measurement_profile") is not None
            ):
                return result_row["measurement_profile"]
            if key in self.profile_cache:
                result = self.profile_cache.pop(key)
                self.profile_cache[key] = result
                return result
            if self.volume is None:
                raise ValueError("Upload a TIFF first.")
            if strut_id not in self.struts:
                raise KeyError(strut_id)
            item = self.struts[strut_id]
            start = self.junctions[int(item["junction0"])]
            end = self.junctions[int(item["junction1"])]
            result = extract_profile(
                self.volume, start, end, float(threshold)
            )
            result.update({
                "profile_source": "live_viewer_preview",
                "source": "standalone_strut_viewer",
                "strut_id": int(strut_id),
                "start_xyz": [float(value) for value in start],
                "end_xyz": [float(value) for value in end],
            })
            self.profile_cache[key] = result
            if len(self.profile_cache) > 128:
                self.profile_cache.popitem(last=False)
            return result

    def crop(self, strut_id, padding=24.0):
        with self.lock:
            if self.volume is None:
                raise ValueError("Upload a TIFF first.")
            item = self.struts.get(int(strut_id))
            if item is None:
                raise KeyError(strut_id)
            start = self.junctions[int(item["junction0"])]
            end = self.junctions[int(item["junction1"])]
            low = np.floor(np.minimum(start, end) - padding).astype(int)
            high = np.ceil(np.maximum(start, end) + padding + 1).astype(int)
            low = np.maximum(low, [0, 0, 0])
            high = np.minimum(
                high, [self.volume.shape[2], self.volume.shape[1], self.volume.shape[0]]
            )
            x0, y0, z0 = map(int, low)
            x1, y1, z1 = map(int, high)
            if x1 <= x0 or y1 <= y0 or z1 <= z0:
                raise ValueError("Registered strut is outside the TIFF volume.")
            source = np.asarray(self.volume[z0:z1, y0:y1, x0:x1])
            dtype_map = {
                ("u", 2): ("<u2", "uint16"),
                ("i", 2): ("<i2", "int16"),
                ("f", 4): ("<f4", "float32"),
            }
            target_dtype, dtype_name = dtype_map.get(
                (source.dtype.kind, source.dtype.itemsize), ("<f4", "float32")
            )
            crop = np.ascontiguousarray(source, dtype=target_dtype)
            sample = crop.ravel()[::max(1, crop.size // 200_000)]
            display_low, display_high = np.percentile(sample, [1.0, 99.7])
            if display_high <= display_low:
                display_low, display_high = float(sample.min()), float(sample.max())
            return {
                "body": crop.tobytes(order="C"),
                "shape": crop.shape,
                "origin": (x0, y0, z0),
                "start": start,
                "end": end,
                "range": (float(display_low), float(display_high)),
                "dtype": dtype_name,
            }


STATE = AppState()
atexit.register(STATE.clear)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(f"[strut-viewer] {self.address_string()} {fmt % args}")

    def send_json(self, payload, status=HTTPStatus.OK):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json({"error": str(message)}, status)

    def read_body(self, maximum=None):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request body is empty.")
        if maximum is not None and length > maximum:
            raise ValueError("Uploaded metadata file is too large.")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Upload ended before all bytes were received.")
        return body

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route == "/api/catalog":
                self.send_json(STATE.catalog())
                return
            if route.startswith("/api/profile/"):
                strut_id = int(route.rsplit("/", 1)[1])
                query = parse_qs(parsed.query)
                threshold = float(query.get("threshold", [STATE.threshold])[0])
                self.send_json(STATE.profile(strut_id, threshold))
                return
            if route.startswith("/api/volume/"):
                strut_id = int(route.rsplit("/", 1)[1])
                crop = STATE.crop(strut_id)
                body = crop["body"]
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Volume-Dtype", crop["dtype"])
                self.send_header("X-Volume-Shape", ",".join(map(str, crop["shape"])))
                self.send_header(
                    "X-Volume-Origin-XYZ", ",".join(map(str, crop["origin"]))
                )
                self.send_header(
                    "X-Strut-Start-XYZ", ",".join(map(str, crop["start"]))
                )
                self.send_header(
                    "X-Strut-End-XYZ", ",".join(map(str, crop["end"]))
                )
                self.send_header(
                    "X-Intensity-Range", ",".join(map(str, crop["range"]))
                )
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/":
                self.path = "/index.html"
            super().do_GET()
        except KeyError:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown strut ID.")
        except (TypeError, ValueError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, exc)
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    def do_POST(self):
        route = urlparse(self.path).path
        try:
            if route == "/api/tiff":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("TIFF upload is empty.")
                temp_dir = tempfile.mkdtemp(prefix="strut-viewer-")
                path = Path(temp_dir) / "uploaded.tiff"
                remaining = length
                try:
                    with path.open("wb") as target:
                        while remaining:
                            chunk = self.rfile.read(min(4 * 1024 * 1024, remaining))
                            if not chunk:
                                raise ValueError("TIFF upload ended early.")
                            target.write(chunk)
                            remaining -= len(chunk)
                    try:
                        volume = tifffile.memmap(path, mode="r")
                    except ValueError:
                        with tifffile.TiffFile(path) as source:
                            shape = source.series[0].shape
                            dtype = source.series[0].dtype
                        decoded_path = Path(temp_dir) / "decoded.tiff"
                        volume = tifffile.memmap(
                            decoded_path, shape=shape, dtype=dtype
                        )
                        tifffile.imread(path, out=volume)
                        volume.flush()
                    if volume.ndim != 3:
                        raise ValueError(
                            f"Expected a 3D TIFF stack, received shape {volume.shape}."
                        )
                    threshold = estimate_threshold(volume)
                    STATE.set_tiff(temp_dir, path, volume, threshold)
                except Exception:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise
                self.send_json({
                    "shape_zyx": list(map(int, volume.shape)),
                    "dtype": str(volume.dtype),
                    "threshold": threshold,
                })
                return
            if route == "/api/registration":
                body = self.read_body(MAX_METADATA_BYTES)
                STATE.set_registration(json.loads(body.decode("utf-8-sig")))
                self.send_json({
                    "junctions": len(STATE.junctions),
                    "struts": len(STATE.struts),
                })
                return
            if route == "/api/results":
                body = self.read_body(MAX_METADATA_BYTES)
                payload = json.loads(body.decode("utf-8-sig"))
                result = STATE.set_result_json(
                    payload, unquote(self.headers.get("X-File-Name", "analysis.json"))
                )
                self.send_json(result)
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route.")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            self.send_error_json(HTTPStatus.BAD_REQUEST, exc)
        except Exception as exc:
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    def do_DELETE(self):
        if urlparse(self.path).path == "/api/session":
            STATE.clear()
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error_json(HTTPStatus.NOT_FOUND, "Unknown API route.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Standalone strut viewer: http://{args.host}:{args.port}/")
    print("Uploaded TIFF data is temporary and is deleted when the server stops.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        STATE.clear()


if __name__ == "__main__":
    main()
