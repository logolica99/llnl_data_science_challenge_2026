"""Extract CT cross-sections perpendicular to one registered lattice strut.

The TIFF is never rotated or rewritten. The tool samples a virtual plane through
the original TIFF, stores measured contours/radii, and creates an interactive
HTML slider for stepping through the strut.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import find_contours


def load_registered_graph(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    junctions = {int(j["id"]): np.asarray(j["position"], dtype=float)
                 for j in data["junctions"]}
    return data, junctions


def make_basis(start, end):
    direction = end - start
    length = np.linalg.norm(direction)
    if length == 0:
        raise ValueError("The selected strut has zero length.")
    direction = direction / length
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(direction, helper))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    v /= np.linalg.norm(v)
    return direction, u, v, float(length)


def sample_nearest(volume, xyz):
    xyz = np.asarray(xyz, dtype=float)
    x = np.clip(np.rint(xyz[..., 0]).astype(int), 0, volume.shape[2] - 1)
    y = np.clip(np.rint(xyz[..., 1]).astype(int), 0, volume.shape[1] - 1)
    z = np.clip(np.rint(xyz[..., 2]).astype(int), 0, volume.shape[0] - 1)
    return np.asarray(volume[z, y, x], dtype=float)


def detect_dense_boundary_limits(volume, threshold, sample_stride=8,
                                 density_threshold=0.12, consecutive=5,
                                 padding_slices=2):
    """Find dense top/bottom CT slabs that are not separable strut material."""
    fractions = np.asarray([
        np.mean(np.asarray(volume[z, ::sample_stride, ::sample_stride]) >= threshold)
        for z in range(volume.shape[0])
    ], dtype=float)
    run = int(max(1, consecutive))
    lower = 0
    for z in range(0, max(1, len(fractions) - run + 1)):
        if np.all(fractions[z:z + run] < density_threshold):
            lower = z
            break
    upper = len(fractions) - 1
    for z in range(len(fractions) - 1, run - 2, -1):
        if np.all(fractions[z - run + 1:z + 1] < density_threshold):
            upper = z
            break
    lower = min(lower + int(padding_slices), volume.shape[0] - 1)
    upper = max(upper - int(padding_slices), 0)
    return int(lower), int(upper), fractions


def choose_component(mask, center_index, center_tolerance_pixels=2.5, axis=None,
                     predicted_uv=(0.0, 0.0), expected_area_pixels=None):
    """Choose the most plausible strut component near a predicted plane center.

    ``center_tolerance_pixels`` is interpreted in voxel/axis units when ``axis``
    is supplied.  This lets a registered centerline be several voxels off
    without allowing the measurement to jump to an arbitrary neighboring
    strut.  Area consistency is a weak tie-breaker; proximity remains dominant.
    """
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    yy, xx = np.indices(mask.shape)
    if axis is None:
        coord_u = xx.astype(float) - float(center_index[1])
        coord_v = yy.astype(float) - float(center_index[0])
    else:
        coord_u = np.asarray(axis, dtype=float)[xx]
        coord_v = np.asarray(axis, dtype=float)[yy]
    predicted_u, predicted_v = map(float, predicted_uv)
    distance = np.sqrt((coord_u - predicted_u) ** 2 + (coord_v - predicted_v) ** 2)

    choices = []
    for label in range(1, count + 1):
        component = labels == label
        area_pixels = int(np.count_nonzero(component))
        nearest = float(distance[component].min())
        if nearest > float(center_tolerance_pixels):
            continue
        centroid_u = float(coord_u[component].mean())
        centroid_v = float(coord_v[component].mean())
        centroid_distance = math.hypot(centroid_u - predicted_u, centroid_v - predicted_v)
        area_penalty = 0.0
        if expected_area_pixels and expected_area_pixels > 0:
            area_penalty = 1.25 * abs(math.log(max(area_pixels, 1) / expected_area_pixels))
        # The nearest component boundary establishes eligibility. The centroid
        # and weak area term make the choice stable when several struts enter a
        # relatively wide search plane.
        score = nearest + 0.45 * centroid_distance + area_penalty
        choices.append((score, nearest, -area_pixels, label))
    if choices:
        return labels == min(choices)[3]
    return np.zeros_like(mask, dtype=bool)


def trim_merged_component(selected, axis, predicted_uv, expected_area_pixels,
                          raw_area_factor=2.4):
    """Extract the cylindrical core when a junction merges several branches.

    Junction planes often form one connected foreground blob, so connected
    component area and centroid are not reliable. A distance-transform seed
    close to the predicted path is used to retain a bounded local core.
    """
    raw_area = int(np.count_nonzero(selected))
    if not raw_area or not expected_area_pixels or raw_area <= raw_area_factor * expected_area_pixels:
        return selected, raw_area, False

    yy, xx = np.indices(selected.shape)
    coord_u = np.asarray(axis, dtype=float)[xx]
    coord_v = np.asarray(axis, dtype=float)[yy]
    predicted_u, predicted_v = map(float, predicted_uv)
    distance_to_prediction = np.hypot(coord_u - predicted_u, coord_v - predicted_v)
    distance_inside = ndimage.distance_transform_edt(selected)
    score = distance_inside - 0.35 * distance_to_prediction / max(abs(axis[1] - axis[0]), 1e-6)
    score[~selected] = -np.inf
    seed_y, seed_x = np.unravel_index(int(np.argmax(score)), selected.shape)
    seed_u = float(axis[seed_x])
    seed_v = float(axis[seed_y])
    expected_radius_pixels = math.sqrt(max(expected_area_pixels, 1.0) / math.pi)
    core_radius_voxels = max(
        2.0,
        1.45 * expected_radius_pixels * abs(float(axis[1] - axis[0])),
    )
    core_disk = np.hypot(coord_u - seed_u, coord_v - seed_v) <= core_radius_voxels
    core = selected & core_disk
    return core if np.count_nonzero(core) >= 3 else selected, raw_area, True


def enumerate_component_candidates(mask, axis, search_radius, spacing, max_candidates=5,
                                   search_uv=(0.0, 0.0),
                                   allow_distant_centroid=False):
    """Return plausible components near a registered or CT-predicted center."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    yy_grid, xx_grid = np.indices(mask.shape)
    coord_u = np.asarray(axis, dtype=float)[xx_grid]
    coord_v = np.asarray(axis, dtype=float)[yy_grid]
    registered_distance = np.hypot(coord_u, coord_v)
    search_u, search_v = map(float, search_uv)
    search_distance = np.hypot(coord_u - search_u, coord_v - search_v)
    candidates = []
    for label in range(1, count + 1):
        component = labels == label
        nearest = float(search_distance[component].min())
        if nearest > search_radius:
            continue
        area_pixels = int(component.sum())
        centroid_u = float(coord_u[component].mean())
        centroid_v = float(coord_v[component].mean())
        centroid_distance = math.hypot(centroid_u - search_u, centroid_v - search_v)
        if (
            not allow_distant_centroid
            and centroid_distance > search_radius + 3.0
        ):
            continue
        boundary = component & ~ndimage.binary_erosion(
            component, structure=np.ones((3, 3), dtype=bool)
        )
        perimeter = max(float(boundary.sum()) * spacing, spacing)
        area = float(area_pixels) * spacing * spacing
        circularity = float(np.clip(4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0))
        candidates.append({
            "selected": component,
            "area_pixels": area_pixels,
            "centroid_u": centroid_u,
            "centroid_v": centroid_v,
            "registered_distance": float(math.hypot(centroid_u, centroid_v)),
            "nearest_registered_distance": float(registered_distance[component].min()),
            "search_distance": centroid_distance,
            "circularity": circularity,
        })
    # Recovery searches are centered on an already-established CT path rather
    # than the registered origin. Rank those candidates by the CT prediction;
    # otherwise a nearer registered neighbor can crowd the target out of the
    # bounded candidate list.
    candidates.sort(key=lambda item: (
        item["search_distance"]
        if search_uv != (0.0, 0.0)
        else item["registered_distance"]
        + 0.35 * item["nearest_registered_distance"],
        -item["circularity"],
    ))
    return candidates[:max_candidates]


def consensus_ct_centerline(candidate_planes, maximum_step=2.75,
                            maximum_offset=8.75, ignore_edge_sections=1):
    """Seed a CT path when registration offset makes the first path all-null.

    A registration error is shared by the sections of one strut. Charging that
    same offset as independent evidence on every plane can make the null state
    cheaper than a clearly repeated component. This fallback looks for one
    transverse component cluster supported across most non-empty planes, then
    interpolates its smooth centerline. It does not widen candidate generation,
    so components still have to intersect the original registration search tube.
    """
    edge = min(
        max(int(ignore_edge_sections), 0),
        max((len(candidate_planes) - 3) // 2, 0),
    )
    interior_start = edge
    interior_stop = len(candidate_planes) - edge if edge else len(candidate_planes)
    interior_planes = candidate_planes[interior_start:interior_stop]
    nonempty = sum(bool(candidates) for candidates in interior_planes)
    required_planes = max(3, int(math.ceil(0.70 * len(interior_planes))))
    if nonempty < required_planes:
        return None
    # This fallback exists for the specific all-null/one-visible-component
    # failure. Multiple candidates are genuinely ambiguous and remain under
    # the original registration-constrained decision.
    # Junction planes commonly contain several connected branches. They are
    # excluded from defect measurements, so ambiguity there must not veto an
    # otherwise unique and continuous interior component.
    if any(len(candidates) > 1 for candidates in interior_planes):
        return None
    anchors = [
        (candidate["centroid_u"], candidate["centroid_v"])
        for candidates in interior_planes
        for candidate in candidates
    ]
    best = None
    for anchor_u, anchor_v in anchors:
        matches = []
        for plane_index, candidates in enumerate(
            interior_planes, start=interior_start
        ):
            if not candidates:
                continue
            candidate = min(
                candidates,
                key=lambda item: math.hypot(
                    item["centroid_u"] - anchor_u,
                    item["centroid_v"] - anchor_v,
                ),
            )
            distance = math.hypot(
                candidate["centroid_u"] - anchor_u,
                candidate["centroid_v"] - anchor_v,
            )
            if distance <= maximum_step:
                matches.append((plane_index, candidate, distance))
        if not matches:
            continue
        areas = np.asarray(
            [item[1]["area_pixels"] for item in matches], dtype=float
        )
        area_mad = float(np.median(np.abs(
            np.log(np.maximum(areas, 1.0)) -
            np.median(np.log(np.maximum(areas, 1.0)))
        )))
        score = (
            len(matches),
            -float(np.median([item[2] for item in matches])),
            -area_mad,
            -float(np.median([
                item[1]["registered_distance"] for item in matches
            ])),
        )
        if best is None or score > best[0]:
            best = (score, matches)
    if best is None or len(best[1]) < required_planes:
        return None
    median_offset = float(np.median([
        item[1]["registered_distance"] for item in best[1]
    ]))
    if median_offset > maximum_offset:
        return None
    indices = np.asarray([item[0] for item in best[1]], dtype=float)
    centers_u = np.asarray(
        [item[1]["centroid_u"] for item in best[1]], dtype=float
    )
    centers_v = np.asarray(
        [item[1]["centroid_v"] for item in best[1]], dtype=float
    )
    target = np.arange(len(candidate_planes), dtype=float)
    u = np.interp(target, indices, centers_u)
    v = np.interp(target, indices, centers_v)
    filter_size = min(5, len(candidate_planes))
    return np.column_stack((
        ndimage.median_filter(u, size=filter_size, mode="nearest"),
        ndimage.median_filter(v, size=filter_size, mode="nearest"),
    ))


def select_registration_constrained_path(candidate_planes, expected_area_pixels,
                                         ct_centerline=None):
    """Choose a smooth path, using registration first and CT continuity second."""
    if not candidate_planes:
        return []
    expected_area = max(float(expected_area_pixels or 1.0), 1.0)
    all_states = []
    for plane_index, candidates in enumerate(candidate_planes):
        states = list(candidates) + [None]
        for candidate in candidates:
            area_penalty = abs(math.log(max(candidate["area_pixels"], 1) / expected_area))
            continuity_distance = 0.0
            if ct_centerline is not None:
                continuity_distance = math.hypot(
                    candidate["centroid_u"] - float(ct_centerline[plane_index, 0]),
                    candidate["centroid_v"] - float(ct_centerline[plane_index, 1]),
                )
            candidate["continuity_distance"] = float(continuity_distance)
            registration_weight = 0.35 if ct_centerline is not None else 1.0
            continuity_weight = 0.65 if ct_centerline is not None else 0.0
            candidate["emission_cost"] = (
                # Keep the JSON line as a soft spatial prior, while letting a
                # continuous CT path win after the first pass establishes it.
                registration_weight * (candidate["registered_distance"] / 2.5) ** 2 +
                continuity_weight * (continuity_distance / 2.5) ** 2 +
                0.45 * area_penalty +
                0.55 * (1.0 - candidate["circularity"])
            )
        all_states.append(states)

    costs = []
    backpointers = []
    for plane_index, states in enumerate(all_states):
        current_costs = np.full(len(states), np.inf, dtype=float)
        current_back = np.full(len(states), -1, dtype=int)
        for state_index, state in enumerate(states):
            emission = 4.5 if state is None else float(state["emission_cost"])
            if plane_index == 0:
                current_costs[state_index] = emission
                continue
            for prior_index, prior in enumerate(all_states[plane_index - 1]):
                if state is None and prior is None:
                    transition = 0.35
                elif state is None or prior is None:
                    transition = 1.15
                else:
                    step = math.hypot(
                        state["centroid_u"] - prior["centroid_u"],
                        state["centroid_v"] - prior["centroid_v"],
                    )
                    area_change = abs(math.log(
                        max(state["area_pixels"], 1) / max(prior["area_pixels"], 1)
                    ))
                    transition = 0.55 * (step / 2.0) ** 2 + 0.35 * area_change
                value = costs[-1][prior_index] + transition + emission
                if value < current_costs[state_index]:
                    current_costs[state_index] = value
                    current_back[state_index] = prior_index
        costs.append(current_costs)
        backpointers.append(current_back)

    selected_indices = [int(np.argmin(costs[-1]))]
    for plane_index in range(len(all_states) - 1, 0, -1):
        selected_indices.append(int(backpointers[plane_index][selected_indices[-1]]))
    selected_indices.reverse()
    path = []
    for plane_index, state_index in enumerate(selected_indices):
        state = all_states[plane_index][state_index]
        if state is None:
            path.append({"candidate": None, "confidence": 0.0})
            continue
        alternatives = [
            float(other["emission_cost"])
            for other in all_states[plane_index]
            if other is not state and other is not None
        ]
        second = (
            min(alternatives)
            if alternatives else float(state["emission_cost"]) + 2.5
        )
        margin = second - float(state["emission_cost"])
        confidence_distance = (
            float(state.get("continuity_distance", state["registered_distance"]))
            if ct_centerline is not None else float(state["registered_distance"])
        )
        confidence = (
            1.0 / (1.0 + math.exp(-margin)) *
            # Once a CT path is established, confidence should measure support
            # for that path rather than punish its legitimate registration
            # offset from the JSON reference line.
            math.exp(-0.015 * confidence_distance ** 2)
        )
        path.append({
            "candidate": state,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
        })
    return path


def recover_bounded_path_gaps(selected_path, sampled_masks, axis, spacing,
                              expected_area_pixels, search_radius,
                              maximum_gap_planes=2, maximum_prediction_error=4.5,
                              ambiguity_margin=0.75):
    """Recover short missing runs only when flanking CT tracking agrees.

    The primary path remains registration constrained.  This second pass is
    intentionally narrower in purpose: when one or two interior planes were
    assigned to the null state despite visible material, interpolate the CT
    center between confident tracked neighbors and search around that CT
    prediction.  A recovery is rejected when competing components have similar
    scores, preserving ``uncertain`` for genuinely ambiguous geometry.
    """
    recovered = [dict(item) for item in selected_path]
    if len(recovered) < 3 or not expected_area_pixels:
        return recovered

    index = 1
    while index < len(recovered) - 1:
        if recovered[index]["candidate"] is not None:
            index += 1
            continue
        start = index
        while index < len(recovered) and recovered[index]["candidate"] is None:
            index += 1
        stop = index
        gap = stop - start
        if (
            gap > int(maximum_gap_planes)
            or start == 0
            or stop >= len(recovered)
            or recovered[start - 1]["candidate"] is None
            or recovered[stop]["candidate"] is None
        ):
            continue

        left = recovered[start - 1]
        right = recovered[stop]
        left_candidate = left["candidate"]
        right_candidate = right["candidate"]
        span = stop - (start - 1)
        expected_du = (
            right_candidate["centroid_u"] - left_candidate["centroid_u"]
        ) / span
        expected_dv = (
            right_candidate["centroid_v"] - left_candidate["centroid_v"]
        ) / span
        plane_options = []
        for plane_index in range(start, stop):
            fraction = (plane_index - (start - 1)) / span
            predicted_u = (
                (1.0 - fraction) * left_candidate["centroid_u"]
                + fraction * right_candidate["centroid_u"]
            )
            predicted_v = (
                (1.0 - fraction) * left_candidate["centroid_v"]
                + fraction * right_candidate["centroid_v"]
            )
            candidates = enumerate_component_candidates(
                sampled_masks[plane_index],
                axis,
                max(float(search_radius), float(maximum_prediction_error)),
                spacing,
                max_candidates=8,
                search_uv=(predicted_u, predicted_v),
                allow_distant_centroid=True,
            )
            scored = []
            for candidate in candidates:
                # Threshold connectivity can temporarily merge the target with
                # nearby lattice material. Isolate a bounded core around the CT
                # prediction before rejecting the component as implausibly
                # large. This core is used only for a short, flank-supported
                # bridge; unconstrained or boundary gaps remain missing.
                if (
                    candidate["area_pixels"]
                    > 1.8 * max(expected_area_pixels, 1.0)
                ):
                    core, _, was_trimmed = trim_merged_component(
                        candidate["selected"],
                        axis,
                        (predicted_u, predicted_v),
                        expected_area_pixels,
                        raw_area_factor=1.8,
                    )
                    if was_trimmed and np.count_nonzero(core) >= 3:
                        yy, xx = np.nonzero(core)
                        candidate = dict(candidate)
                        candidate["selected"] = core
                        candidate["area_pixels"] = int(core.sum())
                        candidate["centroid_u"] = float(axis[xx].mean())
                        candidate["centroid_v"] = float(axis[yy].mean())
                        candidate["registered_distance"] = float(math.hypot(
                            candidate["centroid_u"], candidate["centroid_v"]
                        ))
                        boundary = core & ~ndimage.binary_erosion(
                            core, structure=np.ones((3, 3), dtype=bool)
                        )
                        perimeter = max(
                            float(boundary.sum()) * spacing, spacing
                        )
                        area = float(core.sum()) * spacing * spacing
                        candidate["circularity"] = float(np.clip(
                            4.0 * math.pi * area / (perimeter * perimeter),
                            0.0,
                            1.0,
                        ))
                        candidate["recovery_core_trimmed"] = True
                prediction_error = math.hypot(
                    candidate["centroid_u"] - predicted_u,
                    candidate["centroid_v"] - predicted_v,
                )
                if prediction_error > maximum_prediction_error:
                    continue
                area_ratio = (
                    candidate["area_pixels"] / max(expected_area_pixels, 1.0)
                )
                if not 0.40 <= area_ratio <= 2.50:
                    continue
                area_penalty = abs(math.log(max(area_ratio, 1e-9)))
                score = (
                    prediction_error
                    + 0.70 * area_penalty
                    + 0.35 * (1.0 - candidate["circularity"])
                )
                scored.append((float(score), prediction_error, area_penalty, candidate))
            scored.sort(key=lambda item: item[0])
            if not scored:
                plane_options = []
                break
            plane_options.append(scored)

        if len(plane_options) != gap:
            continue

        # Score complete bridges rather than demanding an unambiguous winner
        # on every plane. A true strut can be locally ambiguous for one slice
        # while still forming the only smooth, area-consistent path between
        # two confident flanks.
        bridges = []
        for combination in itertools.product(*plane_options):
            points = [left_candidate] + [
                item[3] for item in combination
            ] + [right_candidate]
            cost = sum(item[0] for item in combination)
            for prior, current in zip(points, points[1:]):
                step_error = math.hypot(
                    (current["centroid_u"] - prior["centroid_u"]) - expected_du,
                    (current["centroid_v"] - prior["centroid_v"]) - expected_dv,
                )
                area_change = abs(math.log(
                    max(current["area_pixels"], 1.0)
                    / max(prior["area_pixels"], 1.0)
                ))
                cost += 0.55 * step_error + 0.25 * area_change
            bridges.append((float(cost), combination))
        bridges.sort(key=lambda item: item[0])
        bridge_margin = (
            bridges[1][0] - bridges[0][0] if len(bridges) > 1 else 2.5
        )
        if bridge_margin < ambiguity_margin:
            continue

        best = bridges[0][1]
        maximum_error = max(item[1] for item in best)
        mean_area_penalty = float(np.mean([item[2] for item in best]))
        flank_confidence = min(
            float(left.get("confidence", 0.0)),
            float(right.get("confidence", 0.0)),
        )
        recovery_confidence = (
            flank_confidence
            * math.exp(-0.05 * maximum_error ** 2)
            * math.exp(-0.15 * mean_area_penalty)
            * min(1.0, bridge_margin)
        )
        if recovery_confidence < 0.45:
            continue
        for plane_index, item in zip(range(start, stop), best):
            recovered[plane_index] = {
                "candidate": item[3],
                "confidence": float(np.clip(
                    recovery_confidence, 0.0, 1.0
                )),
                "recovered": True,
            }
    return recovered


def _longest_true_run(values):
    longest = run = 0
    for value in values:
        run = run + 1 if bool(value) else 0
        longest = max(longest, run)
    return int(longest)


def add_registration_metrics(sections):
    """Fit the smooth transverse registration correction for one strut."""
    distance = np.asarray([s["distance_voxels"] for s in sections], dtype=float)
    u = np.asarray([s["centroid_u_voxels"] for s in sections], dtype=float)
    v = np.asarray([s["centroid_v_voxels"] for s in sections], dtype=float)
    eligible = np.asarray(
        [s.get("measurement_eligible", True) for s in sections], dtype=bool
    )
    confidence = np.asarray(
        [s.get("tracking_confidence", 1.0) for s in sections], dtype=float
    )
    valid = (
        np.isfinite(distance) & np.isfinite(u) & np.isfinite(v) & eligible &
        (confidence >= 0.45)
    )
    fit_u = np.full(len(sections), np.nan, dtype=float)
    fit_v = np.full(len(sections), np.nan, dtype=float)
    residual = np.full(len(sections), np.nan, dtype=float)

    if valid.sum() >= 2:
        idx = np.flatnonzero(valid)
        keep = np.ones(idx.size, dtype=bool)
        for _ in range(3):
            degree = 1 if keep.sum() >= 2 else 0
            pu = np.polyfit(distance[idx][keep], u[idx][keep], degree)
            pv = np.polyfit(distance[idx][keep], v[idx][keep], degree)
            predicted_u = np.polyval(pu, distance[idx])
            predicted_v = np.polyval(pv, distance[idx])
            errors = np.hypot(u[idx] - predicted_u, v[idx] - predicted_v)
            median = float(np.median(errors))
            mad = float(np.median(np.abs(errors - median)))
            limit = max(1.25, median + 3.0 * 1.4826 * mad)
            updated = errors <= limit
            if updated.sum() < 2 or np.array_equal(updated, keep):
                break
            keep = updated
        fit_u[:] = np.polyval(pu, distance)
        fit_v[:] = np.polyval(pv, distance)
        residual[idx] = np.hypot(u[idx] - fit_u[idx], v[idx] - fit_v[idx])
    elif valid.sum() == 1:
        fit_u[:] = u[valid][0]
        fit_v[:] = v[valid][0]
        residual[valid] = 0.0

    offsets = np.hypot(u[valid], v[valid])
    finite_residual = residual[np.isfinite(residual)]
    confidences = np.asarray(
        [s.get("tracking_confidence", float("nan")) for s in sections],
        dtype=float,
    )
    # Coverage already records how many eligible planes produced a trustworthy
    # tracked center.  Average confidence over those supported planes only;
    # including null-state zeros here double-penalizes the same missing sample
    # and can make an otherwise well-supported path fail both gates.
    eligible_confidences = confidences[valid]
    for i, section in enumerate(sections):
        section["registered_center_u_voxels"] = 0.0
        section["registered_center_v_voxels"] = 0.0
        section["tracked_center_u_voxels"] = section["centroid_u_voxels"]
        section["tracked_center_v_voxels"] = section["centroid_v_voxels"]
        section["registration_fit_u_voxels"] = float(fit_u[i])
        section["registration_fit_v_voxels"] = float(fit_v[i])
        section["registration_residual_voxels"] = float(residual[i])
        section["registered_offset_voxels"] = (
            float(math.hypot(section["centroid_u_voxels"], section["centroid_v_voxels"]))
            if np.isfinite(section["centroid_u_voxels"]) else float("nan")
        )
    return {
        "tracking_coverage": (
            float(valid.sum() / max(int(eligible.sum()), 1)) if len(valid) else 0.0
        ),
        "median_registration_offset_voxels": float(np.median(offsets)) if offsets.size else float("nan"),
        "registration_offset_variation_voxels": (
            float(np.sqrt(np.mean(finite_residual * finite_residual)))
            if finite_residual.size else float("nan")
        ),
        "registration_fit_u_start_voxels": float(fit_u[0]) if fit_u.size else float("nan"),
        "registration_fit_v_start_voxels": float(fit_v[0]) if fit_v.size else float("nan"),
        "registration_fit_u_end_voxels": float(fit_u[-1]) if fit_u.size else float("nan"),
        "registration_fit_v_end_voxels": float(fit_v[-1]) if fit_v.size else float("nan"),
        "mean_tracking_confidence": (
            float(np.mean(eligible_confidences))
            if eligible_confidences.size else 0.0
        ),
    }


def add_curvature_metrics(sections, ignore_edge_sections=0):
    """Estimate centerline curvature from the measured section-center path.

    This is a screening metric in inverse-voxel units, not a paper-calibrated
    pass/fail threshold. The ideal registered strut is straight, so curvature
    comes from changes in the CT-measured section centroid.
    """
    distance = np.asarray([s["distance_voxels"] for s in sections], dtype=float)
    u = np.asarray([s["centroid_u_voxels"] for s in sections], dtype=float)
    v = np.asarray([s["centroid_v_voxels"] for s in sections], dtype=float)
    eligible = np.asarray(
        [s.get("measurement_eligible", True) for s in sections], dtype=bool
    )
    confidence = np.asarray(
        [s.get("tracking_confidence", 1.0) for s in sections], dtype=float
    )
    valid = (
        np.isfinite(distance) & np.isfinite(u) & np.isfinite(v) & eligible &
        (confidence >= 0.45)
    )
    curvature = np.full(len(sections), np.nan, dtype=float)
    if valid.sum() >= 3:
        idx = np.flatnonzero(valid)
        # Polynomial smoothing prevents half-voxel centroid quantization from
        # becoming a high-curvature false positive.
        degree = min(3, idx.size - 1)
        pu = np.polyfit(distance[idx], u[idx], degree)
        pv = np.polyfit(distance[idx], v[idx], degree)
        d2u = np.polyval(np.polyder(pu, 2), distance[idx])
        d2v = np.polyval(np.polyder(pv, 2), distance[idx])
        curvature[idx] = np.hypot(d2u, d2v)
    for section, value in zip(sections, curvature):
        section["curvature_inverse_voxels"] = float(value)
    edge = int(max(0, ignore_edge_sections))
    screened = curvature[edge:len(curvature) - edge] if edge and len(curvature) > 2 * edge else curvature
    interior = screened[np.isfinite(screened)]
    rms = float(np.sqrt(np.mean(interior * interior))) if interior.size else float("nan")
    return rms


def classify_strut(sections, rms_curvature, ideal_diameter_voxels=None,
                   curvature_threshold=0.02, radius_tolerance=0.20,
                   ignore_edge_sections=1, dross_ratio_threshold=1.75,
                   broken_ratio_threshold=0.15, min_broken_sections=2,
                   missing_plane_fraction_threshold=0.05,
                   registration_metrics=None, dross_robust_z_threshold=3.5,
                   min_dross_sections=2):
    edge = int(max(0, ignore_edge_sections))
    topology_available = any("measurement_eligible" in s for s in sections)
    topology_eligible = [s for s in sections if s.get("measurement_eligible", False)]
    eligible_sections = (
        topology_eligible if topology_available else
        (sections[edge:len(sections) - edge]
         if edge and len(sections) > 2 * edge else sections)
    )
    eligible_ids = {id(section) for section in eligible_sections}
    for section in sections:
        section["defect_eligible"] = id(section) in eligible_ids
    radii = np.asarray([s["equivalent_radius_voxels"] for s in eligible_sections], dtype=float)
    material_sections = eligible_sections
    if not material_sections:
        material_sections = (
            sections[edge:len(sections) - edge]
            if edge and len(sections) > 2 * edge else sections
        )
    plane_material = np.asarray(
        [s["plane_material_fraction"] for s in material_sections], dtype=float
    )
    circularity = np.asarray(
        [s.get("cross_section_circularity", float("nan")) for s in eligible_sections],
        dtype=float,
    )
    excess_area = np.asarray(
        [s.get("excess_area_fraction", float("nan")) for s in eligible_sections],
        dtype=float,
    )
    radial_cv = np.asarray(
        [s.get("contour_radius_cv", float("nan")) for s in eligible_sections],
        dtype=float,
    )
    valid = radii[np.isfinite(radii)]
    median_radius = float(np.median(valid)) if valid.size else float("nan")
    radius_mad = (
        float(np.median(np.abs(valid - median_radius))) if valid.size else float("nan")
    )
    robust_sigma = (
        max(1.4826 * radius_mad, 0.10 * median_radius, 0.20)
        if valid.size and median_radius > 0 else float("nan")
    )
    result = "normal"
    reasons = []
    robust_z = (
        (radii - median_radius) / robust_sigma
        if np.isfinite(robust_sigma) and robust_sigma > 0
        else np.full_like(radii, np.nan)
    )
    if np.isfinite(median_radius) and median_radius > 0:
        morphology_outlier = (
            (np.isfinite(circularity) & (circularity < 0.72)) |
            (np.isfinite(excess_area) & (excess_area >= 0.30)) |
            (np.isfinite(radial_cv) & (radial_cv >= 0.28))
        )
        strong_radius_outlier = (
            np.isfinite(robust_z) &
            (robust_z >= dross_robust_z_threshold) &
            (radii >= 1.35 * median_radius)
        )
        morphology_supported_outlier = (
            np.isfinite(robust_z) & (robust_z >= 2.75) &
            (radii >= 1.25 * median_radius) & morphology_outlier
        )
        spike_mask = strong_radius_outlier | morphology_supported_outlier
    else:
        spike_mask = np.zeros_like(radii, dtype=bool)
    longest_spike_run = _longest_true_run(spike_mask)
    for section in sections:
        section["radius_robust_z"] = float("nan")
        section["radius_outlier"] = False
    for section, z_score, is_spike in zip(eligible_sections, robust_z, spike_mask):
        section["radius_robust_z"] = float(z_score)
        section["radius_outlier"] = bool(is_spike)
    empty_sections = int(np.count_nonzero(~np.isfinite(radii) | (radii <= 0)))
    empty_mask = ~np.isfinite(radii) | (radii <= 0)
    longest_empty_run = _longest_true_run(empty_mask)
    if not np.isfinite(median_radius):
        if any(s.get("dense_boundary_interference", False) for s in sections):
            result = "uncertain"
            reasons.append(
                "dense boundary/build-plate material prevents a reliable interior measurement"
            )
        elif plane_material.size and float(np.nanmax(plane_material)) < missing_plane_fraction_threshold:
            result = "potentially_missing"
            reasons.append("no locally tracked component and very little material in any interior plane")
        else:
            result = "uncertain"
            reasons.append("no locally tracked component, but other material is present in the search planes")
    elif ideal_diameter_voxels is not None and ideal_diameter_voxels > 0:
        ideal_radius = ideal_diameter_voxels / 2.0
        ratio = median_radius / ideal_radius
        if ratio < 1.0 - radius_tolerance:
            result = "potentially_thin"
            reasons.append("median radius is below the ideal-radius tolerance")
        elif ratio > 1.0 + radius_tolerance:
            result = "potentially_thick"
            reasons.append("median radius is above the ideal-radius tolerance")
    if np.isfinite(rms_curvature) and rms_curvature > curvature_threshold:
        result = "potentially_bent" if result == "normal" else result + "+bent"
        reasons.append("RMS curvature exceeds the provisional screening threshold")
    # Robust radius outliers must persist across adjacent topology-safe planes.
    # A single spike is more commonly a partial-volume or branch-contamination
    # artifact than true local thickening.
    if longest_spike_run >= min_dross_sections:
        result = "potentially_dross_or_local_thickening" if result == "normal" else result + "+thick_spike"
        reasons.append(
            "multiple adjacent junction-safe sections show robust excess area or irregular local thickening"
        )
    low_mask = (
        radii < broken_ratio_threshold * median_radius
        if valid.size else np.zeros_like(radii, dtype=bool)
    )
    longest_low_run = _longest_true_run(low_mask)
    required_missing = len(eligible_sections)
    boundary_interference = any(
        s.get("dense_boundary_interference", False) for s in sections
    )
    if (not boundary_interference and empty_sections >= required_missing and
            plane_material.size and
            float(np.nanmax(plane_material)) < missing_plane_fraction_threshold):
        if result in ("normal", "uncertain"):
            result = "potentially_missing"
        reasons.append("all eligible sections lack locally tracked material and the planes are mostly empty")
    elif longest_empty_run >= min_broken_sections:
        if result == "normal":
            result = "potentially_broken_or_missing"
        reasons.append("consecutive interior sections have no locally tracked material")
    elif longest_low_run >= min_broken_sections:
        if result == "normal":
            result = "potentially_broken_or_missing"
        reasons.append("consecutive/recurring interior sections have very little tracked material")

    registration_metrics = registration_metrics or {}
    tracking_coverage = float(registration_metrics.get("tracking_coverage", 0.0))
    offset = float(registration_metrics.get("median_registration_offset_voxels", float("nan")))
    variation = float(registration_metrics.get("registration_offset_variation_voxels", float("nan")))
    centerline_residual = np.asarray([
        section.get("registration_residual_voxels", float("nan"))
        for section in eligible_sections
    ], dtype=float)
    localized_bend_mask = (
        np.isfinite(centerline_residual) & (centerline_residual >= 0.75)
    )
    localized_bend = (
        np.any(np.isfinite(centerline_residual)) and
        float(np.nanmax(centerline_residual)) >= 1.50 and
        _longest_true_run(localized_bend_mask) >= 2
    )
    if localized_bend and result == "normal":
        result = "potentially_bent"
        reasons.append(
            "adjacent interior sections deviate from the best-fit straight CT centerline"
        )
    if result == "normal" and tracking_coverage < 0.55:
        result = "uncertain"
        reasons.append("too few planes support a stable locally tracked centerline")
    if np.isfinite(offset) and offset >= 2.0 and tracking_coverage >= 0.55:
        reasons.append(
            "material is continuous after local recentering; registered centerline is laterally offset"
            if result == "normal" else
            "registered centerline has a measurable lateral offset"
        )
    if result == "normal" and np.isfinite(variation) and variation > 2.0:
        result = "uncertain"
        reasons.append("local centerline varies too much for a confident defect decision")
    return result, reasons, median_radius


def extract_cross_sections(volume, start, end, threshold, positions, extent, grid_size,
                           ignore_edge_sections=1, center_tolerance_pixels=2.5,
                           tracking_radius_voxels=None, valid_z_range=None):
    direction, u, v, length = make_basis(start, end)
    axis = np.linspace(-extent, extent, grid_size)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    sample_offsets = uu[..., None] * u + vv[..., None] * v
    spacing = float(axis[1] - axis[0])
    positions = np.asarray(positions, dtype=float)
    sampled_planes = []
    section_centers = []
    for position in positions:
        center = start + position * (end - start)
        section_centers.append(center)
        xyz = center + sample_offsets
        intensities = sample_nearest(
            volume, xyz.reshape(-1, 3)
        ).reshape(grid_size, grid_size)
        sampled_planes.append((intensities, intensities >= threshold))

    search_radius = float(
        tracking_radius_voxels if tracking_radius_voxels is not None
        else center_tolerance_pixels
    )
    candidate_planes = []
    for center, (_, mask) in zip(section_centers, sampled_planes):
        boundary_interference = (
            valid_z_range is not None and
            not (valid_z_range[0] <= center[2] <= valid_z_range[1])
        )
        candidate_planes.append(
            [] if boundary_interference else
            enumerate_component_candidates(mask, axis, search_radius, spacing)
        )
    central_areas = [
        candidates[0]["area_pixels"]
        for position, candidates in zip(positions, candidate_planes)
        if 0.25 <= position <= 0.75 and candidates
    ]
    expected_area_pixels = (
        float(np.median(central_areas)) if central_areas else None
    )
    original_path = select_registration_constrained_path(
        candidate_planes, expected_area_pixels
    )
    original_support = sum(
        item["candidate"] is not None for item in original_path
    )
    edge = min(
        max(int(ignore_edge_sections), 0),
        max((len(candidate_planes) - 3) // 2, 0),
    )
    interior_stop = len(candidate_planes) - edge if edge else len(candidate_planes)
    interior_path = original_path[edge:interior_stop]
    required_interior_support = max(
        3, int(math.ceil(0.70 * len(interior_path)))
    )
    ct_centerline = (
        consensus_ct_centerline(
            candidate_planes,
            ignore_edge_sections=ignore_edge_sections,
        )
        if (
            original_support < 3 or
            sum(item["candidate"] is not None for item in interior_path)
            < required_interior_support
        )
        else None
    )
    selected_path = (
        select_registration_constrained_path(
            candidate_planes, expected_area_pixels, ct_centerline=ct_centerline
        )
        if ct_centerline is not None else original_path
    )
    selected_path = recover_bounded_path_gaps(
        selected_path,
        [mask for _, mask in sampled_planes],
        axis,
        spacing,
        expected_area_pixels,
        search_radius,
    )
    tracked = []
    for path_item in selected_path:
        candidate = path_item["candidate"]
        if candidate is None:
            tracked.append({
                "selected": np.zeros((grid_size, grid_size), dtype=bool),
                "raw_component_area_pixels": 0,
                "junction_contaminated": False,
                "tracking_confidence": 0.0,
                "candidate_count": 0,
                "tracking_recovered": False,
            })
            continue
        selected, raw_area, contaminated = trim_merged_component(
            candidate["selected"], axis,
            (candidate["centroid_u"], candidate["centroid_v"]),
            expected_area_pixels,
        )
        tracked.append({
            "selected": selected,
            "raw_component_area_pixels": int(raw_area),
            "junction_contaminated": bool(contaminated),
            "tracking_confidence": float(path_item["confidence"]),
            "candidate_count": 0,
            "tracking_recovered": bool(path_item.get("recovered", False)),
        })
    for index, candidates in enumerate(candidate_planes):
        if index < len(tracked):
            tracked[index]["candidate_count"] = len(candidates)

    sections = []
    for index, position in enumerate(positions):
        intensities, mask = sampled_planes[index]
        record = tracked[index] or {
            "selected": np.zeros_like(mask, dtype=bool),
            "raw_component_area_pixels": 0,
            "junction_contaminated": False,
            "tracking_confidence": 0.0,
            "candidate_count": 0,
            "tracking_recovered": False,
        }
        selected = record["selected"]
        area = float(selected.sum() * spacing * spacing)
        radius = math.sqrt(area / math.pi) if area else float("nan")
        if selected.any():
            yy, xx = np.nonzero(selected)
            centroid_u = float(axis[xx].mean())
            centroid_v = float(axis[yy].mean())
            contour_list = find_contours(selected.astype(float), 0.5)
            contour = max(contour_list, key=len) if contour_list else np.empty((0, 2))
            contour_uv = [[float(np.interp(point[1], np.arange(grid_size), axis)),
                           float(np.interp(point[0], np.arange(grid_size), axis))]
                          for point in contour]
            contour_array = np.asarray(contour_uv, dtype=float)
            radial = (np.sqrt(((contour_array - np.array([centroid_u, centroid_v])) ** 2).sum(axis=1))
                      if contour_uv else np.array([]))
            radius_min = float(radial.min()) if radial.size else float("nan")
            radius_max = float(radial.max()) if radial.size else float("nan")
            radius_mean = float(radial.mean()) if radial.size else float("nan")
            radial_cv = (
                float(np.std(radial) / max(np.mean(radial), 1e-6))
                if radial.size else float("nan")
            )
            boundary = selected & ~ndimage.binary_erosion(
                selected, structure=np.ones((3, 3), dtype=bool)
            )
            perimeter = max(float(boundary.sum()) * spacing, spacing)
            circularity = float(np.clip(
                4.0 * math.pi * area / (perimeter * perimeter), 0.0, 1.0
            ))
        else:
            centroid_u = centroid_v = float("nan")
            contour_uv = []
            radius_min = radius_max = radius_mean = float("nan")
            radial_cv = circularity = float("nan")
        sections.append({
            "position_fraction": float(position),
            "distance_voxels": float(position * length),
            "area_voxels_squared": area,
            "equivalent_radius_voxels": radius,
            "contour_radius_min_voxels": radius_min,
            "contour_radius_mean_voxels": radius_mean,
            "contour_radius_max_voxels": radius_max,
            "contour_radius_cv": radial_cv,
            "cross_section_circularity": circularity,
            "centroid_u_voxels": centroid_u,
            "centroid_v_voxels": centroid_v,
            "foreground_fraction": float(selected.mean()),
            "plane_material_fraction": float(mask.mean()),
            "raw_component_area_pixels": record["raw_component_area_pixels"],
            "junction_contaminated": record["junction_contaminated"],
            "tracking_confidence": record["tracking_confidence"],
            "tracking_candidate_count": record["candidate_count"],
            "tracking_recovered": record["tracking_recovered"],
            "dense_boundary_interference": bool(
                valid_z_range is not None and
                not (valid_z_range[0] <= section_centers[index][2] <= valid_z_range[1])
            ),
            "contour_uv": contour_uv,
            "intensity_min": float(intensities.min()),
            "intensity_max": float(intensities.max()),
        })

    usable_radii = np.asarray([
        section["equivalent_radius_voxels"]
        if (0.30 <= section["position_fraction"] <= 0.70 and
            not section["junction_contaminated"])
        else float("nan")
        for section in sections
    ])
    usable_radii = usable_radii[np.isfinite(usable_radii)]
    if not usable_radii.size:
        usable_radii = np.asarray([
            section["equivalent_radius_voxels"]
            for section in sections
            if (np.isfinite(section["equivalent_radius_voxels"]) and
                not section["junction_contaminated"])
        ])
    baseline_radius = (
        float(np.median(usable_radii)) if usable_radii.size else float("nan")
    )
    node_exclusion_distance = (
        max(3.0 * baseline_radius, 0.15 * length)
        if np.isfinite(baseline_radius) else 0.15 * length
    )
    node_exclusion_fraction = min(
        0.25, node_exclusion_distance / max(length, 1e-6)
    )
    for section in sections:
        position = section["position_fraction"]
        junction_excluded = (
            position <= node_exclusion_fraction or
            position >= 1.0 - node_exclusion_fraction
        )
        section["junction_excluded"] = bool(junction_excluded)
        section["measurement_eligible"] = bool(
            not junction_excluded and not section["junction_contaminated"] and
            not section["dense_boundary_interference"]
        )
        section["node_exclusion_fraction"] = float(node_exclusion_fraction)
        section["node_exclusion_distance_voxels"] = float(node_exclusion_distance)
        section["interior_baseline_radius_voxels"] = float(baseline_radius)
        baseline_area = (
            math.pi * baseline_radius * baseline_radius
            if np.isfinite(baseline_radius) else float("nan")
        )
        section["excess_area_fraction"] = (
            float(max(section["area_voxels_squared"] - baseline_area, 0.0) /
                  max(baseline_area, 1e-6))
            if np.isfinite(baseline_area) else float("nan")
        )

    registration_metrics = add_registration_metrics(sections)
    rms_curvature = add_curvature_metrics(sections, ignore_edge_sections)
    for section in sections:
        section["tracking_search_radius_voxels"] = search_radius
    # Preserve the historical three-value return contract. Aggregate tracking
    # metrics are attached to every section for callers that need them.
    for section in sections:
        section.update(registration_metrics)
    return sections, length, rms_curvature


def write_html(path, sections, strut_id, threshold, extent, summary):
    payload = json.dumps(sections, separators=(",", ":"))
    template = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Strut STRUT_ID cross-section viewer</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}canvas{border:1px solid #bbb;max-width:100%;height:auto}.row{display:flex;gap:24px;flex-wrap:wrap}label{display:block;margin:8px 0}#metrics{font-family:monospace;margin:10px 0}</style></head>
<body><h1>Strut STRUT_ID cross-section viewer</h1>
<p>Original TIFF is unchanged. The plane is sampled perpendicular to the registered strut axis.</p>
<label>Position along strut: <input id="pos" type="range" min="0" max="COUNT_MINUS_ONE" value="0"><span id="posLabel"></span></label>
<div id="metrics"></div><div class="row"><canvas id="contour" width="520" height="520"></canvas><canvas id="overlay" width="520" height="520"></canvas><canvas id="profile" width="720" height="360"></canvas></div>
<script>
const sections=SECTIONS; const summary=SUMMARY; const extent=EXTENT; const pos=document.getElementById('pos'); const posLabel=document.getElementById('posLabel'); const metrics=document.getElementById('metrics'); const contour=document.getElementById('contour'); const overlay=document.getElementById('overlay'); const profile=document.getElementById('profile');
function path(c,s,cx,cy,scale){c.beginPath();(s.contour_uv||[]).forEach((q,j)=>{let x=cx+q[0]*scale,y=cy-q[1]*scale;if(j)c.lineTo(x,y);else c.moveTo(x,y)});c.closePath();c.stroke()}
function draw(){const i=Number(pos.value),s=sections[i],cx=260,cy=260,scale=220/extent;const deviation=Number.isFinite(s.radius_deviation_fraction)?(s.radius_deviation_fraction*100).toFixed(1)+'%':'N/A (provide ideal diameter in voxels)';posLabel.textContent=(s.position_fraction*100).toFixed(0)+'%';metrics.textContent='classification='+summary.classification+' | distance='+s.distance_voxels.toFixed(1)+' voxels | radius='+s.equivalent_radius_voxels.toFixed(2)+' voxels | curvature='+s.curvature_inverse_voxels.toFixed(5)+' 1/voxel | radius deviation='+deviation;let c=contour.getContext('2d');c.clearRect(0,0,520,520);c.strokeStyle='#9b59b6';c.lineWidth=3;path(c,s,cx,cy,scale);c.fillStyle='#e67e22';c.beginPath();c.arc(cx+s.centroid_u_voxels*scale,cy-s.centroid_v_voxels*scale,4,0,2*Math.PI);c.fill();c.strokeStyle='#e67e22';c.beginPath();c.moveTo(cx+s.centroid_u_voxels*scale,cy-s.centroid_v_voxels*scale);c.lineTo(cx+(s.centroid_u_voxels+s.equivalent_radius_voxels)*scale,cy-s.centroid_v_voxels*scale);c.stroke();c.fillStyle='#222';c.fillText('actual CT contour; orange = measured center/radius',16,24);let o=overlay.getContext('2d');o.clearRect(0,0,520,520);o.strokeStyle='rgba(52,152,219,.30)';o.lineWidth=2;sections.forEach(x=>path(o,x,cx,cy,scale));o.strokeStyle='#e74c3c';o.lineWidth=4;path(o,s,cx,cy,scale);o.fillStyle='#222';o.fillText('all interior contours overlaid',16,24);let p=profile.getContext('2d');p.clearRect(0,0,720,360);p.strokeStyle='#777';p.beginPath();p.moveTo(45,20);p.lineTo(45,320);p.lineTo(700,320);p.stroke();p.strokeStyle='#2980b9';p.lineWidth=2;p.beginPath();sections.forEach((x,j)=>{let xx=45+j/(sections.length-1)*655,yy=320-(x.equivalent_radius_voxels/(extent||1))*280;if(j)p.lineTo(xx,yy);else p.moveTo(xx,yy)});p.stroke();p.fillStyle='#222';p.fillText('radius profile along strut',50,16);p.fillText('position',640,345);p.save();p.translate(12,220);p.rotate(-Math.PI/2);p.fillText('radius (voxels)',0,0);p.restore()}
pos.addEventListener('input',draw);draw();
</script></body></html>"""
    html = (template.replace("STRUT_ID", str(strut_id))
            .replace("COUNT_MINUS_ONE", str(max(0, len(sections) - 1)))
            .replace("SECTIONS", payload)
            .replace("SUMMARY", json.dumps(summary, separators=(",", ":")))
            .replace("EXTENT", str(extent)))
    path.write_text(html, encoding="utf-8")


def write_pngs(outdir, sections, strut_id, extent):
    positions = [s["position_fraction"] for s in sections]
    radii = [s["equivalent_radius_voxels"] for s in sections]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    curvature = [s["curvature_inverse_voxels"] for s in sections]
    fig, (ax, curvature_ax) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax.plot(np.asarray(positions) * 100, radii, marker="o", color="tab:blue")
    ax.set_xlabel("Position along strut (%)")
    ax.set_ylabel("Equivalent radius (voxels)")
    ax.set_title(f"CT-measured radius profile for strut {strut_id}")
    ax.grid(alpha=0.25)
    curvature_ax.plot(np.asarray(positions) * 100, curvature, marker="o", color="tab:orange")
    curvature_ax.set_xlabel("Position along strut (%)")
    curvature_ax.set_ylabel("Curvature (1/voxel)")
    curvature_ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "radius_profile.png", dpi=160)
    plt.close(fig)

    cols = 5
    rows = int(math.ceil(len(sections) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.5 * rows), squeeze=False)
    for ax, section in zip(axes.ravel(), sections):
        contour = np.asarray(section["contour_uv"], dtype=float)
        if contour.size:
            ax.plot(contour[:, 0], contour[:, 1], color="tab:purple")
        ax.set_title(f"{section['position_fraction'] * 100:.0f}%")
        ax.set_aspect("equal")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
    for ax in axes.ravel()[len(sections):]:
        ax.axis("off")
    fig.suptitle(f"CT cross-sections for strut {strut_id}")
    fig.tight_layout()
    fig.savefig(outdir / "cross_sections.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for section in sections:
        contour = np.asarray(section["contour_uv"], dtype=float)
        if contour.size:
            ax.plot(contour[:, 0], contour[:, 1], color="tab:blue", alpha=0.22, linewidth=1.2)
    ax.set_title(f"Interior CT contours overlaid — strut {strut_id}")
    ax.set_xlabel("u (voxels)")
    ax.set_ylabel("v (voxels)")
    ax.set_aspect("equal")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "cross_sections_overlay.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("registered_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--strut-id", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=40129.0)
    parser.add_argument("--positions", type=int, default=19)
    parser.add_argument("--start-fraction", type=float, default=0.10,
                        help="First sampling position along the strut (0 to 1).")
    parser.add_argument("--end-fraction", type=float, default=0.90,
                        help="Last sampling position along the strut (0 to 1).")
    parser.add_argument("--ignore-edge-sections", type=int, default=1,
                        help="Number of sampled sections at each end excluded from defect decisions.")
    parser.add_argument("--extent-pixels", type=float, default=9.0)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--ideal-diameter-voxels", type=float, default=None,
                        help="Optional ideal diameter in TIFF voxel units for radius deviation.")
    parser.add_argument("--curvature-threshold", type=float, default=0.02,
                        help="Provisional RMS-curvature screening threshold (1/voxel).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, junctions = load_registered_graph(args.registered_json)
    strut = next(s for s in data["struts"] if int(s["id"]) == args.strut_id)
    start = junctions[int(strut["junction0"])]
    end = junctions[int(strut["junction1"])]
    volume = tifffile.memmap(args.input_tiff, mode="r")
    if not (0.0 <= args.start_fraction < args.end_fraction <= 1.0):
        parser.error("Require 0 <= start-fraction < end-fraction <= 1.")
    positions = np.linspace(args.start_fraction, args.end_fraction, args.positions)
    sections, length, rms_curvature = extract_cross_sections(
        volume, start, end, args.threshold, positions, args.extent_pixels, args.grid_size,
        args.ignore_edge_sections
    )
    ideal_diameter = args.ideal_diameter_voxels
    ideal_radius = ideal_diameter / 2.0 if ideal_diameter else None
    for section in sections:
        section["ideal_radius_voxels"] = ideal_radius
        section["radius_deviation_fraction"] = ((section["equivalent_radius_voxels"] - ideal_radius) / ideal_radius
                                                 if ideal_radius and np.isfinite(section["equivalent_radius_voxels"])
                                                 else float("nan"))
    classification, reasons, median_radius = classify_strut(
        sections, rms_curvature, ideal_diameter, args.curvature_threshold,
        ignore_edge_sections=args.ignore_edge_sections,
        registration_metrics={
            key: sections[0][key] for key in (
                "tracking_coverage",
                "median_registration_offset_voxels",
                "registration_offset_variation_voxels",
            )
        },
    )
    summary = {
        "classification": classification,
        "classification_reasons": reasons,
        "rms_curvature_inverse_voxels": rms_curvature,
        "median_interior_radius_voxels": median_radius,
        "ideal_diameter_voxels": ideal_diameter,
        "ideal_radius_voxels": ideal_radius,
        "curvature_threshold_inverse_voxels": args.curvature_threshold,
        "tracking_coverage": sections[0]["tracking_coverage"],
        "median_registration_offset_voxels": sections[0]["median_registration_offset_voxels"],
        "registration_offset_variation_voxels": sections[0]["registration_offset_variation_voxels"],
        "defect_sampling_range_fraction": [args.start_fraction, args.end_fraction],
        "ignored_edge_sections_each_end": args.ignore_edge_sections,
        "registered_json_thickness": strut.get("thickness"),
        "registered_json_thickness_note": (
            "Stored for traceability. It is not converted to voxel radius unless "
            "--ideal-diameter-voxels is supplied because the JSON thickness unit "
            "may differ from the TIFF coordinate unit."
        ),
    }
    fields = [k for k in sections[0] if k != "contour_uv"]
    with (args.output_dir / "radius_profile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: s[k] for k in fields} for s in sections)
    (args.output_dir / "cross_section_data.json").write_text(json.dumps({
        "strut_id": args.strut_id,
        "start_xyz": start.tolist(),
        "end_xyz": end.tolist(),
        "length_voxels": length,
        "threshold": args.threshold,
        "plane_extent_pixels": args.extent_pixels,
        "summary": summary,
        "sections": sections,
    }, indent=2), encoding="utf-8")
    write_html(args.output_dir / "cross_section_viewer.html", sections, args.strut_id, args.threshold, args.extent_pixels, summary)
    write_pngs(args.output_dir, sections, args.strut_id, args.extent_pixels)
    print(f"Wrote cross-section demo for strut {args.strut_id} to {args.output_dir}")


if __name__ == "__main__":
    main()
