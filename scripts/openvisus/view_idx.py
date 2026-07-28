#!/usr/bin/env python3
"""Visualize an OpenVisus IDX / TIFF volume.

Modes:
  orthoslices  – axis-aligned XY/XZ/YZ cuts with sliders
  plane111     – fast 2D (111) oblique cut with depth slider (corner view)
  3d           – full isosurface (slower)
  openvisus    – native Qt viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_local_pkgs() -> None:
    """Use workspace wheels only on Python 3.11; strip them otherwise.

    Users often export ``PYTHONPATH=.python_pkgs``. Those wheels are cp311 and
    will break base/miniconda Python 3.14 (numpy ImportError / segfaults).
    """
    pkgs = (_repo_root() / ".python_pkgs").resolve()
    pkgs_s = str(pkgs)

    if sys.version_info[:2] != (3, 11):
        # Drop incompatible path entries inserted via PYTHONPATH.
        sys.path[:] = [
            p
            for p in sys.path
            if Path(p).resolve().as_posix() != pkgs_s
            and not p.startswith(pkgs_s + "/")
        ]
        return

    if pkgs.is_dir() and pkgs_s not in sys.path:
        sys.path.insert(0, pkgs_s)


def _ensure_openvisus() -> None:
    _ensure_local_pkgs()
    pkgs = _repo_root() / ".python_pkgs"
    # Local wheels are cp311; loading them on other Pythons segfaults.
    if pkgs.is_dir() and (pkgs / "OpenVisus").is_dir() and sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"OpenVisus in .python_pkgs requires Python 3.11 "
            f"(you are on {sys.version.split()[0]}).\n\n"
            "Run:\n"
            "  conda activate dssi_env\n"
            "  PYTHONPATH=.python_pkgs python data/missing_struts/openvisus/view_idx.py "
            "--mode 3d --downsample 4\n"
        )
    try:
        import OpenVisus  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "OpenVisus is not installed. From the repo root run:\n"
            "  conda activate dssi_env\n"
            "  pip install --target .python_pkgs OpenVisus"
        ) from exc


def _read_volume(idx_path: Path, downsample: int):
    from OpenVisus import LoadDataset
    import numpy as np

    db = LoadDataset(str(idx_path))
    data = db.read()
    if not isinstance(data, np.ndarray):
        raise RuntimeError("OpenVisus read() did not return a NumPy array")
    # OpenVisus returns (Z, Y, X) for this volume layout.
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
    return data, db


def _read_volume_from_tiff(tiff_path: Path, downsample: int):
    """Fallback volume load without OpenVisus (uses tifffile)."""
    import tifffile
    import numpy as np

    data = tifffile.imread(tiff_path)
    if data.ndim != 3:
        raise SystemExit(f"Expected 3D TIFF, got shape {data.shape}")
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
    return np.asarray(data), None

def _load_volume(
    idx_path: Path,
    downsample: int,
    tiff_path: Path | None = None,
    *,
    prefer_tiff: bool = False,
):
    """Load ZYX volume from IDX when possible, else TIFF.

    If prefer_tiff=True and tiff_path exists, load the TIFF even when IDX is present
    (used when the user explicitly passed --tiff).
    """
    _ensure_local_pkgs()
    pkgs = _repo_root() / ".python_pkgs"
    can_use_ov = not (
        pkgs.is_dir()
        and (pkgs / "OpenVisus").is_dir()
        and sys.version_info[:2] != (3, 11)
    )

    if prefer_tiff and tiff_path is not None and tiff_path.is_file():
        print(f"Reading TIFF:\n  {tiff_path}")
        return _read_volume_from_tiff(tiff_path, downsample)

    if can_use_ov and idx_path.is_file():
        try:
            _ensure_openvisus()
            return _read_volume(idx_path, downsample)
        except SystemExit as exc:
            print(exc)

    if tiff_path is None or not tiff_path.is_file():
        raise SystemExit(
            "Could not load IDX/OpenVisus and no readable --tiff was provided.\n"
            "Pass --tiff /path/to/volume.tif  (or activate dssi_env for IDX)."
        )
    print(f"Reading TIFF:\n  {tiff_path}")
    return _read_volume_from_tiff(tiff_path, downsample)


def oblique_slice(
    volume,
    offset: float,
    normal=(1.0, 1.0, 1.0),
    resolution: int = 512,
):
    """Sample a 2D plane through ``volume`` (Z,Y,X).

    ``offset`` in [0, 1] slides the plane along its normal from the
    near corner of the volume to the far corner — for normal (1,1,1)
    this is the Miller (111) family of planes.
    """
    import numpy as np
    from scipy.ndimage import map_coordinates

    vol = np.asarray(volume)
    nz, ny, nx = vol.shape
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal must be non-zero")
    n = n / n_norm

    # Orthonormal in-plane axes (stable for (111) and nearby normals).
    tmp = np.array([1.0, -1.0, 0.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, -1.0])
    e1 = tmp - np.dot(tmp, n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    corners = np.array(
        [[z, y, x] for z in (0, nz - 1) for y in (0, ny - 1) for x in (0, nx - 1)],
        dtype=float,
    )
    proj = corners @ n
    s = float(proj.min() + np.clip(offset, 0.0, 1.0) * (proj.max() - proj.min()))

    center = np.array([(nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0])
    p0 = center + (s - np.dot(center, n)) * n

    half = 0.75 * max(nz, ny, nx)
    uu = np.linspace(-half, half, resolution)
    vv = np.linspace(-half, half, resolution)
    U, V = np.meshgrid(uu, vv, indexing="xy")
    coords = p0[:, None, None] + e1[:, None, None] * U + e2[:, None, None] * V
    return map_coordinates(vol, coords, order=1, mode="constant", cval=0.0)


def view_plane111(
    idx_path: Path,
    downsample: int,
    save_preview: Path | None,
    tiff_path: Path | None = None,
    normal=(1.0, 1.0, 1.0),
    resolution: int = 512,
    prefer_tiff: bool = False,
) -> None:
    """Fast 2D viewer for (111)-style oblique planes with a depth slider."""
    _ensure_local_pkgs()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    data, _ = _load_volume(
        idx_path, downsample, tiff_path=tiff_path, prefer_tiff=prefer_tiff
    )
    lo, hi = np.percentile(data, (1, 99.5))
    if hi <= lo:
        lo, hi = float(data.min()), float(data.max() or 1)

    offset0 = 0.5
    img0 = oblique_slice(data, offset0, normal=normal, resolution=resolution)

    fig, ax = plt.subplots(figsize=(7, 7))
    nlab = f"({int(normal[0])}{int(normal[1])}{int(normal[2])})"
    fig.suptitle(
        f"{nlab} oblique slice  shape={data.shape}  ds={downsample}\n"
        "slider = depth along plane normal"
    )
    im = ax.imshow(img0, cmap="gray", vmin=lo, vmax=hi, origin="lower")
    ax.set_xlabel("in-plane U")
    ax.set_ylabel("in-plane V")
    ax.tick_params(labelsize=8)
    ax.set_title(f"offset={offset0:.2f}  |  CT axes: volume is Z,Y,X — use mid CT xyz from CSV")
    ax.text(
        0.02,
        0.98,
        f"plane normal ≈ ({normal[0]:g},{normal[1]:g},{normal[2]:g}) in (Z,Y,X)",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="yellow",
        fontsize=9,
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    ax_sl = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_sl, "depth", 0.0, 1.0, valinit=offset0, valstep=0.01)

    def update(_=None):
        off = float(slider.val)
        im.set_data(oblique_slice(data, off, normal=normal, resolution=resolution))
        ax.set_title(f"offset={off:.2f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    if save_preview is not None:
        save_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_preview, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {save_preview}")

    print("Move the depth slider to sweep (111) planes. Close the window to exit.")
    plt.show()


def _load_strut_segments(
    registered_json: Path,
    downsample: int,
    missing_csv: Path | None = None,
):
    """Load strut endpoints in downsampled CT voxel coords (x,y,z)."""
    import csv
    import json
    import numpy as np

    data = json.loads(registered_json.read_text())
    jpos = {j["id"]: np.asarray(j["position"], dtype=float) for j in data["junctions"]}
    scale = float(max(downsample, 1))
    segments = []
    for s in data["struts"]:
        p0 = jpos[s["junction0"]] / scale
        p1 = jpos[s["junction1"]] / scale
        segments.append((int(s["id"]), p0, p1))

    missing_ids: set[int] = set()
    if missing_csv is not None and missing_csv.is_file():
        # utf-8-sig strips BOM; skip blank lines so a leading empty row
        # doesn't become the DictReader header (fieldnames=[]).
        with missing_csv.open(encoding="utf-8-sig", newline="") as f:
            lines = (ln for ln in f if ln.strip())
            reader = csv.DictReader(lines)
            if not reader.fieldnames or "strut_id" not in reader.fieldnames:
                raise ValueError(
                    f"{missing_csv} must have a strut_id column; "
                    f"got fieldnames={reader.fieldnames!r}"
                )
            for row in reader:
                val = (row.get("strut_id") or "").strip()
                if val:
                    missing_ids.add(int(float(val)))
        print(f"Loaded {len(missing_ids)} missing strut ids from {missing_csv}")
    elif missing_csv is not None:
        print(f"WARNING: missing CSV not found: {missing_csv}")
    print(f"Loaded {len(segments)} struts from {registered_json.name} (ds={downsample})")
    return segments, missing_ids


def _clear_artist_list(art_list):
    for a in art_list:
        a.remove()
    art_list.clear()


def _point_to_segment_dist(px, py, x0, y0, x1, y1):
    """Distance from point (px,py) to segment (x0,y0)-(x1,y1)."""
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        return ((px - x0) ** 2 + (py - y0) ** 2) ** 0.5
    t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx = x0 + t * dx
    qy = y0 + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


def _draw_strut_overlays(
    axes,
    artists,
    segments,
    missing_ids,
    x,
    y,
    z,
    tol: float,
    *,
    active_views,
    show_design: bool,
    show_missing: bool,
    show_projections: bool = False,
    highlight_sid: int | None = None,
    highlight_endpoints=None,
):
    """Redraw strut overlays only on active views. Uses LineCollection for speed.

    By default (show_projections=False), only draw struts that are nearly
    coplanar with the current slice — both endpoints within ``tol`` of the
    slice coordinate. That hides axis-aligned "ghost" lines that are just
    projections of out-of-plane struts.

    When show_projections=True, use the older range-overlap rule (draw if the
    strut's span intersects the slice).

    Returns hover_hits: list of 3 lists of (sid, is_missing, x0, y0, x1, y1)
    in each panel's display coordinates, for nearest-strut hover lookup.
    """
    from matplotlib.collections import LineCollection

    for i, art_list in enumerate(artists):
        _clear_artist_list(art_list)

    hover_hits: list[list] = [[], [], []]
    ax_xy, ax_xz, ax_yz = axes
    xy_segs, xz_segs, yz_segs = [], [], []
    xy_miss, xz_miss, yz_miss = [], [], []

    for sid, p0, p1 in segments:
        is_missing = sid in missing_ids
        if is_missing and not show_missing:
            continue
        if (not is_missing) and not show_design:
            continue

        if show_projections:
            # Old behavior: any strut whose axis-range overlaps the slice.
            near_z = active_views[0] and (
                min(p0[2], p1[2]) - tol <= z <= max(p0[2], p1[2]) + tol
            )
            near_y = active_views[1] and (
                min(p0[1], p1[1]) - tol <= y <= max(p0[1], p1[1]) + tol
            )
            near_x = active_views[2] and (
                min(p0[0], p1[0]) - tol <= x <= max(p0[0], p1[0]) + tol
            )
        else:
            # In-plane only: both endpoints close to the slice plane.
            near_z = active_views[0] and (
                abs(float(p0[2]) - z) <= tol and abs(float(p1[2]) - z) <= tol
            )
            near_y = active_views[1] and (
                abs(float(p0[1]) - y) <= tol and abs(float(p1[1]) - y) <= tol
            )
            near_x = active_views[2] and (
                abs(float(p0[0]) - x) <= tol and abs(float(p1[0]) - x) <= tol
            )
        if not (near_z or near_y or near_x):
            continue

        if near_z:
            seg = [(p0[0], p0[1]), (p1[0], p1[1])]
            (xy_miss if is_missing else xy_segs).append(seg)
            hover_hits[0].append(
                (sid, is_missing, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
            )
        if near_y:
            seg = [(p0[0], p0[2]), (p1[0], p1[2])]
            (xz_miss if is_missing else xz_segs).append(seg)
            hover_hits[1].append(
                (sid, is_missing, float(p0[0]), float(p0[2]), float(p1[0]), float(p1[2]))
            )
        if near_x:
            seg = [(p0[1], p0[2]), (p1[1], p1[2])]
            (yz_miss if is_missing else yz_segs).append(seg)
            hover_hits[2].append(
                (sid, is_missing, float(p0[1]), float(p0[2]), float(p1[1]), float(p1[2]))
            )

    def _add_lc(ax, art_list, segs, color, lw, alpha, zorder):
        if not segs:
            return
        lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, zorder=zorder)
        ax.add_collection(lc)
        art_list.append(lc)

    if active_views[0]:
        _add_lc(ax_xy, artists[0], xy_segs, "#00e5ff", 0.5, 0.2, 3)
        _add_lc(ax_xy, artists[0], xy_miss, "#ff3333", 2.0, 0.95, 5)
    if active_views[1]:
        _add_lc(ax_xz, artists[1], xz_segs, "#00e5ff", 0.5, 0.2, 3)
        _add_lc(ax_xz, artists[1], xz_miss, "#ff3333", 2.0, 0.95, 5)
    if active_views[2]:
        _add_lc(ax_yz, artists[2], yz_segs, "#00e5ff", 0.5, 0.2, 3)
        _add_lc(ax_yz, artists[2], yz_miss, "#ff3333", 2.0, 0.95, 5)

    # Always draw searched strut highlight (even if filtered out above).
    if highlight_sid is not None and highlight_endpoints is not None:
        p0, p1 = highlight_endpoints
        is_missing = highlight_sid in missing_ids
        if active_views[0]:
            (ln,) = ax_xy.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                color="#ffff00", lw=3.0, alpha=1.0, zorder=10,
            )
            artists[0].append(ln)
            hover_hits[0].append(
                (highlight_sid, is_missing, float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
            )
        if active_views[1]:
            (ln,) = ax_xz.plot(
                [p0[0], p1[0]], [p0[2], p1[2]],
                color="#ffff00", lw=3.0, alpha=1.0, zorder=10,
            )
            artists[1].append(ln)
            hover_hits[1].append(
                (highlight_sid, is_missing, float(p0[0]), float(p0[2]), float(p1[0]), float(p1[2]))
            )
        if active_views[2]:
            (ln,) = ax_yz.plot(
                [p0[1], p1[1]], [p0[2], p1[2]],
                color="#ffff00", lw=3.0, alpha=1.0, zorder=10,
            )
            artists[2].append(ln)
            hover_hits[2].append(
                (highlight_sid, is_missing, float(p0[1]), float(p0[2]), float(p1[1]), float(p1[2]))
            )

    return hover_hits


def view_orthoslices(
    idx_path: Path,
    downsample: int,
    save_preview: Path | None,
    registered_json: Path | None = None,
    missing_csv: Path | None = None,
    overlay_struts: bool = True,
    slice_tol: float = 1.5,
    tiff_path: Path | None = None,
    prefer_tiff: bool = False,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, CheckButtons, TextBox
    from matplotlib.lines import Line2D

    data, _db = _load_volume(
        idx_path, downsample, tiff_path=tiff_path, prefer_tiff=prefer_tiff
    )
    source_name = (
        tiff_path.name
        if prefer_tiff and tiff_path is not None and tiff_path.is_file()
        else (idx_path.name if idx_path.is_file() else (tiff_path.name if tiff_path else "volume"))
    )
    zmax, ymax, xmax = (s - 1 for s in data.shape)
    z0, y0, x0 = zmax // 2, ymax // 2, xmax // 2

    lo, hi = np.percentile(data, (1, 99.5))
    if hi <= lo:
        lo, hi = float(data.min()), float(data.max() or 1)

    segments = []
    missing_ids: set[int] = set()
    if overlay_struts and registered_json is not None and registered_json.is_file():
        segments, missing_ids = _load_strut_segments(
            registered_json, downsample, missing_csv=missing_csv
        )
    elif overlay_struts:
        print("Strut overlay skipped: registered JSON not found.")

    segments_by_id = {sid: (p0, p1) for sid, p0, p1 in segments}

    # If a missing CSV is provided, default to missing-only (much faster).
    default_show_design = not bool(missing_ids)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle(
        f"{source_name}  shape={data.shape}  ds={downsample}"
        + ("  | red=missing  cyan=design" if segments else "")
    )

    im_xy = axes[0].imshow(data[z0], cmap="gray", vmin=lo, vmax=hi, origin="lower")
    axes[0].set_title(f"XY  z={z0}")
    axes[0].set_xlabel("X →")
    axes[0].set_ylabel("Y →")
    im_xz = axes[1].imshow(data[:, y0, :], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
    axes[1].set_title(f"XZ  y={y0}")
    axes[1].set_xlabel("X →")
    axes[1].set_ylabel("Z →")
    im_yz = axes[2].imshow(data[:, :, x0], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
    axes[2].set_title(f"YZ  x={x0}")
    axes[2].set_xlabel("Y →")
    axes[2].set_ylabel("Z →")
    images = [im_xy, im_xz, im_yz]
    for ax in axes:
        ax.tick_params(labelsize=8)

    overlay_artists: list[list] = [[], [], []]
    hover_hits: list[list] = [[], [], []]
    view_on = [True, True, True]
    show_struts = True
    show_design = default_show_design
    show_missing = True
    show_projections = False  # default: in-plane only (no ghost projections)
    hover_tol = 4.0  # voxels in panel display coords
    hover_annot = None
    last_hover_sid = {"id": None}
    search_highlight_sid = {"id": None}

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.34, right=0.78)
    ax_z = fig.add_axes([0.12, 0.20, 0.50, 0.03])
    ax_y = fig.add_axes([0.12, 0.15, 0.50, 0.03])
    ax_x = fig.add_axes([0.12, 0.10, 0.50, 0.03])
    s_z = Slider(ax_z, "Z", 0, zmax, valinit=z0, valstep=1)
    s_y = Slider(ax_y, "Y", 0, ymax, valinit=y0, valstep=1)
    s_x = Slider(ax_x, "X", 0, xmax, valinit=x0, valstep=1)

    search_ax = fig.add_axes([0.12, 0.03, 0.18, 0.04])
    search_box = TextBox(search_ax, "Strut ID ", initial="")
    search_status = fig.text(0.32, 0.045, "", fontsize=9, va="center")

    labels = ["XY view", "XZ view", "YZ view", "Struts", "All design", "Projections"]
    actives = [True, True, True, bool(segments), default_show_design, False]
    check_ax = fig.add_axes([0.78, 0.02, 0.20, 0.26])
    check = CheckButtons(check_ax, labels, actives)

    if segments:
        fig.legend(
            handles=[
                Line2D([0], [0], color="#00e5ff", lw=2, label="design (if enabled)"),
                Line2D([0], [0], color="#ff3333", lw=2, label="missing CSV"),
            ],
            loc="upper right",
            fontsize=8,
        )

    def _apply_view_visibility():
        for i, ax in enumerate(axes):
            ax.set_visible(view_on[i])

    def _hide_hover():
        nonlocal hover_annot
        last_hover_sid["id"] = None
        if hover_annot is not None:
            hover_annot.remove()
            hover_annot = None

    def _clear_search_highlight():
        search_highlight_sid["id"] = None
        search_status.set_text("")
        search_status.set_color("black")
        try:
            search_box.set_val("")
        except Exception:
            pass
        if getattr(search_box, "capturekeystrokes", False):
            search_box.stop_typing()

    def _redraw_struts(x, y, z):
        nonlocal hover_hits
        _hide_hover()
        if not (segments and show_struts) and search_highlight_sid["id"] is None:
            for art_list in overlay_artists:
                _clear_artist_list(art_list)
            hover_hits = [[], [], []]
            return
        hid = search_highlight_sid["id"]
        hover_hits = _draw_strut_overlays(
            axes,
            overlay_artists,
            segments if show_struts else [],
            missing_ids,
            x,
            y,
            z,
            slice_tol,
            active_views=view_on,
            show_design=show_design,
            show_missing=show_missing,
            show_projections=show_projections,
            highlight_sid=hid,
            highlight_endpoints=segments_by_id.get(hid) if hid is not None else None,
        )

    def update(_=None):
        z, y, x = int(s_z.val), int(s_y.val), int(s_x.val)
        if view_on[0]:
            images[0].set_data(data[z])
            axes[0].set_title(f"XY  z={z}  (X→ horiz, Y→ vert)")
        if view_on[1]:
            images[1].set_data(data[:, y, :])
            axes[1].set_title(f"XZ  y={y}  (X→ horiz, Z→ vert)")
        if view_on[2]:
            images[2].set_data(data[:, :, x])
            axes[2].set_title(f"YZ  x={x}  (Y→ horiz, Z→ vert)")
        _redraw_struts(x, y, z)
        fig.canvas.draw_idle()

    def on_search(text):
        raw = (text or "").strip()
        if not raw:
            _clear_search_highlight()
            update()
            return
        try:
            sid = int(float(raw))
        except ValueError:
            search_status.set_text("Enter a numeric strut_id")
            search_status.set_color("red")
            fig.canvas.draw_idle()
            return
        if sid not in segments_by_id:
            search_highlight_sid["id"] = None
            search_status.set_text(f"strut {sid} not found")
            search_status.set_color("red")
            fig.canvas.draw_idle()
            return

        p0, p1 = segments_by_id[sid]
        mid = 0.5 * (p0 + p1)
        tx = int(np.clip(round(float(mid[0])), 0, xmax))
        ty = int(np.clip(round(float(mid[1])), 0, ymax))
        tz = int(np.clip(round(float(mid[2])), 0, zmax))

        # Which panels are truly in-plane for this strut at the midpoint.
        in_plane = []
        if abs(float(p0[2]) - tz) <= slice_tol and abs(float(p1[2]) - tz) <= slice_tol:
            in_plane.append("XY")
        if abs(float(p0[1]) - ty) <= slice_tol and abs(float(p1[1]) - ty) <= slice_tol:
            in_plane.append("XZ")
        if abs(float(p0[0]) - tx) <= slice_tol and abs(float(p1[0]) - tx) <= slice_tol:
            in_plane.append("YZ")

        kind = "missing" if sid in missing_ids else "design"
        planes = ",".join(in_plane) if in_plane else "none (see yellow projection)"
        search_highlight_sid["id"] = sid
        search_status.set_text(
            f"strut {sid} ({kind}) → X={tx} Y={ty} Z={tz}  in-plane: {planes}"
        )
        search_status.set_color("black")

        # Release text-box focus so arrow/WASD keys work again.
        if getattr(search_box, "capturekeystrokes", False):
            search_box.stop_typing()

        # Jump sliders to the strut midpoint (triggers update via on_changed).
        s_x.set_val(tx)
        s_y.set_val(ty)
        s_z.set_val(tz)
        update()

    def on_check(label):
        nonlocal show_struts, show_design, show_projections
        status = check.get_status()
        # labels: XY, XZ, YZ, Struts, All design, Projections
        view_on[0], view_on[1], view_on[2] = status[0], status[1], status[2]
        show_struts = status[3]
        show_design = status[4]
        show_projections = status[5]
        # Turning off Struts also clears any searched strut highlight.
        if not show_struts:
            _clear_search_highlight()
        _apply_view_visibility()
        update()

    def on_motion(event):
        nonlocal hover_annot
        if event.inaxes not in axes or event.xdata is None or event.ydata is None:
            if last_hover_sid["id"] is not None:
                _hide_hover()
                fig.canvas.draw_idle()
            return
        if not (segments and show_struts):
            return

        ax_i = list(axes).index(event.inaxes)
        if not view_on[ax_i]:
            return

        hits = hover_hits[ax_i]
        if not hits:
            if last_hover_sid["id"] is not None:
                _hide_hover()
                fig.canvas.draw_idle()
            return

        px, py = float(event.xdata), float(event.ydata)
        best = None  # (dist, prefer_missing, sid, is_missing)
        for sid, is_missing, x0s, y0s, x1s, y1s in hits:
            d = _point_to_segment_dist(px, py, x0s, y0s, x1s, y1s)
            if d > hover_tol:
                continue
            # Prefer closer; on tie prefer missing struts.
            cand = (d, 0 if is_missing else 1, sid, is_missing)
            if best is None or cand < best:
                best = cand

        if best is None:
            if last_hover_sid["id"] is not None:
                _hide_hover()
                fig.canvas.draw_idle()
            return

        _, _, sid, is_missing = best
        if sid == last_hover_sid["id"] and hover_annot is not None and hover_annot.axes is event.inaxes:
            hover_annot.xy = (px, py)
            fig.canvas.draw_idle()
            return

        last_hover_sid["id"] = sid
        if hover_annot is not None:
            hover_annot.remove()
            hover_annot = None
        kind = "missing" if is_missing else "design"
        hover_annot = event.inaxes.annotate(
            f"strut {sid}  ({kind})",
            xy=(px, py),
            xytext=(12, 12),
            textcoords="offset points",
            color="black",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.9),
            zorder=20,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        key = event.key
        if key is None:
            return
        # Only suppress shortcuts while the strut-id box is actively capturing
        # keystrokes (not merely because the TextBox widget is "active").
        typing = bool(getattr(search_box, "capturekeystrokes", False))
        if typing:
            if key == "escape":
                search_box.stop_typing()
                _clear_search_highlight()
                update()
            return

        def _bump(slider, delta):
            new_val = float(np.clip(slider.val + delta, slider.valmin, slider.valmax))
            if new_val != slider.val:
                slider.set_val(new_val)

        if key in ("left", "a"):
            _bump(s_x, -1)
        elif key in ("right", "d"):
            _bump(s_x, 1)
        elif key in ("up", "w"):
            _bump(s_z, 1)
        elif key in ("down", "s"):
            _bump(s_z, -1)
        elif key in ("shift+up", "e"):
            _bump(s_y, 1)
        elif key in ("shift+down", "q"):
            _bump(s_y, -1)
        elif key == "1":
            check.set_active(0)
        elif key == "2":
            check.set_active(1)
        elif key == "3":
            check.set_active(2)
        elif key in ("t", "T") and segments:
            check.set_active(3)
        elif key in ("g", "G") and segments:
            check.set_active(4)
        elif key in ("p", "P") and segments:
            check.set_active(5)
        elif key == "escape":
            _clear_search_highlight()
            update()
        else:
            return

    # initial strut draw (missing-only if CSV present)
    _redraw_struts(x0, y0, z0)

    s_z.on_changed(update)
    s_y.on_changed(update)
    s_x.on_changed(update)
    check.on_clicked(on_check)
    search_box.on_submit(on_search)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    if save_preview is not None:
        save_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_preview, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {save_preview}")

    print("Close the window to exit.")
    print(
        "Keys: ←/→ X | ↑/↓ Z | Shift+↑/↓ Y | 1/2/3 views | T struts | G all design | P projections | Esc clear search"
    )
    print("Search: type a strut_id in the box and press Enter to jump X/Y/Z to its midpoint (yellow highlight).")
    print("Hover a strut line to see its strut_id (yellow tooltip).")
    print("Default: in-plane struts only (hides out-of-plane projection ghosts). Toggle 'Projections' or P for old behavior.")
    if missing_ids:
        print("Tip: with a missing CSV, 'All design' is OFF by default (faster). Turn it on only if needed.")
    plt.show()


def view_openvisus_qt(idx_path: Path) -> None:
    """Native OpenVisus Qt viewer (requires PyQt5)."""
    from OpenVisus.gui import PyViewer
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    viewer = PyViewer(title=f"OpenVisus: {idx_path.name}")
    viewer.open(str(idx_path))
    viewer.run()
    del viewer
    app.quit()


def view_3d(
    idx_path: Path,
    downsample: int,
    threshold_percentile: float,
    save_preview: Path | None,
    backend: str = "matplotlib",
    tiff_path: Path | None = None,
    prefer_tiff: bool = False,
) -> None:
    """Interactive 3D isosurface (matplotlib by default)."""
    _ensure_local_pkgs()
    import numpy as np

    print(f"Loading volume (downsample={downsample})...")
    data, _db = _load_volume(
        idx_path, downsample, tiff_path=tiff_path, prefer_tiff=prefer_tiff
    )
    source = (
        tiff_path.name
        if prefer_tiff and tiff_path is not None and tiff_path.is_file()
        else (idx_path.name if idx_path.is_file() else (tiff_path.name if tiff_path else "volume"))
    )

    print(f"Volume shape: {data.shape}, dtype={data.dtype}")
    level = float(np.percentile(data, threshold_percentile))
    print(f"Isosurface threshold: {level:.1f} (p{threshold_percentile})")

    if backend == "pyvista":
        _view_3d_pyvista(Path(source), data, level, downsample, threshold_percentile)
        return

    _view_3d_matplotlib(
        Path(source), data, level, downsample, threshold_percentile, save_preview
    )


def _view_3d_pyvista(
    idx_path: Path,
    data,
    level: float,
    downsample: int,
    threshold_percentile: float,
) -> None:
    import numpy as np

    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"PyVista backend needs Python 3.11 (found {sys.version.split()[0]}).\n"
            "Use the challenge env:\n"
            "  conda activate dssi_env\n"
            "  PYTHONPATH=.python_pkgs python data/missing_struts/openvisus/view_idx.py "
            "--mode 3d --backend pyvista --downsample 4\n"
            "Or stick with the default matplotlib backend (no --backend flag)."
        )

    import pyvista as pv

    grid = pv.ImageData(dimensions=np.array(data.shape[::-1]) + 1)
    grid.cell_data["density"] = data.T.ravel(order="F")
    grid = grid.cell_data_to_point_data()
    mesh = grid.contour([level], scalars="density")
    print(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    print("Opening PyVista window — rotate / zoom, then close to exit.")

    plotter = pv.Plotter(window_size=(1200, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color="#4a7c59",
        opacity=0.85,
        smooth_shading=True,
        specular=0.3,
    )
    plotter.add_axes()
    plotter.add_text(
        f"{idx_path.name}  ds={downsample}  p{threshold_percentile:g}={level:.0f}",
        font_size=10,
    )
    plotter.camera_position = "iso"
    plotter.show()


def _view_3d_matplotlib(
    idx_path: Path,
    data,
    level: float,
    downsample: int,
    threshold_percentile: float,
    save_preview: Path | None,
) -> None:
    from skimage import measure
    import matplotlib.pyplot as plt

    print("Extracting isosurface (matplotlib)...")
    verts, faces, *_ = measure.marching_cubes(data, level=level)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        faces,
        verts[:, 2],
        color="#4a7c59",
        lw=0.05,
        edgecolor="none",
        alpha=0.85,
    )
    ax.set_title(f"{idx_path.name}  ds={downsample}  p{threshold_percentile:g}")
    ax.set_axis_off()
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    if save_preview is not None:
        save_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_preview, dpi=150, bbox_inches="tight")
        print(f"Saved 3D preview: {save_preview}")
        plt.close(fig)
        return
    print("Rotate the matplotlib 3D window. Close it to exit.")
    plt.show()


def main() -> None:
    default_idx = (
        _repo_root()
        / "data/missing_struts/tif_stacks/openvisus_idx"
        / "0point5dash1"
        / "visus.idx"
    )
    default_tiff = (
        _repo_root()
        / "data/missing_struts/tif_stacks"
        / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idx",
        type=Path,
        default=default_idx,
        help=f"OpenVisus IDX path (default: {default_idx.name} under openvisus_idx/0point5dash1)",
    )
    parser.add_argument(
        "--mode",
        choices=("orthoslices", "plane111", "3d", "openvisus"),
        default="plane111",
        help="plane111: fast (111) 2D oblique slices; orthoslices; 3d; openvisus",
    )
    parser.add_argument(
        "--backend",
        choices=("matplotlib", "pyvista"),
        default="matplotlib",
        help="3d renderer (matplotlib is default; pyvista needs dssi_env / Python 3.11)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Volume downsample factor (2 is a good default for plane111)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Output pixel size for plane111 oblique slice",
    )
    parser.add_argument(
        "--normal",
        default="1,1,1",
        help="Plane normal as h,k,l (default 1,1,1 for Miller (111))",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=92.0,
        help="Isosurface level as intensity percentile (3d mode)",
    )
    parser.add_argument(
        "--tiff",
        type=Path,
        default=None,
        help=(
            "CT multipage TIFF to open. If omitted, uses the hardcoded 0point5dash1 "
            f"TIFF ({default_tiff.name}). When you pass --tiff explicitly, that file "
            "is preferred over IDX."
        ),
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Optional PNG path for a static preview (skips interactive window)",
    )
    parser.add_argument(
        "--registered-json",
        type=Path,
        default=(
            _repo_root()
            / "data/missing_struts/registered_jsons"
            / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
        ),
        help="Registered design graph for strut overlay (orthoslices mode)",
    )
    parser.add_argument(
        "--missing-csv",
        type=Path,
        default=None,
        help="CSV with strut_id column; those struts are drawn in red",
    )
    parser.add_argument(
        "--no-strut-overlay",
        action="store_true",
        help="Disable strut overlay on orthoslices",
    )
    parser.add_argument(
        "--slice-tol",
        type=float,
        default=1.5,
        help="How close a strut must be to the slice (in downsampled voxels) to draw",
    )
    args = parser.parse_args()
    _ensure_local_pkgs()  # scrub bad PYTHONPATH before importing numpy/etc.

    # Hardcoded TIFF default when --tiff is not mentioned.
    user_set_tiff = args.tiff is not None
    tiff_path = args.tiff if user_set_tiff else default_tiff

    normal = tuple(float(x) for x in args.normal.split(","))
    if len(normal) != 3:
        raise SystemExit("--normal must be three comma-separated numbers, e.g. 1,1,1")

    if args.mode == "openvisus":
        _ensure_openvisus()
        if not args.idx.is_file():
            raise SystemExit(f"IDX not found: {args.idx}")
        try:
            import PyQt5  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "Native OpenVisus viewer needs PyQt5.\n"
                "  pip install PyQt5\n"
                "Or use: --mode plane111"
            ) from exc
        view_openvisus_qt(args.idx)
    elif args.mode == "3d":
        view_3d(
            args.idx,
            args.downsample,
            args.threshold_percentile,
            args.save_preview,
            backend=args.backend,
            tiff_path=tiff_path,
            prefer_tiff=user_set_tiff,
        )
    elif args.mode == "plane111":
        preview = args.save_preview
        if preview is None:
            preview = args.idx.parent / "plane111_preview.png"
        view_plane111(
            args.idx,
            args.downsample,
            preview if args.save_preview else None,
            tiff_path=tiff_path,
            normal=normal,
            resolution=args.resolution,
            prefer_tiff=user_set_tiff,
        )
    else:
        # orthoslices: IDX if available (unless user passed --tiff), else TIFF
        if user_set_tiff and not tiff_path.is_file():
            raise SystemExit(f"TIFF not found: {tiff_path}")
        if (not user_set_tiff) and (not args.idx.is_file()) and (not tiff_path.is_file()):
            raise SystemExit(
                f"Neither IDX nor default TIFF found.\n  IDX: {args.idx}\n  TIFF: {tiff_path}"
            )
        view_orthoslices(
            args.idx,
            args.downsample,
            args.save_preview,
            registered_json=args.registered_json,
            missing_csv=args.missing_csv,
            overlay_struts=not args.no_strut_overlay,
            slice_tol=args.slice_tol,
            tiff_path=tiff_path,
            prefer_tiff=user_set_tiff,
        )


if __name__ == "__main__":
    main()
