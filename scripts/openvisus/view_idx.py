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
):
    """Load ZYX volume from IDX when possible, else TIFF."""
    _ensure_local_pkgs()
    pkgs = _repo_root() / ".python_pkgs"
    can_use_ov = not (
        pkgs.is_dir()
        and (pkgs / "OpenVisus").is_dir()
        and sys.version_info[:2] != (3, 11)
    )
    if can_use_ov and idx_path.is_file():
        try:
            _ensure_openvisus()
            return _read_volume(idx_path, downsample)
        except SystemExit as exc:
            print(exc)

    if tiff_path is None or not tiff_path.is_file():
        raise SystemExit(
            "Could not load IDX/OpenVisus and no readable --tiff was provided.\n"
            "Activate dssi_env (Python 3.11) or pass a valid --tiff path."
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
) -> None:
    """Fast 2D viewer for (111)-style oblique planes with a depth slider."""
    _ensure_local_pkgs()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    data, _ = _load_volume(idx_path, downsample, tiff_path=tiff_path)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"offset={offset0:.2f}")

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


def view_orthoslices(idx_path: Path, downsample: int, save_preview: Path | None) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    data, _db = _read_volume(idx_path, downsample)
    zmax, ymax, xmax = (s - 1 for s in data.shape)
    z0, y0, x0 = zmax // 2, ymax // 2, xmax // 2

    lo, hi = np.percentile(data, (1, 99.5))
    if hi <= lo:
        lo, hi = float(data.min()), float(data.max() or 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"OpenVisus IDX: {idx_path.name}  shape={data.shape}  ds={downsample}")

    im_xy = axes[0].imshow(data[z0], cmap="gray", vmin=lo, vmax=hi, origin="lower")
    axes[0].set_title(f"XY  z={z0}")
    im_xz = axes[1].imshow(data[:, y0, :], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
    axes[1].set_title(f"XZ  y={y0}")
    im_yz = axes[2].imshow(data[:, :, x0], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto")
    axes[2].set_title(f"YZ  x={x0}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    ax_z = fig.add_axes([0.15, 0.12, 0.7, 0.03])
    ax_y = fig.add_axes([0.15, 0.07, 0.7, 0.03])
    ax_x = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    s_z = Slider(ax_z, "Z", 0, zmax, valinit=z0, valstep=1)
    s_y = Slider(ax_y, "Y", 0, ymax, valinit=y0, valstep=1)
    s_x = Slider(ax_x, "X", 0, xmax, valinit=x0, valstep=1)

    def update(_=None):
        z, y, x = int(s_z.val), int(s_y.val), int(s_x.val)
        im_xy.set_data(data[z])
        im_xz.set_data(data[:, y, :])
        im_yz.set_data(data[:, :, x])
        axes[0].set_title(f"XY  z={z}")
        axes[1].set_title(f"XZ  y={y}")
        axes[2].set_title(f"YZ  x={x}")
        fig.canvas.draw_idle()

    s_z.on_changed(update)
    s_y.on_changed(update)
    s_x.on_changed(update)

    if save_preview is not None:
        save_preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_preview, dpi=150, bbox_inches="tight")
        print(f"Saved preview: {save_preview}")

    print("Close the window to exit.")
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
) -> None:
    """Interactive 3D isosurface (matplotlib by default)."""
    _ensure_local_pkgs()
    import numpy as np

    print(f"Loading volume (downsample={downsample})...")
    data = None
    source = idx_path.name
    pkgs = _repo_root() / ".python_pkgs"
    can_use_ov = not (
        pkgs.is_dir()
        and (pkgs / "OpenVisus").is_dir()
        and sys.version_info[:2] != (3, 11)
    )
    if can_use_ov:
        try:
            _ensure_openvisus()
            if idx_path.is_file():
                data, _db = _read_volume(idx_path, downsample)
        except SystemExit as exc:
            print(exc)
            data = None

    if data is None:
        if tiff_path is None or not tiff_path.is_file():
            raise SystemExit(
                "Could not load IDX/OpenVisus and no readable --tiff was provided.\n"
                "Activate dssi_env (Python 3.11) or pass a valid --tiff path."
            )
        print(f"Reading TIFF:\n  {tiff_path}")
        data, _db = _read_volume_from_tiff(tiff_path, downsample)
        source = tiff_path.name

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx", type=Path, default=default_idx)
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
        default=(
            _repo_root()
            / "data/missing_struts/tif_stacks"
            / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
        ),
        help="Fallback TIFF if OpenVisus/IDX cannot be loaded",
    )
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="Optional PNG path for a static preview (skips interactive window)",
    )
    args = parser.parse_args()
    _ensure_local_pkgs()  # scrub bad PYTHONPATH before importing numpy/etc.

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
            tiff_path=args.tiff,
        )
    elif args.mode == "plane111":
        preview = args.save_preview
        if preview is None:
            preview = args.idx.parent / "plane111_preview.png"
        view_plane111(
            args.idx,
            args.downsample,
            preview if args.save_preview else None,
            tiff_path=args.tiff,
            normal=normal,
            resolution=args.resolution,
        )
    else:
        _ensure_openvisus()
        if not args.idx.is_file():
            raise SystemExit(f"IDX not found: {args.idx}")
        preview = args.save_preview
        if preview is None:
            preview = args.idx.parent / "orthoslice_preview.png"
        view_orthoslices(args.idx, args.downsample, preview)


if __name__ == "__main__":
    main()
