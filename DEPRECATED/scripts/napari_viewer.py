"""Open one NumPy or TIFF dataset in Napari.

Change only FILE_TO_OPEN below. Relative paths are resolved from the repository
root, so the script works regardless of PyCharm's working directory.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Replace this filename with the .npy, .tif, or .tiff file you want to view.
# You may also paste an absolute path here.
# ---------------------------------------------------------------------------
FILE_TO_OPEN = "/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/mask.tif"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_WORDS = ("mask", "segment", "skeleton", "label")


def resolve_input(filename: str) -> Path:
    """Resolve a user-supplied path and ensure that it exists."""
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    return path


def load_dataset(path: Path) -> np.ndarray:
    """Load supported datasets, using memory mapping whenever possible."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if suffix in {".tif", ".tiff"}:
        try:
            return tifffile.memmap(path, mode="r")
        except ValueError:
            # Compressed TIFFs cannot be mapped directly and must be decoded.
            print("TIFF is compressed; decoding it into memory...")
            return tifffile.imread(path)
    raise ValueError(
        f"Unsupported file type {suffix!r}. Use a .npy, .tif, or .tiff file."
    )


def looks_like_labels(path: Path, data: np.ndarray) -> bool:
    """Infer whether the array is a discrete mask from its name and dtype."""
    lower_name = path.stem.lower()
    named_as_mask = any(word in lower_name for word in MASK_WORDS)
    return data.dtype == np.bool_ or (
        named_as_mask and np.issubdtype(data.dtype, np.integer)
    )


def main() -> None:
    path = resolve_input(FILE_TO_OPEN)
    data = load_dataset(path)

    print(f"Opening: {path}")
    print(f"Shape: {data.shape}")
    print(f"Dimensions: {data.ndim}")
    print(f"Data type: {data.dtype}")

    if data.ndim == 0:
        raise ValueError("A scalar array cannot be displayed in Napari.")
    if data.ndim == 1:
        # Napari image layers require at least two dimensions. This makes a 1-D
        # array viewable as a single-row heatmap; use Matplotlib for a line plot.
        print(
            "This is a 1-D array, so Napari will display it as a one-row "
            "heatmap. Use Matplotlib if you need a line graph."
        )
        data = data[np.newaxis, :]

    viewer = napari.Viewer(title=f"Napari — {path.name}")
    if looks_like_labels(path, data):
        viewer.add_labels(data, name=path.stem)
    else:
        viewer.add_image(data, name=path.stem, colormap="gray")

    if data.ndim == 3:
        viewer.dims.axis_labels = ("z", "y", "x")

    napari.run()


if __name__ == "__main__":
    main()
