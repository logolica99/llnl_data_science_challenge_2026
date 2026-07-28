"""Compare segmentation mask slice 380 with the rendered ground truth in Napari."""

from pathlib import Path

from matplotlib.image import imread
import napari
import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "9x9x9_octet_lattice"

SLICE_INDEX = 380
MASK_PATH = DATASET_DIR / "segmentation" / "runs"/ "cad10_upward_eval_20260721T172730Z"/ "proposed_mask_not_adopted.tif"
GROUND_TRUTH_PATH = DATASET_DIR / "ground_truth_segmentation_slice_380.png"


def load_mask_slice(path: Path, slice_index: int) -> np.ndarray:
    """Read one axis-0 mask slice without loading the full 3D TIFF."""
    if not path.is_file():
        raise FileNotFoundError(f"Segmentation mask does not exist: {path}")

    with tifffile.TiffFile(path) as tif:
        shape = tif.series[0].shape
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D mask, found shape {shape}: {path}")
        if not 0 <= slice_index < shape[0]:
            raise IndexError(
                f"Slice {slice_index} is outside axis 0, which has {shape[0]} slices."
            )
        mask_slice = tif.pages[slice_index].asarray()

    unique_values = set(int(value) for value in np.unique(mask_slice))
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Expected a binary mask with values 0 and 1, found {unique_values}."
        )
    return mask_slice.astype(np.uint8, copy=False)


def load_ground_truth(path: Path) -> np.ndarray:
    """Load the supplied rendered ground-truth PNG."""
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth image does not exist: {path}")
    return np.asarray(imread(path))


def main() -> None:
    mask_slice = load_mask_slice(MASK_PATH, SLICE_INDEX)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    print(f"Mask slice shape: {mask_slice.shape}")
    print(f"Ground-truth rendering shape: {ground_truth.shape}")
    print(
        "The ground truth contains plot axes, margins, and a colorbar, so this "
        "viewer uses side-by-side comparison rather than a pixel overlay."
    )

    viewer = napari.Viewer(title=f"Segmentation comparison — slice {SLICE_INDEX}")
    viewer.add_labels(
        mask_slice,
        name=f"Agent mask — slice {SLICE_INDEX}",
    )
    viewer.add_image(
        ground_truth,
        name=f"Ground truth — slice {SLICE_INDEX}",
        rgb=ground_truth.ndim == 3,
    )

    # Grid mode gives each visible layer its own canvas for side-by-side review.
    viewer.grid.enabled = True
    viewer.grid.shape = (1, 2)
    napari.run()


if __name__ == "__main__":
    main()
