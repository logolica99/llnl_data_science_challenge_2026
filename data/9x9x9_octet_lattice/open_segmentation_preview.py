#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import napari

# Use 0/255 version so Image layers look correct with default contrast
path = Path(__file__).with_name("segmentation_preview_vis_u8.npy")
mask = np.load(path)

viewer = napari.Viewer(title="segmentation preview")
layer = viewer.add_image(
    mask,
    name="mask_0_255",
    colormap="gray",
    contrast_limits=(0, 255),
    blending="opaque",
)
viewer.dims.ndisplay = 2
viewer.dims.set_point(0, 10)  # original CT slice 380
print("Opened", path, "shape", mask.shape, "go to slider index 10")
napari.run()
