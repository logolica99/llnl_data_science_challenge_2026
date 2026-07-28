from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

repo = Path("/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026")
histogram_path = (
    repo
    / "data/9x9x9_octet_lattice/segmentation/intensity_histogram.npy"
)

histogram = np.load(histogram_path, allow_pickle=False)
intensities = np.arange(histogram.size)

plt.plot(intensities, histogram)
plt.axvline(34963, color="red", label="Selected threshold: 34,963")
plt.yscale("log")
plt.xlabel("CT intensity")
plt.ylabel("Voxel count")
plt.title("CT intensity histogram")
plt.legend()
plt.show()