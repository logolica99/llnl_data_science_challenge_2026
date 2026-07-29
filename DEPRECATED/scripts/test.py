import numpy as np

v = np.load("/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/unitcell/unitcell.npy", mmap_mode='r')
print(v.shape, v.dtype)                    # (256, 256, 256) float32
print(v.min(), v.max())                    # ← this is how you catch "it's NOT [0,1]"
print(np.histogram(v[::4,::4,::4], bins=10))  # subsample for speed; where's the material peak?

import tifffile
with tifffile.TiffFile("/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif") as tf:
    print(tf.series[0].shape, tf.series[0].dtype)   # (761, 815, 837) uint16 — free, no data read
    z = tf.asarray(key=380)                          # load just slice 380 (~1.4 MB)
print(z.min(), z.max())                              # ~29k–60k → thresholds live here, not at 0.5