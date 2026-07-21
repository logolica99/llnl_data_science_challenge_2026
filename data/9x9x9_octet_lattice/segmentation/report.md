# 9x9x9 Octet Lattice Segmentation Report

## Input and output

- Input: `data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`
- Input shape: `(761, 815, 837)` (axis order: z, y, x)
- Input dtype and range: big-endian `uint16`, 0 to 65,535
- Output mask: `data/9x9x9_octet_lattice/segmentation/mask.tif`
- Output shape and dtype: `(761, 815, 837)`, `uint8`
- Binary convention: 1 = lattice/material foreground; 0 = background
- Reproducible script: `data/9x9x9_octet_lattice/segmentation/segment_lattice.py`
- Evaluation image: `data/9x9x9_octet_lattice/segmentation/slice_380.png`
- Additional QC: `preview.png`, `overlay_slice_380.png`, and per-iteration histogram/preview PNGs

## Final method and parameters

The final method is slice-wise seeded hysteresis segmentation. Pixels at or above the high threshold form reliable material seeds. Pixels at or above the low threshold are retained only when they are 8-connected to a high-threshold seed. This recovers dim struts around reliable nodes without accepting every low-intensity pixel. A disk-shaped binary closing with radius 1 reconnects one-pixel gaps. Connected foreground components smaller than 6 pixels are removed as speckle.

- Low threshold: 40,000 (approximately the 95th percentile of slice 380)
- High threshold: 50,468 (approximately the 98th percentile of slice 380)
- Hysteresis connectivity: 8-connected within each z slice
- Closing radius: 1 pixel
- Minimum retained connected-component area: 6 pixels per slice
- Opening/dilation: none; full-mask opening damaged thin struts and large dilation was intentionally avoided

The volume is processed one z slice at a time and written to a memory-mapped BigTIFF, limiting peak memory use for the approximately 519-million-voxel dataset.

## Iteration history

| Iteration | Low | High | Close | Slice-380 foreground | Assessment |
|---:|---:|---:|---:|---:|---|
| 1 | 37,550 | 50,468 | 1 | 46,765 | Recovered thin struts, but over-connected a second/interior diagonal band not present in the reference. |
| 2 | 36,708 | 50,468 | 1 | 53,304 | More strut continuity, but clear over-segmentation and too many interior diagonals; regression. |
| 3 | 40,000 | 50,468 | 1 | 33,288 | Best balance: substantially improved left-edge diamond continuity with compact nodes and limited interior false positives. Selected. |
| 4 | 41,595 | 50,468 | 1 | 27,923 | Cleaner but noticeably broke the left-edge chain; regression, so iteration 3 was retained. |

Optimization stopped after four bounded attempts because the third candidate gave the best topology/thickness tradeoff and the stricter fourth candidate degraded strut continuity.

## Segmentation statistics

- Total voxels: 519,119,955
- Foreground voxels: 54,828,620
- Background voxels: 464,291,335
- Foreground fraction: 0.10561840 (10.561840%)
- Slice-380 foreground pixels: 33,288 of 682,155 (4.8798%)
- Verified mask values: exactly `{0, 1}`

## Qualitative assessment

The expected node grid is present and aligned. Slice 380 retains most of the left-edge diamond chain while the right side remains primarily isolated nodes, consistent with the supplied reference. Node sizes remain compact rather than ballooning into large blobs. Noise is low because weak pixels must connect to strong seeds and small isolated components are removed. Some very dim diagonal segments remain incomplete; lowering the threshold further was rejected because it produced conspicuous interior false-positive connections.

## Reproduction

From the repository root, run:

```bash
MPLCONFIGDIR=/tmp/llnl-mpl conda run -n dssi_env python data/9x9x9_octet_lattice/segmentation/segment_lattice.py
```

