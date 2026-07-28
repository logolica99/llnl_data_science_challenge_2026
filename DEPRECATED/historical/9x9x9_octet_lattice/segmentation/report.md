# CT lattice segmentation report

## Input and metadata

- Input path: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/9x9x9_octet_lattice.tif`
- Shape (axis 0, 1, 2): `(761, 815, 837)`
- Input dtype: `>u2`
- Total voxels: 519119955
- Ground truth: not inspected or used.

## Final method and parameters

A global high-density mask was initialized by Otsu thresholding on a regular 3-D intensity sample. Lower, MAD-scaled candidate thresholds were admitted only if a downsampled 3-D continuity/noise score improved while foreground stayed within 5% of the Otsu baseline. No morphology was applied, avoiding erosion or artificial thickening of thin struts.

- Final threshold rule: `input >= 40129`
- Parameters: `{"connectivity_sample_strides": [8, 8, 8], "final_threshold": 40129, "foreground_guard_relative_to_otsu": 1.05, "histogram_sample_strides": [24, 4, 4], "refinement_step": 192, "sample_mad": 766.0, "sample_median": 32402.0, "sampled_otsu_threshold": 40705}`

## Final exact voxel statistics

- Foreground voxels: 58324474 (11.235259%)
- Background voxels: 460795481 (88.764741%)

## Iteration history

### Iteration 1

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 40705
- Sampled foreground: 0.112200
- Sampled 3-D components: 753
- Largest sampled 3-D component / sampled foreground: 0.986486
- Small sampled 3-D component burden: 0.008417
- Slice 380 components: 250
- Largest slice component / slice foreground: 0.056006
- Continuity/noise score: 0.978069
- Decision/failure: accepted as Otsu baseline
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_01.png`

### Iteration 2

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 40513
- Sampled foreground: 0.113725
- Sampled 3-D components: 723
- Largest sampled 3-D component / sampled foreground: 0.987651
- Small sampled 3-D component burden: 0.007877
- Slice 380 components: 249
- Largest slice component / slice foreground: 0.056344
- Continuity/noise score: 0.979774
- Decision/failure: accepted: continuity/noise score improved within 5% foreground guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_02.png`

### Iteration 3

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 40321
- Sampled foreground: 0.115290
- Sampled 3-D components: 708
- Largest sampled 3-D component / sampled foreground: 0.987793
- Small sampled 3-D component burden: 0.007584
- Slice 380 components: 251
- Largest slice component / slice foreground: 0.058495
- Continuity/noise score: 0.980209
- Decision/failure: accepted: continuity/noise score improved within 5% foreground guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_03.png`

### Iteration 4

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 40129
- Sampled foreground: 0.116813
- Sampled 3-D components: 679
- Largest sampled 3-D component / sampled foreground: 0.989217
- Small sampled 3-D component burden: 0.007260
- Slice 380 components: 256
- Largest slice component / slice foreground: 0.058397
- Continuity/noise score: 0.981957
- Decision/failure: accepted: continuity/noise score improved within 5% foreground guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_04.png`

### Iteration 5

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 39937
- Sampled foreground: 0.118502
- Sampled 3-D components: 647
- Largest sampled 3-D component / sampled foreground: 0.990553
- Small sampled 3-D component burden: 0.006500
- Slice 380 components: 250
- Largest slice component / slice foreground: 0.058389
- Continuity/noise score: 0.984053
- Decision/failure: rejected: foreground increase exceeded 5% guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_05.png`

### Iteration 6

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 39745
- Sampled foreground: 0.120193
- Sampled 3-D components: 615
- Largest sampled 3-D component / sampled foreground: 0.991795
- Small sampled 3-D component burden: 0.005907
- Slice 380 components: 250
- Largest slice component / slice foreground: 0.066728
- Continuity/noise score: 0.985887
- Decision/failure: rejected: foreground increase exceeded 5% guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_06.png`

### Iteration 7

- Method: global high-density threshold; sampled Otsu followed by MAD-scaled continuity refinement
- Threshold: 39553
- Sampled foreground: 0.121952
- Sampled 3-D components: 586
- Largest sampled 3-D component / sampled foreground: 0.992822
- Small sampled 3-D component burden: 0.005567
- Slice 380 components: 247
- Largest slice component / slice foreground: 0.066596
- Continuity/noise score: 0.987255
- Decision/failure: rejected: foreground increase exceeded 5% guard
- Feedback: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations/iteration_07.png`

## Stopping reason

three consecutive failed attempts without improvement

## Limitations

The result is unsupervised and is not an accuracy estimate. Component statistics use an 8-voxel regular subsample and can undercount diagonal or sub-resolution connections. Global thresholding may miss severe local attenuation changes; visual review of the saved slice and iteration feedback remains appropriate.

## Artifacts

- Program: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/segment_ct.py`
- Binary mask: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/mask.tif`
- Mask slice 380: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/slice_380.png`
- Report: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/report.md`
- Iteration feedback directory: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/iterations`
- Verification record: `/Users/dannyvillanueva/Documents/Livermore/llnl_data_science_challenge_2026/data/9x9x9_octet_lattice/segmentation/verification.json`
