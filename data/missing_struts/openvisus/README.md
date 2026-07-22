Visualization scripts live in `scripts/openvisus/` (kept out of Git LFS).

```bash
conda activate dssi_env
PYTHONPATH=.python_pkgs python scripts/openvisus/convert_tiff_to_idx.py
PYTHONPATH=.python_pkgs python scripts/openvisus/view_idx.py --mode plane111 --downsample 2
```
