---
name: metadata-extractor
description: Loads a .npy volume or mask and prints basic metadata (shape, dtype, min, max, mean) to the terminal.
---

# Metadata Extractor Protocol

You are the **Metadata Extractor**. When this skill is active, inspect a `.npy` array and report its basic properties.

### Step 1: Locate the Input
- Use the `.npy` path provided by the user.
- If none is given, prefer files under `./data/` (e.g. `data/unitcell/unitcell.npy` or a generated mask/skeleton).

### Step 2: Extract Metadata
Run the helper script from this skill directory:

```bash
python .agents/skills/metadata_extractor/scripts/extract_metadata.py <path_to_npy>
```

The script must print at least:
- file path
- shape
- dtype
- minimum value
- maximum value
- mean value
- (optional) nonzero / foreground voxel count for binary-like arrays

### Step 3: Report
Summarize the printed metadata for the user in a short table or bullet list. Do not invent values — only report what the script printed.

# Technical Constraints
- Do not modify the input `.npy` file.
- If the file is missing or not a valid NumPy array, report the error clearly and stop.
- If you create temporary helper scripts outside this skill folder, delete them when finished.
