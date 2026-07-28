# Standalone CT Strut Viewer

This is a local, viewing-only website for inspecting struts listed in a CSV
against an arbitrary 3D CT TIFF and its registered ideal structure. It measures
and displays cross-sectional radius, but it does not classify defects.

## Required files

The viewer asks for three matching files:

1. A 3D `.tif` or `.tiff` volume stored in Z, Y, X order.
2. A registered `.json` file containing:
   - `junctions` with an `id` and voxel position `[x, y, z]`
   - `struts` with an `id` and the IDs of their two junctions
3. A `.csv` file containing a `strut_id` column.

Additional CSV columns are displayed as reference information for the selected
strut.

## Run with PowerShell

Open PowerShell and run these as **two separate commands**:

```powershell
cd C:\Users\dpala\projects\llnl_data_science_challenge_2026\standalone_strut_viewer
powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1
```

You may also put both commands on one line if they are separated by a
semicolon:

```powershell
cd C:\Users\dpala\projects\llnl_data_science_challenge_2026\standalone_strut_viewer; powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1
```

Keep the PowerShell window open. In a browser, open:

```text
http://127.0.0.1:8780/
```

Upload the TIFF, registered JSON, and CSV, then click **Load viewer**.

## Use another port

If port 8780 is already being used:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1 -Port 8781
```

Then open `http://127.0.0.1:8781/`.

## First-time dependency setup

The launcher first tries the Python runtime bundled with Codex. If that is not
available, it uses the `python` command installed on the computer.

From the `standalone_strut_viewer` folder, install the required packages with:

```powershell
python -m pip install -r .\requirements.txt
```

If NumPy reports incompatible compiled extensions, use a clean Python 3.12
Conda environment:

```powershell
conda create -n strut-viewer python=3.12 -y
conda activate strut-viewer
python -m pip install -r .\requirements.txt
powershell -ExecutionPolicy Bypass -File .\start_viewer.ps1
```

## Viewer controls

- Search for a strut using its `strut_id`.
- The main graph shows area-equivalent radius along the strut.
- **Open four views** displays XY, XZ, YZ, and perpendicular cross-sections.
- **CT tracking** toggles the cyan measured/segmented location.
- **Registered position** toggles the coral location expected from the JSON.
- Use each slider to move through its associated view.

## Stop the viewer

Return to the PowerShell window and press:

```text
Ctrl+C
```

You can also click **Clear files** before stopping.

## Data and memory handling

- Files remain on this computer and are not uploaded to an external service.
- The TIFF is streamed to temporary local storage and memory-mapped.
- JSON and CSV metadata are kept in process memory.
- Only the selected strut's local data and cross-sections are calculated.
- Clicking **Clear files** or stopping the server deletes its temporary TIFF.
