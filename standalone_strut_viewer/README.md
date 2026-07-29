# Standalone CT Strut Viewer

This is a local website for reviewing thin, thick, and bent struts described by
analysis JSON files against a 3D CT TIFF and its registered ideal structure. It
uses embedded pipeline cross-sectional radius and centerline-deviation
measurements when available, so the graphs match the uploaded classifications.
Older JSONs without embedded measurements use a clearly labeled live preview.

## Required files

The viewer asks for three matching groups:

1. A 3D `.tif` or `.tiff` volume stored in Z, Y, X order.
2. A registered `.json` file containing:
   - `junctions` with an `id` and voxel position `[x, y, z]`
   - `struts` with an `id` and the IDs of their two junctions
3. One or more analysis `.json` files.

Finding JSONs may use `findings`, `classified_struts`, `entries`, or `results`
arrays; every finding needs a `strut_id` (or `id`). You may select all JSON
artifacts from a pipeline run at once. Recognized threshold, measurement
manifest, and hand-off JSONs are retained as run metadata, while duplicate
strut findings are merged by ID.

For the thin/thick/bent pipeline, select files such as
`findings_thin.json`, `findings_thick.json`, `findings_bent.json`,
`thresholds.json`, `measurement_manifest.json`, and `handoff.json`.
Current class-specific findings embed the exact sampled radii, tracked centers,
deviations, confidence, exclusions, CT threshold, and section-artifact hash.

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

Upload the TIFF, registered JSON, and one or more analysis JSONs, then click
**Load viewer**.

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
- Filter the catalog by thin, thick, bent, normal, or uncertain.
- The radius graph shows area-equivalent radius along the strut and, when
  available, the uploaded peer median as a dashed reference.
- The deviation graph shows distance from the tracked CT center to its best-fit
  straight centerline. A bent threshold is shown when supplied in JSON.
- A green provenance banner identifies embedded pipeline measurements and their
  CT threshold. An amber banner identifies a viewer preview that was not used
  for classification.
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
- JSON metadata is kept in process memory.
- Embedded profiles render without recalculating CT cross-sections. Older
  result JSONs calculate a preview only for the selected strut.
- Profiles use bounded browser/server caches. CT crops keep their native
  16-bit representation when possible and the browser retains only two crops.
- Clicking **Clear files** or stopping the server deletes its temporary TIFF.
