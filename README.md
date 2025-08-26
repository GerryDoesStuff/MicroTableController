# MicroStage App v0.1

A Windows-first Python/Qt application that controls a **Marlin-based microscope stage** (MEGA2560+RAMPS)
and a **RisingCam E3ISPM** camera (ToupTek OEM) via the vendor SDK. Includes **autofocus**, **focus plane correction (multi-area)**,
**raster & timelapse acquisition**, **profiles/presets**, **robust device handling**, and **scripting hooks**.

## Quick start (Windows 10/11, 64‑bit)

1. Install **Python 3.10+ (64‑bit)**.
2. Create & activate a venv, then install deps:
   ```bash
   pip install -r requirements.txt
   ```
   On Linux, OpenCV (`opencv-python`) requires the system library `libGL.so.1`.
   Install it via your package manager, e.g. `sudo apt-get install -y libgl1`,
   or run the helper script `scripts/install_libgl1.sh`. For headless setups,
   you may instead install `opencv-python-headless` to avoid the `libGL`
   dependency.
3. Install the **ToupTek / Toupcam SDK for Windows**. Copy the `toupcam.dll` (x64) next to `main.py` (or put it in your PATH).
   The SDK usually ships `toupcam.py` and examples; this app will auto-import if present.
4. Connect your **Marlin** stage (Mega2560+RAMPS), power it on.
5. Run the app:
   ```bash
   python -m microstage_app
   ```

> No camera? The app falls back to a software **MockCamera** so you can test UI and scans.

## Features
- Device discovery: auto-detects Marlin via `M115` (verifying custom machine name and optional UUID) and ToupCam via SDK enumerate.
- Live preview + jog controls (XY/Z), home, go-to.
- Capture primitives: move → settle → snap.
- Modes: Timelapse, Raster (serpentine), Combined.
- Autofocus: Laplacian & Tenengrad metrics, coarse→fine search.
- Focus planes: planar/quadratic fits; **multiple areas** with priority.
- Profiles & presets: YAML, per-device and per-scan; import/export.
- Robustness: hot-plug (to be expanded), watchdogs (to be expanded), structured logs.
- Scripting: run custom recipes from `microstage_app/scripts/` with a safe API.
- Validated capture directory/filename fields with optional auto-numbering to prevent overwrites.

## Capture directory & file naming

The capture panel lets you choose an output folder and base filename. The fields
are validated: the directory must be writable (it will be created if missing) and
the name cannot contain characters such as `\\ / : * ? \" < > |`. The directory,
base name, and auto-number option are all remembered between runs.

Enabling **Auto-number (_n)** appends an incrementing suffix when a file with the
same name already exists, preventing accidental overwrites.

Example usage:

1. Set directory to `C:/data/run1` and base name `sample`.
2. Check **Auto-number (_n)**.
3. Click **Capture** repeatedly to produce `sample.tif`,
   `sample_1.tif`, `sample_2.tif`, …

## Packaging
```bash
pyinstaller -F -w -n MicroStageApp microstage_app/main.py
```

## License

This project is licensed under the [MIT License](LICENSE).
