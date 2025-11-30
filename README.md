# MicroStage App v0.1

A Windows-first Python/Qt application that controls a **Marlin-based microscope stage** (MEGA2560+RAMPS)
and a **RisingCam E3ISPM** camera (ToupTek OEM) via the vendor SDK. Includes **autofocus**, **focus plane correction (multi-area)**,
**area scan & timelapse acquisition**, **profiles/presets**, **robust device handling**, and **scripting hooks**.

## Quick start (Windows 10/11, 64‑bit)

1. Install **Python 3.10–3.12 (64‑bit)** on Windows; newer releases may fail until wheels are published.
2. Create & activate a venv, then install deps:
   ```bash
   pip install -r requirements.txt
   ```
   On Linux, OpenCV (`opencv-python`) requires the system library `libGL.so.1`.
   Install it via your package manager, e.g. `sudo apt-get install -y libgl1`,
  or run the helper script `scripts/install_libgl1.sh`. For headless setups,
  you may instead install `opencv-python-headless` to avoid the `libGL`
  dependency. The system monitor tab uses `psutil`; NVIDIA GPU metrics also
  require the optional `nvidia-ml-py3` package and appropriate drivers. When
  no GPU is present, the tab simply shows a notice so the app continues to
  run normally. Using unsupported Python versions can trigger source builds (e.g.
  for PySide6 or OpenCV) that need a compiler; stick to 3.10–3.12 or install a
  compatible wheel to avoid that hassle.

   Optional GPU acceleration for captures, autofocus metrics, and scale-bar
   drawing is available when OpenCV is built with CUDA modules. The
   application automatically detects CUDA support, falls back to CPU
   processing if the modules are absent, and logs which path is chosen to aid
   troubleshooting.
3. Install the **ToupTek / Toupcam SDK for Windows**. Copy the `toupcam.dll` (x64) next to `main.py` (or put it in your PATH).
   The SDK usually ships `toupcam.py` and examples; this app will auto-import if present.
   Basic USB webcams are also supported via OpenCV's ``VideoCapture`` and do not
   require this SDK, though only simple streaming and resolution selection are
   available. The app probes a handful of ``VideoCapture`` indices (0–3) to
   discover attached USB cameras. Generic webcam support depends on the
   ``opencv-python`` package, and any camera controls not exposed by a webcam
   (e.g. brightness or gain) will appear greyed out in the UI.
4. Connect your **Marlin** stage (Mega2560+RAMPS), power it on.
5. Run the app:
   ```bash
   python -m microstage_app
   ```
   On Windows you can also double-click `quicklaunch.cmd`, which delegates to the
   new Python launcher discussed below so the GUI starts without a console
   window.

> No camera? The app falls back to a software **MockCamera** so you can test UI and scans.

### Troubleshooting

Certain sliders or checkboxes may be disabled if the connected webcam does not
advertise those capabilities. Only controls for features reported by the camera
will be enabled.

## Features
- Device discovery: auto-detects Marlin via `M115` (verifying custom machine name and optional UUID), ToupCam via SDK enumeration, and probes a couple of OpenCV webcam indices.
- Live preview + jog controls (XY/Z), home, go-to.
- Capture primitives: move → settle → snap.
- Modes: Area (serpentine), Timelapse, Combined.
- Autofocus: Laplacian & Tenengrad metrics, coarse→fine search.
- Focus planes: planar/quadratic fits; **multiple areas** with priority.
- Profiles & presets: YAML, per-device and per-scan; import/export.
- Robustness: hot-plug (to be expanded), watchdogs (to be expanded), structured logs.
- Scripting: run custom recipes from `microstage_app/scripts/` with a safe API.
- Validated capture directory/filename fields with optional auto-numbering to prevent overwrites.
- Optional CUDA acceleration for capture, autofocus metrics, and scale-bar
  drawing when OpenCV is built with CUDA; falls back to CPU otherwise and
  logs the active path.
- System monitor tab displaying CPU load and, when supported, NVIDIA GPU
  utilization via NVML, with a message shown when GPUs are unavailable.

## Capture directory & file naming

The capture panel lets you choose an output folder and base filename. The fields
are validated: the directory must be writable (it will be created if missing) and
the name cannot contain characters such as `\\ / : * ? \" < > |`. The directory,
base name, auto-prefix, and auto-number options are all remembered between runs. Captures
default to PNG to retain embedded metadata, with BMP, TIFF and JPEG also
available.

Enabling **Auto-number (_n)** appends an incrementing suffix when a file with the
same name already exists, preventing accidental overwrites.

Enable **Auto-prefix (yyyymmddhhmmss_)** to prepend a timestamp such as
`20240101123045_` to each capture. When both toggles are active the timestamp
comes first, followed by the auto-number suffix (e.g. `20240101123045_sample_1.tif`).

Example usage:

1. Set directory to `C:/data/run1` and base name `sample`.
2. Check **Auto-number (_n)**.
3. Click **Capture** repeatedly to produce `sample.tif`,
   `sample_1.tif`, `sample_2.tif`, …

## Profiles and persistent settings

The application stores user preferences and scan presets in a YAML file named
`profiles.yaml` located in the application's working directory (alongside
`main.py` when running from source or the packaged executable). The file also
contains a `version` field so that older profiles can be migrated
automatically when new settings are introduced.

Saved fields include:

- **Stage**: feed rate (`feed_mm_s`) and settle delay (`settle_ms`).
- **Camera**: exposure time (`exposure_ms`), gain (1.0–4.0x), and binning.
- **Scan presets**: default area region (`x1_mm`, `y1_mm`, `x2_mm`,
  `y2_mm`, `rows`, `cols`).
- **Capture**: last used directory, base filename, auto-prefix toggle,
  auto-number toggle, and
  file format.
- **Jog panel**: step sizes, feed rates, and absolute positions for each axis.

To reset to factory defaults, delete `profiles.yaml`; the file will be recreated
the next time the app starts. To transfer your settings to another machine, copy
this `profiles.yaml` file to the target system's working directory.

## Packaging
```bash
pyinstaller -F -w -n MicroStageApp microstage_app/main.py
```

## Launcher and interpreter selection

The new `scripts/quicklaunch.py` helper finds the repository root by looking
for `microstage_app/__init__.py`. If the file is not present alongside the
launcher, the script inspects immediate subdirectories named
`MicroTableController*` until it locates the package. Once the root is
identified the script searches for an interpreter in priority order:

1. `python/pythonw.exe` (embedded runtime shipped with the distribution).
2. `.venv/Scripts/pythonw.exe` (virtual environment, GUI-friendly build).
3. `.venv/Scripts/python.exe` (virtual environment console interpreter).
4. The interpreter running `quicklaunch.py` (`sys.executable`).

When the embedded runtime is chosen, `scripts/ensure_embedded_python_ready.cmd`
is invoked before the GUI is started so the bundled dependencies are installed
or refreshed. The GUI is launched with `pythonw.exe` whenever possible so end
users do not see a console window; the code automatically falls back to
`python.exe` in environments that do not provide `pythonw.exe` (for example,
stock CPython installs on Linux or minimal virtual environments).

`quicklaunch.cmd` is now just a thin wrapper that executes
`pythonw.exe scripts\quicklaunch.py`, falling back to `python.exe` when the
windowless binary is not available. Packagers that ship the embedded runtime
should place it in the top-level `python/` directory next to the launcher; the
ensure script will handle the first-run dependency bootstrap. When targeting an
existing Python installation instead, provide a ready-to-use `.venv` directory
or instruct users to install requirements before running `quicklaunch.cmd`.

## License

This project is licensed under the [MIT License](LICENSE).
