# MicroStage App

Windows desktop app (Python + PySide6) for controlling a motorized microscope stage (Arduino MEGA2560 + RAMPS running Marlin) and a RisingCam/ToupTek industrial camera from one UI. It supports live preview, jogging/home, capture, autofocus (with plane subtraction), raster mapping, time-lapse (roadmap), profiles, scripting, and diagnostics.

Note: This repository currently targets Windows for hardware use. You can run tests on any OS. The UI can launch cross‑platform but requires system OpenGL (libEGL) and the ToupTek SDK on the host if you want real camera input; otherwise it falls back to a mock camera.

## Features
- Live camera preview (pull-mode; RGB24 or RAW8 mono), FPS readout
- Stage control: jog X/Y/Z with user step (0.001–1000 mm) and feed (mm/s -> converted to mm/min), Home (G28), and sync (M400) before capture
- Capture: waits for motion to finish (M400), settles briefly, snapshots, and saves TIFFs
- Autofocus: coarse→fine sweep with selectable metric; plane subtraction scaffolding
- Raster mapping: grid (rows×cols, X/Y pitch); can be extended to time‑lapse
- Profiles & presets: YAML-backed defaults; quick recall of settings
- Scripting hooks: example Z‑stack script provided
- Diagnostics: CLI to enumerate Toupcam devices and probe Marlin ports

## Repo Layout
```
microstage_app/
  devices/
    stage_marlin.py      # Marlin serial driver (probe M115; jog G91/G1..F; G90; home G28; wait M400)
    camera_toupcam.py    # Toupcam backend (StartPullModeWithCallback, PullImageV2)
    camera_mock.py       # Mock camera (fallback if SDK missing/no device)
  ui/main_window.py      # PySide6 UI: Jog | Camera | Autofocus | Raster | Scripts
  control/
    autofocus.py         # coarse→fine autofocus + metrics
    raster.py            # raster scanning runner
    profiles.py          # profile persistence (profiles.yaml in repo root)
    focus_planes.py      # per‑area plane models and helpers
  io/storage.py          # TIFF writer (runs/<timestamp> folders)
  tools/diagnose.py      # environment, serial probe (M115), Toupcam enumerate
  scripts/zstack_example.py
```

Threading: GUI on the main thread; workers (serial, autofocus, raster) in QThreads; communicate via signals or polled state.

## Requirements
- Windows 10/11 recommended for hardware use; Python 3.10+
- Python packages (install via requirements.txt): PySide6, numpy, opencv-python, pyserial, tifffile, PyYAML, pytest
- Toupcam SDK (DLL + Python wrapper) if using a ToupTek/RisingCam camera

## Quick start
1) Create venv and install deps
```
python -m venv .venv
. .venv/bin/activate   # on Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

2) Place Toupcam bits (if using real camera)
- Put toupcam.dll and the Python wrapper on PATH or next to the app

3) Run diagnostics
```
python -m microstage_app
# or
microstage-app.tools.diagnose
```

4) Launch UI
```
python -m microstage_app
# or
microstage-app
```
If toupcam is missing or no devices are found, the app uses the mock camera.

## Tests
```
pytest -q microstage_app/tools/tests
```
Current tests cover:
- Camera fallback when toupcam module is unavailable
- Stage probe returns None when there are no serial ports

## Notes on serial probing
- We prefer CH340 (VID:PID 1A86:7523) and deprioritize COM1
- Probe Marlin via M115 at 250000 baud, then fall back to 115200
- The probe now returns (port, baud); the UI passes the detected baud to StageMarlin

## License
MIT. See LICENSE for details. Ensure vendor SDK licenses are followed for camera SDK files.


## Headless/CI notes
- Qt on Linux headless may require QT_QPA_PLATFORM=offscreen and QT_OPENGL=software.
- Real camera requires Toupcam SDK present; otherwise app uses MockCamera and tests still pass.


## Install from source
```
pip install -e .[dev]
```
