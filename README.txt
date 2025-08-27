MicroStage App

Windows desktop app (Python + PySide6) for controlling a motorized microscope stage (Arduino MEGA2560 + RAMPS running Marlin) and a RisingCam/ToupTek industrial camera from one UI. It supports live preview, jogging/home, capture, autofocus (with plane subtraction), area mapping, time-lapse, profiles, scripting, and robust diagnostics.

Stage speaks Marlin G-code over USB serial; camera uses the ToupTek Toupcam SDK (Python wrapper + toupcam.dll). Feedrates are mm/min (Marlin convention).

--------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------
- Live camera preview (pull-mode; RGB24 or RAW8 mono), FPS readout.
- Stage control: jog X/Y/Z with user step (0.001–1000 mm) & feed (mm/s -> converted to mm/min), Home (G28), and sync (M400) before capture.
- Capture: wait for motion to finish (M400), settle briefly, snapshot, and save.
- Area tab: autofocus (coarse->fine sweep with selectable metric and plane subtraction
  for multi-sample runs), grid mapping (rows x cols, X/Y pitch), and optional time-lapse.
- Profiles & presets: persist and quickly recall settings.
- Scripting hooks: run small workflows (e.g., Z-stack example) right from the UI.
- Diagnostics: CLI tool to enumerate Toupcam devices and probe Marlin ports.

--------------------------------------------------------------------------
HARDWARE
--------------------------------------------------------------------------
- Stage: Arduino Mega 2560 + RAMPS 1.4 (USB-to-serial often CH340, VID:PID 1A86:7523).
- Firmware: Marlin (serial baud commonly 250000, 115200 as fallback).
- Camera: RisingCam E3ISPM25000KPA (ToupTek family) via Toupcam SDK. Supports ROI, bit-depth switching, and a speed/bandwidth level.
Note: Arduino boards usually auto-reset when the serial port opens (DTR/RTS). Allow a short delay after opening the port.

--------------------------------------------------------------------------
ARCHITECTURE (DIRECTORY OVERVIEW)
--------------------------------------------------------------------------
microstage_app/
  devices/
    stage_marlin.py      - Marlin serial driver (probe M115, jog G91/G1..F, G90, home G28, wait M400)
    camera_toupcam.py    - Toupcam backend (StartPullModeWithCallback, PullImageV2)
    camera_mock.py       - Mock camera for development without hardware
  ui/
    main_window.py       - PySide6 UI: Jog | Camera | Area | Scripts tabs
  control/
    autofocus.py         - coarse->fine autofocus + plane subtraction tools (global/per-area)
    raster.py            - grid mapping runner (+ optional time-lapse)
    profiles.py          - profiles/presets
  utils/
    serial_worker.py     - persistent QThread loop for non-blocking serial I/O
    workers.py           - run_async helper returning (thread, worker)
    img.py, log.py       - numpy->QImage, structured logging to UI + stdout
  io/
    storage.py           - image writer (run folder, tiles)
  tools/
    diagnose.py          - env, serial probe (M115), Toupcam enumerate
  scripts/
    zstack_example.py    - sample script wired to UI

Threading model: All Qt widgets live on the GUI thread. Workers (serial, autofocus, raster) run in QThreads and communicate via signals or shared data polled by a QTimer.

Camera acquisition (Toupcam): Enumerate -> open -> StartPullModeWithCallback(cb, ctx) -> on TOUPCAM_EVENT_IMAGE, PullImageV2(buffer, bits, None) into a stride-aligned buffer; convert BGR->RGB if needed; store as “latest frame.” The GUI timer paints that frame.

--------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------
- Windows 10/11, Python 3.10+ (developed on 3.13).
- Python packages: PySide6, numpy, opencv-python, pyserial.
- Toupcam SDK (DLL + Python wrapper). Place toupcam.dll where Python can find it (working dir or on PATH).

--------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------
1) Create venv & install deps
   py -3.13 -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt

2) Place Toupcam bits
   Copy toupcam.dll (and toupcam.py if using the vendor wrapper)
   beside the app or add its folder to PATH.

3) Run the app
   python -m microstage_app

4) Diagnostics
   python -m microstage_app.tools.diagnose

Ports & baud: We probe for Marlin by opening the COM port at 250000 and sending M115 to find
    'FIRMWARE_NAME:Marlin' and the expected MACHINE_NAME. If a MACHINE_UUID is provided, it is used
    to choose between multiple compatible boards. If that fails, try 115200.

--------------------------------------------------------------------------
USING THE APP
--------------------------------------------------------------------------
Jog:
- Set Step (0.001–1000 mm) and Feed (mm/s). The app converts to mm/min for Marlin G1 ... F moves.
- Home uses G28. Wait/sync uses M400 before capture.

Camera:
- Exposure (ms) (Auto optional), Gain, RAW8 fast mono (8-bit; ~1/3 bandwidth vs RGB24), Resolution index,
  ROI presets (Full, 2048^2, 1024^2, 512^2), USB speed/bandwidth level, and display decimation (show every Nth frame).
- Smaller ROI / shorter exposure -> higher FPS.

Capture:
- M400 wait -> short settle -> snapshot latest frame -> save to the current run folder.

Area:
- Autofocus: coarse->fine Z sweep using selected metric, with plane subtraction tools
  (global or per-area planes) to handle multiple samples in one run.
- Mapping & time-lapse: configure rows, cols, pitch X/Y; the runner visits each tile,
  captures, and can repeat on a schedule.

Scripts:
- Z-stack example provided; add more scripts that receive (stage, camera, image_writer).

--------------------------------------------------------------------------
DIAGNOSTICS & TROUBLESHOOTING
--------------------------------------------------------------------------
- python -m microstage_app.tools.diagnose prints Python env, confirms module imports,
  enumerates COM ports with VID:PID, probes Marlin (M115), and lists Toupcam devices.

Common camera issues:
- DLL not found -> put toupcam.dll beside the app or on PATH; ensure OS/bitness matches the DLL.
- Slow FPS -> lower exposure, enable RAW8 mono, choose smaller ROI/resolution, adjust USB speed level.

Common stage issues:
- Port opens but no response -> Arduino may auto-reset on open (DTR/RTS). Give it ~1–2 s before sending G-code.
- Wrong baud -> confirm Marlin’s configured baud. Defaults often use 250000; if unstable, try 115200.

--------------------------------------------------------------------------
THREADING & SAFETY
--------------------------------------------------------------------------
- Never create or touch Qt widgets off the GUI thread. Use signals/slots from worker threads and paint
  the preview from the main thread (we poll the latest frame with a QTimer).

--------------------------------------------------------------------------
MARLIN PROTOCOL QUICK REF
--------------------------------------------------------------------------
- Probe: M115 -> parse 'FIRMWARE_NAME:Marlin' and custom MACHINE_NAME to confirm device (UUID optional).
- Home: G28 (all axes by default).
- Jog (relative): G91 -> G1 X/Y/Z... F... -> G90 (feed in mm/min).
- Wait: M400 before capture/critical sequencing.

--------------------------------------------------------------------------
DEVELOPMENT NOTES
--------------------------------------------------------------------------
- Keep GUI updates on the main thread.
- For long-running tasks: run in a worker (QThread) and emit signals for progress.
- Keep logs concise and structured (utils/log.py) — all serial TX (TX >>) and major camera events should be visible.

--------------------------------------------------------------------------
NEAR-TERM ROADMAP
--------------------------------------------------------------------------
- Time-lapse job UI and combined area+time-lapse scheduling.
- Multi-area plane subtraction UX (define ROIs, cache planes per sample).
- Additional autofocus metrics + hill-climb refinement.
- Profiles import/export, per-mode overrides.
- More tools: tools/cam_probe.py, tools/marlin_echo.py smoke tests.

--------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------
TBD (MIT/Apache-2.0 recommended). Ensure bundled SDK files respect vendor licenses.
