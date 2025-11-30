# Vis Spectroscopy Module Design Document

## Context and goals

The current UI exposes device selection through the Devices menu and a View menu, with left-column device controls and right-hand tabs for camera/area/scripts/system functions; there is no spectroscopy support yet.

A new `spectroscopyModuleStructure` folder contains reference screenshots for a spectroscopy workflow, including a main spectroscopy window, a mode selector, and per-mode step flows for absorbance, fluorescence, Raman, reflectance, relative irradiance, and transmittance (with background/reference acquisition, wavelength/config steps, and finish states).

Goal: introduce a Vis Spectroscopy module driven by OceanOptics / Ocean Insight USB2000-class spectrometers, integrated into the existing device architecture, reusing established UX patterns, and ready for later expansion (additional spectrometers, more modes, raster integration, etc.).

---

## Frontend design

### Navigation and entry points

- Insert a **Modules** menu between **Devices** and **View** in the main menu bar.
  - This menu contains a single action: **Vis Spectroscopy…**, which launches the spectroscopy window.
  - The position mirrors the existing Devices menu placement for consistency.

### Devices tab integration

- Extend the **Devices** tab with a new **Spectrometers** section that:
  - Enumerates available OceanOptics spectrometers (USB2000 and future-compatible models).
  - Shows them in a plain list with:
    - model name,
    - serial number,
    - USB path or logical ID.
  - Provides **Connect / Disconnect** controls mirroring the existing Cameras / Stages discovery UX.
- Spectrometer connections are managed by the same central **device manager** as cameras and stages:
  - The spectroscopy window attaches to the spectrometer already connected via the Devices tab.
  - The spectroscopy window does not open its own independent connection; it uses the shared device abstraction.

### Spectroscopy window layout

#### Ownership and modality

- The spectroscopy window is a normal, modeless child window:
  - The main app remains usable while it is open.
  - When the main app closes, the spectroscopy window also closes.
  - Closing the spectroscopy window does **not** disconnect the spectrometer device.

#### Main layout

- **Top bar:**
  - Device drop-down showing connected spectrometer(s) (from device manager), current selection, model, serial/USB ID, connection state, and a simple status indicator (LED-style).
  - Button for **Refresh** (manual re-enumeration in case of hot plugging).
  - Quick indicator of current mode (Absorbance, Transmittance, etc.).
- **Center:**
  - **Live spectrum plot** with:
    - primary axis: wavelength (nm),
    - optional secondary axis: Raman shift (cm⁻¹) in Raman mode,
    - legend showing active traces (current spectrum, dark, reference, overlays).
- **Right/Bottom:**
  - Acquisition controls:
    - Integration time slider + spinbox.
    - Averages spinbox.
    - Dark correction toggle.
    - Smoothing / boxcar controls.
    - Single capture button.
    - Continuous acquisition **Start / Stop** buttons.
  - Mode selector access:
    - Button: **Modes…** (opens mode selector dialog).
  - Shelly Dimmer 2 light source controls.
  - Status/footer:
    - Saturation warning.
    - Acquisition FPS (or update rate).
    - Connection / error messages.

#### Compaction / layout modes

- Support a minimal presentation mode where only:
  - the live plot,
  - minimal status bar,
  - and start/stop controls
  are visible, with advanced controls collapsed into a toggleable side panel.
- The window is resizable, and its last size and layout mode are persisted across runs via the profiles system.

### Spectral plot interactions and tools

The live spectrum plot should provide the standard interaction tools expected in a spectroscopy application:

- **Pan and zoom**
  - Mouse-based panning (e.g. click-drag with middle mouse or Ctrl+drag).
  - Zoom in/out:
    - scroll wheel zoom around cursor,
    - rectangular zoom (click-drag to define zoom region),
    - toolbar buttons for **Zoom In**, **Zoom Out**, **Reset View**.
  - Independent control over:
    - X-axis (wavelength / Raman shift) limits,
    - Y-axis (intensity / absorbance / mode-specific unit) autoscale or fixed limits.

- **Cursor / crosshair**
  - A movable vertical dotted line showing the currently selected wavelength (or Raman shift).
  - Optional horizontal guide line for intensity.
  - Numeric readout area under the plot displaying:
    - selected wavelength (nm) and/or Raman shift (cm⁻¹),
    - corresponding intensity / absorbance / other derived quantity,
    - optionally values for multiple traces at the same X location (e.g. current vs reference).

- **Region-of-interest (ROI) tools**
  - Ability to drag-select a wavelength interval (ROI) on the plot.
  - ROIs used for:
    - band-averaged absorbance,
    - integration under curve (for fluorescence),
    - selecting calibration wavelengths for Beer–Lambert (e.g. choose λ_max or a band).
  - ROI definitions are stored in `SpectroscopySession` and persisted per mode where relevant.

- **Trace management**
  - Toggle visibility of individual traces from the legend (clickable legend entries).
  - Highlight active trace on hover or selection.
  - Option to “pin” a spectrum (e.g. reference) for comparison so it remains visible when new captures arrive.

- **Scaling and display options**
  - Intensity axis:
    - linear vs logarithmic scale toggle (where meaningful).
    - autoscale and “fit to data” functions.
  - Grid toggle (show/hide grid lines).
  - Option to change line thickness, marker visibility (basic appearance controls).

- **Export of plot**
  - Provide “Save plot as image…” (PNG, SVG) using current view (axes limits and visible traces).
  - Respect current zoom, ROI, and legend visibility.

These controls apply both to live data and static views in the final result screens of each mode.

### Mode selector dialog and wizards

- **Mode selector dialog:**
  - Mimic `spectroscopyModeWindow.png` and associated screenshots.
  - Present buttons/cards for:
    - Absorbance
    - Transmittance
    - Reflectance
    - Relative Irradiance
    - Fluorescence
    - Raman
  - Selecting a mode opens the corresponding wizard.

#### Absorbance

- Subflows:
  - “Absorbance only”
  - “Beer–Lambert”
  - “Calibrate on known concentration”
- Steps:
  1. **Wavelength / configuration**
     - Display wavelength axis from device calibration.
     - Allow optional wavelength-range restriction (e.g. 400–800 nm) for plotting and downstream calculations.
  2. **Acquisition parameters**
     - Integration time, averages, smoothing.
     - Show saturation indicator during parameter tuning (via live preview).
  3. **Dark (background) capture**
     - Light off; acquire and store dark spectrum.
  4. **Reference capture**
     - Light on, reference sample in place; acquire and store reference spectrum.
  5. **Optional compound / calibration setup**
     - For Beer–Lambert / known concentration flows:
       - Allow the user to define:
         - one or more wavelengths or a band for absorbance evaluation (using ROI tools or explicit numerical input),
         - concentration points and their measured absorbance (points captured via guided steps).
       - Fit a calibration curve according to user choice (e.g. linear through origin or general linear fit).
  6. **Final result**
     - Display absorbance spectrum \(A(\lambda) = -\log_{10}(I/I_0)\).
     - Show calibration results where applicable.
     - Provide options to overlay raw, dark, and reference spectra and to re-run calibration steps.

#### Transmittance / Reflectance

- Steps:
  1. Wavelength/config.
  2. Acquisition parameters.
  3. Dark capture.
  4. Reference capture.
  5. Final result:
     - Display T(\(\lambda\)) or R(\(\lambda\)) as normalized intensity (I/I₀) or in percentage.
     - Overlay options for raw, dark, reference.

#### Relative irradiance

- Steps:
  1. Wavelength/config.
  2. Acquisition parameters.
  3. Dark capture.
  4. Reference / calibration:
     - Load or capture reference lamp data (using a known-calibration lamp or manufacturer calibration file) for spectral response correction.
  5. Color/irradiance configuration:
     - Option to enable:
       - spectral response correction to obtain relative spectral irradiance \(E(\lambda)\) up to a multiplicative constant,
       - CIE XYZ/xyY and correlated color temperature (CCT) computation over a selected wavelength band.
  6. Final result:
     - Display relative spectral irradiance \(E(\lambda)\) after correction.
     - Display derived color metrics (xy chromaticity, CCT) where configured.

#### Fluorescence

- Steps:
  1. Wavelength/config.
     - Option to define emission observation range (e.g. block out region near excitation if desired).
  2. Acquisition parameters.
  3. Fluorescence configuration:
     - Excitation wavelength (for documentation).
     - Notes on filters (emission filter, dichroic) for metadata.
  4. Final result:
     - Display emission spectrum with metadata.
     - Baseline correction hooks exist but strategy is configured later (no fixed default in v1; capability present in processing backend).

#### Raman

- Steps:
  1. Wavelength/config.
     - Display device wavelength axis.
     - Allow definition of:
       - laser excitation wavelength \(\lambda_{\text{exc}}\),
       - optional notch / edge filter region to ignore.
     - Precompute Raman shift axis:  
       \[
       \Delta \nu \,(\text{cm}^{-1}) = (1/\lambda_{\text{exc}} - 1/\lambda) \times 10^7
       \]
  2. Acquisition parameters.
  3. Raman configuration:
     - Confirm \(\lambda_{\text{exc}}\),
     - specify filter details for metadata.
  4. Final result:
     - Display spectrum vs Raman shift (cm⁻¹) on the main axis, with an optional secondary wavelength axis.
     - Baseline correction hooks present but not configured with a fixed default; methods (e.g. ALS, polynomial) can be plugged in later via configuration.

### Shared controls

- **Device selection:**
  - Device drop-down listing connected spectrometers from the device manager.
  - Connect/Disconnect handled in Devices tab; spectroscopy window reflects state and allows switching between devices that are already connected.

- **Acquisition controls:**
  - Integration time slider + numeric entry.
  - Averages (integer).
  - Boxcar / smoothing controls (with clear indication of whether applied pre- or post-mode computation).
  - Dark correction toggle (apply stored dark spectrum when enabled).

- **Capture / acquisition:**
  - **Single capture** button (one spectrum, processed through pipeline).
  - **Continuous acquisition** Start/Stop buttons.
  - **Save spectrum** button:
    - Save current spectrum (and mode) to individual file (e.g. CSV, plus optional sidecar metadata).
  - **Export data**:
    - Export processed spectra in:
      - CSV (wavelength + values),
      - one or more standard spectral formats (e.g. JCAMP-DX, and/or a documented internal JSON/HDF5 schema),
      - optional raw counts export for debugging.

- **Live plot:**
  - Implemented via matplotlib or QtCharts.
  - Supports:
    - pan, zoom, ROI tools, cursor/crosshair as above,
    - overlay of multiple spectra,
    - toggling traces via legend.

- **Status/footer:**
  - Saturation indication (e.g. “Saturated” if any pixel exceeds threshold).
  - Approximate acquisition FPS.
  - Connection state and last error message.

### Result panes and history

- Final screens per mode present computed metrics (absorbance, transmittance, reflectance, relative irradiance, Raman shift, etc.).
- Provide a side panel listing **recent captures**:
  - Metadata:
    - timestamp,
    - mode,
    - integration time,
    - averages,
    - whether dark/reference applied,
    - device and serial.
  - User can select previous captures to:
    - overlay on the plot,
    - export in desired formats,
    - use as a reference for calibration where applicable.

### Shelly Dimmer 2 integration

- Integrate **Shelly Dimmer 2** light source controls into the spectroscopy window:
  - Controls:
    - On/Off toggle.
    - Brightness slider.
    - Optional presets (e.g. “Dark for background”, “Reference intensity”, “Measurement intensity”).
  - All state changes logged via existing LOG utilities.
  - Mode wizards can prompt scripted light changes:
    - Turn off lamp for dark capture.
    - Set reference preset for reference capture.
    - Set measurement preset for sample measurement.

### Spectral save folder and raster integration

- **Save folder selection**
  - Add a setting for the **default spectral data folder**, configurable via:
    - a preferences entry in the main app, and
    - a shortcut button in the spectroscopy window to quickly change the output folder.
  - All exports (single spectra, CSV, standard formats, raster cubes) default to this folder, with per-save overrides allowed.

- **Raster scan integration and hyperdata cubes**
  - Provide an option in raster/area scan workflows:
    - “Acquire spectrum at each scan position using current spectroscopy settings”.
  - For each scan point, the system stores:
    - stage coordinates (X, Y, optionally Z),
    - full spectrum (wavelength axis shared),
    - mode and acquisition parameters.
  - Data structure:
    - Wavelength axis: 1D array \(\lambda[N_\lambda]\).
    - Raster grid: X × Y (and optionally Z) grid of positions.
    - Spectral cube: \(C[x, y, (z,), \lambda]\) containing intensity (or mode-specific processed quantity) per point.
  - File format:
    - Use an efficient container (e.g. HDF5 or NPZ) that stores:
      - wavelength axis,
      - coordinate arrays (X, Y, Z indices or absolute units),
      - 3D/4D spectral cube,
      - metadata (device, mode, timestamps, parameters).
  - The spectroscopy window and raster modules share access to this saved hypercube structure for later visualization and analysis.

### Time-series logging

- Provide a **time-series acquisition mode**:
  - Acquire spectra at fixed time intervals (e.g. every N seconds) using current settings.
  - Store results as:
    - wavelength axis,
    - time axis,
    - 2D array spectra[t, λ],
    - associated metadata (device, mode, parameters).
  - Save to HDF5/NPZ using a similar schema to raster cubes (time as the additional dimension instead of spatial coordinates).
  - Plot options:
    - view selected wavelengths as intensity vs time.
    - view selected time slices as spectra.
    - optionally a spectrogram-style plot (time vs wavelength, colored by intensity).

---

## Interaction patterns

### Acquisition threading and event flow

- All device I/O and spectrum acquisition run in a **worker thread** (QThread or equivalent):
  - UI remains responsive during continuous acquisition.
- **Single capture:**
  - Issue one `read_spectrum()` in worker thread → pipeline → emit processed spectrum to UI.
- **Continuous acquisition:**
  - Worker loop:
    - Acquire raw spectra respecting integration time and averages.
    - Process through pipeline.
    - Emit spectra at a controlled rate (e.g. capped to a reasonable FPS).
  - Stop conditions:
    - User presses Stop.
    - Device error/disconnect.
    - Application shutdown.
- **UI updates:**
  - Use signals/slots to push new spectra to the plot and history list.
  - Rate limiting to avoid overloading the plot and event loop.

### Hot plugging and refresh

- On application start and when opening the spectroscopy window:
  - Query device manager for spectrometers.
- Support hot plugging:
  - Device manager reacts to USB changes and updates its device list.
  - Spectroscopy window listens for device manager signals and updates its drop-down and status.
  - If the currently selected device disappears:
    - Stop any ongoing acquisition.
    - Mark spectrometer as disconnected and prompt the user to reconnect or select another device.
- Manual **Refresh** button remains available to force re-enumeration if auto-detection fails.

### Wizard navigation and state retention

- Mode wizards use Next/Back/Finish consistent with screenshots.
- **Finish** remains disabled until mandatory steps (dark, reference, required calibration) are completed and valid.
- Session state:
  - Within a given app run, `SpectroscopySession` retains:
    - current mode,
    - dark/reference spectra,
    - calibration data,
    - last acquisition settings,
    - defined ROIs.
  - Across app runs, profiles store:
    - last-used device,
    - last window geometry and layout,
    - last mode and core acquisition parameters,
    - last-selected data output folder.
  - Dark/reference and calibration sets are optionally persistable with names and parameters; on load, the system compares saved parameters to current settings and marks them invalid if incompatible.

---

## Backend design

### Device abstraction and reuse

- Define a generic **SpectrometerDevice** interface, for example:
  - `enumerate() -> List[Descriptor]`
  - `open(descriptor)`
  - `close()`
  - `get_wavelengths() -> np.ndarray`
  - `set_integration_time(microseconds)`
  - `set_averages(n)`
  - `read_raw_spectrum() -> np.ndarray`
  - `get_bit_depth()` or max count
  - optional: `get_serial()`, `get_model()`, `get_usb_path()`.
- **OceanOptics implementation** (`spectrometer_oceanoptics.py`):
  - Use a Python OceanOptics/Ocean Insight library (e.g. seabreeze or vendor SDK) to implement this interface.
  - Handle:
    - enumeration,
    - reading wavelength coefficients from EEPROM,
    - integration time and averaging configuration,
    - raw spectrum acquisition.
  - Implement context-managed resource handling for safe open/close and reconnect logic similar to existing camera/stage flows.
- **Mock backend:**
  - Provide a `MockSpectrometer` implementing the same interface:
    - Generates synthetic spectra (Gaussian peaks, noise, optional drift).
    - Used for unit tests and for a “no hardware available” mode.
  - Device manager can expose the mock device when no real hardware is present or when explicitly requested.
- **Multi-spectrometer support:**
  - Device manager supports more than one spectrometer at once.
  - Spectroscopy window’s device drop-down allows switching between connected spectrometers.
  - Concurrency rules:
    - For a given spectrometer, only one active acquisition loop at a time.
    - Different spectrometers can run acquisitions in parallel (separate worker threads).

### SpectroscopySession and calibration data model

- `SpectroscopySession` responsibilities:
  - Store:
    - wavelength axis (from device),
    - last raw spectrum,
    - dark spectrum,
    - reference spectrum,
    - current mode,
    - per-mode parameters (e.g. \(\lambda_{\text{exc}}\) for Raman, excitation for fluorescence),
    - calibration data (Beer–Lambert curves, lamp calibration for irradiance),
    - ROIs and plot-related settings where useful.
  - Manage validity:
    - Dark and reference spectra are stamped with:
      - device ID,
      - integration time,
      - averages,
      - mode,
      - acquisition timestamp.
    - They become **invalid** if:
      - device changes,
      - integration time changes,
      - averages change,
      - or user explicitly invalidates/toggles them.
  - Provide signals:
    - `darkChanged`, `referenceChanged`, `calibrationChanged`, `parametersChanged`, `roiChanged`.
  - Provide methods:
    - `set_dark_spectrum(...)`, `set_reference_spectrum(...)`.
    - `compute_absorbance(raw)`, `compute_transmittance(raw)`, etc., delegating to processing functions.
- **Persistence of calibration sets:**
  - Allow users to name calibration sets (e.g. “White tile, 2025-12-01”).
  - Store along with acquisition parameters and device metadata.
  - On load:
    - Compare stored parameters to current session; if mismatched, show a warning and mark as “needs verification”.

### Data processing

#### Pipeline order

For every acquisition, the pipeline is:

1. Acquire raw counts \(I_{\text{raw}}(\lambda)\) from device.
2. Optionally apply smoothing (if configured to operate on raw):
   - e.g. boxcar or Savitzky–Golay.
3. Apply dark correction if enabled and dark spectrum available:
   - \(I_{\text{dark-corr}}(\lambda) = I_{\text{raw}}(\lambda) - D(\lambda)\).
4. Apply reference normalization where the mode requires it:
   - \(I_{\text{norm}}(\lambda) = I_{\text{dark-corr}}(\lambda) / (R(\lambda) - D(\lambda))\),
     with safeguards against division by zero or near-zero values.
5. Mode-specific computation:
   - Absorbance: \(A(\lambda) = -\log_{10}(I_{\text{norm}}(\lambda))\).
   - Transmittance/Reflectance: \(T/R(\lambda) = I_{\text{norm}}(\lambda)\) (optionally ×100%).
   - Relative irradiance: apply spectral response correction to get \(E(\lambda)\).
   - Fluorescence, Raman: treat dark-corrected spectrum as primary quantity; optionally apply baseline correction later.
6. Final smoothing / cosmetic operations (if configured to operate on mode-specific results).
7. Plotting and export.

Safeguards:

- Clamp or mask values where \(R(\lambda) - D(\lambda) \le \varepsilon\) to avoid infinities and NaNs.
- Mark saturated regions based on bit depth and threshold.

#### Per-mode specifics

- **Absorbance / Beer–Lambert**
  - Absorbance computed as above.
  - For Beer–Lambert calibration:
    - Allow user-defined selection of wavelengths or a band (e.g. \(\lambda_{\max}\) or average over a band), via ROI tools or explicit numerical input.
    - For each calibration point:
      - store concentration and corresponding measured absorbance (or band-averaged absorbance).
    - Fit:
      - user-selectable: linear through origin or general linear regression.
    - Store resulting slope/intercept and goodness-of-fit metrics.

- **Transmittance / Reflectance**
  - \(T/R(\lambda) = I_{\text{norm}}(\lambda)\), optionally expressed as percentage (100 × \(I_{\text{norm}}\)).
  - Provide both raw intensity vs λ and T/R vs λ views.

- **Relative irradiance and color metrics**
  - Use lamp or manufacturer-provided calibration to obtain a **spectral response correction function** \(S(\lambda)\).
  - Compute corrected relative spectral irradiance:
    - \(E_{\text{rel}}(\lambda) = I_{\text{dark-corr}}(\lambda) / S(\lambda)\) (up to a multiplicative constant).
  - For color metrics (optional but supported in v1):
    - Over a user-selected wavelength range, convolve \(E_{\text{rel}}(\lambda)\) with standard observer color matching functions to compute CIE XYZ.
    - Convert to xyY and derive CCT using a standard method (e.g. CCT from chromaticity).
  - Expose:
    - toggles to enable/disable this color analysis,
    - display of xy chromaticity and CCT where enabled.

- **Fluorescence**
  - Core processing is dark correction + optional smoothing.
  - Store excitation-related metadata for documentation and for potential future mapping (e.g. excitation–emission matrices).
  - Baseline correction:
    - Provide backend hooks for methods such as asymmetric least squares (ALS) or low-order polynomial baseline.
    - UI exposes a simple control:
      - baseline mode: Off / ALS / Polynomial (configurable in preferences or advanced options).

- **Raman**
  - Compute Raman shift axis using \(\lambda_{\text{exc}}\).
  - Dark correction and optional smoothing applied as for other modes.
  - Plot intensity vs shift (cm⁻¹) on the main axis, with an optional secondary wavelength axis.
  - Baseline correction:
    - Backend supports ALS / polynomial baseline options.
    - UI exposes a simple baseline mode selector similar to fluorescence.

### Standard spectral file formats

- Support import/export of spectra in:
  - CSV (primary simple format).
  - A documented internal JSON/HDF5 schema capturing:
    - wavelength axis,
    - spectra,
    - mode,
    - parameters,
    - device metadata.
  - Where practical, add JCAMP-DX export for compatibility with external spectroscopy tools.
- Imported spectra should be displayable in the spectroscopy window with the same plot tools (pan/zoom, ROIs, cursor).

---

## Application integration

- Wire the **Modules → Vis Spectroscopy…** action in `MainWindow` to open the spectroscopy window.
- Ensure the new Modules menu does not interfere with the existing right-hand Camera/Area/Scripts/System tabs.
- **Device manager integration:**
  - Spectroscopy window, raster scan logic, and time-series logging all acquire spectrometer handles via the device manager.
  - Prevent conflicting access (e.g. no two acquisition loops for the same spectrometer at once); the device manager arbitrates.

- **Shelly Dimmer 2 integration:**
  - Expose light source controls via a small component embedded in the spectroscopy window and accessible by the mode wizards.
  - Provide a simple public interface:
    - `set_on(bool)`, `set_brightness(0–100)`, `get_state()`.
- **Data folder:**
  - Profiles store default spectral data directory.
  - UI in spectroscopy window to override for a given session or save operation.
- **Logging:**
  - Reuse existing LOG utilities to record:
    - device discovery and connection/disconnection,
    - acquisition parameters,
    - Shelly state changes,
    - errors and warnings,
    - time-series and raster acquisition events.

---

## Error handling and diagnostics

- **Device/USB errors**
  - Catch all SDK exceptions and USB errors in the worker thread.
  - Emit a signal to:
    - stop acquisition,
    - update status bar with concise message,
    - optionally show a non-blocking toast/dialog.
  - Log the full stack trace and low-level message for debugging.

- **Timeouts / no data**
  - Define a timeout per acquisition; on timeout:
    - mark spectrum as invalid,
    - notify user,
    - attempt one reconnect if appropriate.

- **Saturation handling**
  - Detect if any pixel exceeds saturation threshold (derived from bit depth).
  - Indicate saturation in status bar and optionally highlight saturated regions on the plot.
  - Optionally suggest lowering integration time.

- **Mode preconditions**
  - If a user attempts to finish a wizard without valid dark/reference/calibration:
    - highlight missing steps,
    - prevent Finish,
    - offer to jump directly to the required step.

- **Calibration validity**
  - If acquisition parameters change such that dark/reference/calibration become invalid:
    - automatically mark them invalid,
    - visually indicate that recalibration is required,
    - disable dependent actions (Finish, export of derived quantities) until recalibrated.

---

## Milestones

1. **Backend scaffolding**
   - Implement `SpectrometerDevice` interface.
   - Implement OceanOptics backend and mock backend.
   - Add device manager support and enumeration.
   - Unit-test listing and single spectrum read (mock + real if available).

2. **UI shell**
   - Add Modules menu and Vis Spectroscopy action.
   - Implement spectroscopy window with:
     - device picker tied to device manager,
     - live plot,
     - basic acquisition controls,
     - spectral plot interaction tools (pan/zoom, cursor, ROIs).

3. **Mode wizards**
   - Build per-mode wizard pages consistent with screenshot flows.
   - Wire wizards to `SpectroscopySession` and enforce dark/reference/calibration requirements.

4. **Processing and export**
   - Implement full processing pipeline.
   - Implement CSV and standard-format export for single spectra.
   - Implement raster hypercube export (HDF5/NPZ) with coordinates and metadata.
   - Implement time-series logging and file format.

5. **Shelly and raster integration**
   - Add Shelly Dimmer 2 control panel and hook it into wizards.
   - Integrate spectrometer acquisition into raster scanning to create hypercubes.

6. **Persistence and polish**
   - Implement profiles for:
     - default device,
     - window geometry and layout,
     - default data folder,
     - last acquisition parameters.
   - Add tooltips and align styling with existing dark/light themes.

7. **Baseline and color analysis**
   - Wire baseline correction options (ALS/polynomial) into Raman and fluorescence modes.
   - Validate and refine relative irradiance and color metric computations.

---

## Testing strategy

### Unit tests

- **SpectrometerDevice mock:**
  - enumeration, open/close, read_spectrum, wavelength axis.
- **Processing functions:**
  - dark correction, reference normalization,
  - absorbance computation (including zero/near-zero \(I/I_0\) edge cases),
  - Beer–Lambert calibration fits (slope, intercept, R²),
  - relative irradiance correction and color metric computation (within numerical tolerances),
  - Raman shift calculation,
  - baseline correction routines (ALS, polynomial).
- **Calibration validity logic:**
  - verify that parameter changes invalidate dark/reference/calibration correctly.
- **ROI-related functions:**
  - correct mapping from selected wavelength intervals to index ranges and band-averaged values.

### Integration tests

- Using mock spectrometer:
  - single and continuous acquisition,
  - mode wizards enforcing required steps,
  - parameter changes triggering recalibration requirements,
  - UI remaining responsive under continuous acquisition,
  - time-series logging producing correctly shaped data.
- Raster + spectrometer:
  - acquire small raster,
  - confirm hypercube structure and metadata correctness.
- Multi-spectrometer:
  - attach two mock devices,
  - verify independent acquisition loops and correct routing of data to UI.

### Manual validation with USB2000

- Connect/disconnect and hot-plug behavior.
- Run through each mode, including dark/reference flows.
- Verify:
  - saturation detection,
  - Shelly control integration,
  - exported spectra (CSV, JCAMP/internal formats),
  - raster and time-series files.
- Regression check that existing device menus and Camera/Area/Scripts/System tabs remain functional.

---
