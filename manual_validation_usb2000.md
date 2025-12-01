# USB2000 Manual Validation Checklist

Use this checklist to manually validate USB2000 spectrometer support end-to-end. Mark each step as Pass/Fail with notes.

## Connection and Detection
- Confirm the USB2000 enumerates in the spectrometer list and shows expected serial.
- Connect/disconnect repeatedly; status LED and label update accordingly.
- Verify wavelength axis is populated after connection.

## Acquisition Modes
- Run single-capture and continuous acquisition; ensure live plots refresh smoothly.
- Exercise Absorbance, Transmittance/Reflectance, Relative Irradiance, Fluorescence, and Raman mode wizards, verifying Finish is gated until dark/reference/raw captures are complete.
- Change integration time and averages; ensure captures respect updated parameters and invalidate prior calibration.

## Saturation and Quality Indicators
- Intentionally saturate the detector; confirm peak/saturation warning text appears and clears when intensity is reduced.
- Toggle dark subtraction and smoothing; confirm charts and metrics update without stale data.

## Shelly Illumination Controls
- Connect to configured Shelly dimmer; toggle on/off and adjust brightness from the spectroscopy window.
- Verify presets load, apply, and persist, and that UI remains responsive during dimmer operations.

## Data Export and Logging
- Save recent captures to disk with wavelength axis; confirm filenames and metadata reflect mode, integration time, and averages.
- Capture a reference/dark and apply it via the Recent list shortcuts; verify updates to session state and UI labels.
- Confirm time-series entries cap at 25 while older entries roll off without UI errors.

## Regression of Menus/Tabs
- Open the spectroscopy window from the main application; ensure all tabs/menus still render and controls are enabled as expected.
- Verify raster controls coexist with spectroscopy UI and remain operable after spectrometer interactions.
