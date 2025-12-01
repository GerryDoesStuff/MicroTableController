import os
import sys
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCharts, QtWidgets

# Ensure repository root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from microstage_app.control.raster import RasterConfig, RasterRunner
from microstage_app.spectroscopy.devices import (
    MockSpectrometerProvider,
    SpectrometerDescriptor,
    SpectrometerManager,
)
from microstage_app.ui.spectroscopy_modes import SpectroscopyModeWizard
from microstage_app.ui.spectroscopy_window import SpectroscopyWindow


class _AxisStub:
    def __init__(self):
        self._min = 0.0
        self._max = 1.0

    def setTitleText(self, _text):
        pass

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum

    def max(self):
        return self._max

    def min(self):
        return self._min


def _axis_x_stub(self):
    if not hasattr(self, "_axis_x_stub"):
        self._axis_x_stub = _AxisStub()
    return self._axis_x_stub


def _axis_y_stub(self):
    if not hasattr(self, "_axis_y_stub"):
        self._axis_y_stub = _AxisStub()
    return self._axis_y_stub


QtCharts.QChart.axisX = _axis_x_stub  # type: ignore[assignment]
QtCharts.QChart.axisY = _axis_y_stub  # type: ignore[assignment]

if not hasattr(QtWidgets.QWizard, "pageCount"):
    QtWidgets.QWizard.pageCount = lambda self: len(self.pageIds())  # type: ignore[attr-defined,assignment]


class _StubProfiles:
    def __init__(self, data=None):
        self.data = data or {}
        self.saved = False

    def get(self, key, default=None, expected_type=None, min_value=None, max_value=None):
        value = self.data.get(key, default)
        if expected_type and not isinstance(value, expected_type):
            return default
        if isinstance(value, (int, float)):
            if min_value is not None:
                value = max(value, min_value)
            if max_value is not None:
                value = min(value, max_value)
        return value

    def set(self, key, value):
        self.data[key] = value

    def save(self):
        self.saved = True


class _StageStub:
    def __init__(self):
        self.pos = [0.0, 0.0]

    def get_position(self):
        return tuple(self.pos)

    def move_absolute(self, x=None, y=None, **kwargs):
        if x is not None:
            self.pos[0] = x
        if y is not None:
            self.pos[1] = y

    def move_relative(self, dx=0.0, dy=0.0, **kwargs):
        self.pos[0] += dx
        self.pos[1] += dy

    def wait_for_moves(self):
        pass


class _HypercubeCamera:
    def __init__(self):
        self.counter = 0

    def snap(self):
        self.counter += 1
        # Return a small cube with a spectral dimension
        return np.ones((2, 2, 3)) * self.counter

    def name(self):
        return "HypercubeCam"


class _WriterStub:
    def __init__(self):
        self.saved = []
        self.run_dir = "."

    def save_single(self, img, directory, filename, auto_prefix, auto_number, fmt, metadata):
        self.saved.append((img, metadata))


class _CaptureTracker:
    def __call__(self, *_args, **_kwargs):
        return True


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _build_window_with_mock():
    descriptor = SpectrometerDescriptor("MockSpectrometer", "SIM-INTEGRATION", "mock://integration")
    provider = MockSpectrometerProvider([descriptor])
    manager = SpectrometerManager(providers=[provider])
    profiles = _StubProfiles(
        {
            "spectroscopy.last_mode": "Absorbance",
            "spectroscopy.last_params": {},
            "spectroscopy.compact": False,
        }
    )
    window = SpectroscopyWindow(manager, profiles)
    device = manager.connect(descriptor)
    window.session.set_wavelengths(device.get_wavelengths())
    window._plot_spectrum = lambda *a, **k: None
    return window, device


def test_single_and_continuous_acquisition_with_mock_device():
    window, device = _build_window_with_mock()
    window.integration_spin.setValue(5)
    window.averages_spin.setValue(2)
    window.smoothing_spin.setValue(1)

    window._capture_once()
    assert window.session.raw_spectrum is not None
    assert len(window._recent_captures) == 1
    assert "Captured" in window.status_message.text()

    window._start_continuous()
    assert not window.btn_start.isEnabled()
    window._capture_once()
    window._stop_continuous()
    assert window.btn_start.isEnabled()
    assert len(window._recent_captures) >= 2
    assert any(entry.metadata.get("integration_ms") == 5.0 for entry in window._recent_captures)

    device.disconnect()


def test_wizard_preconditions_and_invalidation():
    window, _device = _build_window_with_mock()
    wizard = SpectroscopyModeWizard("Absorbance", window.session, capture_callback=_CaptureTracker())
    finish_btn = wizard.button(QtWidgets.QWizard.FinishButton)

    assert not finish_btn.isEnabled()
    wizard.state.update({"dark_captured": True, "reference_captured": True, "raw_captured": True})
    wizard._update_finish_state()
    assert finish_btn.isEnabled()

    wizard.invalidate_captures()
    wizard._update_finish_state()
    assert not finish_btn.isEnabled()

    window.session.set_calibration(np.ones_like(window.session.wavelengths))
    window.session.set_mode_params(integration_ms=10)
    assert window.session.requires_recalibration()


def test_time_series_logging_and_ui_feedback():
    window, device = _build_window_with_mock()
    wavelengths = np.linspace(400.0, 500.0, 8)
    window.session.set_wavelengths(wavelengths)

    for idx in range(26):
        window._record_capture("measurement", np.ones_like(wavelengths) * idx)
    assert len(window._recent_captures) == 25
    assert window.recent_list.count() == 25

    window._apply_capture_as("reference")
    assert window.session.reference_spectrum is not None
    device.disconnect()


def test_raster_and_spectrometer_hypercube_stub():
    stage = _StageStub()
    camera = _HypercubeCamera()
    writer = _WriterStub()
    cfg = RasterConfig(rows=2, cols=2, capture=True, serpentine=False)
    runner = RasterRunner(stage, camera, writer, cfg)
    runner.run()

    assert len(writer.saved) == cfg.rows * cfg.cols
    first_img, metadata = writer.saved[0]
    assert first_img.shape == (2, 2, 3)
    assert metadata["Camera"] == "HypercubeCam"


def test_multi_spectrometer_routing_and_switching():
    descriptors = [
        SpectrometerDescriptor("MockSpectrometer", "AA", "mock://aa"),
        SpectrometerDescriptor("MockSpectrometer", "BB", "mock://bb"),
    ]
    provider = MockSpectrometerProvider(descriptors)
    manager = SpectrometerManager(providers=[provider])

    devices = manager.refresh()
    assert len(devices) == 2

    dev_a = manager.connect(descriptors[0])
    assert manager.active is dev_a

    dev_b = manager.connect(descriptors[1])
    assert manager.active is dev_b
    assert dev_a is not dev_b
    manager.disconnect()
