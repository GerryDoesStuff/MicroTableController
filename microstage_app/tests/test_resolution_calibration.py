import os
import pytest
from PySide6 import QtWidgets
import microstage_app.ui.main_window as mw
from microstage_app.analysis import Lens


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_lens_calibration_scales_with_resolution(monkeypatch, qt_app):
    class FakeCamera:
        def __init__(self):
            self.resolutions = [(0, 100, 100), (1, 200, 200)]
            self.current_idx = 0
            self.started = True

        def name(self):
            return "FakeCam"

        def start_stream(self):
            pass

        def list_resolutions(self):
            return self.resolutions

        def set_resolution_index(self, idx):
            self.current_idx = idx

        def get_resolution_index(self):
            return self.current_idx

    cam = FakeCamera()
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    win = mw.MainWindow()
    win.camera = cam
    win._populate_resolutions()

    lens = Lens("10x", 1.0, {"100x100": 1.0})
    win.lenses = {lens.name: lens}
    win.current_lens = lens

    win._apply_resolution(0)
    assert win.current_lens.um_per_px == pytest.approx(1.0)

    win.res_combo.setCurrentIndex(1)
    win._apply_resolution(1)
    assert win.current_lens.um_per_px == pytest.approx(0.5)

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
