import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from PySide6 import QtWidgets, QtCore
from microstage_app.ui import main_window


class FakeCam:
    def name(self):
        return "FakeCam"

    def start_stream(self):
        pass

    def stop_stream(self):
        pass

    def set_exposure_ms(self, ms, auto):
        pass

    def set_gain(self, gain):
        pass


def test_widgets_disabled_when_capability_missing(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = main_window.MainWindow()

    monkeypatch.setattr(main_window, "create_camera", lambda dev_id=None: FakeCam())
    monkeypatch.setattr(main_window.MainWindow, "_populate_speed_levels", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_apply_speed", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_populate_color_depths", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_populate_binning", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_populate_resolutions", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_apply_camera_profile", lambda self: None)
    monkeypatch.setattr(main_window.MainWindow, "_sync_cam_controls", lambda self: None)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda ms, func: func())

    win.preview_timer.start = lambda *a, **k: None
    win.fps_timer.start = lambda *a, **k: None

    win._connect_camera()

    assert win.exp_spin.isEnabled()
    assert not win.brightness_spin.isEnabled()
    assert not win.brightness_slider.isEnabled()
