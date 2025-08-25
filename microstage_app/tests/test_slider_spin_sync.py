import os
import pytest
from PySide6 import QtWidgets
import microstage_app.ui.main_window as mw


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_slider_spin_alignment_and_sync(monkeypatch, qt_app):
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()

    pairs = [
        (win.brightness_slider, win.brightness_spin),
        (win.contrast_slider, win.contrast_spin),
        (win.saturation_slider, win.saturation_spin),
        (win.hue_slider, win.hue_spin),
        (win.gamma_slider, win.gamma_spin),
    ]

    for slider, spin in pairs:
        assert slider.value() == spin.value()

    win.brightness_slider.setValue(10)
    qt_app.processEvents()
    assert win.brightness_spin.value() == 10

    win.contrast_spin.setValue(20)
    qt_app.processEvents()
    assert win.contrast_slider.value() == 20

    class FakeCam:
        def get_brightness(self):
            return 5

        def get_contrast(self):
            return 6

        def get_saturation(self):
            return 7

        def get_hue(self):
            return 8

        def get_gamma(self):
            return 60

    win.camera = FakeCam()
    win._sync_cam_controls()

    expected = [5, 6, 7, 8, 60]
    for (slider, spin), val in zip(pairs, expected):
        assert slider.value() == val
        assert spin.value() == val

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()

