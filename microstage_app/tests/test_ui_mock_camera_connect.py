import os
import pytest
from PySide6 import QtWidgets
import microstage_app.ui.main_window as mw
from microstage_app.devices.camera_mock import MockCamera


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_ui_connects_with_mock_camera(monkeypatch, qt_app):
    cam = MockCamera()
    called = []
    cam.set_color_depth = lambda d: called.append(d)
    monkeypatch.setattr(mw, "create_camera", lambda: cam)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()
    win._connect_camera()

    assert win.camera is not None
    items = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items == ["640×480", "320×240"]
    depth_items = [win.depth_combo.itemText(i) for i in range(win.depth_combo.count())]
    assert depth_items == ["8-bit", "16-bit"]
    win.depth_combo.setCurrentIndex(1)
    assert called[-1] == 16

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
