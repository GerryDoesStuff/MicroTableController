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


def test_dispatch_updates_stage_pos(monkeypatch, qt_app):
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_attach_stage_worker", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None)

    win = mw.MainWindow()
    try:
        pos = (1.0, 2.0, 3.0)
        win._dispatch_stage_result(win._on_stage_position, pos)
        QtWidgets.QApplication.processEvents()
        txt = win.stage_pos.text()
        assert "X1.000000" in txt and "Y2.000000" in txt and "Z3.000000" in txt
    finally:
        win.preview_timer.stop()
        win.fps_timer.stop()
        win.close()


def test_dispatch_partial_pos(monkeypatch, qt_app):
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_attach_stage_worker", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None)

    win = mw.MainWindow()
    try:
        win._dispatch_stage_result(win._on_stage_position, (1.0, 2.0, 3.0))
        QtWidgets.QApplication.processEvents()
        win._dispatch_stage_result(win._on_stage_position, (None, None, 4.0))
        QtWidgets.QApplication.processEvents()
        txt = win.stage_pos.text()
        assert "X1.000000" in txt and "Y2.000000" in txt and "Z4.000000" in txt
    finally:
        win.preview_timer.stop()
        win.fps_timer.stop()
        win.close()
