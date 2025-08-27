import os
import pytest
from PySide6 import QtWidgets

from microstage_app.devices.stage_marlin import StageMarlin
import microstage_app.ui.main_window as mw


class FakeStage(StageMarlin):
    def __init__(self, responses):
        self._responses = list(responses)

    def send(self, cmd, wait_ok=True):  # pragma: no cover - trivial
        return self._responses.pop(0)


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_get_position_and_label(monkeypatch, qt_app):
    responses = [
        "X:1.00 Y:0.00 Z:1.00 E:0.00 count X:0 Y:0 Z:0",
        "X:1.00 Y:0.00 Z:2.00 E:0.00 count X:0 Y:0 Z:0",
    ]
    stage = FakeStage(responses)
    x1, _, z1 = stage.get_position()
    assert x1 == 1.0 and z1 == 1.0
    x2, _, z2 = stage.get_position()
    assert x2 == 1.0 and z2 == 2.0

    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_attach_stage_worker", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None)

    win = mw.MainWindow()
    try:
        win._on_stage_position((x1, 0.0, z1))
        QtWidgets.QApplication.processEvents()
        assert "X1.000000" in win.stage_pos.text()
        assert "Z1.000000" in win.stage_pos.text()
        win._on_stage_position((x1, 0.0, z2))
        QtWidgets.QApplication.processEvents()
        txt = win.stage_pos.text()
        assert "X1.000000" in txt and "Z2.000000" in txt
    finally:
        win.preview_timer.stop()
        win.fps_timer.stop()
        win.close()
