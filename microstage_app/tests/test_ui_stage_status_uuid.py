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


def test_stage_status_multiline(monkeypatch, qt_app):
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_attach_stage_worker", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None)

    win = mw.MainWindow()

    class DummyStage:
        def get_info(self):
            return {"machine_type": "MicroStageController", "uuid": "1234-uuid"}
        def get_bounds(self):
            return {}

    win._on_stage_connect(DummyStage(), None)
    assert win.stage_status.text() == "Stage: MicroStageController\n1234-uuid"

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
