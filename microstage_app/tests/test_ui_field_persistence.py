import os
import pytest
from PySide6 import QtWidgets

import microstage_app.ui.main_window as mw
from microstage_app.control.profiles import Profiles


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_ui_field_persistence(monkeypatch, tmp_path, qt_app):
    monkeypatch.setattr(Profiles, "PATH", str(tmp_path / "profiles.yaml"))

    def fake_init(self, base_dir='runs'):
        self.base_dir = base_dir
        self.run_dir = str(tmp_path / "runs")
    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_init)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None, raising=False)
    monkeypatch.setattr(mw.MainWindow, "_update_cam_buttons", lambda self: None, raising=False)

    win1 = mw.MainWindow()
    win1.stepx_spin.setValue(1.234)
    win1.feedy_spin.setValue(123.0)
    win1.absz_spin.setValue(9.876)
    win1.af_range.setValue(0.1234)
    win1.stack_range.setValue(0.4321)
    qt_app.processEvents()
    win1.preview_timer.stop(); win1.fps_timer.stop(); win1.close()

    win2 = mw.MainWindow()
    assert win2.stepx_spin.value() == pytest.approx(1.234)
    assert win2.feedy_spin.value() == pytest.approx(123.0)
    assert win2.absz_spin.value() == pytest.approx(9.876)
    assert win2.af_range.value() == pytest.approx(0.1234)
    assert win2.stack_range.value() == pytest.approx(0.4321)
    assert win2.af_range.text() == "0.1234"
    assert win2.stack_range.text() == "0.4321"
    win2.preview_timer.stop(); win2.fps_timer.stop(); win2.close()

