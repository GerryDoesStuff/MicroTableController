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


def test_capture_settings_persist(monkeypatch, tmp_path, qt_app):
    # use temp profiles file
    monkeypatch.setattr(Profiles, "PATH", str(tmp_path / "profiles.yaml"))

    # stub ImageWriter to avoid filesystem churn
    def fake_init(self, base_dir='runs'):
        self.base_dir = base_dir
        self.run_dir = str(tmp_path / "runs")
    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_init)

    # prevent auto-connect to hardware
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    # first session: change settings
    win1 = mw.MainWindow()
    dir1 = str(tmp_path / "out")
    win1.capture_dir_edit.setText(dir1)
    win1.capture_name_edit.setText("foo")
    win1.autonumber_chk.setChecked(True)
    assert win1.capture_format == "png"
    assert win1.format_combo.currentText() == "PNG"
    win1.format_combo.setCurrentText("BMP")
    qt_app.processEvents()
    win1.preview_timer.stop(); win1.fps_timer.stop(); win1.close()

    # second session: values should persist
    win2 = mw.MainWindow()
    assert win2.capture_dir == dir1
    assert win2.capture_dir_edit.text() == dir1
    assert win2.capture_name == "foo"
    assert win2.capture_name_edit.text() == "foo"
    assert win2.auto_number is True
    assert win2.autonumber_chk.isChecked()
    assert win2.capture_format == "bmp"
    assert win2.format_combo.currentText() == "BMP"
    win2.preview_timer.stop(); win2.fps_timer.stop(); win2.close()
