import os
import yaml
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


def test_invalid_profile_values(monkeypatch, tmp_path, qt_app):
    # prepare profiles file with invalid entries
    data = {
        "ui": {"jog": {"stepx": "bad", "feedx": 999999}},
        "capture": {"format": "INVALID"},
    }
    pfile = tmp_path / "profiles.yaml"
    pfile.write_text(yaml.safe_dump(data))

    # use temp profile path
    monkeypatch.setattr(Profiles, "PATH", str(pfile))

    # stub ImageWriter to avoid filesystem access
    def fake_init(self, base_dir="runs"):
        self.base_dir = base_dir
        self.run_dir = str(tmp_path / "runs")

    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_init)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()

    # invalid type should fall back to default
    assert win.stepx_spin.value() == pytest.approx(0.100)
    # out-of-range value should fall back to default
    assert win.feedx_spin.value() == pytest.approx(50.0)
    # invalid capture format should fall back to bmp
    assert win.capture_format == "bmp"
    assert win.format_combo.currentText() == "BMP"

    win.preview_timer.stop(); win.fps_timer.stop(); win.close()

