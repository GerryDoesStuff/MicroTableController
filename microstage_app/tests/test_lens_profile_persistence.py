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


def test_lens_profile_persistence(monkeypatch, tmp_path, qt_app):
    pfile = tmp_path / "profiles.yaml"
    monkeypatch.setattr(Profiles, "PATH", str(pfile))
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    # simulate dialogs
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", staticmethod(lambda *a, **k: ("15x", True)))
    monkeypatch.setattr(QtWidgets.QInputDialog, "getDouble", staticmethod(lambda *a, **k: (200.0, True)))

    win1 = mw.MainWindow()
    qt_app.processEvents()
    win1._add_lens()
    qt_app.processEvents()
    win1._on_calibration_done(100.0)
    qt_app.processEvents()
    win1.preview_timer.stop(); win1.fps_timer.stop(); win1.close()

    data = yaml.safe_load(pfile.read_text())
    assert data["measurement"]["lenses"]["15x"]["um_per_px"] == pytest.approx(2.0)
    assert data["measurement"]["lenses"]["15x"]["calibrations"]["default"] == pytest.approx(2.0)

    win2 = mw.MainWindow()
    lens = win2.lenses["15x"]
    assert lens.um_per_px == pytest.approx(2.0)
    assert lens.calibrations["default"] == pytest.approx(2.0)
    win2.preview_timer.stop(); win2.fps_timer.stop(); win2.close()
