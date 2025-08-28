import os
import numpy as np
from types import SimpleNamespace
from PIL import Image
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


def test_default_capture_preserves_metadata(monkeypatch, tmp_path, qt_app):
    # use temp profiles file
    monkeypatch.setattr(Profiles, "PATH", str(tmp_path / "profiles.yaml"))

    # stub ImageWriter to write into tmp_path
    def fake_init(self, base_dir='runs'):
        self.base_dir = str(tmp_path)
        self.run_dir = str(tmp_path)
    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_init)

    # prevent auto-connect
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    # run async immediately
    def fake_run_async(fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        class DummySignal:
            def connect(self, cb):
                cb(res, None)
        return None, SimpleNamespace(finished=DummySignal())
    monkeypatch.setattr(mw, "run_async", fake_run_async)

    win = mw.MainWindow()
    win.stage = SimpleNamespace(wait_for_moves=lambda: None, get_position=lambda: (1, 2, 3))
    win.camera = SimpleNamespace(snap=lambda: np.zeros((5, 5, 3), dtype=np.uint8), name=lambda: "MockCam")
    win.capture_dir = str(tmp_path)
    win.capture_name = "meta_test"
    win.auto_number = False
    win.current_lens = SimpleNamespace(name="LensMock", um_per_px=1.0)

    win._capture()

    path = tmp_path / "meta_test.png"
    assert path.exists()
    img = Image.open(path)
    assert img.info.get("Camera") == "MockCam"
    assert img.info.get("Lens") == "LensMock"
    assert img.info.get("Position") == "(1, 2, 3)"

    win.preview_timer.stop(); win.fps_timer.stop(); win.close()
