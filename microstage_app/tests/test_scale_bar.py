import os
from types import SimpleNamespace

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtWidgets

import microstage_app.ui.main_window as mw
from microstage_app.analysis import Lens
from microstage_app.utils.img import draw_scale_bar


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_draw_scale_bar_basic():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_scale_bar(img, 1.0)
    assert np.all(out[80, 160:180] == 255)


def test_capture_applies_scale_bar(monkeypatch, tmp_path, qt_app):
    win = mw.MainWindow()
    win.stage = SimpleNamespace(wait_for_moves=lambda: None)
    win.camera = SimpleNamespace(snap=lambda: np.zeros((100, 200, 3), dtype=np.uint8))
    win.capture_dir = str(tmp_path)
    win.capture_name = "img"
    win.auto_number = False
    win.capture_format = "bmp"
    win.chk_scale_bar.setChecked(True)
    win.current_lens = Lens("test", 1.0)

    saved = {}
    win.image_writer = SimpleNamespace(save_single=lambda img, **kw: saved.setdefault("img", img))

    def fake_run_async(fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        class DummySignal:
            def connect(self, cb):
                cb(res, None)
        return None, SimpleNamespace(finished=DummySignal())

    monkeypatch.setattr(mw, "run_async", fake_run_async)

    win._capture()
    out = saved["img"]
    assert np.all(out[80, 160:180] == 255)

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
