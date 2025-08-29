import os
from types import SimpleNamespace

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


@pytest.fixture
def make_window(monkeypatch, tmp_path, qt_app):
    monkeypatch.setattr(Profiles, "PATH", str(tmp_path / "profiles.yaml"))
    def fake_writer_init(self, base_dir='runs'):
        self.base_dir = base_dir
        self.run_dir = str(tmp_path / "runs")
    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_writer_init)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    class DummyThread:
        def quit(self):
            pass
        def wait(self):
            pass
    class DummySignal:
        def connect(self, *args, **kwargs):
            pass
    class DummyWorker:
        def __init__(self):
            self.finished = DummySignal()
    def fake_run_async(fn, *args, **kwargs):
        return DummyThread(), DummyWorker()
    monkeypatch.setattr(mw, "run_async", fake_run_async)

    captured = {}
    class DummyRasterRunner:
        def __init__(self, stage, camera, writer, cfg, **kwargs):
            captured['cfg'] = cfg
        def run(self):
            pass
        def stop(self):
            pass
    monkeypatch.setattr(mw, "RasterRunner", DummyRasterRunner)

    win = mw.MainWindow()
    win.stage = object()
    win.camera = object()
    win.stage_worker = SimpleNamespace(enqueue=lambda *args, **kwargs: None)
    win.image_writer = object()
    win.current_lens = SimpleNamespace(name="lens", um_per_px=1.0)
    win.capture_dir = str(tmp_path)
    win.capture_name = "cap"
    win.auto_number = False
    win.capture_format = "bmp"
    yield win, captured
    win.preview_timer.stop(); win.fps_timer.stop(); win.close()


@pytest.mark.parametrize(
    "mode,p3_en,p4_en",
    [
        ("2-point", False, False),
        ("3-point", True, False),
        ("4-point", True, True),
    ],
)
def test_raster_mode_control_enablement(make_window, qt_app, mode, p3_en, p4_en):
    win, _ = make_window
    win.raster_mode_combo.setCurrentText(mode)
    qt_app.processEvents()
    for w in (win.rast_x3_spin, win.rast_y3_spin, win.btn_raster_p3):
        assert w.isEnabled() is p3_en
    for w in (win.rast_x4_spin, win.rast_y4_spin, win.btn_raster_p4):
        assert w.isEnabled() is p4_en

def test_raster_mode_two_point(make_window):
    win, captured = make_window
    win.raster_mode_combo.setCurrentText("2-point")
    win.rast_x1_spin.setValue(1.0)
    win.rast_y1_spin.setValue(2.0)
    win.rast_x2_spin.setValue(5.0)
    win.rast_y2_spin.setValue(6.0)
    win.rows_spin.setValue(3)
    win.cols_spin.setValue(4)
    win.feedx_spin.setValue(11.0)
    win.feedy_spin.setValue(22.0)
    win._run_raster()
    cfg = captured['cfg']
    assert cfg.mode == "rectangle"
    assert cfg.x1_mm == pytest.approx(1.0)
    assert cfg.y1_mm == pytest.approx(2.0)
    assert cfg.x2_mm == pytest.approx(5.0)
    assert cfg.y2_mm == pytest.approx(2.0)
    assert cfg.x3_mm == pytest.approx(1.0)
    assert cfg.y3_mm == pytest.approx(6.0)
    assert cfg.x4_mm == pytest.approx(5.0)
    assert cfg.y4_mm == pytest.approx(6.0)
    assert cfg.feed_x_mm_min == pytest.approx(11.0)
    assert cfg.feed_y_mm_min == pytest.approx(22.0)


def test_raster_mode_three_point(make_window):
    win, captured = make_window
    win.raster_mode_combo.setCurrentText("3-point")
    win.rast_x1_spin.setValue(0.0)
    win.rast_y1_spin.setValue(0.0)
    win.rast_x2_spin.setValue(4.0)
    win.rast_y2_spin.setValue(0.0)
    win.rast_x3_spin.setValue(5.0)
    win.rast_y3_spin.setValue(3.0)
    win._run_raster()
    cfg = captured['cfg']
    assert cfg.mode == "parallelogram"
    assert cfg.x1_mm == pytest.approx(0.0)
    assert cfg.y1_mm == pytest.approx(0.0)
    assert cfg.x2_mm == pytest.approx(4.0)
    assert cfg.y2_mm == pytest.approx(0.0)
    assert cfg.x3_mm == pytest.approx(5.0)
    assert cfg.y3_mm == pytest.approx(3.0)


def test_raster_mode_four_point(make_window):
    win, captured = make_window
    win.raster_mode_combo.setCurrentText("4-point")
    win.rast_x1_spin.setValue(0.0)
    win.rast_y1_spin.setValue(0.0)
    win.rast_x2_spin.setValue(4.0)
    win.rast_y2_spin.setValue(0.0)
    win.rast_x3_spin.setValue(1.0)
    win.rast_y3_spin.setValue(3.0)
    win.rast_x4_spin.setValue(5.0)
    win.rast_y4_spin.setValue(3.0)
    win._run_raster()
    cfg = captured['cfg']
    assert cfg.mode == "trapezoid"
    assert cfg.x1_mm == pytest.approx(0.0)
    assert cfg.y1_mm == pytest.approx(0.0)
    assert cfg.x2_mm == pytest.approx(4.0)
    assert cfg.y2_mm == pytest.approx(0.0)
    assert cfg.x3_mm == pytest.approx(1.0)
    assert cfg.y3_mm == pytest.approx(3.0)
    assert cfg.x4_mm == pytest.approx(5.0)
    assert cfg.y4_mm == pytest.approx(3.0)


def test_raster_stack_config(make_window):
    win, captured = make_window
    win.raster_mode_combo.setCurrentText("2-point")
    win.rast_x1_spin.setValue(0.0)
    win.rast_y1_spin.setValue(0.0)
    win.rast_x2_spin.setValue(1.0)
    win.rast_y2_spin.setValue(1.0)
    win.chk_raster_stack.setChecked(True)
    win.stack_range.setValue(0.7)
    win.stack_step.setValue(0.02)
    win._run_raster()
    cfg = captured['cfg']
    assert cfg.stack is True
    assert cfg.stack_range_mm == pytest.approx(0.7)
    assert cfg.stack_step_mm == pytest.approx(0.02)
