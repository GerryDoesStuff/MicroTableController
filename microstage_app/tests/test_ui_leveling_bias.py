import math
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6", exc_type=ImportError)
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

import microstage_app.ui.main_window as mw
from microstage_app.control.focus_planes import Area, SurfaceKind, SurfaceModel
from microstage_app.control.profiles import Profiles


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def main_window(monkeypatch, tmp_path, qt_app):
    monkeypatch.setattr(Profiles, "PATH", str(tmp_path / "profiles.yaml"))

    def fake_writer_init(self, base_dir="runs"):
        self.base_dir = base_dir
        self.run_dir = str(tmp_path / "runs")

    monkeypatch.setattr(mw.ImageWriter, "__init__", fake_writer_init)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_attach_stage_worker", lambda self: None)
    monkeypatch.setattr(mw.MainWindow, "_update_stage_buttons", lambda self: None)

    win = mw.MainWindow()
    yield win
    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()


def test_stage_position_preserves_requested_z(main_window):
    win = main_window
    initial_request = 1.234
    measured_z = 5.678

    win._last_requested_z = initial_request
    win._on_stage_position((10.0, 20.0, measured_z))

    assert math.isclose(win._last_requested_z, initial_request, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(win._last_pos["z"], measured_z, rel_tol=1e-9, abs_tol=1e-9)


def test_leveling_bias_persists_and_supports_large_offsets(main_window):
    win = main_window
    win.leveling_enabled = True

    commands = []

    def enqueue(fn, *args, callback=None):
        commands.append((fn, args, callback))

    win.stage_worker = SimpleNamespace(enqueue=enqueue)

    def _noop(*_args, **_kwargs):
        return None

    win.stage = SimpleNamespace(
        move_absolute=_noop,
        move_relative=_noop,
        wait_for_moves=_noop,
        get_position=lambda: (0.0, 0.0, 0.0),
    )

    model = SurfaceModel(kind=SurfaceKind.LINEAR)
    model.coeffs = np.array([0.5, 0.002, 0.0])
    area = Area(
        name="all",
        polygon=[(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)],
        model=model,
        priority=1,
    )
    win.focus_mgr.areas = [area]

    z_target = 0.510  # 10 µm above the plane at x=0
    win._apply_focus_bias_for_target(0.0, 0.0, z_target)
    assert math.isclose(win.focus_mgr.z_bias, 0.010, rel_tol=1e-9, abs_tol=1e-9)

    win._last_requested_z = z_target
    win._last_pos.update(x=0.0, y=0.0, z=z_target)
    win._on_stage_position((0.0, 0.0, z_target))
    assert math.isclose(win._last_requested_z, z_target, rel_tol=1e-9, abs_tol=1e-9)

    win._jog(dx=1.0, dy=0.0, dz=0.0, feed=1.0, wait_ok=True)

    assert math.isclose(win.focus_mgr.z_bias, 0.010, rel_tol=1e-9, abs_tol=1e-9)

    move_cmd = next((cmd for cmd in commands if cmd[0] is win.stage.move_absolute), None)
    assert move_cmd is not None
    _, args, _ = move_cmd
    x, y, z, feed, wait_ok = args
    assert math.isclose(x, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(y, 0.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(z, 0.512, rel_tol=1e-9, abs_tol=1e-9)
    assert feed == pytest.approx(1.0)
    assert wait_ok is True
    assert abs(z - z_target) > 0.001  # retains ~2 µm offset when moving to the new point
