import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from PySide6 import QtWidgets
from microstage_app.ui import main_window


class DummySignal:
    def connect(self, *args, **kwargs):
        pass


class DummyThread:
    finished = DummySignal()


class DummyWorker:
    finished = DummySignal()


def dummy_run_async(fn, *args, **kwargs):
    # Return dummy thread/worker without executing fn to keep test fast and
    # avoid modal dialogs.
    return DummyThread(), DummyWorker()


def test_manual_controls_remain_enabled_during_leveling(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = main_window.MainWindow()
    win.stage = object()
    win.level_mode.setCurrentText("Manual")
    win.level_method.setCurrentText("Three-point")

    monkeypatch.setattr(main_window, "run_async", dummy_run_async)

    # Movement controls should be enabled before and after starting leveling.
    assert win.btn_xp.isEnabled()
    assert win.btn_xm.isEnabled()
    assert win.btn_home_x.isEnabled()

    win._run_leveling()

    assert win.btn_xp.isEnabled()
    assert win.btn_xm.isEnabled()
    assert win.btn_home_x.isEnabled()

    # Cleanup should still leave movement controls enabled.
    win._cleanup_leveling_thread()
    assert win.btn_xp.isEnabled()
