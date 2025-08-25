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


def test_res_combo_lists_and_updates(monkeypatch, qt_app):
    class FakeCamera:
        def __init__(self):
            self.resolutions = [
                (0, 1920, 1080),
                (1, 1280, 720),
                (2, 640, 480),
            ]
            self.current_idx = 1

        def name(self):
            return "FakeCam"

        def start_stream(self):
            pass

        def list_resolutions(self):
            return self.resolutions

        def set_resolution_index(self, idx):
            self.current_idx = idx

        def get_resolution_index(self):
            return self.current_idx

    fake = FakeCamera()
    monkeypatch.setattr(mw, "create_camera", lambda: fake)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()
    win._connect_camera()

    items = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items == [f"{w}×{h}" for _, w, h in fake.resolutions]
    assert win.res_combo.currentIndex() == fake.current_idx

    win.res_combo.setCurrentIndex(2)
    win._apply_resolution(2)
    assert fake.current_idx == 2

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
