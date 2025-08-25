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
            self.resolutions_after = [
                (0, 1920, 1080),
                (1, 1280, 720),
                (2, 640, 480),
            ]
            self.current_idx = 0
            self.started = False

        def name(self):
            return "FakeCam"

        def start_stream(self):
            self.started = True

        def list_resolutions(self):
            if self.started:
                return self.resolutions_after
            return []

        def set_resolution_index(self, idx):
            self.current_idx = idx

    fake = FakeCamera()
    monkeypatch.setattr(mw, "create_camera", lambda: fake)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()
    win._connect_camera()

    items = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items == [f"{w}×{h}" for _, w, h in fake.resolutions_after]

    win.res_combo.setCurrentIndex(2)
    win._apply_resolution(2)
    assert fake.current_idx == 2

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()


def test_res_combo_repopulates_on_reconnect(monkeypatch, qt_app):
    class FakeCamera:
        def __init__(self, resolutions):
            self.resolutions = resolutions
            self.started = False

        def name(self):
            return "FakeCam"

        def start_stream(self):
            self.started = True

        def list_resolutions(self):
            return self.resolutions if self.started else []

        def set_resolution_index(self, idx):
            pass

    cam1 = FakeCamera([(0, 800, 600)])
    cam2 = FakeCamera([(0, 1024, 768), (1, 800, 600)])
    cams = iter([cam1, cam2])
    monkeypatch.setattr(mw, "create_camera", lambda: next(cams))
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()
    win._connect_camera()
    items1 = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items1 == [f"{w}×{h}" for _, w, h in cam1.resolutions]

    win._disconnect_camera()
    win._connect_camera()
    items2 = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items2 == [f"{w}×{h}" for _, w, h in cam2.resolutions]

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
