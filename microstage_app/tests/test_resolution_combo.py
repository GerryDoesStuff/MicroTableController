import os
import pytest
from PySide6 import QtWidgets
import microstage_app.ui.main_window as mw
import microstage_app.devices.camera_toupcam as camera_toupcam


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
            self.current_idx = 0
            self.started = False

        def name(self):
            return "FakeCam"

        def start_stream(self):
            self.started = True

        def list_resolutions(self):
            return self.resolutions if self.started else []

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


def test_toupcam_probe_resolutions(monkeypatch, qt_app):
    class FakeCam:
        def __init__(self):
            self.allowed = [(1920, 1080), (960, 540), (480, 270)]
            self.size = self.allowed[0]

        # enumeration fails
        def get_ResolutionNumber(self):
            return 0

        def get_Size(self):
            return self.size

        def put_Size(self, w, h):
            if (w, h) not in self.allowed:
                raise RuntimeError("unsupported")
            self.size = (w, h)

    class FakeTp:
        class Toupcam:
            @staticmethod
            def Open(_):
                return FakeCam()

    # avoid SDK-specific init
    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_force_rgb_or_raw", lambda self: None)
    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_init_usb_and_speed", lambda self: None)

    def make_cam():
        cam = camera_toupcam.ToupcamCamera(FakeTp, 0, "Fake")
        cam.start_stream = lambda: None
        cam.stop_stream = lambda: None
        return cam

    cam = make_cam()
    expected = cam.list_resolutions()
    assert expected == [
        (0, 1920, 1080),
        (1, 960, 540),
        (2, 480, 270),
    ]

    monkeypatch.setattr(mw, "create_camera", make_cam)
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()
    win._connect_camera()
    items = [win.res_combo.itemText(i) for i in range(win.res_combo.count())]
    assert items == [f"{w}×{h}" for _, w, h in expected]
    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()


def test_toupcam_dedup_and_halved(monkeypatch, qt_app):
    class FakeCam:
        def __init__(self):
            self.allowed = [(2464, 2464), (1232, 1232), (616, 616)]
            self.size = self.allowed[0]
            self.idx = 0

        def get_ResolutionNumber(self):
            return 2

        def get_Resolution(self, _):
            return (2464, 2464)

        def get_eSize(self):
            return self.idx

        def put_eSize(self, idx):
            if idx != 0:
                raise RuntimeError("bad idx")
            self.idx = idx
            self.size = (2464, 2464)

        def get_Size(self):
            return self.size

        def put_Size(self, w, h):
            if (w, h) not in self.allowed:
                raise RuntimeError("unsupported")
            self.size = (w, h)

    class FakeTp:
        class Toupcam:
            @staticmethod
            def Open(_):
                return FakeCam()

    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_force_rgb_or_raw", lambda self: None)
    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_init_usb_and_speed", lambda self: None)

    cam = camera_toupcam.ToupcamCamera(FakeTp, 0, "Fake")
    cam.start_stream = lambda: None
    cam.stop_stream = lambda: None

    res = cam.list_resolutions()
    assert res == [
        (0, 2464, 2464),
        (1, 1232, 1232),
        (2, 616, 616),
    ]
