import types

import pytest
from microstage_app.devices import camera_toupcam


class DummyCam:
    def __init__(self):
        self.sensor_w = 3000
        self.sensor_h = 3000
        self.w = self.sensor_w
        self.h = self.sensor_h
        self.started = False

    # API used by ToupcamCamera
    def get_Size(self):
        return self.w, self.h

    def get_FinalSize(self):
        return self.w, self.h

    def put_Roi(self, x, y, w, h):
        if w == 0 or h == 0:
            self.w = self.sensor_w
            self.h = self.sensor_h
        else:
            self.w = w
            self.h = h

    def put_Size(self, w, h):
        self.w = w
        self.h = h

    def Stop(self):
        self.started = False

    def StartPullModeWithCallback(self, cb, ctx=None):
        self.started = True

    def PullImageV2(self, buf, bits, something):
        pass

    def put_AutoExpoEnable(self, x):
        pass

    def put_Option(self, opt, val):
        pass


class DummyTP:
    TOUPCAM_EVENT_IMAGE = 0x0001
    TOUPCAM_OPTION_RAW = 0

    class Toupcam:
        @staticmethod
        def EnumV2():
            item = types.SimpleNamespace(id=1, displayname='dummy')
            return [item]

        @staticmethod
        def Open(idx):
            return DummyCam()


def test_roi_presets(monkeypatch):
    # Use the dummy SDK so tests do not require hardware
    monkeypatch.setattr(camera_toupcam, '_import_toupcam', lambda: DummyTP)
    cam = camera_toupcam.create_camera()

    # Verify full frame capture works
    cam._on_event(DummyTP.TOUPCAM_EVENT_IMAGE)
    img = cam.get_latest_frame()
    assert img.shape == (3000, 3000, 3)
    assert len(cam._buf) == cam._stride * cam._h

    # Apply ROI presets and ensure dimensions/buffers update
    for side in (2048, 1024, 512):
        cam.set_center_roi(side, side)
        cam._on_event(DummyTP.TOUPCAM_EVENT_IMAGE)
        img = cam.get_latest_frame()
        exp = side & ~1  # even-aligned dimension
        assert img.shape == (exp, exp, 3)
        assert len(cam._buf) == cam._stride * cam._h

    # Clear ROI back to full frame
    cam.set_center_roi(0, 0)
    cam._on_event(DummyTP.TOUPCAM_EVENT_IMAGE)
    img = cam.get_latest_frame()
    assert img.shape == (3000, 3000, 3)
    assert len(cam._buf) == cam._stride * cam._h
