import sys
import types
import numpy as np
import pytest

import microstage_app.devices.camera_toupcam as camera_toupcam


def test_webcam_list_and_create(monkeypatch):
    # Simulate absence of toupcam SDK
    def raise_import():
        raise ImportError
    monkeypatch.setattr(camera_toupcam, '_import_toupcam', raise_import)

    # Dummy cv2 with a single working webcam at index 0
    class DummyCap:
        def __init__(self, idx):
            self.idx = idx
            self._open = idx == 0
        def isOpened(self):
            return self._open
        def release(self):
            pass
        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        def set(self, prop, value):
            pass

    dummy_cv2 = types.SimpleNamespace(
        VideoCapture=lambda idx: DummyCap(idx),
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        COLOR_BGR2RGB=None,
        cvtColor=lambda frame, code: frame,
    )

    monkeypatch.setitem(sys.modules, 'cv2', dummy_cv2)
    monkeypatch.setattr(camera_toupcam, 'cv2', dummy_cv2)

    import microstage_app.devices.camera_webcam as camera_webcam
    monkeypatch.setattr(camera_webcam, 'cv2', dummy_cv2)

    cams = camera_toupcam.list_cameras()
    assert ("webcam:0", "Webcam 0") in cams

    cam = camera_toupcam.create_camera('webcam:0')
    assert isinstance(cam, camera_webcam.WebcamCamera)
    img = cam.snap()
    assert img.shape == (480, 640, 3)
