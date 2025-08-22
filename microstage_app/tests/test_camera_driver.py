import types
from microstage_app.devices import camera_toupcam
from microstage_app.devices.camera_mock import MockCamera


def test_create_camera_no_devices(monkeypatch):
    class DummyTP:
        class Toupcam:
            @staticmethod
            def EnumV2():
                return []
    monkeypatch.setattr(camera_toupcam, '_import_toupcam', lambda: DummyTP)
    cam = camera_toupcam.create_camera()
    assert isinstance(cam, MockCamera)


def test_mock_camera_snap_shape():
    cam = MockCamera()
    img = cam.snap()
    assert img.shape == (480, 640, 3)
    assert img.dtype.name == 'uint8'
