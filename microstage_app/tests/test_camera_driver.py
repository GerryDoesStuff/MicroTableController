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


def test_list_color_depths_from_flags(monkeypatch):
    tp = types.SimpleNamespace(
        TOUPCAM_FLAG_RAW10=1 << 0,
        TOUPCAM_FLAG_RAW12=1 << 1,
        TOUPCAM_FLAG_RAW14=1 << 2,
        TOUPCAM_FLAG_RAW16=1 << 3,
    )

    def fake_open(self):
        self._color_depths = [8]
        for depth, flag in [
            (10, "TOUPCAM_FLAG_RAW10"),
            (12, "TOUPCAM_FLAG_RAW12"),
            (14, "TOUPCAM_FLAG_RAW14"),
            (16, "TOUPCAM_FLAG_RAW16"),
        ]:
            if self._flags & getattr(self._tp, flag, 0):
                self._color_depths.append(depth)
        self._color_depth = self._color_depths[0]
        self._cam = None

    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_open", fake_open)
    monkeypatch.setattr(camera_toupcam.ToupcamCamera, "_query_binning_options", lambda self: None)
    cam = camera_toupcam.ToupcamCamera(tp, "id", "name", tp.TOUPCAM_FLAG_RAW10 | tp.TOUPCAM_FLAG_RAW14)
    assert cam.list_color_depths() == [8, 10, 14]
