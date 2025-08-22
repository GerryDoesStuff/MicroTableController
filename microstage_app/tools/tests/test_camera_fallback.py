def test_toupcam_import_fallback(monkeypatch):
    import importlib
    from microstage_app.devices.camera_toupcam import create_camera
    def fake_import(name):
        if name == "toupcam":
            raise ImportError("no toupcam")
        return importlib.import_module(name)
    monkeypatch.setattr(importlib, "import_module", fake_import, raising=False)
    cam = create_camera()
    from microstage_app.devices.camera_mock import MockCamera
    assert isinstance(cam, MockCamera)