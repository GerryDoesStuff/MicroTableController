import importlib

from PySide6 import QtCore
import pytest

from microstage_app.spectroscopy.devices import OceanOpticsSpectrometer, SpectrometerManager


@pytest.fixture(autouse=True)
def _qt_core_app():
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtCore.QCoreApplication([])
    yield app


def test_ocean_optics_backend_handles_system_exit(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def boom(_name):
        raise SystemExit("boom")

    monkeypatch.setattr(importlib, "import_module", boom)

    backend = OceanOpticsSpectrometer._get_backend()
    assert backend is None


def test_manager_refresh_ignores_provider_crash(monkeypatch):
    class BoomProvider:
        def list_devices(self):
            raise SystemExit("boom")

        def connect(self, descriptor):
            pytest.fail("connect should not be called when enumeration fails")

    mgr = SpectrometerManager(providers=[BoomProvider()])
    try:
        assert mgr.refresh() == []
    finally:
        mgr.shutdown()
