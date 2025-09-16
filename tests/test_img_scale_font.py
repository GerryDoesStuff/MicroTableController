"""Tests for the scale-bar font loading helpers."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def qt_stubs():
    """Provide minimal PySide6 stubs required by :mod:`img` during tests."""

    saved_modules = {}
    for name in ("PySide6", "PySide6.QtGui", "PySide6.QtWidgets"):
        if name in sys.modules:
            saved_modules[name] = sys.modules[name]

    qtgui_module = ModuleType("PySide6.QtGui")
    qtwidgets_module = ModuleType("PySide6.QtWidgets")

    class DummyFont:
        def __init__(self, family: str = "Sans Serif", point_size: float = 12.0,
                     pixel_size: int = 12) -> None:
            self._family = family
            self._point_size = point_size
            self._pixel_size = pixel_size

        def pointSizeF(self) -> float:
            return float(self._point_size)

        def pixelSize(self) -> int:
            return int(self._pixel_size)

        def family(self) -> str:
            return self._family

    class DummyQGuiApplication:
        _instance: "DummyQGuiApplication | None" = None

        def __init__(self, *args, font: DummyFont | None = None, **kwargs) -> None:
            self._font = font or DummyFont()
            type(self)._instance = self

        @classmethod
        def instance(cls) -> "DummyQGuiApplication | None":
            return cls._instance

        def font(self) -> DummyFont:
            return self._font

        def setFont(self, font: DummyFont) -> None:  # pragma: no cover - helper
            self._font = font

    class DummyQImage:
        Format_Grayscale8 = 0
        Format_RGB888 = 1

        def __init__(self, *args, **kwargs) -> None:
            self._args = args
            self._kwargs = kwargs

    class DummyMessageBox:
        warnings: list[tuple[object | None, str, str]] = []

        @classmethod
        def warning(cls, parent, title: str, message: str) -> None:
            cls.warnings.append((parent, title, message))

    qtgui_module.QGuiApplication = DummyQGuiApplication
    qtgui_module.QImage = DummyQImage
    qtwidgets_module.QMessageBox = DummyMessageBox

    pyside_module = ModuleType("PySide6")
    pyside_module.QtGui = qtgui_module
    pyside_module.QtWidgets = qtwidgets_module

    sys.modules["PySide6"] = pyside_module
    sys.modules["PySide6.QtGui"] = qtgui_module
    sys.modules["PySide6.QtWidgets"] = qtwidgets_module

    context = SimpleNamespace(
        QGuiApplication=DummyQGuiApplication,
        DummyFont=DummyFont,
        message_box=DummyMessageBox,
    )

    try:
        yield context
    finally:
        DummyQGuiApplication._instance = None
        DummyMessageBox.warnings.clear()
        for name in ("PySide6.QtWidgets", "PySide6.QtGui", "PySide6"):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            sys.modules[name] = module


@pytest.fixture
def img_module(qt_stubs):
    """Import :mod:`microstage_app.utils.img` with a clean cache state."""

    module_name = "microstage_app.utils.img"

    saved_cv2 = sys.modules.get("cv2")
    cv2_stub = ModuleType("cv2")
    cv2_stub.cuda = SimpleNamespace(getCudaEnabledDeviceCount=lambda: 0)
    sys.modules["cv2"] = cv2_stub

    sys.modules.pop(module_name, None)
    img = importlib.import_module(module_name)
    img._scale_font_cache = None
    img._font_error_reported = False
    try:
        yield img
    finally:
        img._scale_font_cache = None
        img._font_error_reported = False
        sys.modules.pop(module_name, None)
        if saved_cv2 is not None:
            sys.modules["cv2"] = saved_cv2
        else:
            sys.modules.pop("cv2", None)


def _force_missing_font(monkeypatch, img):
    """Point ``DEJAVU_SANS_PATH`` at a temporary, non-existent location."""

    missing_path = Path("/nonexistent/DejaVuSans.ttf")
    monkeypatch.setattr(img, "DEJAVU_SANS_PATH", missing_path)
    return missing_path


def test_load_scale_font_uses_fallback_when_packaged_font_missing(
    img_module, qt_stubs, monkeypatch, caplog
) -> None:
    """If the bundled DejaVu Sans font is missing a fallback should be loaded."""

    qt_stubs.QGuiApplication(font=qt_stubs.DummyFont(point_size=11.0))

    img = img_module
    img._scale_font_cache = None

    missing_path = _force_missing_font(monkeypatch, img)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("fc-match not available")

    monkeypatch.setattr(img.subprocess, "run", fake_run)

    original_load_default = img.ImageFont.load_default
    original_truetype = img.ImageFont.truetype

    def fake_truetype(path, size, **kwargs):
        if hasattr(path, "read"):
            return original_truetype(path, size, **kwargs)
        if str(path) == str(missing_path):
            raise OSError("bundled font missing")
        if str(path) == "DejaVuSans.ttf":
            return original_load_default()
        raise AssertionError(f"Unexpected font lookup for {path}")

    monkeypatch.setattr(img.ImageFont, "truetype", fake_truetype)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=img.logger.name):
        font = img._load_scale_font()

    assert hasattr(font, "getbbox")
    assert font.getbbox("scale")

    fallback_warning = [
        record.message for record in caplog.records
        if "fallback font 'DejaVuSans.ttf'" in record.message
    ]
    assert fallback_warning, "Expected fallback warning was not logged"


def test_font_error_dialog_only_shown_once_on_load_failure(
    img_module, qt_stubs, monkeypatch
) -> None:
    """Ensure the font error dialog is not displayed repeatedly."""

    qt_stubs.QGuiApplication(font=qt_stubs.DummyFont(point_size=9.5))

    img = img_module
    img._scale_font_cache = None
    img._font_error_reported = False

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("fc-match missing")

    monkeypatch.setattr(img.subprocess, "run", fake_run)

    def fake_truetype(path, size, **kwargs):
        raise OSError("truetype loading failed")

    monkeypatch.setattr(img.ImageFont, "truetype", fake_truetype)

    def fake_load_default():
        raise RuntimeError("load_default failed")

    monkeypatch.setattr(img.ImageFont, "load_default", fake_load_default)

    call_count = 0

    def fake_show_font_error_dialog(message: str) -> None:
        nonlocal call_count
        if img._font_error_reported:
            return
        call_count += 1
        img._font_error_reported = True

    monkeypatch.setattr(img, "_show_font_error_dialog", fake_show_font_error_dialog)

    with pytest.raises(RuntimeError):
        img._load_scale_font()

    assert call_count == 1

    with pytest.raises(RuntimeError):
        img._load_scale_font()

    assert call_count == 1
