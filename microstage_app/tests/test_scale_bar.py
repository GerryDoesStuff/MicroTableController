import os
from types import SimpleNamespace
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtWidgets, QtGui, QtCore

import microstage_app.ui.main_window as mw
from microstage_app.analysis import Lens
from microstage_app.utils.img import (
    draw_scale_bar,
    VERT_SCALE,
    TEXT_SCALE,
    _scale_bar_geometry,
    _scaled_font,
)


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_draw_scale_bar_length_and_label(monkeypatch, qt_app):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    captured = {}

    original_draw_text = QtGui.QPainter.drawText

    def spy_draw_text(self, *args, **kwargs):
        text = kwargs.get("text")
        if text is None and args:
            candidate = args[-1]
            if isinstance(candidate, str):
                text = candidate
        if text is not None:
            captured["text"] = text
        return original_draw_text(self, *args, **kwargs)

    monkeypatch.setattr(QtGui.QPainter, "drawText", spy_draw_text)

    out = draw_scale_bar(img, 1.0)

    bar_row = out[80]
    bar_pixels = np.where(np.all(bar_row == 255, axis=1))[0]
    assert bar_pixels.size > 0

    _, length_px, x0, _ = _scale_bar_geometry(out.shape[1], out.shape[0], 1.0)
    tolerance = max(1, VERT_SCALE)
    assert abs(bar_pixels[0] - x0) <= tolerance
    assert abs(bar_pixels[-1] - (x0 + length_px)) <= tolerance

    left_clear = max(0, bar_pixels[0] - tolerance)
    right_clear = min(out.shape[1], bar_pixels[-1] + tolerance + 1)
    assert np.all(bar_row[:left_clear] == 0)
    assert np.all(bar_row[right_clear:] == 0)

    assert captured["text"] == "20 µm"


def test_capture_contains_scale_bar(monkeypatch, tmp_path, qt_app):
    class DummyMainWindow(mw.MainWindow):
        def _update_raster_controls(self):
            pass

        def _update_stage_buttons(self):
            pass

        def _update_cam_buttons(self):
            pass

    monkeypatch.setattr(mw, "MainWindow", DummyMainWindow)
    win = mw.MainWindow()
    win.stage = SimpleNamespace(wait_for_moves=lambda: None, get_position=lambda: (0, 0, 0))
    win.camera = SimpleNamespace(
        snap=lambda: np.zeros((100, 200, 3), dtype=np.uint8),
        name=lambda: "CameraMock",
    )
    win.capture_dir = str(tmp_path)
    win.capture_name = "img"
    win.auto_number = False
    win.chk_scale_bar.setChecked(True)
    win.current_lens = Lens("test", 1.0)

    saved = {}
    win.image_writer = SimpleNamespace(save_single=lambda img, **kw: saved.setdefault("img", img))

    def fake_run_async(fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        class DummySignal:
            def connect(self, cb):
                cb(res, None)
        return None, SimpleNamespace(finished=DummySignal())

    monkeypatch.setattr(mw, "run_async", fake_run_async)

    win._capture()
    out = saved["img"]

    bar_row = out[80]
    bar_pixels = np.where(np.all(bar_row == 255, axis=1))[0]
    assert bar_pixels.size > 0

    _, length_px, x0, _ = _scale_bar_geometry(out.shape[1], out.shape[0], 1.0)
    tolerance = max(1, VERT_SCALE)
    assert abs(bar_pixels[0] - x0) <= tolerance
    assert abs(bar_pixels[-1] - (x0 + length_px)) <= tolerance

    left_clear = max(0, bar_pixels[0] - tolerance)
    right_clear = min(out.shape[1], bar_pixels[-1] + tolerance + 1)
    assert np.all(bar_row[:left_clear] == 0)
    assert np.all(bar_row[right_clear:] == 0)

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()


def test_preview_scale_bar_pen_and_font(monkeypatch, qt_app):
    view = mw.MeasureView()
    view.set_scale_bar(True, 1.0)
    img = QtGui.QImage(200, 100, QtGui.QImage.Format_RGB32)
    img.fill(QtCore.Qt.black)
    view.set_image(img)

    captured_pen_widths = []
    original_set_pen = QtGui.QPainter.setPen

    def spy_set_pen(self, pen):
        if isinstance(pen, QtGui.QPen) and pen.color() == QtCore.Qt.white and pen.width() > 0:
            captured_pen_widths.append(pen.width())
        return original_set_pen(self, pen)

    draw_calls = []
    original_draw_text = QtGui.QPainter.drawText

    def spy_draw_text(self, *args, **kwargs):
        text = kwargs.get("text")
        if text is None and args:
            candidate = args[-1]
            if isinstance(candidate, str):
                text = candidate
        metrics = QtGui.QFontMetricsF(self.font())
        draw_calls.append(
            {
                "text": text,
                "point_size": self.font().pointSizeF(),
                "pixel_size": self.font().pixelSize(),
                "height": metrics.height(),
                "descent": metrics.descent(),
            }
        )
        return original_draw_text(self, *args, **kwargs)

    monkeypatch.setattr(QtGui.QPainter, "setPen", spy_set_pen)
    monkeypatch.setattr(QtGui.QPainter, "drawText", spy_draw_text)

    target = QtGui.QImage(200, 100, QtGui.QImage.Format_RGB32)
    painter = QtGui.QPainter(target)
    view.drawForeground(painter, QtCore.QRectF(target.rect()))
    painter.end()

    assert captured_pen_widths[0] == 2 * VERT_SCALE
    assert draw_calls
    preview_call = draw_calls[0]

    app_font = QtGui.QFont(qt_app.font())
    base_point = app_font.pointSizeF()
    if base_point > 0:
        assert preview_call["point_size"] == pytest.approx(base_point * TEXT_SCALE)
    else:
        expected_pixel = app_font.pixelSize() * TEXT_SCALE
        assert preview_call["pixel_size"] == expected_pixel

    out = draw_scale_bar(np.zeros((100, 200, 3), dtype=np.uint8), 1.0)
    assert out.shape == (100, 200, 3)
    assert len(draw_calls) >= 2
    capture_call = draw_calls[-1]

    assert capture_call["text"] == preview_call["text"]
    assert capture_call["height"] == pytest.approx(preview_call["height"])
    assert capture_call["descent"] == pytest.approx(preview_call["descent"])
    if base_point > 0:
        assert capture_call["point_size"] == pytest.approx(preview_call["point_size"])
    else:
        assert capture_call["pixel_size"] == preview_call["pixel_size"]

    view.close()


def test_scale_bar_mu_character_renders(qt_app):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_scale_bar(img, 0.05)

    h, w, _ = out.shape
    um_per_px = 0.05
    nice_um, length_px, x0, y0 = _scale_bar_geometry(w, h, um_per_px)

    font = _scaled_font(QtGui.QFont(qt_app.font()))
    metrics = QtGui.QFontMetricsF(font)
    label = (
        f"{nice_um/1000:.2f} mm" if nice_um >= 1000 else f"{nice_um:.0f} µm"
    )

    baseline = y0 - (7 * TEXT_SCALE) - metrics.descent()
    prefix_width = metrics.horizontalAdvance(label.split("µ")[0])
    mu_width = metrics.horizontalAdvance("µ")

    top = int(max(0, math.floor(baseline - metrics.ascent())))
    bottom = int(min(h, math.ceil(baseline + metrics.descent())))
    left = int(max(0, math.floor(x0 + prefix_width)))
    right = int(min(w, math.ceil(x0 + prefix_width + mu_width)))

    mu_region = out[top:bottom, left:right]
    assert mu_region.size > 0 and np.any(mu_region == 255)


def test_selecting_lens_updates_scale_bar(monkeypatch, qt_app):
    """Changing the lens selection updates the scale bar calibration."""

    class DummyMainWindow(mw.MainWindow):
        def _auto_connect_async(self):
            pass

        def _update_stage_buttons(self):
            pass

        def _update_cam_buttons(self):
            pass

    monkeypatch.setattr(mw, "MainWindow", DummyMainWindow)
    win = mw.MainWindow()

    lens_a = Lens("5x", 2.0)
    lens_b = Lens("10x", 1.0)
    win.lenses = {lens_a.name: lens_a, lens_b.name: lens_b}
    win.current_lens = lens_a
    win._refresh_lens_combo()

    win.chk_scale_bar.setChecked(True)
    captured = {}

    def fake_set_scale_bar(enabled, um_per_px):
        captured["enabled"] = enabled
        captured["um_per_px"] = um_per_px

    monkeypatch.setattr(win.measure_view, "set_scale_bar", fake_set_scale_bar)

    idx = win.lens_combo.findData("10x")
    win.lens_combo.setCurrentIndex(idx)

    assert captured["enabled"] is True
    assert captured["um_per_px"] == pytest.approx(1.0)

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()
