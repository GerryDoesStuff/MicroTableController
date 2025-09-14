import os
from types import SimpleNamespace
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from PySide6 import QtWidgets, QtGui, QtCore

import microstage_app.ui.main_window as mw
from microstage_app.analysis import Lens
from microstage_app.utils.img import draw_scale_bar, VERT_SCALE, TEXT_SCALE


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_draw_scale_bar_length_and_label(monkeypatch):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    captured = {}

    orig_text = ImageDraw.ImageDraw.text

    def fake_text(self, xy, text, fill=None, font=None):
        captured["text"] = text
        return orig_text(self, xy, text, fill=fill, font=font)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", fake_text)

    orig_truetype = ImageFont.truetype

    def fake_truetype(font, size=10, *args, **kwargs):
        if isinstance(font, (str, bytes)):
            raise OSError("missing font")
        return orig_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", fake_truetype)

    out = draw_scale_bar(img, 1.0)

    # Bar is 20 µm => 20 px long starting at x=160 for this image size
    bar_row = out[80]
    bar_pixels = np.where(np.all(bar_row == 255, axis=1))[0]
    assert bar_pixels[0] == 160
    assert bar_pixels[-1] - bar_pixels[0] == 20
    assert np.all(bar_row[:160] == 0)
    assert np.all(bar_row[181:] == 0)

    assert captured["text"] == "20 µm"


def test_capture_contains_scale_bar(monkeypatch, tmp_path, qt_app):
    monkeypatch.setattr(
        mw.MainWindow, "_update_raster_controls", lambda self: None, raising=False
    )
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

    orig_truetype = ImageFont.truetype

    def fake_truetype(font, size=10, *args, **kwargs):
        if isinstance(font, (str, bytes)):
            raise OSError("missing font")
        return orig_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", fake_truetype)

    win._capture()
    out = saved["img"]

    # Scale bar drawn at bottom-right with length 20 px
    bar_row = out[80]
    bar_pixels = np.where(np.all(bar_row == 255, axis=1))[0]
    assert bar_pixels[0] == 160
    assert bar_pixels[-1] - bar_pixels[0] == 20
    assert np.all(bar_row[:160] == 0)
    assert np.all(bar_row[181:] == 0)

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()


def test_preview_scale_bar_pen_and_font(monkeypatch, qt_app):
    view = mw.MeasureView()
    view.set_scale_bar(True, 1.0)
    img = QtGui.QImage(200, 100, QtGui.QImage.Format_RGB32)
    img.fill(QtCore.Qt.black)
    view.set_image(img)

    captured = {}
    orig_setPen = QtGui.QPainter.setPen
    orig_setFont = QtGui.QPainter.setFont

    def fake_setPen(self, pen):
        captured["pen_width"] = pen.width()
        orig_setPen(self, pen)

    def fake_setFont(self, font):
        captured["font_size"] = font.pointSizeF()
        orig_setFont(self, font)

    monkeypatch.setattr(QtGui.QPainter, "setPen", fake_setPen)
    monkeypatch.setattr(QtGui.QPainter, "setFont", fake_setFont)

    target = QtGui.QImage(200, 100, QtGui.QImage.Format_RGB32)
    painter = QtGui.QPainter(target)
    view.drawForeground(painter, QtCore.QRectF(target.rect()))
    painter.end()

    assert captured["pen_width"] == 2 * VERT_SCALE
    base_size = QtGui.QFont().pointSizeF()
    assert captured["font_size"] == pytest.approx(base_size * TEXT_SCALE)
    view.close()


def test_scale_bar_mu_character_renders(monkeypatch, tmp_path, qt_app):
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    used = {}

    orig_truetype = ImageFont.truetype

    def spy_truetype(font, size=10, *args, **kwargs):
        used["path"] = font
        return orig_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", spy_truetype)

    out = draw_scale_bar(img, 0.2)
    Image.fromarray(out).save(tmp_path / "scale.png")

    font_path = used["path"]
    assert isinstance(font_path, (str, bytes)) and font_path

    # restore original truetype for analysis
    monkeypatch.setattr(ImageFont, "truetype", orig_truetype)

    h, w, _ = img.shape
    um_per_px = 0.2
    max_um = 0.2 * w * um_per_px
    exp = math.floor(math.log10(max_um)) if max_um > 0 else 0
    nice_um = 10 ** exp
    for m in (5, 2, 1):
        candidate = m * (10 ** exp)
        if candidate <= max_um:
            nice_um = candidate
            break
    length_px = int(round(nice_um / um_per_px))
    x0 = int(round(w - 20 - length_px))
    y0 = int(round(h - 20))

    base_font = ImageFont.load_default()
    font_size = base_font.size * TEXT_SCALE
    font = ImageFont.truetype(str(Path(font_path).resolve()), font_size)
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    label = f"{nice_um:.0f} µm"
    bbox = draw.textbbox((0, 0), label, font=font)
    pre = draw.textlength(f"{nice_um:.0f} ", font=font)
    mu_w = draw.textlength("µ", font=font)
    th = bbox[3] - bbox[1]
    y_text = y0 - (7 * TEXT_SCALE) - th

    x_start = int(x0 + pre)
    x_end = int(min(w, x0 + pre + mu_w))
    mu_region = out[y_text : y_text + th, x_start:x_end]
    assert mu_region.size > 0 and np.any(mu_region == 255)


def test_selecting_lens_updates_scale_bar(monkeypatch, qt_app):
    """Changing the lens selection updates the scale bar calibration."""
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)
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
