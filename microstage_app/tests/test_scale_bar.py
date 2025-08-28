import os
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import ImageDraw, ImageFont
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
        if isinstance(font, (str, bytes)) and os.path.basename(font) == "DejaVuSans.ttf":
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
        if isinstance(font, (str, bytes)) and os.path.basename(font) == "DejaVuSans.ttf":
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
