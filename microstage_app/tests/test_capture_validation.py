import os

import pytest
from PySide6 import QtWidgets

import microstage_app.ui.main_window as mw


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def win(monkeypatch, qt_app):
    w = mw.MainWindow()
    w.stage = object()
    w.camera = object()
    monkeypatch.setattr(mw, "run_async", lambda fn: pytest.fail("run_async called"))
    yield w
    w.preview_timer.stop()
    w.fps_timer.stop()
    w.close()


def test_capture_empty_filename(monkeypatch, tmp_path, win):
    win.capture_dir = str(tmp_path)
    win.capture_name = ""
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: messages.append(args[2])
    )
    win._capture()
    assert messages == ["Filename cannot be empty."]


def test_capture_illegal_filename(monkeypatch, tmp_path, win):
    win.capture_dir = str(tmp_path)
    win.capture_name = "bad:name"
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: messages.append(args[2])
    )
    win._capture()
    assert "illegal characters" in messages[0]


def test_capture_uncreatable_directory(monkeypatch, tmp_path, win):
    win.capture_dir = str(tmp_path / "sub")
    win.capture_name = "ok"
    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "critical", lambda *args: messages.append(args[2])
    )

    def fail_makedirs(path, exist_ok=False):
        raise OSError("fail")

    monkeypatch.setattr(mw.os, "makedirs", fail_makedirs)
    win._capture()
    assert "Unable to create directory" in messages[0]

