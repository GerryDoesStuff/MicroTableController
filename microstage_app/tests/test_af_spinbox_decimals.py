import os
import pytest
from PySide6 import QtWidgets, QtCore, QtTest
import microstage_app.ui.main_window as mw


@pytest.fixture
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_af_spinboxes_six_decimals(monkeypatch, qt_app):
    monkeypatch.setattr(mw.MainWindow, "_auto_connect_async", lambda self: None)

    win = mw.MainWindow()

    # Ensure autofocus controls live in a group box and not a tab
    tabs = win.findChildren(QtWidgets.QTabWidget)
    assert all(
        tab.tabText(i) != "Autofocus" for tab in tabs for i in range(tab.count())
    )
    af_boxes = [b for b in win.findChildren(QtWidgets.QGroupBox) if b.title() == "Autofocus"]
    assert af_boxes, "Autofocus group box not found"

    for box in (win.af_coarse, win.af_fine):
        assert box.decimals() == 6

        line = box.lineEdit()
        line.selectAll()
        QtTest.QTest.keyClicks(line, "0.1234567")
        QtTest.QTest.keyClick(line, QtCore.Qt.Key_Return)
        qt_app.processEvents()
        assert box.text() == "0.123457"

        box.setValue(0.0015)
        qt_app.processEvents()
        assert box.text() == "0.001500"

        box.setValue(0.1)
        qt_app.processEvents()
        assert box.text() == "0.100000"

    win.preview_timer.stop()
    win.fps_timer.stop()
    win.close()

