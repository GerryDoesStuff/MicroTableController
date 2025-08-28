import os
import sys
from pathlib import Path

# Ensure offscreen platform for Qt on headless environments
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets, QtGui, QtCore

# Add repository root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from microstage_app.ui.main_window import MeasureView


def _mouse_event(event_type, pos, button, buttons):
    return QtGui.QMouseEvent(event_type, QtCore.QPointF(*pos), button, buttons, QtCore.Qt.NoModifier)


def _draw_line(view: MeasureView, start, end):
    view.mousePressEvent(_mouse_event(QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton))
    view.mouseMoveEvent(_mouse_event(QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton, QtCore.Qt.LeftButton))
    view.mouseReleaseEvent(_mouse_event(QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton))


def test_start_ruler_appends_lines():
    # Ensure a QApplication exists
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    view = MeasureView()
    view.start_ruler(1.0)
    _draw_line(view, (0, 0), (10, 0))
    assert len(view._lines) == 1

    view.start_ruler(1.0)
    _draw_line(view, (0, 10), (10, 10))
    assert len(view._lines) == 2
