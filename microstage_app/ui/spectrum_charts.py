from __future__ import annotations

from typing import Optional, Tuple

from PySide6 import QtCharts, QtCore, QtGui, QtWidgets


class SpectrumChartView(QtCharts.QChartView):
    cursorMoved = QtCore.Signal(float)
    roiSelected = QtCore.Signal(float, float)

    def __init__(self, chart: QtCharts.QChart, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(chart, parent)
        self.setRubberBand(QtCharts.QChartView.RectangleRubberBand)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self._last_pos: Optional[QtCore.QPoint] = None
        self._roi_origin: Optional[QtCore.QPoint] = None
        self._roi_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._vline = QtWidgets.QGraphicsLineItem()
        self._vline.setPen(QtGui.QPen(QtGui.QColor("orange"), 1, QtCore.Qt.DotLine))
        self.scene().addItem(self._vline)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (QtCore.Qt.MiddleButton,) or (
            event.button() == QtCore.Qt.LeftButton and event.modifiers() & QtCore.Qt.ControlModifier
        ):
            self._last_pos = event.pos()
        elif event.button() == QtCore.Qt.LeftButton and event.modifiers() & QtCore.Qt.ShiftModifier:
            self._roi_origin = event.pos()
            self._roi_band.setGeometry(QtCore.QRect(self._roi_origin, QtCore.QSize()))
            self._roi_band.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._last_pos is not None:
            delta = event.pos() - self._last_pos
            self.chart().scroll(-delta.x(), delta.y())
            self._last_pos = event.pos()
        elif self._roi_origin is not None:
            rect = QtCore.QRect(self._roi_origin, event.pos()).normalized()
            self._roi_band.setGeometry(rect)
        mapped = self.chart().mapToValue(event.position())
        self._update_crosshair(mapped.x())
        self.cursorMoved.emit(mapped.x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() in (QtCore.Qt.MiddleButton, QtCore.Qt.LeftButton):
            self._last_pos = None
        if event.button() == QtCore.Qt.LeftButton and self._roi_origin is not None:
            rect = QtCore.QRect(self._roi_origin, event.pos()).normalized()
            self._roi_band.hide()
            self._roi_origin = None
            if rect.width() > 5:
                p1 = self.chart().mapToValue(rect.topLeft())
                p2 = self.chart().mapToValue(rect.bottomRight())
                start, end = sorted([p1.x(), p2.x()])
                self.roiSelected.emit(start, end)
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._update_crosshair(None)
        return super().leaveEvent(event)

    def _update_crosshair(self, x: Optional[float]) -> None:
        if x is None:
            self._vline.setVisible(False)
            return
        chart = self.chart()
        axis_y = chart.axisY()
        if axis_y is None:
            self._vline.setVisible(False)
            return
        plot_area = chart.plotArea()
        top = chart.mapToPosition(QtCore.QPointF(x, axis_y.max()))
        bottom = chart.mapToPosition(QtCore.QPointF(x, axis_y.min()))
        self._vline.setLine(QtCore.QLineF(top, bottom))
        self._vline.setVisible(plot_area.contains(top) or plot_area.contains(bottom))


def create_spectrum_chart(
    *,
    x_title: str = "Wavelength (nm)",
    y_title: str = "Intensity (counts)",
    legend_visible: bool = False,
    margins: QtCore.QMargins = QtCore.QMargins(4, 4, 4, 4),
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
) -> tuple[QtCharts.QChart, SpectrumChartView, QtCharts.QValueAxis, QtCharts.QValueAxis]:
    chart = QtCharts.QChart()
    chart.setAnimationOptions(QtCharts.QChart.NoAnimation)
    chart.legend().setVisible(legend_visible)
    chart.setMargins(margins)

    x_axis = QtCharts.QValueAxis()
    x_axis.setTitleText(x_title)
    if x_range:
        x_axis.setRange(*x_range)
    chart.addAxis(x_axis, QtCore.Qt.AlignBottom)

    y_axis = QtCharts.QValueAxis()
    y_axis.setTitleText(y_title)
    if y_range:
        y_axis.setRange(*y_range)
    chart.addAxis(y_axis, QtCore.Qt.AlignLeft)

    view = SpectrumChartView(chart)
    return chart, view, x_axis, y_axis
