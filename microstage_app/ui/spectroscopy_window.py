from __future__ import annotations

import time

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtCharts

from ..spectroscopy.devices import SpectrometerDescriptor, SpectrometerManager
from ..spectroscopy.processing import smooth_boxcar, subtract_dark
from ..spectroscopy.session import SpectroscopySession
from .spectroscopy_modes import ModeSelectorDialog, SpectroscopyModeWizard
from ..utils.log import LOG


@dataclass
class SpectrumTrace:
    label: str
    color: QtGui.QColor
    visible: bool = True


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


class SpectroscopyWindow(QtWidgets.QMainWindow):
    MODES = [
        "Absorbance",
        "Transmittance",
        "Reflectance",
        "Relative Irradiance",
        "Fluorescence",
        "Raman",
    ]

    def __init__(
        self,
        manager: SpectrometerManager,
        profiles,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Vis Spectroscopy")
        self.spectrometer_manager = manager
        self.profiles = profiles
        self.session = SpectroscopySession()
        self.current_mode = "Absorbance"
        self._compact = bool(self.profiles.get("spectroscopy.compact", False, expected_type=bool))
        self._last_capture_ts = 0.0

        self._chart = QtCharts.QChart()
        self._chart.setAnimationOptions(QtCharts.QChart.AllAnimations)
        self._chart.legend().setVisible(True)
        self._chart.createDefaultAxes()
        self._chart.axisX().setTitleText("Wavelength (nm)")
        self._chart.axisY().setTitleText("Intensity (a.u.)")

        self._chart_view = SpectrumChartView(self._chart)
        self._traces: Dict[str, QtCharts.QLineSeries] = {}
        self._trace_meta: Dict[str, SpectrumTrace] = {}

        self.capture_timer = QtCore.QTimer(self)
        self.capture_timer.setInterval(250)
        self.capture_timer.timeout.connect(self._capture_once)

        self._build_ui()
        self._restore_geometry()
        self._connect_signals()

        QtCore.QTimer.singleShot(0, self._refresh_devices)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # top bar
        top = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.status_led = QtWidgets.QLabel("●")
        self.status_led.setStyleSheet("color: red;")
        self.status_label = QtWidgets.QLabel("No spectrometer")
        self.mode_label = QtWidgets.QLabel("Mode: Absorbance")
        self.btn_modes = QtWidgets.QPushButton("Modes…")
        self.compact_toggle = QtWidgets.QToolButton()
        self.compact_toggle.setText("Compact")
        self.compact_toggle.setCheckable(True)
        self.compact_toggle.setChecked(self._compact)
        top.addWidget(QtWidgets.QLabel("Spectrometer:"))
        top.addWidget(self.device_combo, 1)
        top.addWidget(self.btn_refresh)
        top.addWidget(self.btn_connect)
        top.addWidget(self.btn_disconnect)
        top.addWidget(self.status_led)
        top.addWidget(self.status_label)
        top.addStretch(1)
        top.addWidget(self.mode_label)
        top.addWidget(self.btn_modes)
        top.addWidget(self.compact_toggle)
        layout.addLayout(top)

        self.chart_toolbar = QtWidgets.QToolBar()
        self.act_zoom_in = self.chart_toolbar.addAction("Zoom In")
        self.act_zoom_out = self.chart_toolbar.addAction("Zoom Out")
        self.act_reset = self.chart_toolbar.addAction("Reset")
        self.act_export = self.chart_toolbar.addAction("Save Plot…")
        layout.addWidget(self.chart_toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        chart_container = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_container)
        chart_layout.addWidget(self._chart_view, 1)
        info = QtWidgets.QHBoxLayout()
        self.cursor_label = QtWidgets.QLabel("Cursor: —")
        self.roi_label = QtWidgets.QLabel("ROI: —")
        info.addWidget(self.cursor_label)
        info.addWidget(self.roi_label)
        info.addStretch(1)
        chart_layout.addLayout(info)
        splitter.addWidget(chart_container)

        # control panel
        self.controls_panel = QtWidgets.QWidget()
        ctrl = QtWidgets.QVBoxLayout(self.controls_panel)
        acq_box = QtWidgets.QGroupBox("Acquisition")
        acq = QtWidgets.QGridLayout(acq_box)
        row = 0
        self.integration_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.integration_slider.setRange(1, 1000)
        self.integration_spin = QtWidgets.QDoubleSpinBox()
        self.integration_spin.setRange(1.0, 1000.0)
        self.integration_spin.setValue(10.0)
        self.integration_spin.setSuffix(" ms")
        acq.addWidget(QtWidgets.QLabel("Integration:"), row, 0)
        acq.addWidget(self.integration_slider, row, 1)
        acq.addWidget(self.integration_spin, row, 2)
        row += 1
        self.averages_spin = QtWidgets.QSpinBox()
        self.averages_spin.setRange(1, 100)
        self.averages_spin.setValue(1)
        acq.addWidget(QtWidgets.QLabel("Averages:"), row, 0)
        acq.addWidget(self.averages_spin, row, 1, 1, 2)
        row += 1
        self.smoothing_spin = QtWidgets.QSpinBox()
        self.smoothing_spin.setRange(1, 101)
        self.smoothing_spin.setValue(5)
        acq.addWidget(QtWidgets.QLabel("Smoothing (boxcar):"), row, 0)
        acq.addWidget(self.smoothing_spin, row, 1, 1, 2)
        row += 1
        self.dark_chk = QtWidgets.QCheckBox("Subtract dark")
        self.btn_capture_dark = QtWidgets.QPushButton("Capture dark")
        acq.addWidget(self.dark_chk, row, 0, 1, 2)
        acq.addWidget(self.btn_capture_dark, row, 2)
        row += 1
        self.btn_single = QtWidgets.QPushButton("Single capture")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        acq.addWidget(self.btn_single, row, 0, 1, 3)
        row += 1
        acq.addWidget(self.btn_start, row, 0, 1, 2)
        acq.addWidget(self.btn_stop, row, 2)
        ctrl.addWidget(acq_box)

        mode_box = QtWidgets.QGroupBox("Modes")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        mode_layout.addWidget(self.btn_modes)
        ctrl.addWidget(mode_box)

        shelly_box = QtWidgets.QGroupBox("Shelly controls")
        shelly_layout = QtWidgets.QVBoxLayout(shelly_box)
        shelly_layout.addWidget(QtWidgets.QLabel("Light controls coming soon."))
        ctrl.addWidget(shelly_box)

        footer = QtWidgets.QVBoxLayout()
        self.status_message = QtWidgets.QLabel("Idle")
        self.fps_label = QtWidgets.QLabel("—")
        footer.addWidget(self.status_message)
        footer.addWidget(self.fps_label)
        footer.addStretch(1)
        ctrl.addLayout(footer)

        splitter.addWidget(self.controls_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.controls_panel.setVisible(not self._compact)

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_refresh.clicked.connect(self._refresh_devices)
        self.btn_connect.clicked.connect(self._connect_selected)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.device_combo.currentIndexChanged.connect(self._update_status_from_selection)
        self.spectrometer_manager.devices_changed.connect(self._on_devices_changed)
        self.spectrometer_manager.device_connected.connect(self._on_device_connected)

        self.integration_slider.valueChanged.connect(self.integration_spin.setValue)
        self.integration_spin.valueChanged.connect(self.integration_slider.setValue)
        self.btn_single.clicked.connect(self._capture_once)
        self.btn_start.clicked.connect(self._start_continuous)
        self.btn_stop.clicked.connect(self._stop_continuous)
        self.btn_capture_dark.clicked.connect(self._capture_dark)
        self.compact_toggle.toggled.connect(self._toggle_compact)
        self.btn_modes.clicked.connect(self._open_modes_dialog)

        self.act_zoom_in.triggered.connect(self._chart.zoomIn)
        self.act_zoom_out.triggered.connect(self._chart.zoomOut)
        self.act_reset.triggered.connect(self._chart.zoomReset)
        self.act_export.triggered.connect(self._export_plot)

        self._chart_view.cursorMoved.connect(self._update_cursor)
        self._chart_view.roiSelected.connect(self._apply_roi)
        if hasattr(self._chart.legend(), "markersChanged"):
            self._chart.legend().markersChanged.connect(self._install_legend_handlers)
        self._install_legend_handlers()

    # ------------------------------------------------------------------
    def _restore_geometry(self) -> None:
        geo = self.profiles.get("spectroscopy.geometry", "", expected_type=str)
        state = self.profiles.get("spectroscopy.window_state", "", expected_type=str)
        if geo:
            try:
                self.restoreGeometry(bytes.fromhex(geo))
            except Exception:
                pass
        if state:
            try:
                self.restoreState(bytes.fromhex(state))
            except Exception:
                pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.capture_timer.stop()
        try:
            self.profiles.set("spectroscopy.compact", self.compact_toggle.isChecked())
            self.profiles.set("spectroscopy.geometry", self.saveGeometry().hex())
            self.profiles.set("spectroscopy.window_state", self.saveState().hex())
            self.profiles.save()
        finally:
            super().closeEvent(event)

    # ------------------------------------------------------------------
    def _install_legend_handlers(self) -> None:
        for marker in self._chart.legend().markers():
            try:
                marker.clicked.disconnect()
            except Exception:
                pass
            marker.clicked.connect(lambda checked=False, m=marker: self._toggle_series(m))

    def _toggle_series(self, marker: QtCharts.QLegendMarker) -> None:
        series = marker.series()
        series.setVisible(not series.isVisible())
        marker.setVisible(True)

    # ------------------------------------------------------------------
    def _toggle_compact(self, checked: bool) -> None:
        self.controls_panel.setVisible(not checked)
        self._compact = checked

    def _refresh_devices(self) -> None:
        devices = self.spectrometer_manager.refresh()
        self._populate_devices(devices)

    def _on_devices_changed(self, devices: list) -> None:
        self._populate_devices(devices)

    def _populate_devices(self, devices: list[SpectrometerDescriptor]) -> None:
        current = self._current_descriptor()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        for desc in devices:
            label = f"{desc.model} {desc.serial_number} [{desc.path}]"
            self.device_combo.addItem(label, desc)
        self.device_combo.blockSignals(False)
        if current:
            self._set_current_descriptor(current)
        self._update_status_from_selection()

    def _current_descriptor(self) -> Optional[SpectrometerDescriptor]:
        idx = self.device_combo.currentIndex()
        if idx < 0:
            return None
        data = self.device_combo.currentData()
        return data if isinstance(data, SpectrometerDescriptor) else None

    def _set_current_descriptor(self, desc: SpectrometerDescriptor) -> None:
        for i in range(self.device_combo.count()):
            data = self.device_combo.itemData(i)
            if isinstance(data, SpectrometerDescriptor) and data == desc:
                self.device_combo.setCurrentIndex(i)
                break

    def _on_device_connected(self, device) -> None:
        if device is None:
            self.status_led.setStyleSheet("color: red;")
            self.status_label.setText("Disconnected")
        else:
            self.status_led.setStyleSheet("color: green;")
            label = getattr(device, "descriptor", None)
            self.status_label.setText(getattr(label, "label", lambda: "Connected")())
            try:
                wavelengths = device.get_wavelengths()
                self.session.set_wavelengths(wavelengths)
            except Exception as exc:
                LOG.warning("Failed to read wavelengths: %s", exc)
        self._update_status_from_selection()

    def _update_status_from_selection(self) -> None:
        active = self.spectrometer_manager.active
        desc = self._current_descriptor()
        is_active = active and desc and getattr(active, "descriptor", None) == desc
        self.btn_connect.setEnabled(bool(desc) and not is_active)
        self.btn_disconnect.setEnabled(bool(active))
        if is_active:
            self.status_led.setStyleSheet("color: green;")
            self.status_label.setText("Connected")
        elif desc:
            self.status_led.setStyleSheet("color: orange;")
            self.status_label.setText("Not connected")
        else:
            self.status_led.setStyleSheet("color: red;")
            self.status_label.setText("No spectrometer")

    def _connect_selected(self) -> None:
        desc = self._current_descriptor()
        if not desc:
            return
        try:
            dev = self.spectrometer_manager.connect(desc)
            wavelengths = dev.get_wavelengths()
            self.session.set_wavelengths(wavelengths)
        except Exception as exc:
            self.status_message.setText(f"Connect failed: {exc}")
            LOG.warning("Spectrometer connect failed: %s", exc)
        self._update_status_from_selection()

    def _disconnect(self) -> None:
        self.capture_timer.stop()
        self.spectrometer_manager.disconnect()
        self._update_status_from_selection()

    # ------------------------------------------------------------------
    def _capture_dark(self) -> None:
        dev = self.spectrometer_manager.active
        if dev is None:
            self.status_message.setText("No spectrometer connected")
            return
        try:
            dev.set_integration_time_ms(self.integration_spin.value())
            dev.set_averages(self.averages_spin.value())
            dark = dev.capture()
            self.session.set_dark(dark)
            self._plot_spectrum("dark", dark, color=QtGui.QColor("gray"))
            self.status_message.setText("Dark captured")
        except Exception as exc:
            self.status_message.setText(f"Dark capture failed: {exc}")

    def _capture_once(self) -> None:
        dev = self.spectrometer_manager.active
        if dev is None:
            self.status_message.setText("No spectrometer connected")
            self._stop_continuous()
            return
        start = time.time()
        try:
            dev.set_integration_time_ms(self.integration_spin.value())
            dev.set_averages(self.averages_spin.value())
            raw = dev.capture()
            self.session.set_raw(raw)
            data = smooth_boxcar(raw, window=self.smoothing_spin.value())
            if self.dark_chk.isChecked() and self.session.dark_spectrum is not None:
                try:
                    data = subtract_dark(data, self.session.dark_spectrum)
                except Exception:
                    pass
            self._plot_spectrum("live", data, color=QtGui.QColor("deepskyblue"))
            self.status_message.setText("Captured spectrum")
        except Exception as exc:
            self.status_message.setText(f"Capture failed: {exc}")
            self._stop_continuous()
            return
        duration = time.time() - start
        if duration > 0:
            fps = 1.0 / duration
            self.fps_label.setText(f"{fps:.1f} Hz")
        self._last_capture_ts = time.time()

    def _start_continuous(self) -> None:
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.capture_timer.start()
        self.status_message.setText("Continuous capture running")

    def _stop_continuous(self) -> None:
        self.capture_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_message.setText("Stopped")

    # ------------------------------------------------------------------
    def _plot_spectrum(self, key: str, data: np.ndarray, color: QtGui.QColor) -> None:
        wavelengths = self.session.wavelengths
        if wavelengths is None:
            return
        series = self._traces.get(key)
        if series is None:
            series = QtCharts.QLineSeries()
            meta = SpectrumTrace(key.title(), color)
            series.setName(meta.label)
            series.setColor(meta.color)
            self._chart.addSeries(series)
            axis_x = self._chart.axisX()
            axis_y = self._chart.axisY()
            if axis_x is None or axis_y is None:
                self._chart.createDefaultAxes()
                axis_x = self._chart.axisX()
                axis_y = self._chart.axisY()
            if axis_x:
                series.attachAxis(axis_x)
            if axis_y:
                series.attachAxis(axis_y)
            self._traces[key] = series
            self._trace_meta[key] = meta
            self._install_legend_handlers()
        series.replace([QtCore.QPointF(x, y) for x, y in zip(wavelengths, data)])
        axis_x = self._chart.axisX()
        axis_y = self._chart.axisY()
        if axis_x:
            axis_x.setRange(float(wavelengths.min()), float(wavelengths.max()))
        ymin = float(np.min(data))
        ymax = float(np.max(data))
        if ymin == ymax:
            ymax += 1.0
        if axis_y:
            axis_y.setRange(ymin, ymax)

    def _apply_roi(self, start: float, end: float) -> None:
        self.session.clear_rois()
        self.session.add_roi(start, end)
        self.roi_label.setText(f"ROI: {start:.1f}–{end:.1f} nm")

    def _update_cursor(self, x: float) -> None:
        y_values = []
        for series in self._traces.values():
            points = series.pointsVector()
            if not points:
                continue
            xs = [p.x() for p in points]
            ys = [p.y() for p in points]
            idx = np.searchsorted(xs, x)
            if 0 < idx < len(xs):
                y = np.interp(x, [xs[idx - 1], xs[idx]], [ys[idx - 1], ys[idx]])
            else:
                y = ys[min(idx, len(ys) - 1)]
            y_values.append(y)
        if y_values:
            self.cursor_label.setText(f"Cursor: {x:.1f} nm | {np.mean(y_values):.1f}")
        else:
            self.cursor_label.setText(f"Cursor: {x:.1f} nm")

    def _export_plot(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "spectrum.png",
            "Images (*.png *.jpg *.svg)",
        )
        if not path:
            return
        img = self._chart_view.grab()
        img.save(path)

    # ------------------------------------------------------------------
    def _open_modes_dialog(self) -> None:
        dlg = ModeSelectorDialog(self.MODES, parent=self)
        if dlg.exec() == QtWidgets.QDialog.Accepted and dlg.selected_mode:
            self._launch_wizard(dlg.selected_mode)

    def _launch_wizard(self, mode: str) -> None:
        wizard = SpectroscopyModeWizard(
            mode,
            self.session,
            capture_callback=self._wizard_capture,
            initial_acquisition={
                "integration": float(self.integration_spin.value()),
                "averages": int(self.averages_spin.value()),
                "smoothing": int(self.smoothing_spin.value()),
            },
            parent=self,
        )
        if wizard.exec() == QtWidgets.QDialog.Accepted:
            self.current_mode = mode
            self.mode_label.setText(f"Mode: {mode}")
            self.status_message.setText(f"{mode} wizard completed")
        else:
            self.status_message.setText(f"{mode} wizard cancelled")

    def _wizard_capture(self, kind: str, integration: float, averages: int) -> Optional[np.ndarray]:
        dev = self.spectrometer_manager.active
        if dev is None:
            self.status_message.setText("No spectrometer connected")
            return None
        try:
            dev.set_integration_time_ms(integration)
            dev.set_averages(averages)
            spectrum = dev.capture()
            return spectrum
        except Exception as exc:
            self.status_message.setText(f"Capture failed: {exc}")
            return None

