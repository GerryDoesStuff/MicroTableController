from __future__ import annotations

import contextlib
import io
import os
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets, QtCharts

from ..devices.shelly_dimmer import ShellyDimmer, ShellyState
from ..spectroscopy.devices import SpectrometerDescriptor, SpectrometerDevice, SpectrometerManager
from ..spectroscopy.io import (
    default_data_directory,
    ensure_data_directory,
    save_spectrum_csv,
    save_spectrum_hdf5,
    save_spectrum_jcamp,
    save_time_series_hdf5,
    save_time_series_npz,
)
from ..spectroscopy.processing import smooth_boxcar, subtract_dark
from ..spectroscopy.session import AcquisitionMetadata, SpectroscopySession
from .spectroscopy_modes import ModeSelectorDialog, SpectroscopyModeWizard
from ..utils.log import LOG, log
from ..utils.workers import run_async
import matplotlib.pyplot as plt


@dataclass
class CaptureJob:
    token: object
    device: SpectrometerDevice
    device_id: str
    lock: QtCore.QMutex
    integration_ms: float
    averages: int
    smoothing: int
    subtract_dark: bool
    dark_spectrum: Optional[np.ndarray]
    kind: str
    timestamp: float


class CaptureWorker(QtCore.QObject):
    capture_ready = QtCore.Signal(object, np.ndarray, np.ndarray, float, float)
    capture_failed = QtCore.Signal(object, str)
    capture_started = QtCore.Signal(object)

    @QtCore.Slot(object)
    def perform_capture(self, job: CaptureJob) -> None:
        self.capture_started.emit(job.token)
        start = time.time()
        try:
            job.lock.lock()
            try:
                job.device.set_integration_time_ms(job.integration_ms)
                job.device.set_averages(job.averages)
                raw = job.device.capture()
            finally:
                job.lock.unlock()
            processed = smooth_boxcar(np.asarray(raw, dtype=float), window=job.smoothing)
            if job.subtract_dark and job.dark_spectrum is not None:
                try:
                    processed = subtract_dark(processed, job.dark_spectrum)
                except Exception:
                    pass
            duration = time.time() - start
            peak = float(np.nanmax(raw)) if raw is not None else float("nan")
            self.capture_ready.emit(job, processed, np.asarray(raw, dtype=float), duration, peak)
        except Exception as exc:  # pragma: no cover - runtime safety
            self.capture_failed.emit(job, str(exc))


@dataclass
class SpectrumTrace:
    label: str
    color: QtGui.QColor
    visible: bool = True


@dataclass
class CapturedSpectrum:
    key: str
    label: str
    timestamp: float
    data: np.ndarray
    mode: str
    metadata: Dict[str, object]
    kind: str = "measurement"
    raw_data: Optional[np.ndarray] = None


@dataclass
class SaveOptions:
    path: str
    format: str
    include_processed: bool
    include_raw: bool


class SaveCaptureDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        default_dir: str,
        default_name: str,
        raw_available: bool,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save capture")
        layout = QtWidgets.QFormLayout(self)

        dir_row = QtWidgets.QHBoxLayout()
        self.dir_edit = QtWidgets.QLineEdit(default_dir)
        browse = QtWidgets.QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._choose_directory)
        dir_row.addWidget(self.dir_edit, 1)
        dir_row.addWidget(browse)
        layout.addRow("Directory", dir_row)

        self.name_edit = QtWidgets.QLineEdit(default_name)
        layout.addRow("File name", self.name_edit)

        self.format_combo = QtWidgets.QComboBox()
        self.format_combo.addItem("CSV", "csv")
        self.format_combo.addItem("HDF5", "h5")
        self.format_combo.addItem("JCAMP-DX", "jdx")
        layout.addRow("Format", self.format_combo)

        self.chk_processed = QtWidgets.QCheckBox("Save processed spectrum")
        self.chk_processed.setChecked(True)
        self.chk_raw = QtWidgets.QCheckBox("Include raw counts")
        self.chk_raw.setChecked(raw_available)
        self.chk_raw.setEnabled(raw_available)
        layout.addRow(self.chk_processed)
        layout.addRow(self.chk_raw)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("color: red;")
        layout.addRow(self.status)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self._options: Optional[SaveOptions] = None

    def _choose_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select directory", self.dir_edit.text())
        if directory:
            self.dir_edit.setText(directory)

    def _on_accept(self) -> None:
        directory = ensure_data_directory(self.dir_edit.text())
        filename = self.name_edit.text().strip()
        ext = self.format_combo.currentData()
        include_processed = self.chk_processed.isChecked()
        include_raw = self.chk_raw.isChecked()
        if not filename:
            self.status.setText("File name is required")
            return
        if not include_processed and not include_raw:
            self.status.setText("Select at least one spectrum to save")
            return
        path = os.path.join(directory, f"{filename}.{ext}")
        self._options = SaveOptions(
            path=path,
            format=str(ext),
            include_processed=include_processed,
            include_raw=include_raw,
        )
        self.accept()

    @property
    def options(self) -> Optional[SaveOptions]:
        return self._options


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
        self.current_mode = str(self.profiles.get("spectroscopy.last_mode", "Absorbance", expected_type=str))
        self.mode_params = dict(self.profiles.get("spectroscopy.last_params", {}, expected_type=dict) or {})
        self._compact = bool(self.profiles.get("spectroscopy.compact", False, expected_type=bool))
        self._last_capture_ts = 0.0
        stored_dir = str(self.profiles.get("spectroscopy.data_dir", "", expected_type=str))
        self._data_dir = ensure_data_directory(stored_dir or default_data_directory())
        self._default_device = str(self.profiles.get("spectroscopy.default_device", "", expected_type=str))
        self._recent_captures: List[CapturedSpectrum] = []
        self._time_series_active = False
        self._time_series_records: List[tuple[float, np.ndarray, Dict[str, object]]] = []
        self._time_series_start = 0.0
        self._time_series_sample_count = 0
        self._ts_traces: Dict[str, QtCharts.QLineSeries] = {}
        self._time_series_timer = QtCore.QTimer(self)
        self._time_series_timer.setSingleShot(True)
        self._time_series_timer.timeout.connect(self._trigger_time_series_capture)
        self._recent_colors = [
            QtGui.QColor("#1f77b4"),
            QtGui.QColor("#ff7f0e"),
            QtGui.QColor("#2ca02c"),
            QtGui.QColor("#d62728"),
            QtGui.QColor("#9467bd"),
            QtGui.QColor("#8c564b"),
        ]

        # Shelly dimmer state
        self.dimmer: Optional[ShellyDimmer] = None
        self._dimmer_conn_thread = None
        self._dimmer_conn_worker = None
        self._dimmer_op_thread = None
        self._dimmer_op_worker = None
        self._updating_dimmer_ui = False
        self._shelly_presets: Dict[str, ShellyState] = self._load_shelly_presets()
        self._step_presets: Dict[str, str] = self._load_step_presets()

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
        self.capture_timer.timeout.connect(self._trigger_continuous_capture)

        self._capture_thread = QtCore.QThread(self)
        self._capture_worker = CaptureWorker()
        self._capture_worker.moveToThread(self._capture_thread)
        self._capture_worker.capture_ready.connect(self._on_capture_ready)
        self._capture_worker.capture_failed.connect(self._on_capture_failed)
        self._capture_worker.capture_started.connect(self._on_capture_started)
        self._capture_thread.start()
        self._capture_in_flight = False
        self._capture_token = object()
        self._continuous = False
        self._rate_limit_ms = 150

        self._build_ui()
        self._restore_geometry()
        self._connect_signals()
        self._update_session_context()
        self._refresh_validity_state()

        if self.mode_params:
            self.session.set_mode_params(**self.mode_params)

        QtCore.QTimer.singleShot(0, self._refresh_devices)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        # top bar
        top = QtWidgets.QHBoxLayout()
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.setToolTip("Select a spectrometer device to connect")
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setToolTip("Scan for connected spectrometers")
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.setToolTip("Connect to the selected spectrometer")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_disconnect.setToolTip("Disconnect the active spectrometer")
        self.status_led = QtWidgets.QLabel("●")
        self.status_led.setStyleSheet("color: red;")
        self.status_label = QtWidgets.QLabel("No spectrometer")
        self.status_label.setToolTip("Connection status")
        self.mode_label = QtWidgets.QLabel(f"Mode: {self.current_mode}")
        self.btn_modes = QtWidgets.QPushButton("Modes…")
        self.btn_modes.setToolTip("Launch mode wizard and presets")
        self.compact_toggle = QtWidgets.QToolButton()
        self.compact_toggle.setText("Compact")
        self.compact_toggle.setCheckable(True)
        self.compact_toggle.setChecked(self._compact)
        self.compact_toggle.setToolTip("Toggle compact layout (hides control panel)")
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
        self.act_zoom_in.setToolTip("Zoom in on the spectrum")
        self.act_zoom_out = self.chart_toolbar.addAction("Zoom Out")
        self.act_zoom_out.setToolTip("Zoom out from the spectrum")
        self.act_reset = self.chart_toolbar.addAction("Reset")
        self.act_reset.setToolTip("Reset zoom and pan")
        self.act_export = self.chart_toolbar.addAction("Save Plot…")
        self.act_export.setToolTip("Export the current plot as an image")
        layout.addWidget(self.chart_toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter = splitter
        layout.addWidget(splitter, 1)

        chart_container = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_container)
        chart_layout.addWidget(self._chart_view, 1)
        info = QtWidgets.QHBoxLayout()
        self.cursor_label = QtWidgets.QLabel("Cursor: —")
        self.roi_label = QtWidgets.QLabel("ROI: —")
        self.cursor_label.setToolTip("Hover over the plot to inspect values")
        self.roi_label.setToolTip("Drag with Shift to set a region of interest")
        info.addWidget(self.cursor_label)
        info.addWidget(self.roi_label)
        info.addStretch(1)
        chart_layout.addLayout(info)
        ts_group = QtWidgets.QGroupBox("Time series views")
        ts_layout = QtWidgets.QVBoxLayout(ts_group)
        self._ts_chart = QtCharts.QChart()
        self._ts_chart.legend().setVisible(True)
        self._ts_chart.createDefaultAxes()
        self._ts_chart.axisX().setTitleText("Time (s)")
        self._ts_chart.axisY().setTitleText("Intensity (a.u.)")
        self._ts_chart_view = QtCharts.QChartView(self._ts_chart)
        self._ts_chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        ts_layout.addWidget(self._ts_chart_view, 1)
        self.spectrogram_label = QtWidgets.QLabel("Spectrogram hidden")
        self.spectrogram_label.setAlignment(QtCore.Qt.AlignCenter)
        self.spectrogram_label.setMinimumHeight(120)
        ts_layout.addWidget(self.spectrogram_label)
        chart_layout.addWidget(ts_group)
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
        self.integration_spin.setToolTip("Integration time per capture (ms)")
        self.integration_slider.setToolTip("Integration time per capture (ms)")
        acq.addWidget(QtWidgets.QLabel("Integration:"), row, 0)
        acq.addWidget(self.integration_slider, row, 1)
        acq.addWidget(self.integration_spin, row, 2)
        row += 1
        self.averages_spin = QtWidgets.QSpinBox()
        self.averages_spin.setRange(1, 100)
        self.averages_spin.setValue(1)
        self.averages_spin.setToolTip("Number of captures to average")
        acq.addWidget(QtWidgets.QLabel("Averages:"), row, 0)
        acq.addWidget(self.averages_spin, row, 1, 1, 2)
        row += 1
        self.smoothing_spin = QtWidgets.QSpinBox()
        self.smoothing_spin.setRange(1, 101)
        self.smoothing_spin.setValue(5)
        self.smoothing_spin.setToolTip("Boxcar smoothing window (odd values recommended)")
        acq.addWidget(QtWidgets.QLabel("Smoothing (boxcar):"), row, 0)
        acq.addWidget(self.smoothing_spin, row, 1, 1, 2)
        row += 1
        self.dark_chk = QtWidgets.QCheckBox("Subtract dark")
        self.btn_capture_dark = QtWidgets.QPushButton("Capture dark")
        self.dark_chk.setToolTip("Subtract the stored dark spectrum from captures")
        self.btn_capture_dark.setToolTip("Capture a new dark spectrum")
        acq.addWidget(self.dark_chk, row, 0, 1, 2)
        acq.addWidget(self.btn_capture_dark, row, 2)
        row += 1
        self.btn_single = QtWidgets.QPushButton("Single capture")
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_single.setToolTip("Capture one spectrum and add to recents")
        self.btn_start.setToolTip("Start continuous capture")
        self.btn_stop.setToolTip("Stop continuous capture")
        acq.addWidget(self.btn_single, row, 0, 1, 3)
        row += 1
        acq.addWidget(self.btn_start, row, 0, 1, 2)
        acq.addWidget(self.btn_stop, row, 2)
        ctrl.addWidget(acq_box)

        mode_box = QtWidgets.QGroupBox("Modes")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        mode_layout.addWidget(self.btn_modes)
        ctrl.addWidget(mode_box)

        data_box = QtWidgets.QGroupBox("Data")
        data_layout = QtWidgets.QFormLayout(data_box)
        self.data_dir_edit = QtWidgets.QLineEdit(self._data_dir)
        self.data_dir_edit.setPlaceholderText("Select a folder for saved spectra")
        browse = QtWidgets.QToolButton()
        browse.setText("…")
        browse.setToolTip("Choose a folder to save capture exports")
        browse.clicked.connect(self._choose_data_dir)
        dir_row = QtWidgets.QHBoxLayout()
        dir_row.addWidget(self.data_dir_edit, 1)
        dir_row.addWidget(browse)
        data_layout.addRow("Data folder", dir_row)
        self.btn_save_selected = QtWidgets.QPushButton("Save selected capture")
        self.btn_save_selected.setToolTip("Export the selected capture with format and data options")
        data_layout.addRow(self.btn_save_selected)
        self.time_series_interval = QtWidgets.QDoubleSpinBox()
        self.time_series_interval.setRange(0.05, 3600.0)
        self.time_series_interval.setValue(1.0)
        self.time_series_interval.setSuffix(" s")
        self.time_series_interval.setToolTip("Interval between captures when recording a time series")
        data_layout.addRow("Interval", self.time_series_interval)
        self.time_series_duration = QtWidgets.QDoubleSpinBox()
        self.time_series_duration.setRange(0.0, 86400.0)
        self.time_series_duration.setValue(30.0)
        self.time_series_duration.setSuffix(" s (0 = no limit)")
        self.time_series_duration.setToolTip("Maximum duration of a time series recording; 0 disables duration stop")
        data_layout.addRow("Duration limit", self.time_series_duration)
        self.time_series_samples = QtWidgets.QSpinBox()
        self.time_series_samples.setRange(0, 1000000)
        self.time_series_samples.setValue(0)
        self.time_series_samples.setToolTip("Maximum number of captures to record; 0 disables sample limit")
        data_layout.addRow("Max samples", self.time_series_samples)
        self.btn_start_timeseries = QtWidgets.QPushButton("Start time series")
        self.btn_stop_timeseries = QtWidgets.QPushButton("Stop time series")
        self.btn_stop_timeseries.setEnabled(False)
        self.chk_plot_timeseries = QtWidgets.QCheckBox("Save intensity vs time plot")
        self.chk_plot_spectrogram = QtWidgets.QCheckBox("Save spectrogram")
        self.chk_show_timeseries = QtWidgets.QCheckBox("Show intensity vs time")
        self.chk_show_timeseries.setChecked(True)
        self.chk_show_spectrogram = QtWidgets.QCheckBox("Show spectrogram")
        ts_row = QtWidgets.QHBoxLayout()
        ts_row.addWidget(self.btn_start_timeseries)
        ts_row.addWidget(self.btn_stop_timeseries)
        data_layout.addRow("Time series", ts_row)
        self.timeseries_wavelengths = QtWidgets.QLineEdit("450, 550, 650")
        self.timeseries_wavelengths.setPlaceholderText("Comma-separated wavelengths to track")
        data_layout.addRow("Tracked wavelengths", self.timeseries_wavelengths)
        data_layout.addRow(self.chk_plot_timeseries)
        data_layout.addRow(self.chk_plot_spectrogram)
        data_layout.addRow(self.chk_show_timeseries)
        data_layout.addRow(self.chk_show_spectrogram)
        ctrl.addWidget(data_box)

        shelly_box = QtWidgets.QGroupBox("Shelly controls")
        shelly_layout = QtWidgets.QVBoxLayout(shelly_box)
        self.shelly_status = QtWidgets.QLabel("Illumination: —")
        self.shelly_status.setTextFormat(QtCore.Qt.PlainText)
        shelly_layout.addWidget(self.shelly_status)

        host_row = QtWidgets.QHBoxLayout()
        host_row.addWidget(QtWidgets.QLabel("Host:"))
        self.shelly_host_edit = QtWidgets.QLineEdit(
            self.profiles.get("illumination.dimmer.host", "", expected_type=str)
        )
        host_row.addWidget(self.shelly_host_edit, 1)
        self.btn_shelly_connect = QtWidgets.QPushButton("Connect")
        self.btn_shelly_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_shelly_disconnect.setEnabled(False)
        host_row.addWidget(self.btn_shelly_connect)
        host_row.addWidget(self.btn_shelly_disconnect)
        shelly_layout.addLayout(host_row)

        toggle_row = QtWidgets.QHBoxLayout()
        self.shelly_toggle = QtWidgets.QCheckBox("On")
        self.shelly_toggle.setChecked(bool(self.profiles.get("illumination.dimmer.on", False)))
        self.shelly_toggle.setEnabled(False)
        toggle_row.addWidget(self.shelly_toggle)
        self.shelly_brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.shelly_brightness_slider.setRange(0, 100)
        self.shelly_brightness_spin = QtWidgets.QSpinBox()
        self.shelly_brightness_spin.setRange(0, 100)
        self.shelly_brightness_spin.setValue(
            int(self.profiles.get("illumination.dimmer.brightness", 0, expected_type=int, min_value=0, max_value=100))
        )
        self.shelly_brightness_slider.setValue(self.shelly_brightness_spin.value())
        self.shelly_brightness_slider.setEnabled(False)
        self.shelly_brightness_spin.setEnabled(False)
        toggle_row.addWidget(self.shelly_brightness_slider, 1)
        toggle_row.addWidget(self.shelly_brightness_spin)
        shelly_layout.addLayout(toggle_row)

        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.shelly_preset_combo = QtWidgets.QComboBox()
        self.step_presets: Dict[str, QtWidgets.QComboBox] = {}
        self._refresh_preset_combo()
        preset_row.addWidget(self.shelly_preset_combo, 1)
        self.btn_apply_preset = QtWidgets.QPushButton("Apply")
        self.btn_save_preset = QtWidgets.QPushButton("Save current")
        preset_row.addWidget(self.btn_apply_preset)
        preset_row.addWidget(self.btn_save_preset)
        shelly_layout.addLayout(preset_row)

        step_form = QtWidgets.QFormLayout()
        for key, label in (("dark", "Dark step"), ("reference", "Reference"), ("raw", "Measurement")):
            combo = QtWidgets.QComboBox()
            combo.addItems(self._shelly_presets.keys())
            combo.setCurrentText(self._step_presets.get(key, next(iter(self._shelly_presets), "")))
            self.step_presets[key] = combo
            step_form.addRow(label, combo)
        shelly_layout.addLayout(step_form)
        ctrl.addWidget(shelly_box)

        footer = QtWidgets.QVBoxLayout()
        self.status_message = QtWidgets.QLabel("Idle")
        self.validity_label = QtWidgets.QLabel("Baseline: unknown")
        self.fps_label = QtWidgets.QLabel("—")
        self.saturation_label = QtWidgets.QLabel("Peak: —")
        footer.addWidget(self.status_message)
        footer.addWidget(self.validity_label)
        footer.addWidget(self.fps_label)
        footer.addWidget(self.saturation_label)
        footer.addStretch(1)
        ctrl.addLayout(footer)

        splitter.addWidget(self.controls_panel)
        self.recent_panel = self._build_recent_panel()
        splitter.addWidget(self.recent_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        self.controls_panel.setVisible(not self._compact)

    def _build_recent_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Recent captures"))
        self.btn_clear_recent = QtWidgets.QToolButton()
        self.btn_clear_recent.setText("Clear")
        self.btn_clear_recent.setToolTip("Clear the recent capture list")
        header.addWidget(self.btn_clear_recent)
        header.addStretch(1)
        layout.addLayout(header)

        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.recent_list.setAlternatingRowColors(True)
        layout.addWidget(self.recent_list, 1)

        self.recent_meta = QtWidgets.QLabel("No captures yet")
        self.recent_meta.setWordWrap(True)
        layout.addWidget(self.recent_meta)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_use_reference = QtWidgets.QPushButton("Use as reference")
        self.btn_use_reference.setToolTip("Apply the selected capture as reference")
        self.btn_use_dark = QtWidgets.QPushButton("Use as dark")
        self.btn_use_dark.setToolTip("Apply the selected capture as dark")
        btn_row.addWidget(self.btn_use_reference)
        btn_row.addWidget(self.btn_use_dark)
        layout.addLayout(btn_row)
        return panel

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.btn_refresh.clicked.connect(self._refresh_devices)
        self.btn_connect.clicked.connect(self._connect_selected)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.device_combo.currentIndexChanged.connect(self._update_status_from_selection)
        self.spectrometer_manager.devices_changed.connect(self._on_devices_changed)
        self.spectrometer_manager.device_connected.connect(self._on_device_connected)
        self.session.validity_changed.connect(self._refresh_validity_state)

        self.integration_slider.valueChanged.connect(self.integration_spin.setValue)
        self.integration_spin.valueChanged.connect(self.integration_slider.setValue)
        self.integration_spin.valueChanged.connect(lambda _=None: self._persist_acquisition_settings())
        self.averages_spin.valueChanged.connect(lambda _=None: self._persist_acquisition_settings())
        self.smoothing_spin.valueChanged.connect(lambda _=None: self._persist_acquisition_settings())
        self.btn_single.clicked.connect(self._capture_once)
        self.btn_start.clicked.connect(self._start_continuous)
        self.btn_stop.clicked.connect(self._stop_continuous)
        self.btn_capture_dark.clicked.connect(self._capture_dark)
        self.dark_chk.toggled.connect(lambda _=None: self._persist_acquisition_settings())
        self.compact_toggle.toggled.connect(self._toggle_compact)
        self.btn_modes.clicked.connect(self._open_modes_dialog)
        self.btn_clear_recent.clicked.connect(self._clear_recent)
        self.recent_list.itemSelectionChanged.connect(self._on_recent_selected)
        self.recent_list.itemChanged.connect(self._on_recent_item_changed)
        self.btn_use_reference.clicked.connect(lambda: self._apply_capture_as("reference"))
        self.btn_use_dark.clicked.connect(lambda: self._apply_capture_as("dark"))
        self.btn_save_selected.clicked.connect(self._save_selected_capture)
        self.btn_start_timeseries.clicked.connect(self._start_time_series)
        self.btn_stop_timeseries.clicked.connect(self._stop_time_series)
        self.chk_show_timeseries.toggled.connect(self._update_time_series_views)
        self.chk_show_spectrogram.toggled.connect(self._update_time_series_views)
        self.timeseries_wavelengths.editingFinished.connect(self._update_time_series_views)
        self.data_dir_edit.editingFinished.connect(self._persist_data_dir)

        self.btn_shelly_connect.clicked.connect(self._connect_dimmer_async)
        self.btn_shelly_disconnect.clicked.connect(self._disconnect_dimmer)
        self.shelly_toggle.toggled.connect(self._on_shelly_toggle)
        self.shelly_brightness_slider.valueChanged.connect(self.shelly_brightness_spin.setValue)
        self.shelly_brightness_spin.valueChanged.connect(self.shelly_brightness_slider.setValue)
        self.shelly_brightness_slider.sliderReleased.connect(self._on_shelly_brightness_changed)
        self.shelly_brightness_spin.editingFinished.connect(self._on_shelly_brightness_changed)
        self.btn_apply_preset.clicked.connect(self._apply_selected_preset)
        self.btn_save_preset.clicked.connect(self._save_selected_preset)
        for key, combo in self.step_presets.items():
            combo.currentTextChanged.connect(lambda name, k=key: self._on_step_preset_changed(k, name))

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
        splitter_state = self.profiles.get("spectroscopy.splitter_state", "", expected_type=str)
        if splitter_state:
            try:
                self.splitter.restoreState(bytes.fromhex(splitter_state))
            except Exception:
                pass
        # acquisition defaults
        self.integration_spin.setValue(float(self.profiles.get("spectroscopy.integration_ms", self.integration_spin.value(), expected_type=float)))
        self.averages_spin.setValue(int(self.profiles.get("spectroscopy.averages", self.averages_spin.value(), expected_type=int)))
        self.smoothing_spin.setValue(int(self.profiles.get("spectroscopy.smoothing", self.smoothing_spin.value(), expected_type=int)))
        self.dark_chk.setChecked(bool(self.profiles.get("spectroscopy.subtract_dark", self.dark_chk.isChecked(), expected_type=bool)))
        self.integration_slider.setValue(int(self.integration_spin.value()))
        self.mode_label.setText(f"Mode: {self.current_mode}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self.capture_timer.stop()
        self._capture_token = object()
        if self._capture_thread.isRunning():
            self._capture_thread.quit()
            self._capture_thread.wait(1000)
        try:
            self._disconnect_dimmer()
            self.profiles.set("spectroscopy.compact", self.compact_toggle.isChecked())
            self.profiles.set("spectroscopy.geometry", self.saveGeometry().hex())
            self.profiles.set("spectroscopy.window_state", self.saveState().hex())
            self.profiles.set("spectroscopy.splitter_state", self.splitter.saveState().hex())
            self._persist_acquisition_settings(save=False)
            self._persist_data_dir(save=False)
            self.profiles.set("spectroscopy.last_mode", self.current_mode)
            self.profiles.set("spectroscopy.last_params", dict(self.session.mode_params))
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

    def _remove_trace(self, key: str) -> None:
        series = self._traces.pop(key, None)
        if series:
            self._chart.removeSeries(series)
            self._trace_meta.pop(key, None)
            self._install_legend_handlers()

    # ------------------------------------------------------------------
    def _toggle_compact(self, checked: bool) -> None:
        self.controls_panel.setVisible(not checked)
        self._compact = checked

    def _refresh_validity_state(self) -> None:
        if not hasattr(self, "validity_label"):
            return
        valid = not self.session.requires_recalibration()
        dark_ok = self.session.dark_valid
        ref_ok = self.session.reference_valid
        cal_ok = self.session.calibration_valid
        parts = []
        parts.append("Dark ✓" if dark_ok else "Dark ✕")
        parts.append("Reference ✓" if ref_ok else "Reference ✕")
        parts.append("Calibration ✓" if cal_ok else "Calibration ✕")
        status_text = " | ".join(parts)
        prefix = "Baseline: "
        self.validity_label.setText(prefix + status_text)
        style = "color: green;" if valid else "color: orange;"
        self.validity_label.setStyleSheet(style)
        self.act_export.setEnabled(valid)
        self.btn_save_selected.setEnabled(valid and self.recent_list.count() > 0 and bool(self._data_dir))
        can_record = valid and bool(self._data_dir)
        self.btn_start_timeseries.setEnabled(can_record and not self._time_series_active)
        self.btn_stop_timeseries.setEnabled(can_record and self._time_series_active)

    def _persist_acquisition_settings(self, save: bool = True) -> None:
        self.profiles.set("spectroscopy.integration_ms", float(self.integration_spin.value()))
        self.profiles.set("spectroscopy.averages", int(self.averages_spin.value()))
        self.profiles.set("spectroscopy.smoothing", int(self.smoothing_spin.value()))
        self.profiles.set("spectroscopy.subtract_dark", bool(self.dark_chk.isChecked()))
        self._update_session_context()
        if save:
            self.profiles.save()

    def _update_session_context(self) -> None:
        self.session.set_acquisition_context(
            device_id=self._current_device_id(),
            integration_ms=float(self.integration_spin.value()),
            averages=int(self.averages_spin.value()),
            mode=self.current_mode,
            timestamp=time.time(),
        )

    def _choose_data_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select spectroscopy data folder",
            self.data_dir_edit.text() or self._data_dir or "",
        )
        if directory:
            self.data_dir_edit.setText(directory)
            self._persist_data_dir()

    def _persist_data_dir(self, save: bool = True) -> None:
        entered = (self.data_dir_edit.text() or "").strip()
        self._data_dir = ensure_data_directory(entered or default_data_directory())
        self.profiles.set("spectroscopy.data_dir", self._data_dir)
        self._refresh_validity_state()
        if save:
            self.profiles.save()

    # ------------------------------------------------------------------
    def _load_shelly_presets(self) -> Dict[str, ShellyState]:
        defaults = {
            "Dark": ShellyState(on=False, brightness=0),
            "Reference": ShellyState(on=True, brightness=25),
            "Measurement": ShellyState(on=True, brightness=70),
        }
        stored = self.profiles.get("spectroscopy.shelly.presets", {}, expected_type=dict) or {}
        presets: Dict[str, ShellyState] = dict(defaults)
        for name, payload in stored.items():
            if not isinstance(payload, dict):
                continue
            brightness = payload.get("brightness", presets.get(name, defaults["Dark"]).brightness)
            on_state = payload.get("on", presets.get(name, defaults["Dark"]).on)
            try:
                presets[name] = ShellyState(on=bool(on_state), brightness=max(0, min(100, int(brightness))))
            except Exception:
                continue
        return presets

    def _save_shelly_presets(self) -> None:
        payload = {name: {"on": state.on, "brightness": state.brightness} for name, state in self._shelly_presets.items()}
        self.profiles.set("spectroscopy.shelly.presets", payload)
        self.profiles.save()

    def _load_step_presets(self) -> Dict[str, str]:
        stored = self.profiles.get("spectroscopy.shelly.step_presets", {}, expected_type=dict) or {}
        defaults = {"dark": "Dark", "reference": "Reference", "raw": "Measurement"}
        result = {}
        for key, default in defaults.items():
            val = stored.get(key, default)
            result[key] = val if val in self._shelly_presets else default
        return result

    def _save_step_presets(self) -> None:
        self.profiles.set("spectroscopy.shelly.step_presets", dict(self._step_presets))
        self.profiles.save()

    def _refresh_preset_combo(self, select: Optional[str] = None) -> None:
        self.shelly_preset_combo.blockSignals(True)
        current = select or self.shelly_preset_combo.currentText()
        self.shelly_preset_combo.clear()
        for name in sorted(self._shelly_presets.keys()):
            self.shelly_preset_combo.addItem(name)
        if current:
            idx = self.shelly_preset_combo.findText(current)
            if idx >= 0:
                self.shelly_preset_combo.setCurrentIndex(idx)
        self.shelly_preset_combo.blockSignals(False)
        self._refresh_step_presets_widgets()

    def _refresh_step_presets_widgets(self) -> None:
        for key, combo in self.step_presets.items():
            with QtCore.QSignalBlocker(combo):
                combo.clear()
                combo.addItems(sorted(self._shelly_presets.keys()))
                combo.setCurrentText(self._step_presets.get(key, ""))

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
        elif self._default_device:
            self._apply_default_device(devices)
        elif self.spectrometer_manager.active_descriptor:
            self._set_current_descriptor(self.spectrometer_manager.active_descriptor)
        self._update_status_from_selection()

    def _apply_default_device(self, devices: list[SpectrometerDescriptor]) -> None:
        for desc in devices:
            if self._device_key(desc) == self._default_device:
                self._set_current_descriptor(desc)
                break

    def _current_descriptor(self) -> Optional[SpectrometerDescriptor]:
        idx = self.device_combo.currentIndex()
        if idx < 0:
            return None
        data = self.device_combo.currentData()
        return data if isinstance(data, SpectrometerDescriptor) else None

    def _device_key(self, desc: SpectrometerDescriptor) -> str:
        return f"{desc.vendor}:{desc.serial_number}:{desc.path}"

    def _current_device_id(self) -> str:
        desc = self._current_descriptor() or self.spectrometer_manager.active_descriptor
        return self._device_key(desc) if desc else ""

    def _set_current_descriptor(self, desc: SpectrometerDescriptor) -> None:
        for i in range(self.device_combo.count()):
            data = self.device_combo.itemData(i)
            if isinstance(data, SpectrometerDescriptor) and data == desc:
                self.device_combo.setCurrentIndex(i)
                break

    def _on_device_connected(self, descriptor, device) -> None:
        selected = self._current_descriptor()
        if device is None:
            if selected == descriptor:
                self.status_led.setStyleSheet("color: red;")
                self.status_label.setText("Disconnected")
        else:
            if descriptor and selected is None:
                self._set_current_descriptor(descriptor)
            if descriptor == self._current_descriptor():
                self.status_led.setStyleSheet("color: green;")
                self.status_label.setText(descriptor.label())
            try:
                if descriptor == self._current_descriptor():
                    wavelengths = device.get_wavelengths()
                    self.session.set_wavelengths(wavelengths)
            except Exception as exc:
                LOG.warning("Failed to read wavelengths: %s", exc)
        self._update_status_from_selection()

    def _update_status_from_selection(self) -> None:
        desc = self._current_descriptor()
        active = self.spectrometer_manager.get_active(desc) if desc else None
        self.btn_connect.setEnabled(bool(desc) and active is None)
        self.btn_disconnect.setEnabled(active is not None)
        if active:
            self.status_led.setStyleSheet("color: green;")
            self.status_label.setText("Connected")
        elif desc:
            self.status_led.setStyleSheet("color: orange;")
            self.status_label.setText("Not connected")
        else:
            self.status_led.setStyleSheet("color: red;")
            self.status_label.setText("No spectrometer")

    def _active_device_with_lock(self):
        desc = self._current_descriptor() or self.spectrometer_manager.active_descriptor
        if desc is None:
            return None, None, None
        device = self.spectrometer_manager.get_active(desc)
        lock = self.spectrometer_manager.acquisition_lock(desc) if device else None
        return device, lock, desc

    def _acquisition_context(self, lock: Optional[QtCore.QMutex]):
        return QtCore.QMutexLocker(lock) if lock is not None else contextlib.nullcontext()

    # ------------------------------------------------------------------
    def _update_dimmer_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.shelly_toggle,
            self.shelly_brightness_slider,
            self.shelly_brightness_spin,
            self.btn_apply_preset,
            self.btn_save_preset,
            self.shelly_preset_combo,
            *self.step_presets.values(),
        ):
            widget.setEnabled(enabled)
        self.btn_shelly_disconnect.setEnabled(enabled and self.dimmer is not None)

    def _connect_dimmer_async(self) -> None:
        host = (self.shelly_host_edit.text() or "").strip()
        if not host:
            self.shelly_status.setText("Illumination: host required")
            return
        self.shelly_status.setText("Illumination: connecting…")
        self._update_dimmer_controls_enabled(False)
        if self._dimmer_conn_thread:
            self._dimmer_conn_thread.quit()
            self._dimmer_conn_thread.wait()

        def do_connect():
            dimmer = ShellyDimmer(host)
            state = dimmer.connect()
            return dimmer, state

        self._dimmer_conn_thread, self._dimmer_conn_worker = run_async(do_connect)
        self._dimmer_conn_worker.finished.connect(self._on_dimmer_connect)

    @QtCore.Slot(object, object)
    def _on_dimmer_connect(self, result, err):
        thread = self._dimmer_conn_thread
        self._dimmer_conn_thread = self._dimmer_conn_worker = None
        if err or result is None:
            log(f"Shelly connect failed: {err}")
            self._handle_dimmer_failure(err)
        else:
            dimmer, state = result
            self.dimmer = dimmer
            self._apply_dimmer_state(state)
            self.btn_shelly_disconnect.setEnabled(True)
            log(f"Shelly connected: {dimmer.base_url}")
        if thread and thread != QtCore.QThread.currentThread():
            thread.wait()

    def _disconnect_dimmer(self) -> None:
        for attr in ("_dimmer_conn_thread", "_dimmer_op_thread"):
            thread = getattr(self, attr)
            worker_attr = f"{attr.replace('thread', 'worker')}"
            if thread:
                thread.quit()
                thread.wait()
            setattr(self, attr, None)
            setattr(self, worker_attr, None)
        self.dimmer = None
        self._update_dimmer_controls_enabled(False)
        try:
            self._updating_dimmer_ui = True
            self.shelly_toggle.setChecked(False)
            self.shelly_brightness_slider.setValue(self.shelly_brightness_spin.value())
        finally:
            self._updating_dimmer_ui = False
        self.shelly_status.setText("Illumination: disconnected")

    def _run_dimmer_action(self, fn):
        if not self.dimmer:
            self.shelly_status.setText("Illumination: not connected")
            return
        self.shelly_status.setText("Illumination: updating…")
        self._update_dimmer_controls_enabled(False)
        if self._dimmer_op_thread:
            self._dimmer_op_thread.quit()
            self._dimmer_op_thread.wait()
        self._dimmer_op_thread, self._dimmer_op_worker = run_async(fn)
        self._dimmer_op_worker.finished.connect(self._on_dimmer_action_done)

    @QtCore.Slot(object, object)
    def _on_dimmer_action_done(self, state, err):
        thread = self._dimmer_op_thread
        self._dimmer_op_thread = self._dimmer_op_worker = None
        if err or state is None:
            log(f"Shelly command failed: {err}")
            self._handle_dimmer_failure(err)
        else:
            self._apply_dimmer_state(state)
        if thread and thread != QtCore.QThread.currentThread():
            thread.wait()

    def _apply_dimmer_state(self, state: ShellyState):
        if state is None:
            self._handle_dimmer_failure("no state")
            return
        self._updating_dimmer_ui = True
        try:
            self.shelly_toggle.setChecked(state.on)
            self.shelly_brightness_spin.setValue(state.brightness)
            self.shelly_brightness_slider.setValue(state.brightness)
        finally:
            self._updating_dimmer_ui = False
        status = "on" if state.on else "off"
        self.shelly_status.setText(f"Illumination: {status} ({state.brightness}%)")
        self._update_dimmer_controls_enabled(True)
        self.btn_shelly_disconnect.setEnabled(True)
        self.profiles.set("illumination.dimmer.on", state.on)
        self.profiles.set("illumination.dimmer.brightness", state.brightness)
        self.profiles.save()
        log(f"Shelly state -> {status} @ {state.brightness}%")

    def _on_shelly_toggle(self, checked: bool) -> None:
        if self._updating_dimmer_ui:
            return
        log(f"Shelly toggle -> {'on' if checked else 'off'}")
        self._run_dimmer_action(lambda: self.dimmer.set_on(checked) if self.dimmer else None)

    def _on_shelly_brightness_changed(self) -> None:
        if self._updating_dimmer_ui:
            return
        value = int(self.shelly_brightness_spin.value())
        log(f"Shelly brightness -> {value}%")
        self._run_dimmer_action(lambda: self.dimmer.set_brightness(value) if self.dimmer else None)

    def _handle_dimmer_failure(self, err=None):
        self._update_dimmer_controls_enabled(False)
        if err:
            self.shelly_status.setText(f"Illumination: error ({err})")
        else:
            self.shelly_status.setText("Illumination: unavailable")

    def _apply_selected_preset(self) -> None:
        name = self.shelly_preset_combo.currentText()
        if name:
            self._apply_preset(name)

    def _apply_preset(self, name: str) -> None:
        state = self._shelly_presets.get(name)
        if state is None:
            self.shelly_status.setText(f"Illumination: preset '{name}' missing")
            return

        def do_apply():
            if not self.dimmer:
                return None
            self.dimmer.set_on(state.on)
            return self.dimmer.set_brightness(state.brightness)

        log(f"Applying Shelly preset '{name}' -> {state.brightness}% {'on' if state.on else 'off'}")
        self._run_dimmer_action(do_apply)

    def _save_selected_preset(self) -> None:
        name = self.shelly_preset_combo.currentText() or "Preset"
        state = ShellyState(on=self.shelly_toggle.isChecked(), brightness=int(self.shelly_brightness_spin.value()))
        self._shelly_presets[name] = state
        self._refresh_preset_combo(select=name)
        self._save_shelly_presets()
        log(f"Saved Shelly preset '{name}' -> {state.brightness}% {'on' if state.on else 'off'}")

    def _on_step_preset_changed(self, step: str, preset: str) -> None:
        self._step_presets[step] = preset
        self._save_step_presets()
        self._refresh_step_presets_widgets()
        log(f"Wizard step '{step}' now uses preset '{preset}'")

    def _apply_step_preset(self, step: str) -> None:
        preset = self._step_presets.get(step)
        if preset:
            self._apply_preset(preset)

    def _connect_selected(self) -> None:
        desc = self._current_descriptor()
        if not desc:
            return
        try:
            dev = self.spectrometer_manager.connect(desc)
            wavelengths = dev.get_wavelengths()
            self.session.set_wavelengths(wavelengths)
            self._default_device = self._device_key(desc)
            self.profiles.set("spectroscopy.default_device", self._default_device)
            self.profiles.save()
        except Exception as exc:
            self.status_message.setText(f"Connect failed: {exc}")
            LOG.warning("Spectrometer connect failed: %s", exc)
        self._update_status_from_selection()
        self._update_session_context()

    def _disconnect(self) -> None:
        self._stop_continuous()
        self._new_capture_token()
        desc = self._current_descriptor()
        self.spectrometer_manager.disconnect(desc)
        self._update_status_from_selection()
        self._update_session_context()

    # ------------------------------------------------------------------
    def _record_capture(
        self,
        kind: str,
        data: np.ndarray,
        metadata: Optional[Dict[str, object]] = None,
        *,
        raw_data: Optional[np.ndarray] = None,
    ) -> None:
        if self.session.wavelengths is None or data is None:
            return
        ts = time.time()
        metadata = dict(metadata or {})
        metadata.setdefault("device_id", self._current_device_id())
        metadata.setdefault("mode", self.current_mode)
        metadata.setdefault("timestamp", ts)
        label = f"{datetime.fromtimestamp(ts).strftime('%H:%M:%S')} | {kind.title()}"
        entry = CapturedSpectrum(
            key=f"capture-{int(ts * 1000)}-{len(self._recent_captures)}",
            label=label,
            timestamp=ts,
            data=np.asarray(data, dtype=float),
            mode=self.current_mode,
            metadata=metadata,
            kind=kind,
            raw_data=None if raw_data is None else np.asarray(raw_data, dtype=float),
        )
        self._recent_captures.insert(0, entry)
        while len(self._recent_captures) > 25:
            dropped = self._recent_captures.pop()
            self._remove_trace(dropped.key)
        self._refresh_recent_list()

    def _refresh_recent_list(self) -> None:
        checked: Dict[str, bool] = {}
        for i in range(self.recent_list.count()):
            item = self.recent_list.item(i)
            checked[item.data(QtCore.Qt.UserRole)] = item.checkState() == QtCore.Qt.Checked
        self.recent_list.blockSignals(True)
        self.recent_list.clear()
        for entry in self._recent_captures:
            item = QtWidgets.QListWidgetItem(entry.label)
            item.setData(QtCore.Qt.UserRole, entry.key)
            item.setToolTip(
                f"Mode: {entry.mode}\nIntegration: {entry.metadata.get('integration_ms', '—')} ms\n"
                f"Averages: {entry.metadata.get('averages', '—')}\nSmoothing: {entry.metadata.get('smoothing', '—')}"
            )
            item.setCheckState(QtCore.Qt.Checked if checked.get(entry.key) else QtCore.Qt.Unchecked)
            self.recent_list.addItem(item)
        if self.recent_list.count() and not self.recent_list.selectedItems():
            self.recent_list.setCurrentRow(0)
        self.recent_list.blockSignals(False)
        self._on_recent_selected()
        self._refresh_validity_state()

    def _on_recent_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        key = item.data(QtCore.Qt.UserRole)
        entry = next((c for c in self._recent_captures if c.key == key), None)
        if not entry:
            return
        if item.checkState() == QtCore.Qt.Checked:
            color = self._recent_colors[self._recent_captures.index(entry) % len(self._recent_colors)]
            self._plot_spectrum(entry.key, entry.data, color=color)
        else:
            self._remove_trace(entry.key)

    def _on_recent_selected(self) -> None:
        items = self.recent_list.selectedItems()
        if not items:
            self.recent_meta.setText("No captures yet")
            return
        key = items[0].data(QtCore.Qt.UserRole)
        entry = next((c for c in self._recent_captures if c.key == key), None)
        if not entry:
            self.recent_meta.setText("No captures yet")
            return
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        meta_parts = [
            f"Timestamp: {ts}",
            f"Mode: {entry.mode}",
        ]
        if entry.metadata.get("device_id"):
            meta_parts.append(f"Device: {entry.metadata['device_id']}")
        for label, key_name, suffix in (
            ("Integration", "integration_ms", "ms"),
            ("Averages", "averages", ""),
            ("Smoothing", "smoothing", ""),
        ):
            if key_name in entry.metadata:
                meta_parts.append(f"{label}: {entry.metadata[key_name]}{suffix}")
        if entry.metadata.get("subtract_dark"):
            meta_parts.append("Dark subtraction applied")
        self.recent_meta.setText(" | ".join(meta_parts))

    def _current_capture(self) -> Optional[CapturedSpectrum]:
        items = self.recent_list.selectedItems()
        if not items:
            return None
        key = items[0].data(QtCore.Qt.UserRole)
        return next((c for c in self._recent_captures if c.key == key), None)

    def _apply_capture_as(self, target: str) -> None:
        capture = self._current_capture()
        if not capture:
            return
        acquisition = AcquisitionMetadata(
            device_id=str(capture.metadata.get("device_id", "")),
            integration_ms=float(capture.metadata.get("integration_ms", 0.0)),
            averages=int(capture.metadata.get("averages", 0)),
            mode=str(capture.metadata.get("mode", self.current_mode)),
            timestamp=float(capture.metadata.get("timestamp", capture.timestamp)),
        )
        try:
            if target == "reference":
                self.session.set_reference(capture.data, acquisition=acquisition)
                self.status_message.setText("Applied capture as reference")
            else:
                self.session.set_dark(capture.data, acquisition=acquisition)
                self.status_message.setText("Applied capture as dark")
        except Exception as exc:
            self.status_message.setText(f"Failed to apply capture: {exc}")

    def _save_selected_capture(self) -> None:
        capture = self._current_capture()
        if not capture:
            self.status_message.setText("No capture selected to save")
            return
        if not self._data_dir:
            self.status_message.setText("Set a data folder first")
            return
        if self.session.wavelengths is None:
            self.status_message.setText("No wavelength axis available for export")
            return
        directory = ensure_data_directory(self._data_dir)
        safe_mode = capture.mode.replace(" ", "_").lower()
        default_name = f"{safe_mode}_{int(capture.timestamp)}"
        dialog = SaveCaptureDialog(self, directory, default_name, capture.raw_data is not None)
        if dialog.exec() != QtWidgets.QDialog.Accepted or not dialog.options:
            return
        options = dialog.options
        directory = os.path.dirname(options.path)
        self._data_dir = directory
        self._persist_data_dir()
        metadata = dict(capture.metadata)
        metadata.update(
            {
                "mode": capture.mode,
                "device_id": capture.metadata.get("device_id", self._current_device_id()),
                "integration_ms": capture.metadata.get("integration_ms", self.integration_spin.value()),
                "averages": capture.metadata.get("averages", self.averages_spin.value()),
                "dark_applied": self.session.dark_valid,
                "reference_applied": self.session.reference_valid,
                "timestamp": capture.timestamp,
            }
        )
        save_kwargs = {
            "raw_counts": capture.raw_data,
            "include_processed": options.include_processed,
            "include_raw": options.include_raw,
        }
        try:
            if options.format == "csv":
                save_spectrum_csv(options.path, self.session.wavelengths, capture.data, metadata, **save_kwargs)
            elif options.format == "h5":
                save_spectrum_hdf5(options.path, self.session.wavelengths, capture.data, metadata, **save_kwargs)
            else:
                save_spectrum_jcamp(options.path, self.session.wavelengths, capture.data, metadata, **save_kwargs)
            self.status_message.setText(f"Saved capture to {options.path}")
        except Exception as exc:
            self.status_message.setText(f"Save failed: {exc}")

    def _clear_recent(self) -> None:
        for entry in list(self._recent_captures):
            self._remove_trace(entry.key)
        self._recent_captures.clear()
        self.recent_list.clear()
        self.recent_meta.setText("No captures yet")

    def _new_capture_token(self) -> object:
        self._capture_token = object()
        return self._capture_token

    def _trigger_continuous_capture(self) -> None:
        if not self._continuous or self._capture_in_flight:
            return
        self._schedule_capture("measurement")

    def _schedule_capture(self, kind: str) -> None:
        if self._capture_in_flight:
            return
        dev, lock, desc = self._active_device_with_lock()
        if dev is None or lock is None:
            self.status_message.setText("No spectrometer connected")
            self._stop_continuous()
            if self._time_series_active:
                self._stop_time_series()
            return
        self._update_session_context()
        job = CaptureJob(
            token=self._capture_token,
            device=dev,
            device_id=self._device_key(desc) if desc else "",
            lock=lock,
            integration_ms=float(self.integration_spin.value()),
            averages=int(self.averages_spin.value()),
            smoothing=int(self.smoothing_spin.value()),
            subtract_dark=bool(self.dark_chk.isChecked()) if kind != "dark" else False,
            dark_spectrum=self.session.dark_spectrum,
            kind=kind,
            timestamp=time.time(),
        )
        self._capture_in_flight = True
        QtCore.QMetaObject.invokeMethod(
            self._capture_worker,
            "perform_capture",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(object, job),
        )

    @QtCore.Slot(object)
    def _on_capture_started(self, token: object) -> None:
        if token != self._capture_token:
            return
        self.status_message.setText("Capturing…")

    @QtCore.Slot(object, np.ndarray, np.ndarray, float, float)
    def _on_capture_ready(
        self, job: CaptureJob, processed: np.ndarray, raw: np.ndarray, duration: float, peak: float
    ) -> None:
        if job.token != self._capture_token:
            return
        self._capture_in_flight = False
        now = time.time()
        if self._continuous and (now - self._last_capture_ts) * 1000 < self._rate_limit_ms:
            QtCore.QTimer.singleShot(self.capture_timer.interval(), self._trigger_continuous_capture)
            return
        self._last_capture_ts = now
        acquisition = AcquisitionMetadata(
            device_id=job.device_id,
            integration_ms=float(job.integration_ms),
            averages=int(job.averages),
            mode=self.current_mode,
            timestamp=job.timestamp,
        )
        if job.kind == "dark":
            self.session.set_dark(raw, acquisition=acquisition)
            self._plot_spectrum("dark", raw, color=QtGui.QColor("gray"))
            meta = {
                "integration_ms": float(job.integration_ms),
                "averages": int(job.averages),
                "smoothing": int(job.smoothing),
            }
            self._record_capture("dark", raw, meta, raw_data=raw)
            self.status_message.setText("Dark captured")
        else:
            self.session.set_raw(raw)
            self._plot_spectrum("live", processed, color=QtGui.QColor("deepskyblue"))
            metadata = {
                "integration_ms": float(job.integration_ms),
                "averages": int(job.averages),
                "smoothing": int(job.smoothing),
                "subtract_dark": bool(job.subtract_dark),
            }
            self._record_capture("measurement", processed, metadata, raw_data=raw)
            self._append_time_series(acquisition, processed, metadata)
            self.status_message.setText("Captured spectrum")
        if duration > 0:
            fps = 1.0 / duration
            self.fps_label.setText(f"{fps:.1f} Hz")
            if fps < 1:
                self.status_message.setText(f"Capture slow ({fps:.2f} Hz)")
        if not np.isnan(peak):
            self.saturation_label.setText(f"Peak: {peak:.0f}")
            if peak >= 60000:
                self.status_message.setText("Warning: signal near saturation")
        if self._continuous:
            QtCore.QTimer.singleShot(self.capture_timer.interval(), self._trigger_continuous_capture)

    @QtCore.Slot(object, str)
    def _on_capture_failed(self, job: CaptureJob, message: str) -> None:
        if job.token != self._capture_token:
            return
        self._capture_in_flight = False
        self.status_message.setText(f"Capture failed: {message}")
        if self._time_series_active:
            self._stop_time_series()
        self._stop_continuous()

    def _capture_dark(self) -> None:
        self._continuous = False
        self.capture_timer.stop()
        self._new_capture_token()
        self._schedule_capture("dark")

    def _capture_once(self) -> None:
        self._continuous = False
        self.capture_timer.stop()
        self._new_capture_token()
        self._schedule_capture("measurement")

    def _start_continuous(self) -> None:
        self._continuous = True
        self._new_capture_token()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.capture_timer.start()
        self.status_message.setText("Continuous capture running")
        self._trigger_continuous_capture()

    def _stop_continuous(self) -> None:
        self._continuous = False
        self.capture_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._capture_in_flight = False
        self._new_capture_token()
        self.status_message.setText("Stopped")

    def _append_time_series(
        self, acquisition: AcquisitionMetadata, spectrum: np.ndarray, metadata: Optional[Dict[str, object]] = None
    ) -> None:
        if not self._time_series_active:
            return
        if self._time_series_start <= 0:
            self._time_series_start = acquisition.timestamp
        rel_ts = float(acquisition.timestamp - self._time_series_start)
        entry_meta = dict(metadata or {})
        entry_meta.setdefault("device_id", acquisition.device_id)
        entry_meta.setdefault("mode", acquisition.mode)
        entry_meta.setdefault("integration_ms", acquisition.integration_ms)
        entry_meta.setdefault("averages", acquisition.averages)
        entry_meta.setdefault("timestamp", acquisition.timestamp)
        entry_meta.setdefault("dark_applied", self.session.dark_valid)
        entry_meta.setdefault("reference_applied", self.session.reference_valid)
        self._time_series_records.append((rel_ts, np.asarray(spectrum, dtype=float), entry_meta))
        self._time_series_sample_count = len(self._time_series_records)
        self._update_time_series_views()
        self._check_time_series_limits(acquisition.timestamp)

    def _start_time_series(self) -> None:
        if self.session.wavelengths is None:
            self.status_message.setText("Set wavelengths before recording time series")
            return
        self._time_series_records.clear()
        self._clear_time_series_views()
        self._time_series_start = time.time()
        self._time_series_active = True
        self._time_series_sample_count = 0
        self._continuous = False
        self.capture_timer.stop()
        self._new_capture_token()
        self.status_message.setText("Time series recording started")
        self._trigger_time_series_capture()
        self._refresh_validity_state()

    def _stop_time_series(self) -> None:
        if not self._time_series_active:
            return
        self._time_series_active = False
        self._time_series_timer.stop()
        self.status_message.setText("Time series recording stopped")
        self._persist_time_series()
        self._refresh_validity_state()

    def _persist_time_series(self) -> None:
        if not self._time_series_records:
            self.status_message.setText("No time series samples captured")
            return
        wavelengths = self.session.wavelengths
        if wavelengths is None:
            self.status_message.setText("No wavelength axis available for export")
            return
        directory = ensure_data_directory(self._data_dir)
        times = np.array([t for t, _, _ in self._time_series_records], dtype=float)
        spectra = np.vstack([np.asarray(s, dtype=float) for _, s, _ in self._time_series_records])
        start_ts = float(self._time_series_start)
        metadata: Dict[str, object] = {
            "mode": self.current_mode,
            "device_id": self._current_device_id(),
            "integration_ms": float(self.integration_spin.value()),
            "averages": int(self.averages_spin.value()),
            "dark_applied": self.session.dark_valid,
            "reference_applied": self.session.reference_valid,
            "start_timestamp": start_ts,
            "samples": len(self._time_series_records),
            "interval_s": float(self.time_series_interval.value()),
            "duration_limit_s": float(self.time_series_duration.value()),
            "max_samples": int(self.time_series_samples.value()),
            "elapsed_s": float(times.max()) if times.size else 0.0,
            "tracked_wavelengths": self._selected_wavelengths(wavelengths),
        }
        base = os.path.join(directory, f"time_series_{int(start_ts)}")
        try:
            save_time_series_npz(base + ".npz", wavelengths, spectra, times, metadata)
            save_time_series_hdf5(base + ".h5", wavelengths, spectra, times, metadata)
            if self.chk_plot_timeseries.isChecked():
                fig, ax = plt.subplots()
                ax.plot(times, spectra.mean(axis=1))
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Mean intensity (a.u.)")
                ax.set_title("Intensity vs time")
                fig.savefig(base + "_intensity.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            if self.chk_plot_spectrogram.isChecked():
                fig, ax = plt.subplots()
                t_min, t_max = float(times.min()), float(times.max())
                if t_max <= t_min:
                    t_max = t_min + 1e-6
                w_min, w_max = float(wavelengths.min()), float(wavelengths.max())
                if w_max <= w_min:
                    w_max = w_min + 1e-6
                im = ax.imshow(
                    spectra.T,
                    aspect="auto",
                    origin="lower",
                    extent=[t_min, t_max, w_min, w_max],
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Wavelength (nm)")
                ax.set_title("Spectrogram")
                fig.colorbar(im, ax=ax, label="Intensity (a.u.)")
                fig.savefig(base + "_spectrogram.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            self.status_message.setText(f"Saved time series to {base}.[npz|h5]")
        except Exception as exc:  # pragma: no cover - runtime safety
            self.status_message.setText(f"Failed to save time series: {exc}")

    def _schedule_next_time_series_tick(self) -> None:
        if not self._time_series_active:
            return
        interval_ms = int(max(50.0, float(self.time_series_interval.value()) * 1000.0))
        self._time_series_timer.start(interval_ms)

    def _trigger_time_series_capture(self) -> None:
        if not self._time_series_active:
            return
        if self._capture_in_flight:
            self._schedule_next_time_series_tick()
            return
        self._schedule_capture("measurement")

    def _selected_wavelengths(self, axis: np.ndarray) -> List[float]:
        entered = self.timeseries_wavelengths.text()
        values: List[float] = []
        for token in entered.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        if values:
            return values
        if axis.size >= 3:
            return [float(axis[0]), float(axis[len(axis) // 2]), float(axis[-1])]
        return [float(x) for x in axis[:1]]

    def _clear_time_series_views(self) -> None:
        for series in list(self._ts_traces.values()):
            self._ts_chart.removeSeries(series)
        self._ts_traces.clear()
        self.spectrogram_label.setPixmap(QtGui.QPixmap())
        self.spectrogram_label.setText("Spectrogram hidden")

    def _update_time_series_views(self) -> None:
        if not self._time_series_records:
            self._clear_time_series_views()
            if self._time_series_active:
                self.spectrogram_label.setText("Recording…")
            return
        wavelengths = self.session.wavelengths
        if wavelengths is None:
            return
        times = np.array([t for t, _, _ in self._time_series_records], dtype=float)
        spectra = np.vstack([np.asarray(s, dtype=float) for _, s, _ in self._time_series_records])

        if self.chk_show_timeseries.isChecked():
            desired = self._selected_wavelengths(wavelengths)
            keep_keys = set()
            for wl in desired:
                key = f"{wl:.2f} nm"
                keep_keys.add(key)
                if key not in self._ts_traces:
                    series = QtCharts.QLineSeries()
                    series.setName(key)
                    self._ts_chart.addSeries(series)
                    axis_x = self._ts_chart.axisX()
                    axis_y = self._ts_chart.axisY()
                    if axis_x is None or axis_y is None:
                        self._ts_chart.createDefaultAxes()
                        axis_x = self._ts_chart.axisX()
                        axis_y = self._ts_chart.axisY()
                    if axis_x:
                        series.attachAxis(axis_x)
                    if axis_y:
                        series.attachAxis(axis_y)
                    self._ts_traces[key] = series
                intensities = [float(np.interp(wl, wavelengths, row)) for row in spectra]
                points = [QtCore.QPointF(t, i) for t, i in zip(times, intensities)]
                self._ts_traces[key].replace(points)
            for key in list(self._ts_traces.keys()):
                if key not in keep_keys:
                    series = self._ts_traces.pop(key)
                    self._ts_chart.removeSeries(series)
            axis_x = self._ts_chart.axisX()
            axis_y = self._ts_chart.axisY()
            if axis_x:
                axis_x.setRange(float(times.min()), float(times.max()))
            if axis_y:
                ymin = float(np.nanmin(spectra))
                ymax = float(np.nanmax(spectra))
                if ymin == ymax:
                    ymax = ymin + 1.0
                axis_y.setRange(ymin, ymax)
        else:
            self._clear_time_series_views()

        if self.chk_show_spectrogram.isChecked():
            pixmap = self._render_spectrogram(times, wavelengths, spectra)
            if pixmap is not None:
                self.spectrogram_label.setPixmap(pixmap)
                self.spectrogram_label.setText("")
            else:
                self.spectrogram_label.setText("Unable to render spectrogram")
        else:
            self.spectrogram_label.setPixmap(QtGui.QPixmap())
            self.spectrogram_label.setText("Spectrogram hidden")

    def _render_spectrogram(
        self, times: np.ndarray, wavelengths: np.ndarray, spectra: np.ndarray
    ) -> Optional[QtGui.QPixmap]:
        if times.size == 0 or wavelengths.size == 0 or spectra.size == 0:
            return None
        fig, ax = plt.subplots()
        try:
            t_min, t_max = float(times.min()), float(times.max())
            if t_max <= t_min:
                t_max = t_min + 1e-6
            w_min, w_max = float(wavelengths.min()), float(wavelengths.max())
            if w_max <= w_min:
                w_max = w_min + 1e-6
            im = ax.imshow(
                spectra.T,
                aspect="auto",
                origin="lower",
                extent=[t_min, t_max, w_min, w_max],
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Wavelength (nm)")
            ax.set_title("Spectrogram")
            fig.colorbar(im, ax=ax, label="Intensity (a.u.)")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            buf.seek(0)
            image = QtGui.QImage.fromData(buf.getvalue(), "PNG")
            return QtGui.QPixmap.fromImage(image)
        except Exception:
            return None
        finally:
            plt.close(fig)

    def _check_time_series_limits(self, timestamp: float) -> None:
        if not self._time_series_active:
            return
        duration_limit = float(self.time_series_duration.value())
        max_samples = int(self.time_series_samples.value())
        elapsed = timestamp - self._time_series_start
        if (duration_limit > 0 and elapsed >= duration_limit) or (
            max_samples > 0 and self._time_series_sample_count >= max_samples
        ):
            self._stop_time_series()
        else:
            self._schedule_next_time_series_tick()

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
            preset_names=sorted(self._shelly_presets.keys()),
            apply_preset=self._apply_step_preset,
            step_presets=dict(self._step_presets),
            on_step_preset_changed=self._on_step_preset_changed,
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
            self.mode_params = dict(self.session.mode_params)
            self.profiles.set("spectroscopy.last_mode", mode)
            self.profiles.set("spectroscopy.last_params", dict(self.mode_params))
            self.profiles.save()
            self.status_message.setText(f"{mode} wizard completed")
            self._update_session_context()
        else:
            self.status_message.setText(f"{mode} wizard cancelled")

    def _wizard_capture(self, kind: str, integration: float, averages: int) -> Optional[np.ndarray]:
        dev, lock, _desc = self._active_device_with_lock()
        if dev is None or lock is None:
            self.status_message.setText("No spectrometer connected")
            return None
        self._update_session_context()
        self._apply_step_preset(kind)
        try:
            with self._acquisition_context(lock):
                dev.set_integration_time_ms(integration)
                dev.set_averages(averages)
                spectrum = dev.capture()
            ts = time.time()
            self._record_capture(
                kind,
                spectrum,
                {
                    "integration_ms": float(integration),
                    "averages": int(averages),
                    "smoothing": int(self.smoothing_spin.value()),
                    "device_id": self._current_device_id(),
                    "mode": self.current_mode,
                    "timestamp": ts,
                },
                raw_data=spectrum,
            )
            return spectrum
        except Exception as exc:
            self.status_message.setText(f"Capture failed: {exc}")
            return None

