from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCharts, QtCore, QtGui, QtWidgets

from ..spectroscopy import processing
from ..spectroscopy.session import SpectroscopySession
from .spectrum_charts import SpectrumChartView, create_spectrum_chart

CaptureCallable = Callable[[str, float, int], Optional[np.ndarray]]


class ModeSelectorDialog(QtWidgets.QDialog):
    def __init__(self, modes: List[str], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Select spectroscopy mode")
        self.selected_mode: Optional[str] = None
        layout = QtWidgets.QGridLayout(self)
        for idx, mode in enumerate(modes):
            button = QtWidgets.QCommandLinkButton(mode)
            button.clicked.connect(lambda _=False, m=mode: self._select(m))
            row, col = divmod(idx, 2)
            layout.addWidget(button, row, col)
        self.setLayout(layout)

    def _select(self, mode: str) -> None:
        self.selected_mode = mode
        self.accept()


class BaseWizardPage(QtWidgets.QWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.wizard_ref = wizard

    @property
    def session(self) -> SpectroscopySession:
        return self.wizard_ref.session

    def mark_dirty(self) -> None:
        self.wizard_ref.invalidate_results()
        self.completeChanged.emit()


def _series_from_data(label: str, x: np.ndarray, y: np.ndarray, color: QtGui.QColor) -> QtCharts.QLineSeries:
    series = QtCharts.QLineSeries()
    series.setName(label)
    series.setColor(color)
    points = [QtCore.QPointF(float(xv), float(yv)) for xv, yv in zip(x, y)]
    series.replace(points)
    return series


def _attach_series(
    chart: QtCharts.QChart,
    x_axis: QtCharts.QAbstractAxis,
    y_axis: QtCharts.QAbstractAxis,
    label: str,
    x: np.ndarray,
    y: np.ndarray,
    color: QtGui.QColor,
) -> QtCharts.QLineSeries:
    series = _series_from_data(label, x, y, color)
    chart.addSeries(series)
    series.attachAxis(x_axis)
    series.attachAxis(y_axis)
    return series


class AcquisitionSetupPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Acquisition and wavelength setup")
        layout = QtWidgets.QVBoxLayout(self)

        params_row = QtWidgets.QHBoxLayout()

        wl_group = QtWidgets.QGroupBox("Wavelength range")
        wl_form = QtWidgets.QFormLayout(wl_group)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setSuffix(" nm")
            spin.setRange(0, 2000)
            spin.valueChanged.connect(self._on_param_changed)
        wl_form.addRow("Min wavelength", self.min_spin)
        wl_form.addRow("Max wavelength", self.max_spin)

        acq_group = QtWidgets.QGroupBox("Acquisition")
        acq_form = QtWidgets.QFormLayout(acq_group)
        self.integration = QtWidgets.QDoubleSpinBox()
        self.integration.setRange(1.0, 10000.0)
        self.integration.setSuffix(" ms")
        self.integration.setValue(wizard.initial_acquisition.get("integration", 10.0))
        self.averages = QtWidgets.QSpinBox()
        self.averages.setRange(1, 100)
        self.averages.setValue(wizard.initial_acquisition.get("averages", 1))
        self.smoothing = QtWidgets.QSpinBox()
        self.smoothing.setRange(1, 101)
        self.smoothing.setValue(wizard.initial_acquisition.get("smoothing", 5))
        for widget in (self.integration, self.averages, self.smoothing):
            widget.valueChanged.connect(self._on_param_changed)
        acq_form.addRow("Integration time", self.integration)
        acq_form.addRow("Averages", self.averages)
        acq_form.addRow("Smoothing (boxcar)", self.smoothing)

        params_row.addWidget(wl_group, 1)
        params_row.addWidget(acq_group, 1)
        layout.addLayout(params_row)

        self.chart = QtCharts.QChart()
        self.chart.setAnimationOptions(QtCharts.QChart.NoAnimation)
        self.chart.legend().setVisible(True)
        self.x_axis = QtCharts.QValueAxis()
        self.y_axis = QtCharts.QValueAxis()
        self.x_axis.setTitleText("Wavelength (nm)")
        self.y_axis.setTitleText("Intensity")
        self.chart.addAxis(self.x_axis, QtCore.Qt.AlignBottom)
        self.chart.addAxis(self.y_axis, QtCore.Qt.AlignLeft)
        self.chart_view = QtCharts.QChartView(self.chart)
        self.chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        layout.addWidget(self.chart_view, 2)
        self.status_label = QtWidgets.QLabel("No spectra captured yet.")
        layout.addWidget(self.status_label)

        wl_range = wizard.mode_params.get("wavelength_range")
        if wl_range and len(wl_range) == 2:
            self.min_spin.setValue(float(wl_range[0]))
            self.max_spin.setValue(float(wl_range[1]))
        elif wizard.session.wavelengths is not None:
            wl = wizard.session.wavelengths
            self.min_spin.setValue(float(wl.min()))
            self.max_spin.setValue(float(wl.max()))

        wizard.session.spectra_changed.connect(self._refresh_chart)
        wizard.session.wavelengths_changed.connect(lambda _: self._refresh_chart())
        self._sync_params()

    def initializePage(self) -> None:  # type: ignore[override]
        self._refresh_chart()

    def _on_param_changed(self) -> None:
        self._sync_params()
        self.wizard_ref.invalidate_captures()
        self.mark_dirty()

    def _sync_params(self) -> None:
        self.wizard_ref.mode_params.update(
            integration=float(self.integration.value()),
            averages=int(self.averages.value()),
            smoothing=int(self.smoothing.value()),
        )
        valid_range = self.min_spin.value() < self.max_spin.value()
        if valid_range:
            self.setSubTitle("")
            self.wizard_ref.mode_params["wavelength_range"] = (
                self.min_spin.value(),
                self.max_spin.value(),
            )
        else:
            self.setSubTitle("Min must be < Max")
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        if valid_range:
            self.x_axis.setRange(self.min_spin.value(), self.max_spin.value())
        self._refresh_chart()

    def _refresh_chart(self) -> None:
        self.chart.removeAllSeries()
        wl = self.session.wavelengths
        if wl is None:
            self.status_label.setText("Set wavelength calibration to preview spectra.")
            return
        spectra_added = False
        ymin, ymax = float("inf"), float("-inf")
        for label, data, color in (
            ("Dark", self.session.dark_spectrum, QtGui.QColor("gray")),
            ("Reference", self.session.reference_spectrum, QtGui.QColor("green")),
            ("Raw", self.session.raw_spectrum, QtGui.QColor("deepskyblue")),
        ):
            if data is None:
                continue
            display = (
                processing.smooth_boxcar(np.asarray(data, dtype=float), window=int(self.smoothing.value()))
                if label == "Raw"
                else np.asarray(data, dtype=float)
            )
            series = _attach_series(self.chart, self.x_axis, self.y_axis, label, wl, display, color)
            ymin = min(ymin, float(np.nanmin(display)))
            ymax = max(ymax, float(np.nanmax(display)))
            spectra_added = True
            series.setUseOpenGL(True)
        if not spectra_added:
            self.status_label.setText("Capture a spectrum to preview settings.")
            return
        self.status_label.setText("")
        if ymin == float("inf") or ymax == float("-inf"):
            ymin, ymax = 0.0, 1.0
        if ymin == ymax:
            ymax += 1.0
        self.y_axis.setRange(ymin, ymax)
        if self.min_spin.value() < self.max_spin.value():
            self.x_axis.setRange(self.min_spin.value(), self.max_spin.value())
        else:
            self.x_axis.setRange(float(np.nanmin(wl)), float(np.nanmax(wl)))

    def isComplete(self) -> bool:  # type: ignore[override]
        return self.min_spin.value() < self.max_spin.value()


class CapturePage(BaseWizardPage):
    CAPTURE_TYPE = "capture"

    def __init__(self, wizard: 'SpectroscopyModeWizard', title: str, capture_key: str):
        super().__init__(wizard)
        self.capture_key = capture_key
        self.setTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        self.info = QtWidgets.QLabel("Ready")
        self.btn_capture = QtWidgets.QPushButton("Capture with current settings")
        self.btn_capture.clicked.connect(self._capture)
        layout.addWidget(self.info)
        if wizard.preset_names:
            preset_row = QtWidgets.QHBoxLayout()
            preset_row.addWidget(QtWidgets.QLabel("Illumination preset:"))
            self.preset_combo = QtWidgets.QComboBox()
            self.preset_combo.addItems(wizard.preset_names)
            current = wizard.get_step_preset(self.capture_key)
            if current:
                self.preset_combo.setCurrentText(current)
            self.preset_combo.currentTextChanged.connect(self._preset_changed)
            preset_row.addWidget(self.preset_combo, 1)
            layout.addLayout(preset_row)
        else:
            self.preset_combo = None
        layout.addWidget(self.btn_capture)
        charts_column = QtWidgets.QVBoxLayout()
        live_group = QtWidgets.QGroupBox("Live spectrum")
        live_layout = QtWidgets.QVBoxLayout(live_group)
        (
            self.live_chart,
            self.live_chart_view,
            self.live_x_axis,
            self.live_y_axis,
        ) = self._build_chart()
        self.live_status = QtWidgets.QLabel("Waiting for live data…")
        live_layout.addWidget(self.live_chart_view)
        live_layout.addWidget(self.live_status)
        stored_group = QtWidgets.QGroupBox("Captured spectrum")
        stored_layout = QtWidgets.QVBoxLayout(stored_group)
        (
            self.stored_chart,
            self.stored_chart_view,
            self.stored_x_axis,
            self.stored_y_axis,
        ) = self._build_chart()
        self.stored_status = QtWidgets.QLabel("No capture yet.")
        stored_layout.addWidget(self.stored_chart_view)
        stored_layout.addWidget(self.stored_status)
        charts_column.addWidget(live_group, 1)
        charts_column.addWidget(stored_group, 1)
        layout.addLayout(charts_column)
        layout.addStretch(1)

        self.session.spectra_changed.connect(self._on_session_spectrum_changed)
        self.session.wavelengths_changed.connect(lambda _=None: self._sync_axes())
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setInterval(750)
        self._live_timer.timeout.connect(self._refresh_live_chart)
        self._live_timer.start()
        self._live_capture_in_flight = False
        self._live_spectrum: Optional[np.ndarray] = None

    def _capture(self) -> None:
        self.wizard_ref.apply_preset_for_step(self.capture_key)
        params = self.wizard_ref.mode_params
        spectrum = self.wizard_ref.capture_callback(
            self.capture_key,
            float(params.get("integration", 10.0)),
            int(params.get("averages", 1)),
        )
        if spectrum is None:
            self.info.setText("Capture failed or unavailable.")
            return
        try:
            if self.capture_key == "dark":
                self.session.set_dark(spectrum)
                self.wizard_ref.state["dark_captured"] = True
            elif self.capture_key == "reference":
                self.session.set_reference(spectrum)
                self.wizard_ref.state["reference_captured"] = True
            else:
                self.session.set_raw(spectrum)
                self.wizard_ref.state["raw_captured"] = True
            self.info.setText("Capture stored in session.")
            self._refresh_stored_chart()
        except Exception as exc:
            self.info.setText(f"Capture failed: {exc}")
        self.wizard_ref._update_finish_state()
        self.completeChanged.emit()

    def _preset_changed(self, preset: str) -> None:
        self.wizard_ref.set_step_preset(self.capture_key, preset)

    def isComplete(self) -> bool:  # type: ignore[override]
        return bool(self.wizard_ref.state.get(f"{self.capture_key}_captured", False))

    def initializePage(self) -> None:  # type: ignore[override]
        self._sync_axes()
        self._refresh_live_chart()
        self._refresh_stored_chart()

    def _on_session_spectrum_changed(self) -> None:
        self._refresh_live_chart()
        self._refresh_stored_chart()

    def _build_chart(self) -> tuple[QtCharts.QChart, SpectrumChartView, QtCharts.QValueAxis, QtCharts.QValueAxis]:
        return create_spectrum_chart(
            x_title=self._x_axis_title(),
            y_title=self._y_axis_title(),
            legend_visible=True,
        )

    def _x_axis_title(self) -> str:
        return "Raman shift (cm⁻¹)" if self.wizard_ref.mode == "Raman" else "Wavelength (nm)"

    def _y_axis_title(self) -> str:
        return "Intensity (counts)"

    def _refresh_live_chart(self) -> None:
        if self._live_capture_in_flight:
            return
        params = self.wizard_ref.mode_params
        integration = float(params.get("integration", 10.0))
        averages = int(params.get("averages", 1))
        self._live_capture_in_flight = True
        try:
            data = self.wizard_ref.capture_callback("preview", integration, averages)
        finally:
            self._live_capture_in_flight = False
        if data is None:
            data = self._live_spectrum
        else:
            self._live_spectrum = data
        label = "Live"
        color = QtGui.QColor("deepskyblue")
        self._plot_chart(self.live_chart, self.live_x_axis, self.live_y_axis, self.live_status, label, data, color)

    def _refresh_stored_chart(self) -> None:
        if self.capture_key == "dark":
            data = self.session.dark_spectrum
            label = "Dark capture"
            color = QtGui.QColor("gray")
        elif self.capture_key == "reference":
            data = self.session.reference_spectrum
            label = "Reference capture"
            color = QtGui.QColor("green")
        else:
            data = self.session.raw_spectrum
            label = "Raw capture"
            color = QtGui.QColor("#1f77b4")
        self._plot_chart(
            self.stored_chart,
            self.stored_x_axis,
            self.stored_y_axis,
            self.stored_status,
            label,
            data,
            color,
            apply_smoothing=self.capture_key == "raw",
        )

    def _plot_chart(
        self,
        chart: QtCharts.QChart,
        x_axis: QtCharts.QValueAxis,
        y_axis: QtCharts.QValueAxis,
        status_label: QtWidgets.QLabel,
        label: str,
        data: Optional[np.ndarray],
        color: QtGui.QColor,
        *,
        apply_smoothing: bool = False,
    ) -> None:
        chart.removeAllSeries()
        wl = self.session.wavelengths
        if wl is None:
            status_label.setText("Set wavelength calibration to preview spectra.")
            return
        if data is None:
            status_label.setText("Waiting for data…")
            return
        array = np.asarray(data, dtype=float)
        if apply_smoothing:
            array = processing.smooth_boxcar(array, window=int(self.wizard_ref.mode_params.get("smoothing", 1)))
        series = _attach_series(chart, x_axis, y_axis, label, wl, array, color)
        series.setUseOpenGL(True)
        ymin = float(np.nanmin(array)) if array.size else 0.0
        ymax = float(np.nanmax(array)) if array.size else 1.0
        if not np.isfinite(ymin) or not np.isfinite(ymax):
            ymin, ymax = 0.0, 1.0
        if ymin == ymax:
            ymax += 1.0
        x_range = self._wavelength_range()
        if x_range is None:
            x_range = (float(np.nanmin(wl)), float(np.nanmax(wl)))
        x_axis.setRange(*x_range)
        y_axis.setRange(ymin, ymax)
        x_axis.setTitleText(self._x_axis_title())
        y_axis.setTitleText(self._y_axis_title())
        status_label.setText("")

    def _wavelength_range(self) -> Optional[Tuple[float, float]]:
        range_vals = self.wizard_ref.mode_params.get("wavelength_range")
        if isinstance(range_vals, (list, tuple)) and len(range_vals) == 2:
            try:
                low, high = float(range_vals[0]), float(range_vals[1])
            except (TypeError, ValueError):
                return None
            if low < high:
                return low, high
        return None

    def _sync_axes(self) -> None:
        self.live_x_axis.setTitleText(self._x_axis_title())
        self.stored_x_axis.setTitleText(self._x_axis_title())
        self.live_y_axis.setTitleText(self._y_axis_title())
        self.stored_y_axis.setTitleText(self._y_axis_title())


class BeerLambertPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Beer–Lambert calibration")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.path_length = QtWidgets.QDoubleSpinBox()
        self.path_length.setRange(0.01, 100.0)
        self.path_length.setValue(float(wizard.mode_params.get("path_length_cm", 1.0)))
        self.path_length.setSuffix(" cm")
        self.concentration = QtWidgets.QDoubleSpinBox()
        self.concentration.setRange(0.0, 1e6)
        self.concentration.setDecimals(6)
        self.concentration.setValue(float(wizard.mode_params.get("concentration_m", 0.0)))
        self.concentration.setSuffix(" mol/L")
        self.fit_checkbox = QtWidgets.QCheckBox("Compute molar absorptivity from captured sample")
        self.fit_checkbox.setChecked(bool(wizard.mode_params.get("compute_ext_coeff", True)))
        self.band_start = QtWidgets.QDoubleSpinBox()
        self.band_end = QtWidgets.QDoubleSpinBox()
        for spin in (self.band_start, self.band_end):
            spin.setRange(0, 2000)
            spin.setSuffix(" nm")
        self.band_label = QtWidgets.QLineEdit()
        add_band_btn = QtWidgets.QPushButton("Add ROI/band")
        add_band_btn.clicked.connect(self._add_band)
        self.roi_combo = QtWidgets.QComboBox()
        self._refresh_rois()
        self.point_btn = QtWidgets.QPushButton("Add calibration point from current capture")
        self.point_btn.clicked.connect(self._capture_point)
        self.points_table = QtWidgets.QTableWidget(0, 3)
        self.points_table.setHorizontalHeaderLabels(["Concentration", "Absorbance", "ROI"])
        self.points_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.fit_label = QtWidgets.QLabel("No calibration points yet.")
        for widget in (self.path_length, self.concentration):
            widget.valueChanged.connect(self._changed)
        self.fit_checkbox.stateChanged.connect(self._changed)
        form.addRow("Path length", self.path_length)
        form.addRow("Concentration", self.concentration)
        form.addRow(self.fit_checkbox)
        form.addRow(QtWidgets.QLabel("Add measurement band for Beer–Lambert fit"))
        form.addRow("Start", self.band_start)
        form.addRow("End", self.band_end)
        form.addRow("Label", self.band_label)
        form.addRow(add_band_btn)
        form.addRow("Select ROI", self.roi_combo)
        layout.addLayout(form)
        layout.addWidget(self.point_btn)
        layout.addWidget(self.points_table)
        layout.addWidget(self.fit_label)
        layout.addStretch(1)
        self._restore_points()

    def _refresh_rois(self) -> None:
        self.roi_combo.clear()
        if not self.session.rois:
            self.roi_combo.addItem("(full spectrum)", userData=None)
        for idx, roi in enumerate(self.session.rois):
            label = roi.label or f"ROI {idx+1}"
            self.roi_combo.addItem(f"{label}: {roi.start_nm:.1f}-{roi.end_nm:.1f} nm", userData=roi)

    def _restore_points(self) -> None:
        points = self.wizard_ref.mode_params.get("beer_lambert_points", [])
        for conc, absorb, roi_label in points:
            self._append_point_row(conc, absorb, roi_label)
        self._update_fit_label()

    def _append_point_row(self, conc: float, absorb: float, roi_label: str) -> None:
        row = self.points_table.rowCount()
        self.points_table.insertRow(row)
        self.points_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{conc:.6g}"))
        self.points_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{absorb:.4f}"))
        self.points_table.setItem(row, 2, QtWidgets.QTableWidgetItem(roi_label))

    def _capture_point(self) -> None:
        measurement = self._measure_absorbance()
        if measurement is None:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Capture dark/reference/raw first to compute absorbance.")
            return
        conc = float(self.concentration.value())
        roi_label, absorb = measurement
        points = self.wizard_ref.mode_params.setdefault("beer_lambert_points", [])
        points.append((conc, absorb, roi_label))
        self._append_point_row(conc, absorb, roi_label)
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self._update_fit_label()
        self.mark_dirty()

    def _measure_absorbance(self) -> Optional[Tuple[str, float]]:
        wl = self.session.wavelengths
        raw = self.session.raw_spectrum
        ref = self.session.reference_spectrum
        dark = self.session.dark_spectrum
        if wl is None or raw is None or ref is None:
            return None
        smoothed = processing.smooth_boxcar(raw, window=int(self.wizard_ref.mode_params.get("smoothing", 1)))
        if dark is not None:
            smoothed = processing.subtract_dark(smoothed, dark)
        absorb = processing.compute_absorbance(smoothed, ref)
        roi = self.roi_combo.currentData()
        if roi is None:
            value = float(np.nanmax(absorb))
            roi_label = "Full spectrum"
        else:
            mask = (wl >= roi.start_nm) & (wl <= roi.end_nm)
            value = float(np.nanmean(absorb[mask])) if np.any(mask) else float("nan")
            roi_label = roi.label or "ROI"
        return roi_label, value

    def _add_band(self) -> None:
        roi = self.session.add_roi(self.band_start.value(), self.band_end.value(), self.band_label.text())
        self.wizard_ref.mode_params.setdefault("rois", []).append(roi.as_tuple())
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self._refresh_rois()
        self.mark_dirty()

    def _update_fit_label(self) -> None:
        points = self.wizard_ref.mode_params.get("beer_lambert_points", [])
        if len(points) < 2:
            self.fit_label.setText("Add at least two points to fit Beer–Lambert line.")
            return
        try:
            slope, intercept, r2 = processing.beer_lambert_fit([(p[0], p[1]) for p in points])
            self.fit_label.setText(f"Fit: absorbance = {slope:.4g} * c + {intercept:.3f}  (R²={r2:.3f})")
        except Exception as exc:
            self.fit_label.setText(f"Fit error: {exc}")

    def _changed(self) -> None:
        self.wizard_ref.mode_params.update(
            path_length_cm=float(self.path_length.value()),
            concentration_m=float(self.concentration.value()),
            compute_ext_coeff=bool(self.fit_checkbox.isChecked()),
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def isComplete(self) -> bool:  # type: ignore[override]
        return self.path_length.value() > 0 and self.concentration.value() >= 0


class TransReflectPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle(f"{wizard.mode} options")
        layout = QtWidgets.QVBoxLayout(self)
        self.percent_checkbox = QtWidgets.QCheckBox("Display result as percentage")
        self.percent_checkbox.setChecked(bool(wizard.mode_params.get("as_percent", True)))
        self.percent_checkbox.stateChanged.connect(self._changed)
        self.offset_checkbox = QtWidgets.QCheckBox("Clamp negative values to zero")
        self.offset_checkbox.setChecked(bool(wizard.mode_params.get("clamp_zero", True)))
        self.offset_checkbox.stateChanged.connect(self._changed)
        layout.addWidget(QtWidgets.QLabel("Configure how transmittance/reflectance is reported."))
        layout.addWidget(self.percent_checkbox)
        layout.addWidget(self.offset_checkbox)
        layout.addStretch(1)

    def _changed(self) -> None:
        self.wizard_ref.mode_params.update(
            as_percent=bool(self.percent_checkbox.isChecked()),
            clamp_zero=bool(self.offset_checkbox.isChecked()),
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


class IrradianceCalibrationPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Relative irradiance calibration")
        layout = QtWidgets.QFormLayout(self)
        self.apply_checkbox = QtWidgets.QCheckBox("Apply response correction")
        self.apply_checkbox.setChecked(bool(wizard.mode_params.get("apply_response", True)))
        self.apply_checkbox.stateChanged.connect(self._changed)
        self.cal_target = QtWidgets.QLineEdit()
        self.cal_target.setPlaceholderText("Calibration source (e.g., halogen lamp)")
        self.cal_target.setText(str(wizard.mode_params.get("calibration_target", "")))
        self.cal_target.textChanged.connect(self._changed)
        self.metric_centroid = QtWidgets.QCheckBox("Report centroid wavelength")
        color_metrics = wizard.mode_params.get("color_metrics", ["centroid", "peak"])
        self.metric_centroid.setChecked("centroid" in color_metrics)
        self.metric_cct = QtWidgets.QCheckBox("Report peak wavelength")
        self.metric_cct.setChecked("peak" in color_metrics)
        self.metric_cie = QtWidgets.QCheckBox("Compute CIE-like metrics")
        self.metric_cie.setChecked("cie" in color_metrics)
        for box in (self.metric_centroid, self.metric_cct, self.metric_cie):
            box.stateChanged.connect(self._changed)
        self.calibrate_btn = QtWidgets.QPushButton("Capture lamp and build response curve")
        self.calibrate_btn.clicked.connect(self._capture_calibration)
        self.status_label = QtWidgets.QLabel("Calibration pending.")
        layout.addRow(self.apply_checkbox)
        layout.addRow("Calibration target", self.cal_target)
        layout.addRow(QtWidgets.QLabel("Color metric options"))
        layout.addRow(self.metric_centroid)
        layout.addRow(self.metric_cct)
        layout.addRow(self.metric_cie)
        layout.addRow(self.calibrate_btn)
        layout.addRow(self.status_label)

    def initializePage(self) -> None:  # type: ignore[override]
        if self.session.calibration_valid:
            self.status_label.setText("Calibration already valid.")
        else:
            self.status_label.setText("Calibration pending.")

    def _changed(self) -> None:
        metrics = []
        if self.metric_centroid.isChecked():
            metrics.append("centroid")
        if self.metric_cct.isChecked():
            metrics.append("peak")
        if self.metric_cie.isChecked():
            metrics.append("cie")
        self.wizard_ref.mode_params.update(
            apply_response=bool(self.apply_checkbox.isChecked()),
            calibration_target=self.cal_target.text(),
            color_metrics=metrics,
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def _capture_calibration(self) -> None:
        params = self.wizard_ref.mode_params
        spectrum = self.wizard_ref.capture_callback(
            "calibration",
            float(params.get("integration", 10.0)),
            int(params.get("averages", 1)),
        )
        if spectrum is None:
            self.status_label.setText("Calibration capture failed.")
            return
        try:
            smoothed = processing.smooth_boxcar(spectrum, window=int(params.get("smoothing", 1)))
            normalized = smoothed / max(np.nanmax(smoothed), 1e-9)
            response_curve = np.where(np.isfinite(normalized) & (normalized > 0), 1.0 / normalized, 1.0)
            self.session.set_calibration(
                response_curve,
                calibration_target=self.cal_target.text(),
                color_metrics=params.get("color_metrics", []),
            )
            self.wizard_ref.state["calibration_valid"] = True
            self.status_label.setText("Calibration response stored.")
        except Exception as exc:
            self.status_label.setText(f"Calibration error: {exc}")
        self.wizard_ref._update_finish_state()
        self.completeChanged.emit()

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


class FluorescenceMetadataPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Fluorescence metadata")
        layout = QtWidgets.QFormLayout(self)
        self.excitation_spin = QtWidgets.QDoubleSpinBox()
        self.excitation_spin.setRange(200, 1200)
        self.excitation_spin.setSuffix(" nm")
        self.excitation_spin.setValue(float(wizard.mode_params.get("excitation_nm", 405.0)))
        self.emission_filter = QtWidgets.QLineEdit()
        self.emission_filter.setPlaceholderText("Emission filter description")
        self.emission_filter.setText(str(wizard.mode_params.get("emission_filter", "")))
        self.baseline_combo = QtWidgets.QComboBox()
        self.baseline_combo.addItems(["None", "Median", "Edge mean"])
        self.baseline_combo.setCurrentText(str(wizard.mode_params.get("baseline", "None")).title())
        self.excitation_spin.valueChanged.connect(self._changed)
        self.emission_filter.textChanged.connect(self._changed)
        self.baseline_combo.currentTextChanged.connect(self._changed)
        layout.addRow("Excitation wavelength", self.excitation_spin)
        layout.addRow("Emission filter", self.emission_filter)
        layout.addRow("Baseline removal", self.baseline_combo)

    def _changed(self) -> None:
        self.wizard_ref.mode_params.update(
            excitation_nm=float(self.excitation_spin.value()),
            emission_filter=self.emission_filter.text(),
            baseline=self.baseline_combo.currentText().lower(),
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


class RamanConfigPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Raman configuration")
        layout = QtWidgets.QFormLayout(self)
        self.excitation_spin = QtWidgets.QDoubleSpinBox()
        self.excitation_spin.setRange(100.0, 2000.0)
        self.excitation_spin.setSuffix(" nm")
        self.excitation_spin.setValue(float(wizard.mode_params.get("excitation_nm", 532.0)))
        self.shift_offset = QtWidgets.QDoubleSpinBox()
        self.shift_offset.setRange(-5000.0, 5000.0)
        self.shift_offset.setSuffix(" cm⁻¹")
        self.shift_offset.setValue(float(wizard.mode_params.get("shift_offset", 0.0)))
        for widget in (self.excitation_spin, self.shift_offset):
            widget.valueChanged.connect(self._changed)
        self.filter_start = QtWidgets.QDoubleSpinBox()
        self.filter_end = QtWidgets.QDoubleSpinBox()
        for spin in (self.filter_start, self.filter_end):
            spin.setRange(0, 2000)
            spin.setSuffix(" nm")
        self.filter_btn = QtWidgets.QPushButton("Add notch/band-stop")
        self.filter_btn.clicked.connect(self._add_filter)
        self.filter_list = QtWidgets.QListWidget()
        self._refresh_filters()
        layout.addRow("Excitation wavelength", self.excitation_spin)
        layout.addRow("Shift offset", self.shift_offset)
        layout.addRow(QtWidgets.QLabel("Add filter stop band (nm)"))
        layout.addRow("Start", self.filter_start)
        layout.addRow("End", self.filter_end)
        layout.addRow(self.filter_btn)
        layout.addRow(self.filter_list)

    def _changed(self) -> None:
        self.wizard_ref.mode_params.update(
            excitation_nm=float(self.excitation_spin.value()),
            shift_offset=float(self.shift_offset.value()),
            filter_bands=self.wizard_ref.mode_params.get("filter_bands", []),
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def _add_filter(self) -> None:
        band = (float(self.filter_start.value()), float(self.filter_end.value()))
        filters = self.wizard_ref.mode_params.setdefault("filter_bands", [])
        filters.append(band)
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self._refresh_filters()
        self.mark_dirty()

    def _refresh_filters(self) -> None:
        self.filter_list.clear()
        for start, end in self.wizard_ref.mode_params.get("filter_bands", []):
            self.filter_list.addItem(f"{min(start, end):.1f}-{max(start, end):.1f} nm")

    def isComplete(self) -> bool:  # type: ignore[override]
        return self.excitation_spin.value() > 0


class ModeConfigPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle(f"{wizard.mode} configuration")
        layout = QtWidgets.QVBoxLayout(self)
        self.roi_list = QtWidgets.QListWidget()
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.start_spin, self.end_spin):
            spin.setRange(0, 2000)
            spin.setSuffix(" nm")
        self.label_edit = QtWidgets.QLineEdit()
        add_btn = QtWidgets.QPushButton("Add ROI")
        del_btn = QtWidgets.QPushButton("Remove selected")
        add_btn.clicked.connect(self._add_roi)
        del_btn.clicked.connect(self._remove_roi)
        layout.addWidget(QtWidgets.QLabel("Define regions of interest (optional)"))
        form = QtWidgets.QFormLayout()
        form.addRow("Start", self.start_spin)
        form.addRow("End", self.end_spin)
        form.addRow("Label", self.label_edit)
        layout.addLayout(form)
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(add_btn)
        btns.addWidget(del_btn)
        layout.addLayout(btns)
        layout.addWidget(self.roi_list)
        layout.addStretch(1)
        self._refresh_list()

    def _add_roi(self) -> None:
        roi = self.session.add_roi(self.start_spin.value(), self.end_spin.value(), self.label_edit.text())
        self.wizard_ref.mode_params.setdefault("rois", []).append(roi.as_tuple())
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self._refresh_list()
        self.mark_dirty()

    def _remove_roi(self) -> None:
        row = self.roi_list.currentRow()
        self.session.remove_roi(row)
        try:
            self.wizard_ref.mode_params.get("rois", []).pop(row)
        except Exception:
            pass
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self._refresh_list()
        self.mark_dirty()

    def _refresh_list(self) -> None:
        self.roi_list.clear()
        for roi in self.session.rois:
            text = f"{roi.label or 'ROI'}: {roi.start_nm:.1f}-{roi.end_nm:.1f} nm"
            self.roi_list.addItem(text)

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


class ResultsPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Results and overlays")
        layout = QtWidgets.QVBoxLayout(self)
        self.info_label = QtWidgets.QLabel("")
        self.info_label.setWordWrap(True)
        self.status_label = QtWidgets.QLabel("")
        layout.addWidget(self.info_label)
        layout.addWidget(self.status_label)
        layout.addStretch(1)
        self.session.validity_changed.connect(self._refresh_status)

    def initializePage(self) -> None:  # type: ignore[override]
        mode_label = self.wizard_ref.mode.lower()
        self.info_label.setText(
            "Live %s is calculated continuously on the main Vis Spectroscopy graph. "
            "Finish the wizard to apply the acquisition and wavelength settings." % mode_label
        )
        self._refresh_status()

    def _refresh_status(self) -> None:
        parts = []
        if self.session.dark_valid:
            parts.append("Dark ✓")
        else:
            parts.append("Dark ✕")
        if self.wizard_ref.mode in {"Absorbance", "Transmittance", "Reflectance"}:
            parts.append("Reference ✓" if self.session.reference_valid else "Reference ✕")
        if self.wizard_ref.mode == "Relative Irradiance":
            parts.append("Calibration ✓" if self.session.calibration_valid else "Calibration ✕")
        self.status_label.setText(" | ".join(parts))

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


class SpectroscopyModeWizard(QtWidgets.QWizard):
    def __init__(
        self,
        mode: str,
        session: SpectroscopySession,
        capture_callback: CaptureCallable,
        preset_names: Optional[List[str]] = None,
        apply_preset: Optional[Callable[[str], None]] = None,
        step_presets: Optional[Dict[str, str]] = None,
        on_step_preset_changed: Optional[Callable[[str, str], None]] = None,
        initial_acquisition: Optional[Dict[str, float]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.mode = mode
        self.session = session
        self.capture_callback = capture_callback
        self.initial_acquisition = initial_acquisition or {}
        self.preset_names = preset_names or []
        self.apply_preset_cb = apply_preset
        self.step_presets = step_presets or {}
        self.on_step_preset_changed = on_step_preset_changed
        if self.preset_names:
            fallback = self.preset_names[0]
            for key in ("dark", "reference", "raw"):
                self.step_presets.setdefault(key, fallback)
        if session.mode_params.get("mode") == mode:
            self.mode_params = dict(session.mode_params)
        else:
            self.mode_params = {"mode": mode}
        if "rois" in self.mode_params:
            with contextlib.suppress(Exception):
                self.session._set_rois_from_iterable(self.mode_params.get("rois", []))
        self.state: Dict[str, bool] = {
            "dark_captured": session.dark_spectrum is not None,
            "reference_captured": session.reference_spectrum is not None,
            "calibration_valid": session.calibration_valid,
        }
        self.last_metrics: Dict[str, float] = {}
        self.setWindowTitle(f"{mode} wizard")
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage)
        self.setButtonLayout(
            [
                QtWidgets.QWizard.Stretch,
                QtWidgets.QWizard.BackButton,
                QtWidgets.QWizard.NextButton,
                QtWidgets.QWizard.FinishButton,
                QtWidgets.QWizard.CancelButton,
            ]
        )
        self.setStyleSheet(
            "\n".join(
                [
                    "QWizardPage QLabel {",
                    "  color: palette(WindowText);",
                    "}",
                    "QWizardPage QGroupBox::title {",
                    "  color: palette(WindowText);",
                    "}",
                ]
            )
        )
        self._build_pages()

    def _build_pages(self) -> None:
        self.addPage(AcquisitionSetupPage(self))
        if self.mode in {"Absorbance", "Transmittance"}:
            self.addPage(CapturePage(self, "Capture reference", "reference"))
            self.addPage(CapturePage(self, "Capture dark", "dark"))
        else:
            self.addPage(CapturePage(self, "Capture dark", "dark"))
            self.addPage(CapturePage(self, "Capture reference", "reference"))
        if self.mode in {"Transmittance", "Reflectance"}:
            self.addPage(TransReflectPage(self))
        elif self.mode == "Relative Irradiance":
            self.addPage(IrradianceCalibrationPage(self))
        elif self.mode == "Fluorescence":
            self.addPage(FluorescenceMetadataPage(self))
        elif self.mode == "Raman":
            self.addPage(RamanConfigPage(self))
        if self.mode != "Absorbance":
            self.addPage(ModeConfigPage(self))
        self.addPage(ResultsPage(self))
        self.button(QtWidgets.QWizard.FinishButton).setEnabled(False)
        self.currentIdChanged.connect(self._update_finish_state)
        self.finished.connect(self._persist_params)
        self._update_finish_state()
        self.session.validity_changed.connect(self._on_session_validity_changed)
        for page_id in self.pageIds():
            page = self.page(page_id)
            if page is not None:
                page.completeChanged.connect(self._update_finish_state)

    def _persist_params(self) -> None:
        self.mode_params["rois"] = [roi.as_tuple() for roi in self.session.rois]
        self.session.set_mode_params(**self.mode_params)

    def invalidate_results(self) -> None:
        self.state["reference_captured"] = False
        self.state["dark_captured"] = False
        self.state["calibration_valid"] = self.session.calibration_valid

    def invalidate_captures(self) -> None:
        self.state["reference_captured"] = False
        self.state["dark_captured"] = False

    def apply_preset_for_step(self, step: str) -> None:
        if self.apply_preset_cb and self.step_presets.get(step):
            self.apply_preset_cb(step)

    def get_step_preset(self, step: str) -> str:
        return self.step_presets.get(step, "")

    def set_step_preset(self, step: str, preset: str) -> None:
        self.step_presets[step] = preset
        if self.on_step_preset_changed:
            self.on_step_preset_changed(step, preset)

    def _update_finish_state(self) -> None:
        finish_enabled = all(
            (
                (not self._requires_reference() or self.state.get("reference_captured")),
                (not self._requires_dark() or self.state.get("dark_captured")),
                (not self._requires_reference() or self.session.reference_valid),
                (not self._requires_dark() or self.session.dark_valid),
                (not self._requires_calibration() or self.session.calibration_valid),
            )
        )
        self.button(QtWidgets.QWizard.FinishButton).setEnabled(bool(finish_enabled))

    def _on_session_validity_changed(self) -> None:
        if not self.session.dark_valid:
            self.state["dark_captured"] = False
        if not self.session.reference_valid:
            self.state["reference_captured"] = False
        self.state["calibration_valid"] = self.session.calibration_valid
        self._update_finish_state()

    def _requires_reference(self) -> bool:
        return self.mode in {"Absorbance", "Transmittance", "Reflectance"}

    def _requires_dark(self) -> bool:
        return True

    def _requires_calibration(self) -> bool:
        return self.mode == "Relative Irradiance"

    def format_metrics(self, result: Tuple[np.ndarray, np.ndarray]) -> str:
        if not self.last_metrics:
            return "No metrics computed."
        parts = [f"{k}: {v:.3f}" for k, v in self.last_metrics.items()]
        return f"Metrics ({self.mode}): " + ", ".join(parts)

    def accept(self) -> None:  # type: ignore[override]
        self._persist_params()
        super().accept()
