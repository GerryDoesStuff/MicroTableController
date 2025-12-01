from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCharts, QtCore, QtGui, QtWidgets

from ..spectroscopy import processing
from ..spectroscopy.session import SpectroscopySession

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


class WavelengthConfigPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Wavelength configuration")
        layout = QtWidgets.QFormLayout(self)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        self.min_spin.setSuffix(" nm")
        self.max_spin.setSuffix(" nm")
        self.min_spin.setRange(0, 2000)
        self.max_spin.setRange(0, 2000)
        self.min_spin.valueChanged.connect(self._sync)
        self.max_spin.valueChanged.connect(self._sync)
        layout.addRow("Min wavelength", self.min_spin)
        layout.addRow("Max wavelength", self.max_spin)
        if wizard.session.wavelengths is not None:
            wl = wizard.session.wavelengths
            self.min_spin.setValue(float(wl.min()))
            self.max_spin.setValue(float(wl.max()))

    def _sync(self) -> None:
        if self.min_spin.value() >= self.max_spin.value():
            self.setSubTitle("Min must be < Max")
        else:
            self.setSubTitle("")
            self.wizard_ref.mode_params["wavelength_range"] = (
                self.min_spin.value(),
                self.max_spin.value(),
            )
            self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.mark_dirty()

    def isComplete(self) -> bool:  # type: ignore[override]
        return self.min_spin.value() < self.max_spin.value()


class AcquisitionPage(BaseWizardPage):
    def __init__(self, wizard: 'SpectroscopyModeWizard') -> None:
        super().__init__(wizard)
        self.setTitle("Acquisition parameters")
        layout = QtWidgets.QFormLayout(self)
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
            widget.valueChanged.connect(self._changed)
        layout.addRow("Integration time", self.integration)
        layout.addRow("Averages", self.averages)
        layout.addRow("Smoothing (boxcar)", self.smoothing)

    def _changed(self) -> None:
        self.wizard_ref.mode_params.update(
            integration=float(self.integration.value()),
            averages=int(self.averages.value()),
            smoothing=int(self.smoothing.value()),
        )
        self.session.set_mode_params(**self.wizard_ref.mode_params)
        self.wizard_ref.invalidate_captures()
        self.mark_dirty()

    def isComplete(self) -> bool:  # type: ignore[override]
        return True


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
        layout.addStretch(1)

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
        except Exception as exc:
            self.info.setText(f"Capture failed: {exc}")
        self.wizard_ref._update_finish_state()
        self.completeChanged.emit()

    def _preset_changed(self, preset: str) -> None:
        self.wizard_ref.set_step_preset(self.capture_key, preset)

    def isComplete(self) -> bool:  # type: ignore[override]
        return bool(self.wizard_ref.state.get(f"{self.capture_key}_captured", False))


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
        self.chart = QtCharts.QChart()
        self.chart.setAnimationOptions(QtCharts.QChart.NoAnimation)
        self.chart.legend().setVisible(True)
        self.chart.createDefaultAxes()
        self.chart.axisX().setTitleText("Wavelength (nm)")
        self.chart.axisY().setTitleText("Intensity")
        self.chart_view = QtCharts.QChartView(self.chart)
        self.chart_view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.history_list = QtWidgets.QListWidget()
        self.history_list.itemChanged.connect(self._toggle_series)
        self.metrics = QtWidgets.QLabel("No metrics computed yet.")
        self.capture_btn = QtWidgets.QPushButton("Capture sample for result")
        self.capture_btn.clicked.connect(self._capture_sample)
        layout.addWidget(self.chart_view, 2)
        layout.addWidget(QtWidgets.QLabel("History (toggle to overlay):"))
        layout.addWidget(self.history_list)
        layout.addWidget(self.metrics)
        layout.addWidget(self.capture_btn)
        layout.addStretch(1)
        self.series_for_item: Dict[QtWidgets.QListWidgetItem, QtCharts.QLineSeries] = {}

    def initializePage(self) -> None:  # type: ignore[override]
        self._recompute()

    def _capture_sample(self) -> None:
        params = self.wizard_ref.mode_params
        spectrum = self.wizard_ref.capture_callback(
            "raw",
            float(params.get("integration", 10.0)),
            int(params.get("averages", 1)),
        )
        if spectrum is not None:
            self.session.set_raw(spectrum)
            self.wizard_ref.state["raw_captured"] = True
            self._recompute()
            self.wizard_ref._update_finish_state()
            self.completeChanged.emit()

    def _toggle_series(self, item: QtWidgets.QListWidgetItem) -> None:
        series = self.series_for_item.get(item)
        if series:
            series.setVisible(item.checkState() == QtCore.Qt.Checked)

    def _recompute(self) -> None:
        self.chart.removeAllSeries()
        self.series_for_item.clear()
        self.history_list.blockSignals(True)
        self.history_list.clear()
        self.history_list.blockSignals(False)
        if self.session.wavelengths is None:
            self.metrics.setText("Missing wavelength calibration.")
            return
        result = self._compute_mode_result()
        if result is None:
            self.metrics.setText("No result computed; capture sample, dark, and reference first.")
            return
        x_axis = self.chart.axisX()
        y_axis = self.chart.axisY()
        if x_axis:
            x_axis.setRange(float(self.session.wavelengths.min()), float(self.session.wavelengths.max()))
        ymin, ymax = np.min(result[1]), np.max(result[1])
        if ymin == ymax:
            ymax += 1.0
        if y_axis:
            y_axis.setRange(float(ymin), float(ymax))
        # current result
        series = self._series_from_data("Current", result[0], result[1], QtGui.QColor("deepskyblue"))
        self.chart.addSeries(series)
        if x_axis:
            series.attachAxis(x_axis)
        if y_axis:
            series.attachAxis(y_axis)
        item = QtWidgets.QListWidgetItem("Current result")
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.history_list.addItem(item)
        self.series_for_item[item] = series
        self._add_overlay_series("Dark", self.session.dark_spectrum, QtGui.QColor("gray"))
        self._add_overlay_series("Reference", self.session.reference_spectrum, QtGui.QColor("green"))
        self._add_overlay_series("Raw", self.session.raw_spectrum, QtGui.QColor("orange"))
        self.metrics.setText(self.wizard_ref.format_metrics(result))

    def _series_from_data(
        self, label: str, x: np.ndarray, y: np.ndarray, color: QtGui.QColor
    ) -> QtCharts.QLineSeries:
        series = QtCharts.QLineSeries()
        series.setName(label)
        series.setColor(color)
        points = [QtCore.QPointF(float(xv), float(yv)) for xv, yv in zip(x, y)]
        series.replace(points)
        return series

    def _add_overlay_series(self, name: str, data: Optional[np.ndarray], color: QtGui.QColor) -> None:
        if data is None or self.session.wavelengths is None:
            return
        series = self._series_from_data(name, self.session.wavelengths, data, color)
        self.chart.addSeries(series)
        x_axis = self.chart.axisX()
        y_axis = self.chart.axisY()
        if x_axis:
            series.attachAxis(x_axis)
        if y_axis:
            series.attachAxis(y_axis)
        item = QtWidgets.QListWidgetItem(name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Unchecked)
        self.history_list.addItem(item)
        self.series_for_item[item] = series

    def _compute_mode_result(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        wl = self.session.wavelengths
        raw = self.session.raw_spectrum
        ref = self.session.reference_spectrum
        dark = self.session.dark_spectrum
        if wl is None or raw is None:
            return None
        params = self.wizard_ref.mode_params
        smoothed = processing.smooth_boxcar(raw, window=int(params.get("smoothing", 1)))
        if dark is not None:
            try:
                smoothed = processing.subtract_dark(smoothed, dark)
            except Exception:
                pass
        if self.wizard_ref.mode == "Absorbance":
            if ref is None:
                return None
            arr = processing.compute_absorbance(smoothed, ref)
            metrics = np.max(arr)
            self.wizard_ref.last_metrics = {"Max absorbance": metrics}
            return wl, arr
        if self.wizard_ref.mode in {"Transmittance", "Reflectance"}:
            if ref is None:
                return None
            arr = processing.normalize_reference(smoothed, ref)
            self.wizard_ref.last_metrics = {"Mean": float(np.mean(arr))}
            return wl, arr
        if self.wizard_ref.mode == "Relative Irradiance":
            curve = self.session.calibration.response_curve
            arr = processing.compute_irradiance(
                smoothed,
                float(params.get("integration", 10.0)),
                response_curve=curve,
            )
            self.wizard_ref.last_metrics = {"Total irradiance": float(np.trapz(arr, wl))}
            return wl, arr
        if self.wizard_ref.mode == "Fluorescence":
            area = float(np.trapz(smoothed, wl))
            self.wizard_ref.last_metrics = {"Integrated intensity": area}
            return wl, smoothed
        if self.wizard_ref.mode == "Raman":
            excitation = float(params.get("excitation_nm", 532.0))
            shift = processing.raman_shift_cm(wl, excitation)
            self.wizard_ref.last_metrics = {"Max shift": float(np.max(shift))}
            return shift, smoothed
        return wl, smoothed

    def isComplete(self) -> bool:  # type: ignore[override]
        return bool(self.wizard_ref.state.get("raw_captured"))


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
        self.mode_params: Dict[str, object] = {"mode": mode}
        self.state: Dict[str, bool] = {
            "dark_captured": session.dark_spectrum is not None,
            "reference_captured": session.reference_spectrum is not None,
            "raw_captured": session.raw_spectrum is not None,
        }
        self.last_metrics: Dict[str, float] = {}
        self.setWindowTitle(f"{mode} wizard")
        self.setOption(QtWidgets.QWizard.NoBackButtonOnStartPage)
        self._build_pages()

    def _build_pages(self) -> None:
        self.addPage(WavelengthConfigPage(self))
        self.addPage(AcquisitionPage(self))
        self.addPage(CapturePage(self, "Capture dark", "dark"))
        self.addPage(CapturePage(self, "Capture reference", "reference"))
        # mode specific config
        self.addPage(ModeConfigPage(self))
        self.addPage(ResultsPage(self))
        self.button(QtWidgets.QWizard.FinishButton).setEnabled(False)
        self.currentIdChanged.connect(self._update_finish_state)
        self.finished.connect(self._persist_params)
        self._update_finish_state()
        self.session.validity_changed.connect(self._on_session_validity_changed)
        for i in range(self.pageCount()):
            page = self.page(i)
            if page is not None:
                page.completeChanged.connect(self._update_finish_state)

    def _persist_params(self) -> None:
        self.session.set_mode_params(**self.mode_params)

    def invalidate_results(self) -> None:
        self.state["raw_captured"] = False
        self.state["reference_captured"] = False
        self.state["dark_captured"] = False

    def invalidate_captures(self) -> None:
        self.state["raw_captured"] = False
        self.state["reference_captured"] = False

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
                self.state.get("raw_captured"),
                self.state.get("reference_captured"),
                self.state.get("dark_captured"),
                self.session.reference_valid,
                self.session.dark_valid,
            )
        )
        self.button(QtWidgets.QWizard.FinishButton).setEnabled(bool(finish_enabled))

    def _on_session_validity_changed(self) -> None:
        if not self.session.dark_valid:
            self.state["dark_captured"] = False
        if not self.session.reference_valid:
            self.state["reference_captured"] = False
        self._update_finish_state()

    def format_metrics(self, result: Tuple[np.ndarray, np.ndarray]) -> str:
        if not self.last_metrics:
            return "No metrics computed."
        parts = [f"{k}: {v:.3f}" for k, v in self.last_metrics.items()]
        return f"Metrics ({self.mode}): " + ", ".join(parts)

    def accept(self) -> None:  # type: ignore[override]
        self._persist_params()
        super().accept()

