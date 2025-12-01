from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6 import QtCore


@dataclass
class ROI:
    start_nm: float
    end_nm: float
    label: str = ""

    def as_tuple(self) -> Tuple[float, float]:
        return (float(self.start_nm), float(self.end_nm))


@dataclass
class CalibrationData:
    response_curve: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    acquisition: Optional["AcquisitionMetadata"] = None


@dataclass(frozen=True)
class AcquisitionMetadata:
    device_id: str
    integration_ms: float
    averages: int
    mode: str
    timestamp: float

    def matches(self, other: Optional["AcquisitionMetadata"]) -> bool:
        if other is None:
            return False
        return (
            self.device_id == other.device_id
            and self.mode == other.mode
            and int(self.averages) == int(other.averages)
            and float(self.integration_ms) == float(other.integration_ms)
        )


class SpectroscopySession(QtCore.QObject):
    wavelengths_changed = QtCore.Signal(object)
    spectra_changed = QtCore.Signal()
    calibration_changed = QtCore.Signal(object)
    mode_params_changed = QtCore.Signal(object)
    rois_changed = QtCore.Signal(list)
    validity_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.wavelengths: Optional[np.ndarray] = None
        self.raw_spectrum: Optional[np.ndarray] = None
        self.dark_spectrum: Optional[np.ndarray] = None
        self.reference_spectrum: Optional[np.ndarray] = None
        self.mode_params: Dict[str, object] = {}
        self.calibration: CalibrationData = CalibrationData()
        self.calibration_valid: bool = False
        self.rois: List[ROI] = []
        self.acquisition_context: Optional[AcquisitionMetadata] = None
        self.dark_metadata: Optional[AcquisitionMetadata] = None
        self.reference_metadata: Optional[AcquisitionMetadata] = None
        self.dark_valid: bool = False
        self.reference_valid: bool = False

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        arr = np.asarray(wavelengths, dtype=float).ravel()
        self.wavelengths = arr
        self._invalidate_dark()
        self._invalidate_reference()
        self._invalidate_calibration()
        self.wavelengths_changed.emit(np.copy(arr))
        self.calibration_changed.emit(self.calibration)
        self.validity_changed.emit()

    def set_acquisition_context(
        self, device_id: str, integration_ms: float, averages: int, mode: str, timestamp: float
    ) -> None:
        meta = AcquisitionMetadata(
            device_id=str(device_id),
            integration_ms=float(integration_ms),
            averages=int(averages),
            mode=str(mode),
            timestamp=float(timestamp),
        )
        changed = not meta.matches(self.acquisition_context)
        self.acquisition_context = meta
        if changed:
            if not meta.matches(self.dark_metadata):
                self._invalidate_dark()
            if not meta.matches(self.reference_metadata):
                self._invalidate_reference()
            if not meta.matches(self.calibration.acquisition):
                self._invalidate_calibration()
            self.validity_changed.emit()

    def _validate_shape(self, spectrum: np.ndarray) -> np.ndarray:
        arr = np.asarray(spectrum, dtype=float).ravel()
        if self.wavelengths is not None and arr.shape != self.wavelengths.shape:
            raise ValueError("Spectrum length does not match wavelength axis")
        return arr

    def set_raw(self, spectrum: np.ndarray) -> None:
        self.raw_spectrum = self._validate_shape(spectrum)
        self.spectra_changed.emit()

    def set_dark(self, spectrum: np.ndarray, acquisition: Optional[AcquisitionMetadata] = None) -> None:
        self.dark_spectrum = self._validate_shape(spectrum)
        self.dark_metadata = acquisition or self.acquisition_context
        self.dark_valid = True
        self._invalidate_calibration()
        self.spectra_changed.emit()
        self.calibration_changed.emit(self.calibration)
        self.validity_changed.emit()

    def set_reference(
        self, spectrum: np.ndarray, acquisition: Optional[AcquisitionMetadata] = None
    ) -> None:
        self.reference_spectrum = self._validate_shape(spectrum)
        self.reference_metadata = acquisition or self.acquisition_context
        self.reference_valid = True
        self._invalidate_calibration()
        self.spectra_changed.emit()
        self.calibration_changed.emit(self.calibration)
        self.validity_changed.emit()

    def set_mode_params(self, **params) -> None:
        if params != self.mode_params:
            self.mode_params = dict(params)
            self._invalidate_calibration()
            self.mode_params_changed.emit(dict(params))
            self.calibration_changed.emit(self.calibration)
            self.validity_changed.emit()

    def set_calibration(
        self,
        response_curve: Optional[np.ndarray],
        acquisition: Optional[AcquisitionMetadata] = None,
        **metadata,
    ) -> None:
        if response_curve is None:
            self.calibration = CalibrationData(None, metadata, acquisition)
            self._invalidate_calibration()
            self.calibration_changed.emit(self.calibration)
            self.validity_changed.emit()
            return
        arr = np.asarray(response_curve, dtype=float).ravel()
        if self.wavelengths is not None and arr.shape != self.wavelengths.shape:
            raise ValueError("Calibration length does not match wavelengths")
        self.calibration = CalibrationData(arr, metadata, acquisition or self.acquisition_context)
        self.calibration_valid = True
        self.calibration_changed.emit(self.calibration)
        self.validity_changed.emit()

    def add_roi(self, start_nm: float, end_nm: float, label: str = "") -> ROI:
        roi = ROI(start_nm=min(start_nm, end_nm), end_nm=max(start_nm, end_nm), label=label)
        self.rois.append(roi)
        self.rois_changed.emit(list(self.rois))
        return roi

    def clear_rois(self) -> None:
        self.rois.clear()
        self.rois_changed.emit([])

    def remove_roi(self, index: int) -> None:
        if 0 <= index < len(self.rois):
            self.rois.pop(index)
            self.rois_changed.emit(list(self.rois))

    def is_ready_for_processing(self) -> bool:
        if self.wavelengths is None or self.raw_spectrum is None:
            return False
        if self.dark_spectrum is not None and self.dark_spectrum.shape != self.wavelengths.shape:
            return False
        if self.reference_spectrum is not None and self.reference_spectrum.shape != self.wavelengths.shape:
            return False
        return True

    def requires_recalibration(self) -> bool:
        return not (self.calibration_valid and self.dark_valid and self.reference_valid)

    # ------------------------------------------------------------------
    def _invalidate_dark(self) -> None:
        self.dark_valid = False

    def _invalidate_reference(self) -> None:
        self.reference_valid = False

    def _invalidate_calibration(self) -> None:
        self.calibration_valid = False
