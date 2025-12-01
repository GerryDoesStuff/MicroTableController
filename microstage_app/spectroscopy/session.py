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


class SpectroscopySession(QtCore.QObject):
    wavelengths_changed = QtCore.Signal(object)
    spectra_changed = QtCore.Signal()
    calibration_changed = QtCore.Signal(object)
    mode_params_changed = QtCore.Signal(object)
    rois_changed = QtCore.Signal(list)

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

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        arr = np.asarray(wavelengths, dtype=float).ravel()
        self.wavelengths = arr
        self.calibration_valid = False
        self.wavelengths_changed.emit(np.copy(arr))
        self.calibration_changed.emit(self.calibration)

    def _validate_shape(self, spectrum: np.ndarray) -> np.ndarray:
        arr = np.asarray(spectrum, dtype=float).ravel()
        if self.wavelengths is not None and arr.shape != self.wavelengths.shape:
            raise ValueError("Spectrum length does not match wavelength axis")
        return arr

    def set_raw(self, spectrum: np.ndarray) -> None:
        self.raw_spectrum = self._validate_shape(spectrum)
        self.spectra_changed.emit()

    def set_dark(self, spectrum: np.ndarray) -> None:
        self.dark_spectrum = self._validate_shape(spectrum)
        self.calibration_valid = False
        self.spectra_changed.emit()
        self.calibration_changed.emit(self.calibration)

    def set_reference(self, spectrum: np.ndarray) -> None:
        self.reference_spectrum = self._validate_shape(spectrum)
        self.calibration_valid = False
        self.spectra_changed.emit()
        self.calibration_changed.emit(self.calibration)

    def set_mode_params(self, **params) -> None:
        if params != self.mode_params:
            self.mode_params = dict(params)
            self.calibration_valid = False
            self.mode_params_changed.emit(dict(params))
            self.calibration_changed.emit(self.calibration)

    def set_calibration(self, response_curve: Optional[np.ndarray], **metadata) -> None:
        if response_curve is None:
            self.calibration = CalibrationData(None, metadata)
            self.calibration_valid = False
            self.calibration_changed.emit(self.calibration)
            return
        arr = np.asarray(response_curve, dtype=float).ravel()
        if self.wavelengths is not None and arr.shape != self.wavelengths.shape:
            raise ValueError("Calibration length does not match wavelengths")
        self.calibration = CalibrationData(arr, metadata)
        self.calibration_valid = True
        self.calibration_changed.emit(self.calibration)

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
        return not self.calibration_valid
