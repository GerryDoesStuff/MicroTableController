"""Spectroscopy module with device abstractions and processing utilities."""

from .devices import (
    SpectrometerDescriptor,
    SpectrometerDevice,
    SpectrometerManager,
    OceanOpticsSpectrometer,
    MockSpectrometer,
    MockSpectrometerProvider,
    OceanOpticsSpectrometerProvider,
)
from .session import AcquisitionMetadata, SpectroscopySession, CalibrationData, ROI

__all__ = [
    "SpectrometerDescriptor",
    "SpectrometerDevice",
    "SpectrometerManager",
    "OceanOpticsSpectrometer",
    "MockSpectrometer",
    "MockSpectrometerProvider",
    "OceanOpticsSpectrometerProvider",
    "AcquisitionMetadata",
    "SpectroscopySession",
    "CalibrationData",
    "ROI",
]
