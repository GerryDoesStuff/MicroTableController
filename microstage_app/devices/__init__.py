"""Device helpers and factory functions."""

from ..spectroscopy.devices import (
    MockSpectrometerProvider,
    OceanOpticsSpectrometerProvider,
    SpectrometerManager,
)


def create_spectrometer_manager() -> SpectrometerManager:
    """Create a spectrometer manager with OceanOptics and mock providers."""
    return SpectrometerManager(
        providers=[OceanOpticsSpectrometerProvider(), MockSpectrometerProvider()]
    )


__all__ = [
    "create_spectrometer_manager",
    "MockSpectrometerProvider",
    "OceanOpticsSpectrometerProvider",
    "SpectrometerManager",
]
