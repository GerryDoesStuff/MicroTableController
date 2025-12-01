import numpy as np

from microstage_app.spectroscopy.devices import (
    MockSpectrometerProvider,
    SpectrometerDescriptor,
    SpectrometerManager,
)


def test_mock_enumeration_and_acquisition():
    descriptor = SpectrometerDescriptor(
        model="MockSpectrometer",
        serial_number="SIM001",
        path="mock://1",
        vendor="Mock",
    )
    provider = MockSpectrometerProvider([descriptor])
    manager = SpectrometerManager(providers=[provider])

    devices = manager.refresh()
    assert devices == [descriptor]

    device = manager.connect(descriptor)
    assert manager.active is device
    assert device.is_connected()

    wavelengths = device.get_wavelengths()
    assert isinstance(wavelengths, np.ndarray)
    assert wavelengths.ndim == 1
    spectrum = device.capture()
    assert spectrum.shape == wavelengths.shape
    assert spectrum.mean() > 0

    device.set_integration_time_ms(20.0)
    device.set_averages(3)
    spectrum_avg = device.capture()
    assert not np.array_equal(spectrum, spectrum_avg)

    manager.disconnect()
    assert manager.active is None
