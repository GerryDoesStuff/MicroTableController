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


def test_mock_capture_respects_parameters():
    descriptor = SpectrometerDescriptor(
        model="MockSpectrometer",
        serial_number="SIM002",
        path="mock://2",
        vendor="Mock",
    )
    device = MockSpectrometerProvider([descriptor]).connect(descriptor)
    device.connect()

    default_capture = device.capture()
    device.set_integration_time_ms(50.0)
    device.set_averages(5)
    adjusted_capture = device.capture()

    assert adjusted_capture.mean() > default_capture.mean()
    assert adjusted_capture.std() < (default_capture.std() * 2)
    device.disconnect()


def test_manager_routes_multiple_spectrometers():
    first = SpectrometerDescriptor("MockSpectrometer", "A", "mock://a")
    second = SpectrometerDescriptor("MockSpectrometer", "B", "mock://b")

    provider = MockSpectrometerProvider([first, second])
    manager = SpectrometerManager(providers=[provider])

    devices = manager.refresh()
    assert devices == [first, second]

    dev_a = manager.connect(first)
    assert manager.active is dev_a
    assert dev_a.descriptor.serial_number == "A"

    dev_b = manager.connect(second)
    assert manager.active is dev_b
    assert dev_b.descriptor.serial_number == "B"
    assert dev_a is not dev_b

    manager.disconnect()
    assert manager.active is None
