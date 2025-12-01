from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import importlib
import importlib.util
import numpy as np
from PySide6 import QtCore

from ..utils.log import LOG


@dataclass(frozen=True)
class SpectrometerDescriptor:
    model: str
    serial_number: str
    path: str
    vendor: str = "OceanOptics"

    def label(self) -> str:
        return f"{self.model} ({self.serial_number})"


class SpectrometerDevice(Protocol):
    descriptor: SpectrometerDescriptor

    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def is_connected(self) -> bool:
        ...

    def get_wavelengths(self) -> np.ndarray:
        ...

    def capture(self) -> np.ndarray:
        ...

    def set_integration_time_ms(self, ms: float) -> None:
        ...

    def set_averages(self, averages: int) -> None:
        ...


class _SpectrometerProvider(Protocol):
    def list_devices(self) -> List[SpectrometerDescriptor]:
        ...

    def connect(self, descriptor: SpectrometerDescriptor) -> SpectrometerDevice:
        ...


class SpectrometerManager(QtCore.QObject):
    devices_changed = QtCore.Signal(list)
    device_connected = QtCore.Signal(object, object)

    def __init__(self, providers: Optional[Sequence[_SpectrometerProvider]] = None):
        super().__init__()
        self._providers: Sequence[_SpectrometerProvider] = (
            providers
            if providers is not None
            else [OceanOpticsSpectrometerProvider(), MockSpectrometerProvider()]
        )
        self._devices: List[SpectrometerDescriptor] = []
        self._active: dict[SpectrometerDescriptor, SpectrometerDevice] = {}
        self._locks: dict[SpectrometerDescriptor, QtCore.QMutex] = {}
        self._last_active: Optional[SpectrometerDescriptor] = None

    @property
    def devices(self) -> List[SpectrometerDescriptor]:
        return list(self._devices)

    def refresh(self) -> List[SpectrometerDescriptor]:
        devices: List[SpectrometerDescriptor] = []
        for provider in self._providers:
            try:
                devices.extend(provider.list_devices())
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to enumerate spectrometers: %s", exc)
        self._devices = devices
        self.devices_changed.emit(list(devices))
        return list(devices)

    def connect(self, descriptor: SpectrometerDescriptor) -> SpectrometerDevice:
        if descriptor in self._active:
            device = self._active[descriptor]
            if not device.is_connected():
                device.connect()
            self._last_active = descriptor
            self.device_connected.emit(descriptor, device)
            return device
        for provider in self._providers:
            for candidate in provider.list_devices():
                if candidate == descriptor:
                    device = provider.connect(descriptor)
                    device.connect()
                    self._active[descriptor] = device
                    self._locks.setdefault(descriptor, QtCore.QMutex())
                    self._last_active = descriptor
                    self.device_connected.emit(descriptor, device)
                    return device
        raise RuntimeError("Device not found: %s" % descriptor.label())

    def disconnect(self, descriptor: Optional[SpectrometerDescriptor] = None) -> None:
        targets = [descriptor] if descriptor else list(self._active.keys())
        for desc in targets:
            device = self._active.pop(desc, None)
            self._locks.pop(desc, None)
            if device is None:
                self.device_connected.emit(desc, None)
                continue
            try:
                device.disconnect()
            finally:
                self.device_connected.emit(desc, None)
        if descriptor is None:
            self._last_active = None
        elif self._last_active == descriptor:
            self._last_active = next(iter(self._active), None)

    def get_active(self, descriptor: Optional[SpectrometerDescriptor] = None) -> Optional[SpectrometerDevice]:
        if descriptor is not None:
            return self._active.get(descriptor)
        if self._last_active is not None:
            return self._active.get(self._last_active)
        return next(iter(self._active.values()), None) if self._active else None

    @property
    def active(self) -> Optional[SpectrometerDevice]:
        return self.get_active()

    @property
    def active_descriptor(self) -> Optional[SpectrometerDescriptor]:
        return self._last_active

    def acquisition_lock(self, descriptor: SpectrometerDescriptor) -> QtCore.QMutex:
        lock = self._locks.get(descriptor)
        if lock is None:
            lock = QtCore.QMutex()
            self._locks[descriptor] = lock
        return lock


class OceanOpticsSpectrometer:
    def __init__(self, descriptor: SpectrometerDescriptor):
        self.descriptor = descriptor
        self._device = None
        self._connected = False
        self._integration_time_ms = 10.0
        self._averages = 1

    @staticmethod
    def _get_backend():
        spec = importlib.util.find_spec("seabreeze.spectrometers")
        if spec is None:
            return None
        return importlib.import_module("seabreeze.spectrometers")

    @classmethod
    def enumerate(cls) -> List[SpectrometerDescriptor]:
        backend = cls._get_backend()
        if backend is None:
            return []
        descriptors: List[SpectrometerDescriptor] = []
        for dev in backend.list_devices():
            descriptors.append(
                SpectrometerDescriptor(
                    model=getattr(dev, "model", "OceanOptics"),
                    serial_number=str(getattr(dev, "serial_number", "")),
                    path=str(getattr(dev, "path", getattr(dev, "device_node", ""))),
                )
            )
        return descriptors

    def connect(self) -> None:
        backend = self._get_backend()
        if backend is None:
            raise RuntimeError("seabreeze backend unavailable for OceanOptics spectrometers")
        if self._device is not None:
            self.disconnect()
        target = self._find_matching_device()
        if target is None:
            raise RuntimeError(f"Spectrometer not found: {self.descriptor.label()}")
        self._device = backend.Spectrometer(target)
        self._connected = True
        self._device.integration_time_micros(int(self._integration_time_ms * 1000))

    def _find_matching_device(self):
        backend = self._get_backend()
        if backend is None:
            return None
        desired_serial = str(self.descriptor.serial_number or "")
        desired_path = str(self.descriptor.path or "")
        for dev in backend.list_devices():
            dev_serial = str(getattr(dev, "serial_number", ""))
            dev_path = str(getattr(dev, "path", getattr(dev, "device_node", "")))
            if desired_serial and dev_serial == desired_serial:
                return dev
            if desired_path and dev_path == desired_path:
                return dev
        return None

    def disconnect(self) -> None:
        if self._device is not None:
            try:
                self._device.close()
            finally:
                self._device = None
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def get_wavelengths(self) -> np.ndarray:
        if self._device is None:
            self.connect()
        return np.asarray(self._device.wavelengths(), dtype=float)

    def capture(self) -> np.ndarray:
        if self._device is None:
            raise RuntimeError("Spectrometer is not connected")
        spectrum = np.zeros_like(self.get_wavelengths())
        for _ in range(max(1, int(self._averages))):
            spectrum += np.asarray(self._device.intensities(), dtype=float)
        return spectrum / max(1, int(self._averages))

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_time_ms = float(ms)
        if self._device is not None:
            self._device.integration_time_micros(int(self._integration_time_ms * 1000))

    def set_averages(self, averages: int) -> None:
        self._averages = max(1, int(averages))


class MockSpectrometer:
    def __init__(self, descriptor: SpectrometerDescriptor):
        self.descriptor = descriptor
        self._connected = False
        self._integration_time_ms = 10.0
        self._averages = 1
        self._wavelengths = np.linspace(350.0, 850.0, 1024)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected)

    def get_wavelengths(self) -> np.ndarray:
        return np.copy(self._wavelengths)

    def capture(self) -> np.ndarray:
        base = 2000 * np.exp(-((self._wavelengths - 550.0) ** 2) / (2 * 40.0**2))
        modulation = 100 * np.sin(self._wavelengths / 30.0)
        spectrum = base + modulation + 10 * self._integration_time_ms
        noise = np.random.RandomState(0).normal(scale=5.0, size=self._wavelengths.shape)
        averaged = np.zeros_like(spectrum)
        for _ in range(max(1, int(self._averages))):
            averaged += spectrum + noise
        return averaged / max(1, int(self._averages))

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_time_ms = float(ms)

    def set_averages(self, averages: int) -> None:
        self._averages = max(1, int(averages))


class OceanOpticsSpectrometerProvider:
    def list_devices(self) -> List[SpectrometerDescriptor]:
        return OceanOpticsSpectrometer.enumerate()

    def connect(self, descriptor: SpectrometerDescriptor) -> OceanOpticsSpectrometer:
        return OceanOpticsSpectrometer(descriptor)


class MockSpectrometerProvider:
    def __init__(self, descriptors: Optional[Iterable[SpectrometerDescriptor]] = None):
        self._descriptors = list(
            descriptors
            if descriptors is not None
            else [
                SpectrometerDescriptor(
                    model="MockSpectrometer",
                    serial_number="SIM0001",
                    path="mock://0",
                    vendor="Mock",
                )
            ]
        )

    def list_devices(self) -> List[SpectrometerDescriptor]:
        return list(self._descriptors)

    def connect(self, descriptor: SpectrometerDescriptor) -> MockSpectrometer:
        return MockSpectrometer(descriptor)
