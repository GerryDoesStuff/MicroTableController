from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import importlib
import importlib.util
import numpy as np
from PySide6 import QtCore

from ..utils.log import LOG
from ..utils.workers import run_async


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

    def connect(self, descriptor: SpectrometerDescriptor) -> Optional[SpectrometerDevice]:
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
        self._poll_thread: Optional[QtCore.QThread] = None
        self._poll_worker: Optional[QtCore.QObject] = None
        self._poll_in_flight = False
        self._monitor_timer = QtCore.QTimer(self)
        self._monitor_timer.setInterval(2000)
        self._monitor_timer.timeout.connect(self._poll_devices)
        self._monitor_timer.start()

    def shutdown(self) -> None:
        """Stop monitoring timers and ensure background polls are finished."""
        try:
            self._monitor_timer.timeout.disconnect(self._poll_devices)
        except (TypeError, RuntimeError):
            pass
        self._monitor_timer.stop()

        worker = self._poll_worker
        if worker is not None:
            try:
                worker.finished.disconnect(self._on_polled_devices)
            except (TypeError, RuntimeError):
                pass

        thread = self._poll_thread
        if thread is not None:
            if thread.isRunning():
                thread.quit()
                thread.wait()
            self._poll_thread = None
        self._poll_worker = None
        self._poll_in_flight = False

    close = shutdown

    @property
    def devices(self) -> List[SpectrometerDescriptor]:
        return list(self._devices)

    def _enumerate_devices(self) -> List[SpectrometerDescriptor]:
        devices: List[SpectrometerDescriptor] = []
        for provider in self._providers:
            try:
                devices.extend(provider.list_devices())
            except BaseException as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to enumerate spectrometers: %s", exc)
        return devices

    def _poll_devices(self) -> None:
        if self._poll_in_flight:
            return
        self._poll_in_flight = True
        thread, worker = run_async(self._enumerate_devices, parent=self)
        self._poll_thread = thread
        self._poll_worker = worker
        worker.finished.connect(self._on_polled_devices)

    @QtCore.Slot(object, object)
    def _on_polled_devices(self, devices, err) -> None:
        self._poll_in_flight = False
        self._poll_thread = None
        self._poll_worker = None
        if err:
            LOG.warning("Spectrometer poll failed: %s", err)
            return
        self._update_devices(devices)

    def _update_devices(self, devices: List[SpectrometerDescriptor]) -> None:
        if devices == self._devices:
            return
        removed = [desc for desc in self._active if desc not in devices]
        for desc in removed:
            self.disconnect(desc)
        self._devices = list(devices)
        self.devices_changed.emit(list(devices))

    def refresh(self) -> List[SpectrometerDescriptor]:
        devices = self._enumerate_devices()
        self._update_devices(devices)
        return list(self._devices)

    def connect(self, descriptor: SpectrometerDescriptor) -> Optional[SpectrometerDevice]:
        if descriptor in self._active:
            device = self._active[descriptor]
            if not device.is_connected():
                try:
                    device.connect()
                except BaseException as exc:
                    LOG.warning("Failed to connect spectrometer %s: %s", descriptor.label(), exc)
                    self.device_connected.emit(descriptor, None)
                    return None
            self._last_active = descriptor
            self.device_connected.emit(descriptor, device)
            return device
        for provider in self._providers:
            try:
                candidates = provider.list_devices()
            except BaseException as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to enumerate spectrometers: %s", exc)
                continue
            for candidate in candidates:
                if candidate == descriptor:
                    try:
                        device = provider.connect(descriptor)
                    except BaseException as exc:
                        LOG.warning(
                            "Failed to create spectrometer %s: %s", descriptor.label(), exc
                        )
                        device = None
                    if device is None:
                        self.device_connected.emit(descriptor, None)
                        return None
                    try:
                        device.connect()
                    except BaseException as exc:
                        LOG.warning(
                            "Failed to connect spectrometer %s: %s", descriptor.label(), exc
                        )
                        self.device_connected.emit(descriptor, None)
                        return None
                    self._active[descriptor] = device
                    self._locks.setdefault(descriptor, QtCore.QMutex())
                    self._last_active = descriptor
                    self.device_connected.emit(descriptor, device)
                    return device
        LOG.warning("Device not found: %s", descriptor.label())
        self.device_connected.emit(descriptor, None)
        return None

    def disconnect(self, descriptor: Optional[SpectrometerDescriptor] = None) -> None:
        targets = [descriptor] if descriptor else list(self._active.keys())
        for desc in targets:
            device = self._active.pop(desc, None)
            self._locks.pop(desc, None)
            if device is None:
                self.device_connected.emit(desc, None)
                continue
            try:
                try:
                    device.disconnect()
                except BaseException as exc:  # pragma: no cover - defensive
                    LOG.warning("Failed to disconnect spectrometer %s: %s", desc.label(), exc)
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
        try:
            spec = importlib.util.find_spec("seabreeze.spectrometers")
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to check for seabreeze backend: %s", exc)
            return None
        if spec is None:
            return None
        try:
            return importlib.import_module("seabreeze.spectrometers")
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to import seabreeze backend: %s", exc)
            return None

    @classmethod
    def enumerate(cls) -> List[SpectrometerDescriptor]:
        backend = cls._get_backend()
        if backend is None:
            return []
        descriptors: List[SpectrometerDescriptor] = []
        try:
            devices = backend.list_devices()
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to enumerate OceanOptics spectrometers: %s", exc)
            return []
        for dev in devices:
            descriptors.append(
                SpectrometerDescriptor(
                    model=getattr(dev, "model", "OceanOptics"),
                    serial_number=str(getattr(dev, "serial_number", "")),
                    path=str(getattr(dev, "path", getattr(dev, "device_node", ""))),
                )
            )
        return descriptors

    def connect(self) -> Optional[object]:
        backend = self._get_backend()
        if backend is None:
            LOG.warning("seabreeze backend unavailable for OceanOptics spectrometers")
            return None
        if self._device is not None:
            self.disconnect()
        try:
            target = self._find_matching_device()
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning(
                "Failed to locate OceanOptics spectrometer %s: %s", self.descriptor.label(), exc
            )
            return None
        if target is None:
            LOG.warning("Spectrometer not found: %s", self.descriptor.label())
            return None
        try:
            self._device = backend.Spectrometer(target)
            self._connected = True
            self._device.integration_time_micros(int(self._integration_time_ms * 1000))
            return self._device
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to connect OceanOptics spectrometer %s: %s", self.descriptor.label(), exc)
            self._device = None
            self._connected = False
            return None

    def _find_matching_device(self):
        backend = self._get_backend()
        if backend is None:
            return None
        desired_serial = str(self.descriptor.serial_number or "")
        desired_path = str(self.descriptor.path or "")
        try:
            devices = backend.list_devices()
        except BaseException as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to enumerate OceanOptics spectrometers: %s", exc)
            return None
        for dev in devices:
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
        if self._device is None:
            raise RuntimeError("Spectrometer is not connected")
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
