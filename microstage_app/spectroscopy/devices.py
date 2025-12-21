from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import importlib
import importlib.util
import logging
import threading
import traceback
import numpy as np
from PySide6 import QtCore

from ..utils.workers import run_async


logger = logging.getLogger(__name__)


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


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
    connect_failed = QtCore.Signal(object, str)
    refresh_completed = QtCore.Signal(bool, str)

    def __init__(
        self,
        providers: Optional[Sequence[_SpectrometerProvider]] = None,
        *,
        auto_start: bool = True,
    ):
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
        self._monitor_paused_for_active = False
        self._refresh_thread: Optional[QtCore.QThread] = None
        self._refresh_worker: Optional[QtCore.QObject] = None
        self._refresh_timeout_timer: Optional[QtCore.QTimer] = None
        self._refresh_in_flight = False
        self._refresh_token: Optional[object] = None
        self._refresh_was_paused = False
        self._refresh_start_monitoring = False
        self._monitor_timer = QtCore.QTimer(self)
        self._monitor_timer.setInterval(2000)
        self._monitor_timer.timeout.connect(self._poll_devices)
        if auto_start:
            self.start_monitoring()

    def start_monitoring(self, *, force: bool = False) -> None:
        """Start periodic device polling if not already running."""
        if self._monitor_paused_for_active and not force:
            return
        if not self._monitor_timer.isActive():
            self._monitor_timer.start()

    def stop_monitoring(self) -> None:
        """Stop periodic device polling."""
        if self._monitor_timer.isActive():
            self._monitor_timer.stop()

    def _pause_monitoring_for_active_use(self) -> None:
        """Pause monitoring while a device is actively connected."""
        self._monitor_paused_for_active = True
        self.stop_monitoring()

    def _resume_monitoring_if_idle(self) -> None:
        """Restart monitoring once all active devices are disconnected."""
        if not self._active:
            self._monitor_paused_for_active = False
            self.start_monitoring()

    def shutdown(self) -> None:
        """Stop monitoring timers and ensure background polls are finished."""
        try:
            self._monitor_timer.timeout.disconnect(self._poll_devices)
        except (TypeError, RuntimeError):
            pass
        self.stop_monitoring()

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
        logger.info(
            "[thread=%s] Enumerating spectrometer devices",
            threading.get_ident(),
        )
        devices: List[SpectrometerDescriptor] = []
        for provider in self._providers:
            try:
                devices.extend(provider.list_devices())
            except BaseException as exc:  # pragma: no cover - defensive
                logger.warning("Failed to enumerate spectrometers: %s\n%s", exc, traceback.format_exc())
        logger.info(
            "[thread=%s] Enumerated %d spectrometer device(s)",
            threading.get_ident(),
            len(devices),
        )
        return devices

    def _reset_poll_state(self, *, stop_thread: bool = False) -> None:
        thread = self._poll_thread
        self._poll_thread = None
        self._poll_worker = None
        self._poll_in_flight = False
        if stop_thread and thread is not None and thread.isRunning():
            try:
                thread.quit()
                thread.wait()
            except BaseException:  # pragma: no cover - defensive
                logger.error("Failed to stop spectrometer poll thread:\n%s", traceback.format_exc())

    def _reset_refresh_state(self) -> None:
        self._refresh_thread = None
        self._refresh_worker = None
        self._refresh_in_flight = False
        self._refresh_token = None
        self._refresh_was_paused = False
        self._refresh_start_monitoring = False

    def _poll_devices(self) -> None:
        if self._monitor_paused_for_active:
            return
        if self._poll_in_flight:
            return
        self._poll_in_flight = True
        try:
            thread, worker = run_async(self._enumerate_devices, parent=self)
            self._poll_thread = thread
            self._poll_worker = worker
            worker.finished.connect(self._on_polled_devices)
        except BaseException:
            logger.warning("Unhandled error starting spectrometer poll:\n%s", traceback.format_exc())
            self._reset_poll_state(stop_thread=True)

    @QtCore.Slot(object, object)
    def _on_polled_devices(self, devices, err) -> None:
        try:
            if err:
                logger.warning("Spectrometer poll failed: %s\n%s", err, _format_exception(err))
                return
            self._update_devices(devices)
        except BaseException:
            logger.warning("Unhandled error handling polled devices:\n%s", traceback.format_exc())
        finally:
            self._reset_poll_state()

    def _update_devices(self, devices: List[SpectrometerDescriptor]) -> None:
        if devices == self._devices:
            return
        removed = [desc for desc in self._active if desc not in devices]
        for desc in removed:
            self.disconnect(desc)
        self._devices = list(devices)
        self.devices_changed.emit(list(devices))

    def refresh(self, *, start_monitoring: bool = False) -> List[SpectrometerDescriptor]:
        was_paused = self._monitor_paused_for_active
        if was_paused:
            self._monitor_paused_for_active = False
        if start_monitoring:
            self.start_monitoring(force=True)
        devices = self._enumerate_devices()
        self._update_devices(devices)
        if was_paused and self._active:
            self._pause_monitoring_for_active_use()
        elif not start_monitoring:
            self.stop_monitoring()
        return list(self._devices)

    def refresh_async(self, *, start_monitoring: bool = False, timeout_ms: int = 10000) -> bool:
        if self._refresh_in_flight:
            return False
        self._refresh_in_flight = True
        self._refresh_was_paused = self._monitor_paused_for_active
        self._refresh_start_monitoring = start_monitoring
        if self._refresh_was_paused:
            self._monitor_paused_for_active = False
        if start_monitoring:
            self.start_monitoring(force=True)
        token = object()
        self._refresh_token = token
        try:
            thread, worker = run_async(self._enumerate_devices, parent=self)
            self._refresh_thread = thread
            self._refresh_worker = worker
            worker.finished.connect(
                lambda devices, err, token=token: self._on_refresh_finished(token, devices, err)
            )
        except BaseException:
            logger.warning("Unhandled error starting spectrometer refresh:\n%s", traceback.format_exc())
            self._finalize_refresh(False, "Failed to start refresh")
            return True
        if timeout_ms > 0:
            if self._refresh_timeout_timer is None:
                self._refresh_timeout_timer = QtCore.QTimer(self)
                self._refresh_timeout_timer.setSingleShot(True)
            else:
                try:
                    self._refresh_timeout_timer.timeout.disconnect()
                except (TypeError, RuntimeError):
                    pass
            self._refresh_timeout_timer.timeout.connect(
                lambda token=token: self._on_refresh_timeout(token)
            )
            self._refresh_timeout_timer.start(timeout_ms)
        return True

    def _finalize_refresh(self, success: bool, message: str) -> None:
        if self._refresh_was_paused and self._active:
            self._pause_monitoring_for_active_use()
        elif not self._refresh_start_monitoring:
            self.stop_monitoring()
        self.refresh_completed.emit(success, message)
        self._reset_refresh_state()

    def _on_refresh_timeout(self, token: object) -> None:
        if token != self._refresh_token or not self._refresh_in_flight:
            return
        logger.warning("Spectrometer refresh timed out")
        self._finalize_refresh(False, "Device refresh timed out")

    @QtCore.Slot(object, object)
    def _on_refresh_finished(self, token: object, devices, err) -> None:
        if token != self._refresh_token:
            return
        if self._refresh_timeout_timer is not None:
            self._refresh_timeout_timer.stop()
        if err:
            logger.warning("Spectrometer refresh failed: %s\n%s", err, _format_exception(err))
            message = str(err) or "Device refresh failed"
            self._finalize_refresh(False, message)
            return
        try:
            self._update_devices(devices)
        except BaseException:
            logger.warning("Unhandled error applying refreshed devices:\n%s", traceback.format_exc())
            self._finalize_refresh(False, "Device refresh failed")
            return
        self._finalize_refresh(True, "Device refresh complete")

    def connect(self, descriptor: SpectrometerDescriptor) -> Optional[SpectrometerDevice]:
        if descriptor in self._active:
            device = self._active[descriptor]
            if not device.is_connected():
                try:
                    logger.info(
                        "[thread=%s] Connecting active spectrometer via %s: %s (%s)",
                        threading.get_ident(),
                        type(device).__name__,
                        descriptor.label(),
                        descriptor.path,
                    )
                    device.connect()
                    logger.debug(
                        "[thread=%s] Connected active spectrometer: %s (%s)",
                        threading.get_ident(),
                        descriptor.label(),
                        descriptor.path,
                    )
                except BaseException as exc:
                    logger.warning(
                        "Failed to connect spectrometer %s: %s\n%s",
                        descriptor.label(),
                        exc,
                        traceback.format_exc(),
                    )
                    self.connect_failed.emit(descriptor, str(exc))
                    self.device_connected.emit(descriptor, None)
                    return None
            self._last_active = descriptor
            self._pause_monitoring_for_active_use()
            self.device_connected.emit(descriptor, device)
            return device
        for provider in self._providers:
            try:
                candidates = provider.list_devices()
            except BaseException as exc:  # pragma: no cover - defensive
                logger.warning("Failed to enumerate spectrometers: %s\n%s", exc, traceback.format_exc())
                continue
            for candidate in candidates:
                if candidate == descriptor:
                    try:
                        logger.info(
                            "[thread=%s] Provider %s connecting spectrometer: %s (%s)",
                            threading.get_ident(),
                            type(provider).__name__,
                            descriptor.label(),
                            descriptor.path,
                        )
                        device = provider.connect(descriptor)
                        logger.debug(
                            "[thread=%s] Provider %s finished connect attempt for %s (%s)",
                            threading.get_ident(),
                            type(provider).__name__,
                            descriptor.label(),
                            descriptor.path,
                        )
                    except BaseException as exc:
                        logger.warning(
                            "Failed to create spectrometer %s: %s\n%s",
                            descriptor.label(),
                            exc,
                            traceback.format_exc(),
                        )
                        self.connect_failed.emit(descriptor, str(exc))
                        device = None
                    if device is None:
                        self.connect_failed.emit(descriptor, "Device unavailable after provider connect")
                        self.device_connected.emit(descriptor, None)
                        return None
                    try:
                        device.connect()
                    except BaseException as exc:
                        logger.warning(
                            "Failed to connect spectrometer %s: %s\n%s",
                            descriptor.label(),
                            exc,
                            traceback.format_exc(),
                        )
                        self.connect_failed.emit(descriptor, str(exc))
                        self.device_connected.emit(descriptor, None)
                        return None
                    self._active[descriptor] = device
                    self._locks.setdefault(descriptor, QtCore.QMutex())
                    self._last_active = descriptor
                    self._pause_monitoring_for_active_use()
                    self.device_connected.emit(descriptor, device)
                    return device
        logger.warning("Device not found: %s", descriptor.label())
        self.connect_failed.emit(descriptor, "Device not found during connect")
        self.device_connected.emit(descriptor, None)
        return None

    def disconnect(self, descriptor: Optional[SpectrometerDescriptor] = None) -> None:
        targets = [descriptor] if descriptor else list(self._active.keys())
        for desc in targets:
            device = self._active.pop(desc, None)
            lock = self._locks.get(desc)
            if device is None:
                self.device_connected.emit(desc, None)
                continue
            try:
                locker = QtCore.QMutexLocker(lock) if lock is not None else None
                try:
                    logger.info(
                        "[thread=%s] Disconnecting spectrometer via %s: %s (%s)",
                        threading.get_ident(),
                        type(device).__name__,
                        desc.label(),
                        desc.path,
                    )
                    device.disconnect()
                    logger.debug(
                        "[thread=%s] Disconnected spectrometer: %s (%s)",
                        threading.get_ident(),
                        desc.label(),
                        desc.path,
                    )
                except BaseException as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to disconnect spectrometer %s: %s", desc.label(), exc)
                finally:
                    if locker is not None:
                        locker.unlock()
            finally:
                self._locks.pop(desc, None)
                self.device_connected.emit(desc, None)
        if descriptor is None:
            self._last_active = None
        elif self._last_active == descriptor:
            self._last_active = next(iter(self._active), None)
        self._resume_monitoring_if_idle()

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
            logger.warning("Failed to check for seabreeze backend: %s", exc)
            return None
        if spec is None:
            return None
        try:
            return importlib.import_module("seabreeze.spectrometers")
        except BaseException as exc:  # pragma: no cover - defensive
            logger.warning("Failed to import seabreeze backend: %s", exc)
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
            logger.warning("Failed to enumerate OceanOptics spectrometers: %s", exc)
            return []
        for dev in devices:
            serial_number = str(getattr(dev, "serial_number", ""))
            path = str(getattr(dev, "path", getattr(dev, "device_node", "")))
            if not path:
                serial_stub = serial_number or getattr(dev, "model", "unknown")
                path = f"usb://{serial_stub}"
            descriptors.append(
                SpectrometerDescriptor(
                    model=getattr(dev, "model", "OceanOptics"),
                    serial_number=serial_number,
                    path=path,
                )
            )
        return descriptors

    def connect(self) -> Optional[object]:
        backend = self._get_backend()
        if backend is None:
            logger.warning("seabreeze backend unavailable for OceanOptics spectrometers")
            return None
        if self._device is not None:
            self.disconnect()
        try:
            target = self._find_matching_device()
        except BaseException as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to locate OceanOptics spectrometer %s: %s", self.descriptor.label(), exc
            )
            return None
        if target is None:
            logger.warning("Spectrometer not found: %s", self.descriptor.label())
            return None
        try:
            self._device = backend.Spectrometer(target)
            self._connected = True
            self._device.integration_time_micros(int(self._integration_time_ms * 1000))
            return self._device
        except BaseException as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to connect OceanOptics spectrometer %s: %s", self.descriptor.label(), exc
            )
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
            logger.warning("Failed to enumerate OceanOptics spectrometers: %s", exc)
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
