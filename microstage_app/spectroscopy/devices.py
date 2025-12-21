from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import importlib
import importlib.util
import logging
import multiprocessing
import queue
import sys
import threading
import time
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
        self._device_providers: dict[SpectrometerDescriptor, _SpectrometerProvider] = {}
        self._provider_devices: dict[_SpectrometerProvider, list[SpectrometerDescriptor]] = {}
        self._last_active: Optional[SpectrometerDescriptor] = None
        self._poll_thread: Optional[QtCore.QThread] = None
        self._poll_worker: Optional[QtCore.QObject] = None
        self._poll_in_flight = False
        self._monitor_paused_for_active = False
        self._refresh_thread: Optional[QtCore.QThread] = None
        self._refresh_worker: Optional[QtCore.QObject] = None
        self._refresh_timeout_timer: Optional[QtCore.QTimer] = None
        self._refresh_in_flight = False
        self._refresh_was_paused = False
        self._refresh_start_monitoring = False
        self._last_refresh_successful = False
        self._last_enumeration_timeouts: list[str] = []
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
            self._stop_qthread(thread, "poll")
            self._poll_thread = None
        self._poll_worker = None
        self._poll_in_flight = False

        refresh_worker = self._refresh_worker
        if refresh_worker is not None:
            try:
                refresh_worker.finished.disconnect(self._on_refresh_finished)
            except (TypeError, RuntimeError):
                pass

        if self._refresh_timeout_timer is not None:
            try:
                self._refresh_timeout_timer.timeout.disconnect(self._on_refresh_timeout)
            except (TypeError, RuntimeError):
                pass
            self._refresh_timeout_timer.stop()

        refresh_thread = self._refresh_thread
        if refresh_thread is not None:
            self._stop_qthread(refresh_thread, "refresh")
            self._refresh_thread = None
        self._refresh_worker = None
        self._refresh_in_flight = False
        self._refresh_was_paused = False
        self._refresh_start_monitoring = False
        self._device_providers.clear()
        self._provider_devices.clear()

    close = shutdown

    @property
    def devices(self) -> List[SpectrometerDescriptor]:
        return list(self._devices)

    def _enumerate_devices(self, *, capture_timeouts: bool = False) -> List[SpectrometerDescriptor]:
        logger.info(
            "[thread=%s] Enumerating spectrometer devices",
            threading.get_ident(),
        )
        devices: List[SpectrometerDescriptor] = []
        device_providers: dict[SpectrometerDescriptor, _SpectrometerProvider] = {}
        timeouts: list[str] = []
        slow_threshold_s = 2.0
        for provider in self._providers:
            provider_name = provider.__class__.__name__
            logger.info("Starting spectrometer enumeration for %s", provider_name)
            start_time = time.perf_counter()
            try:
                provider_devices = provider.list_devices()
                if getattr(provider, "timed_out", False):
                    timeouts.append(provider_name)
                    provider_devices = self._provider_devices.get(provider, [])
                else:
                    self._provider_devices[provider] = list(provider_devices)
                devices.extend(provider_devices)
                for descriptor in provider_devices:
                    device_providers[descriptor] = provider
            except BaseException as exc:  # pragma: no cover - defensive
                provider_devices = self._provider_devices.get(provider, [])
                logger.warning(
                    "Spectrometer enumeration failed for %s (%s: %s)",
                    provider_name,
                    type(exc).__name__,
                    exc,
                )
                logger.debug(
                    "Spectrometer enumeration traceback for %s:\n%s",
                    provider_name,
                    traceback.format_exc(),
                )
                devices.extend(provider_devices)
                for descriptor in provider_devices:
                    device_providers[descriptor] = provider
            finally:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Completed spectrometer enumeration for %s in %.2fs",
                    provider_name,
                    elapsed,
                )
                if elapsed > slow_threshold_s:
                    logger.warning(
                        "Spectrometer enumeration slow: %s took %.1fs",
                        provider_name,
                        elapsed,
                    )
        logger.info(
            "[thread=%s] Enumerated %d spectrometer device(s)",
            threading.get_ident(),
            len(devices),
        )
        self._device_providers = device_providers
        if capture_timeouts:
            self._last_enumeration_timeouts = timeouts
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

    def _stop_qthread(self, thread: Optional[QtCore.QThread], label: str) -> None:
        if thread is None:
            return
        if QtCore.QThread.currentThread() is self.thread():
            self._stop_qthread_on_owner(thread, label)
        else:
            QtCore.QMetaObject.invokeMethod(
                self,
                "_stop_qthread_on_owner",
                QtCore.Qt.BlockingQueuedConnection,
                QtCore.Q_ARG(object, thread),
                QtCore.Q_ARG(str, label),
            )

    @QtCore.Slot(object, str)
    def _stop_qthread_on_owner(self, thread: Optional[QtCore.QThread], label: str) -> None:
        if thread is None:
            return
        if not thread.isRunning():
            return
        thread.quit()
        if not thread.wait(2000):
            logger.warning("Spectrometer %s thread did not stop; terminating", label)
            thread.terminate()
            thread.wait()

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

    def refresh(
        self,
        *,
        start_monitoring: bool = False,
        timeout_ms: int = 10000,
    ) -> List[SpectrometerDescriptor]:
        if QtCore.QCoreApplication.instance() is None:
            was_paused = self._monitor_paused_for_active
            if was_paused:
                self._monitor_paused_for_active = False
            if start_monitoring:
                self.start_monitoring(force=True)
            devices: List[SpectrometerDescriptor] = []
            err: Optional[BaseException] = None
            done = threading.Event()

            def worker() -> None:
                nonlocal devices, err
                try:
                    devices = self._enumerate_devices(capture_timeouts=True)
                except BaseException as exc:  # pragma: no cover - defensive
                    err = exc
                finally:
                    done.set()

            thread = threading.Thread(target=worker, name="SpectrometerRefresh", daemon=True)
            thread.start()
            if not done.wait(max(0.0, timeout_ms / 1000.0)):
                logger.warning("Spectrometer refresh timed out")
                if was_paused and self._active:
                    self._pause_monitoring_for_active_use()
                elif not start_monitoring:
                    self.stop_monitoring()
                return list(self._devices)
            if err:
                logger.warning("Spectrometer refresh failed: %s\n%s", err, _format_exception(err))
                if was_paused and self._active:
                    self._pause_monitoring_for_active_use()
                elif not start_monitoring:
                    self.stop_monitoring()
                return list(self._devices)
            self._update_devices(devices)
            if was_paused and self._active:
                self._pause_monitoring_for_active_use()
            elif not start_monitoring:
                self.stop_monitoring()
            return list(self._devices)

        loop = QtCore.QEventLoop()
        completed = {"done": False}

        def _on_completed(_success: bool, _message: str) -> None:
            completed["done"] = True
            loop.quit()

        self.refresh_completed.connect(_on_completed)
        try:
            started = self.refresh_async(start_monitoring=start_monitoring, timeout_ms=timeout_ms)
            if not started:
                return list(self._devices)
            if not completed["done"]:
                loop.exec()
        finally:
            try:
                self.refresh_completed.disconnect(_on_completed)
            except (TypeError, RuntimeError):
                pass
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
        try:
            thread, worker = run_async(self._enumerate_devices, capture_timeouts=True, parent=self)
            self._refresh_thread = thread
            self._refresh_worker = worker
            worker.finished.connect(self._on_refresh_finished)
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
            self._refresh_timeout_timer.timeout.connect(self._on_refresh_timeout)
            self._refresh_timeout_timer.start(timeout_ms)
        return True

    def _finalize_refresh(self, success: bool, message: str) -> None:
        self._last_refresh_successful = success
        if self._refresh_timeout_timer is not None:
            self._refresh_timeout_timer.stop()
        if self._refresh_was_paused and self._active:
            self._pause_monitoring_for_active_use()
        elif not self._refresh_start_monitoring:
            self.stop_monitoring()
        self.refresh_completed.emit(success, message)
        self._reset_refresh_state()

    def _on_refresh_timeout(self) -> None:
        if not self._refresh_in_flight or self._refresh_worker is None:
            return
        logger.warning("Spectrometer refresh timed out")
        self._stop_qthread(self._refresh_thread, "refresh")
        self._finalize_refresh(False, "Driver did not respond")

    @QtCore.Slot(object, object)
    def _on_refresh_finished(self, devices, err) -> None:
        if not self._refresh_in_flight or self.sender() is not self._refresh_worker:
            return
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
        if "OceanOpticsSpectrometerProvider" in self._last_enumeration_timeouts:
            self._finalize_refresh(True, "Ocean Optics driver did not respond; refresh timed out")
        else:
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
        provider = self._device_providers.get(descriptor)
        if provider is None:
            logger.warning("Device not found in provider cache: %s", descriptor.label())
            self.connect_failed.emit(descriptor, "Device not found during connect")
            self.device_connected.emit(descriptor, None)
            return None
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


def _ocean_optics_enumerate_worker(
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        devices = OceanOpticsSpectrometer.enumerate()
    except BaseException as exc:  # pragma: no cover - defensive
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    result_queue.put(("ok", devices))


class OceanOpticsSpectrometerProvider:
    def __init__(self, *, timeout_s: float = 5.0) -> None:
        self._timeout_s = float(timeout_s)
        self._timed_out = False

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    def list_devices(self) -> List[SpectrometerDescriptor]:
        self._timed_out = False
        is_main_thread = threading.current_thread() is threading.main_thread()
        is_windows = sys.platform == "win32"
        if not is_main_thread:
            app = QtCore.QCoreApplication.instance() if is_windows else None
            if app is not None:
                result: dict[str, object] = {"devices": None, "error": None}

                def _enumerate_on_main_thread() -> None:
                    try:
                        result["devices"] = list(OceanOpticsSpectrometer.enumerate())
                    except BaseException as exc:  # pragma: no cover - defensive
                        result["error"] = exc

                QtCore.QMetaObject.invokeMethod(
                    app,
                    _enumerate_on_main_thread,
                    QtCore.Qt.BlockingQueuedConnection,
                )
                if result["error"] is not None:
                    exc = result["error"]
                    logger.warning(
                        "Failed to enumerate OceanOptics spectrometers: %s",
                        f"{type(exc).__name__}: {exc}",
                    )
                    return []
                return list(result["devices"] or [])
            try:
                return list(OceanOpticsSpectrometer.enumerate())
            except BaseException as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to enumerate OceanOptics spectrometers: %s",
                    f"{type(exc).__name__}: {exc}",
                )
                return []
        ctx = multiprocessing.get_context("spawn")
        result_queue: multiprocessing.Queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_ocean_optics_enumerate_worker,
            args=(result_queue,),
            name="OceanOpticsEnumerate",
            daemon=True,
        )
        process.start()
        process.join(self._timeout_s)
        if process.is_alive():
            self._timed_out = True
            process.terminate()
            process.join()
            logger.warning(
                "Ocean Optics backend unresponsive; enumeration timed out after %.1fs",
                self._timeout_s,
            )
            return []
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty:
            return []
        if status == "error":
            logger.warning("Failed to enumerate OceanOptics spectrometers: %s", payload)
            return []
        return list(payload)

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
