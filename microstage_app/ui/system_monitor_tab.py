from PySide6 import QtWidgets, QtCore

import psutil

try:  # Optional GPU monitoring
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetMemoryInfo,
        NVMLError,
    )
    nvmlInit()
    _NVML_AVAILABLE = True
    _NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover - library or device missing
    _NVML_AVAILABLE = False
    _NVML_HANDLE = None


class SystemMonitorTab(QtWidgets.QWidget):
    """Display simple CPU/GPU utilization metrics."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)

        # CPU widgets
        self.cpu_label = QtWidgets.QLabel("CPU Usage: 0%")
        self.cpu_bar = QtWidgets.QProgressBar()
        self.cpu_bar.setRange(0, 100)
        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_bar)

        # GPU widgets (optional)
        if _NVML_AVAILABLE:
            self.gpu_label = QtWidgets.QLabel("GPU Usage: 0%")
            self.gpu_bar = QtWidgets.QProgressBar()
            self.gpu_bar.setRange(0, 100)

            self.gpu_mem_label = QtWidgets.QLabel("GPU Memory: 0 MiB / 0 MiB")
            self.gpu_mem_bar = QtWidgets.QProgressBar()
            self.gpu_mem_bar.setRange(0, 100)

            layout.addWidget(self.gpu_label)
            layout.addWidget(self.gpu_bar)
            layout.addWidget(self.gpu_mem_label)
            layout.addWidget(self.gpu_mem_bar)
        else:
            self.gpu_label = QtWidgets.QLabel("GPU metrics unavailable")
            layout.addWidget(self.gpu_label)

        layout.addStretch(1)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_metrics)

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.update_metrics()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        if _NVML_AVAILABLE:
            try:  # pragma: no cover - defensive cleanup
                nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def update_metrics(self) -> None:
        cpu = psutil.cpu_percent()
        self.cpu_bar.setValue(int(cpu))
        self.cpu_label.setText(f"CPU Usage: {cpu:.1f}%")

        if _NVML_AVAILABLE and _NVML_HANDLE is not None:
            try:
                util = nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
                mem = nvmlDeviceGetMemoryInfo(_NVML_HANDLE)

                self.gpu_bar.setValue(int(util.gpu))
                self.gpu_label.setText(f"GPU Usage: {util.gpu}%")

                # Memory info in MiB
                used = int(mem.used / 1024**2)
                total = int(mem.total / 1024**2)
                self.gpu_mem_bar.setMaximum(total)
                self.gpu_mem_bar.setValue(used)
                self.gpu_mem_label.setText(
                    f"GPU Memory: {used} MiB / {total} MiB"
                )
            except NVMLError:
                # NVML can occasionally fail if driver resets; ignore
                pass
