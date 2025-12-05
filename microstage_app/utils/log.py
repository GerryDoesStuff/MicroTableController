import logging
import sys
from PySide6 import QtCore


class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

    def emit(self, line: str) -> None:
        """Forward a formatted log line to the UI."""

        self.message.emit(line)

    def info(self, msg: str, *args) -> None:
        logging.getLogger("microstage_app").info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        logging.getLogger("microstage_app").warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        logging.getLogger("microstage_app").error(msg, *args)


class LogBusHandler(logging.Handler):
    """Logging handler that forwards records to the Qt log bus."""

    def __init__(self, bus: LogBus):
        super().__init__()
        self._bus = bus

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - Qt signal
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        try:
            self._bus.emit(line)
        except RuntimeError:
            # The Qt event loop may be shutting down; ignore safely.
            pass


_QT_LEVEL_MAP = {
    QtCore.QtMsgType.QtDebugMsg: logging.DEBUG,
    QtCore.QtMsgType.QtInfoMsg: logging.INFO,
    QtCore.QtMsgType.QtWarningMsg: logging.WARNING,
    QtCore.QtMsgType.QtCriticalMsg: logging.ERROR,
    QtCore.QtMsgType.QtFatalMsg: logging.CRITICAL,
}


def _qt_message_handler(mode, context, message):  # pragma: no cover - Qt callback
    level = _QT_LEVEL_MAP.get(mode, logging.INFO)
    logger = logging.getLogger("qt")
    prefix = context.category if context and context.category else "Qt"
    logger.log(level, "%s: %s", prefix, message)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure Python logging to forward to stdout and the UI log bus."""

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    bus_handler = LogBusHandler(LOG)
    bus_handler.setFormatter(fmt)
    root.addHandler(bus_handler)

    QtCore.qInstallMessageHandler(_qt_message_handler)


def log(msg: str, *args) -> None:
    logging.getLogger("microstage_app").info(msg, *args)


LOG = LogBus()
