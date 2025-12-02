import datetime
import sys
from PySide6 import QtCore


class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

    def _log(self, level: str, msg: str, *args) -> None:
        if args:
            msg = msg % args
        ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        prefix = f"[{ts}] {level}: " if level else f"[{ts}] "
        line = f"{prefix}{msg}"
        print(line, file=sys.stdout, flush=True)
        self.message.emit(line)

    def info(self, msg: str, *args) -> None:
        self._log("INFO", msg, *args)

    def warning(self, msg: str, *args) -> None:
        self._log("WARNING", msg, *args)

    def error(self, msg: str, *args) -> None:
        self._log("ERROR", msg, *args)


def log(msg: str) -> None:
    LOG.info(msg)


LOG = LogBus()
