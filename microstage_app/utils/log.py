import sys, threading, datetime
from PySide6 import QtCore

class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

LOG = LogBus()

def log(msg: str):
    ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{ts}] {msg}"
    print(line, file=sys.stdout, flush=True)
    LOG.message.emit(line)