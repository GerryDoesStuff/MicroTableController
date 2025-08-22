import sys, threading, datetime, os
from pathlib import Path
from PySide6 import QtCore
import logging
from logging.handlers import RotatingFileHandler

class LogBus(QtCore.QObject):
    message = QtCore.Signal(str)

LOG = LogBus()

# File logger setup (rotating)
_logger = logging.getLogger("microstage")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _logdir = Path('logs')
    try:
        _logdir.mkdir(parents=True, exist_ok=True)
        _fh = RotatingFileHandler(_logdir / 'microstage.log', maxBytes=1_000_000, backupCount=5, encoding='utf-8')
        _fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        _logger.addHandler(_fh)
    except Exception:
        pass

def log(msg: str):
    ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    line = f"[{ts}] {msg}"
    print(line, file=sys.stdout, flush=True)
    try:
        _logger.info(msg)
    except Exception:
        pass

    LOG.message.emit(line)