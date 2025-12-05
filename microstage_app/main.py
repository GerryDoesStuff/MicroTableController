import argparse
import logging
import sys
import threading

from PySide6 import QtWidgets

from .ui.main_window import MainWindow
from .utils.log import setup_logging


def _log_uncaught(exc_type, exc_value, exc_traceback):  # pragma: no cover - global hook
    logger = logging.getLogger("microstage_app")
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="MicroStage controller UI")
    parser.set_defaults(auto_connect_on_start=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--auto-connect",
        dest="auto_connect_on_start",
        action="store_true",
        help="Force auto-connecting devices on startup.",
    )
    group.add_argument(
        "--no-auto-connect",
        dest="auto_connect_on_start",
        action="store_false",
        help="Skip auto-connecting devices on startup.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity for stdout and the UI log panel.",
    )
    args, qt_args = parser.parse_known_args(argv)

    setup_logging(level=args.log_level)
    sys.excepthook = _log_uncaught
    threading.excepthook = lambda args: _log_uncaught(args.exc_type, args.exc_value, args.exc_traceback)

    app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("MicroStage App")
    win = MainWindow(auto_connect_on_start=args.auto_connect_on_start)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
