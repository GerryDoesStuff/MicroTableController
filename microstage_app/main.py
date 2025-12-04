import argparse
import sys

from PySide6 import QtWidgets

from .ui.main_window import MainWindow

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
    args, qt_args = parser.parse_known_args(argv)

    app = QtWidgets.QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("MicroStage App")
    win = MainWindow(auto_connect_on_start=args.auto_connect_on_start)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
