from PySide6 import QtCore
from queue import Queue, Empty

class SerialWorker(QtCore.QObject):
    finished = QtCore.Signal()
    errored = QtCore.Signal(str)

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self._q = Queue()
        self._running = True

    @QtCore.Slot()
    def loop(self):
        try:
            while self._running:
                try:
                    fn, args, kwargs = self._q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    fn(*args, **kwargs)
                except Exception as e:
                    self.errored.emit(str(e))
        finally:
            self.finished.emit()

    def enqueue(self, fn, *args, **kwargs):
        self._q.put((fn, args, kwargs))

    def stop(self):
        self._running = False
