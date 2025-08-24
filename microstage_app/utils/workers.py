from PySide6 import QtCore

class FuncWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)  # result, error
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn; self.args = args; self.kw = kwargs
    @QtCore.Slot()
    def run(self):
        try:
            res = self.fn(*self.args, **self.kw)
            self.finished.emit(res, None)
        except Exception as e:
            self.finished.emit(None, e)

def run_async(fn, *args, **kwargs):
    """Run fn(*) on a fresh QThread; return (thread, worker). Caller must keep refs until finished."""
    thread = QtCore.QThread()
    worker = FuncWorker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
