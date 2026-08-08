import threading
import weakref
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal


class RequestSignals(QObject):
    success = Signal(object)
    error = Signal(str)
    finished = Signal(object)


class RequestTask(QRunnable):
    def __init__(self, call):
        super().__init__()
        self.call = call
        self.signals = RequestSignals()
        self.cancelled = threading.Event()
        self._parent_ref = None

    def set_parent(self, parent):
        self._parent_ref = weakref.ref(parent)
        self.signals.setParent(parent)

    def cancel(self):
        self.cancelled.set()

    def run(self):
        try:
            result = self.call()
            error = None
        except Exception as exc:
            result = None
            error = str(exc)

        if self.cancelled.is_set():
            return

        # Check if parent still exists
        if self._parent_ref and self._parent_ref() is None:
            return

        if error is None:
            self.signals.success.emit(result)
        else:
            self.signals.error.emit(error)
        self.signals.finished.emit(self)


class AsyncApi(QObject):
    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.pool = QThreadPool(self)
        self._tasks = set()
        self._closing = False

    def submit(self, call, on_success=None, on_error=None):
        if self._closing:
            return None
        task = RequestTask(call)
        task.set_parent(self)
        self._tasks.add(task)
        if on_success:
            task.signals.success.connect(on_success)
        if on_error:
            task.signals.error.connect(on_error)
        task.signals.finished.connect(self._finished)
        self.pool.start(task)
        return task

    def _finished(self, task):
        self._tasks.discard(task)

    def shutdown(self, timeout_ms=2000):
        if self._closing:
            return
        self._closing = True
        # Cancel all pending tasks
        for task in tuple(self._tasks):
            task.cancel()
            try:
                task.signals.finished.disconnect(self._finished)
            except Exception:
                pass
        # Clear pool and wait
        self.pool.clear()
        self.pool.waitForDone(timeout_ms)
        self._tasks.clear()
