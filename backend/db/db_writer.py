import queue
from PySide6.QtCore import QThread, Signal

from backend.core.logger import get_logger

log = get_logger("db.writer")


class DBWriter(QThread):
    """
    Database yozish alohida queue thread.
    UI hech qachon qotmaydi.
    """

    task_done = Signal(str, object)
    task_failed = Signal(str, str)

    def __init__(self, db, maxsize: int = 10000):
        super().__init__()
        self.db = db
        self.queue = queue.Queue(maxsize=maxsize)
        self._running = False

    def submit(self, method_name: str, **kwargs):
        try:
            self.queue.put_nowait((method_name, kwargs))
        except queue.Full:
            log.error("DBWriter queue full: %s", method_name)

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        log.info("DBWriter started")

        while self._running or not self.queue.empty():
            try:
                method_name, kwargs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                method = getattr(self.db, method_name)
                result = method(**kwargs)
                self.task_done.emit(method_name, result)
            except Exception as e:
                log.exception("DBWriter task failed: %s", method_name)
                self.task_failed.emit(method_name, str(e))

        log.info("DBWriter stopped")