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

    def submit(self, method_name: str, *args, **kwargs):
        """
        Queue a DB operation.
        Supports both: submit("add_event", {...}) and submit("add_event", key=val)
        """
        if args and isinstance(args[0], dict):
            # Dict format: submit("add_event", {"key": "val"})
            self.queue.put((method_name, args[0]))
        elif kwargs:
            # Kwargs format: submit("add_event", key="val")
            self.queue.put((method_name, kwargs))
        else:
            self.queue.put((method_name, args))

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
                if isinstance(kwargs, dict):
                    try:
                        result = method(kwargs)
                    except TypeError:
                        result = method(**kwargs)
                else:
                    result = method(*kwargs) if kwargs else method()
                self.task_done.emit(method_name, result)
            except Exception as e:
                log.exception("DBWriter task failed: %s", method_name)
                self.task_failed.emit(method_name, str(e))

        log.info("DBWriter stopped")

