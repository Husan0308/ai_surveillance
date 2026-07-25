import os
import logging

from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger, ROOT_LOGGER

log = get_logger("core.log_service")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LogService(QObject):
    """
    Log service.

    - Qt signal bilan loglarni UI ga yuboradi
    - recent logs ni o'qiydi
    """

    log_message = Signal(str)

    def __init__(self, config):
        super().__init__()

        logs_dir = config.get("logging.dir", "logs")

        if not os.path.isabs(logs_dir):
            logs_dir = os.path.join(BASE_DIR, logs_dir)

        os.makedirs(logs_dir, exist_ok=True)

        self.log_file = os.path.join(logs_dir, "app.log")

        self.handler = _QtLogHandler(self)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.handler.setFormatter(formatter)

        root = logging.getLogger(ROOT_LOGGER)
        root.addHandler(self.handler)

        log.info("LogService started")

    def get_recent_logs(self, lines: int = 200):
        try:
            if not os.path.exists(self.log_file):
                return []

            with open(self.log_file, "r", encoding="utf-8", errors="ignore") as f:
                return f.readlines()[-lines:]

        except Exception as e:
            log.error("get_recent_logs error: %s", e)
            return []

    def get_log_file_path(self):
        return self.log_file


class _QtLogHandler(logging.Handler):
    def __init__(self, service: LogService):
        super().__init__()
        self.service = service

    def emit(self, record):
        try:
            msg = self.format(record)
            self.service.log_message.emit(msg)
        except Exception:
            pass