import os
import threading

from PySide6.QtCore import QObject, Signal, QTimer

from backend.core.logger import get_logger

log = get_logger("core.performance")


try:
    import psutil
except Exception:
    psutil = None


class PerformanceMonitor(QObject):
    """
    Performance monitor.

    - active threads
    - DB writer queue
    - process CPU / RAM
    - camera health
    - AI workers
    """

    performance_updated = Signal(dict)

    def __init__(
        self,
        config,
        db_writer=None,
        camera_manager=None,
        identity_manager=None,
    ):
        super().__init__()

        self.config = config
        self.db_writer = db_writer
        self.camera_manager = camera_manager
        self.identity_manager = identity_manager

        self.ai_workers = {}

        self.process = None

        if psutil:
            try:
                self.process = psutil.Process(os.getpid())
                self.process.cpu_percent(interval=None)
            except Exception as e:
                log.error("psutil process init error: %s", e)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        log.info("PerformanceMonitor started")

    def set_ai_workers(self, ai_workers: dict):
        self.ai_workers = ai_workers or {}

    def _process_cpu(self):
        if self.process is None:
            return 0.0

        try:
            return float(self.process.cpu_percent(interval=None))
        except Exception:
            return 0.0

    def _process_ram_mb(self):
        if self.process is None:
            return 0.0

        try:
            return round(self.process.memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return 0.0

    def _tick(self):
        data = {
            "threads": threading.active_count(),
            "db_queue": 0,
            "process_cpu": self._process_cpu(),
            "process_ram_mb": self._process_ram_mb(),
            "ai_workers_total": len(self.ai_workers),
            "ai_workers_running": 0,
            "cameras": {},
            "occupancy": {},
        }

        # db queue
        try:
            if self.db_writer is not None and hasattr(self.db_writer, "queue"):
                data["db_queue"] = int(self.db_writer.queue.qsize())
        except Exception:
            pass

        # ai workers
        try:
            running = 0

            for worker in self.ai_workers.values():
                if worker is not None and worker.isRunning():
                    running += 1

            data["ai_workers_running"] = running
        except Exception:
            pass

        # cameras
        try:
            if self.camera_manager is not None:
                for camera_id, worker in self.camera_manager.workers.items():
                    data["cameras"][camera_id] = worker.health.metrics()
        except Exception:
            pass

        # occupancy
        try:
            if self.identity_manager is not None:
                for camera_id, state in self.identity_manager.states.items():
                    data["occupancy"][camera_id] = {
                        "occupancy": state.occupancy,
                        "known": state.known,
                        "unknown": state.unknown,
                        "active_visits": state.active_visits,
                    }
        except Exception:
            pass

        self.performance_updated.emit(data)

    def shutdown(self):
        self._timer.stop()
        log.info("PerformanceMonitor stopped")