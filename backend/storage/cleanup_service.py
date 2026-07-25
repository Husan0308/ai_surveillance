import os
import time
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer

from backend.core.logger import get_logger

log = get_logger("storage.cleanup")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CleanupService(QObject):
    """
    Storage cleanup service.

    - retention days
    - max disk policy
    - old events cleanup
    """

    cleanup_done = Signal(dict)
    message = Signal(str, str)

    def __init__(self, config, db=None):
        super().__init__()

        self.config = config
        self.db = db

        self.snapshots_dir = self._abs(config.get("storage.snapshots_dir", "snapshots"))
        self.recordings_dir = self._abs(config.get("storage.recordings_dir", "recordings"))
        self.exports_dir = self._abs(config.get("storage.exports_dir", "exports"))
        self.logs_dir = self._abs(config.get("logging.dir", "logs"))
        self.backups_dir = self._abs(config.get("storage.backups_dir", "backups"))

        self.retention_days = int(config.get("storage.retention_days", 14))
        self.events_retention_days = int(config.get("events.retention_days", 30))

        self.auto_delete = bool(config.get("storage.auto_delete_old", True))
        self.max_disk_gb = int(config.get("storage.max_disk_gb", 500))

        # run every hour
        self._timer = QTimer(self)
        self._timer.setInterval(60 * 60 * 1000)
        self._timer.timeout.connect(self.cleanup_now)
        self._timer.start()

        log.info("CleanupService started")

    def _abs(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(BASE_DIR, path)

    # ---------------- cleanup ----------------
    def cleanup_now(self):
        deleted_count = 0
        freed_bytes = 0

        try:
            if self.auto_delete:
                c, b = self._delete_old(self.snapshots_dir, self.retention_days)
                deleted_count += c
                freed_bytes += b

                c, b = self._delete_old(self.recordings_dir, self.retention_days)
                deleted_count += c
                freed_bytes += b

                c, b = self._delete_old(self.exports_dir, self.retention_days)
                deleted_count += c
                freed_bytes += b

                c, b = self._delete_old(self.logs_dir, 7)
                deleted_count += c
                freed_bytes += b

                c, b = self._delete_old(self.backups_dir, 30)
                deleted_count += c
                freed_bytes += b

            # enforce max disk for recordings + snapshots
            c, b = self._enforce_max_disk()
            deleted_count += c
            freed_bytes += b

            # old events
            if self.db is not None:
                try:
                    self.db.delete_old_events(self.events_retention_days)
                except Exception as e:
                    log.error("delete_old_events error: %s", e)

            result = {
                "deleted_files": deleted_count,
                "freed_bytes": freed_bytes,
                "freed_mb": round(freed_bytes / (1024 * 1024), 1),
                "time": datetime.now().isoformat(),
            }

            self.cleanup_done.emit(result)
            log.info("Cleanup done: %s files, %s MB", deleted_count, result["freed_mb"])

        except Exception as e:
            log.error("cleanup_now error: %s", e)
            self.message.emit(f"Cleanup error: {e}", "error")

    def _delete_old(self, directory: str, days: int):
        deleted_count = 0
        freed_bytes = 0

        if not os.path.exists(directory):
            return deleted_count, freed_bytes

        now = time.time()
        max_age = days * 24 * 3600

        for root, dirs, files in os.walk(directory):
            for f in files:
                fp = os.path.join(root, f)

                try:
                    mtime = os.path.getmtime(fp)

                    if now - mtime > max_age:
                        size = os.path.getsize(fp)
                        os.remove(fp)

                        deleted_count += 1
                        freed_bytes += size

                except Exception as e:
                    log.error("delete_old file error: %s %s", fp, e)

        return deleted_count, freed_bytes

    def _dir_size(self, directory: str) -> int:
        total = 0

        if not os.path.exists(directory):
            return 0

        for root, dirs, files in os.walk(directory):
            for f in files:
                fp = os.path.join(root, f)

                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass

        return total

    def _list_files_by_age(self, directory: str):
        out = []

        if not os.path.exists(directory):
            return out

        for root, dirs, files in os.walk(directory):
            for f in files:
                fp = os.path.join(root, f)

                try:
                    out.append((os.path.getmtime(fp), os.path.getsize(fp), fp))
                except Exception:
                    pass

        out.sort(key=lambda x: x[0])
        return out

    def _enforce_max_disk(self):
        deleted_count = 0
        freed_bytes = 0

        max_bytes = self.max_disk_gb * 1024 * 1024 * 1024

        used = self._dir_size(self.recordings_dir) + self._dir_size(self.snapshots_dir)

        if used <= max_bytes:
            return deleted_count, freed_bytes

        # delete oldest recordings first
        files = self._list_files_by_age(self.recordings_dir)

        for _, size, fp in files:
            if used <= max_bytes:
                break

            try:
                os.remove(fp)
                used -= size
                deleted_count += 1
                freed_bytes += size
            except Exception as e:
                log.error("enforce_max_disk delete error: %s %s", fp, e)

        # then oldest snapshots if still over
        if used > max_bytes:
            files = self._list_files_by_age(self.snapshots_dir)

            for _, size, fp in files:
                if used <= max_bytes:
                    break

                try:
                    os.remove(fp)
                    used -= size
                    deleted_count += 1
                    freed_bytes += size
                except Exception as e:
                    log.error("enforce_max_disk snapshot delete error: %s %s", fp, e)

        return deleted_count, freed_bytes

    # ---------------- shutdown ----------------
    def shutdown(self):
        self._timer.stop()
        log.info("CleanupService stopped")