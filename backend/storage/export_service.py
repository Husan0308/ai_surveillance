import os
import csv
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger

log = get_logger("storage.export")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExportService(QObject):
    """
    Export service.

    - events CSV
    - visits CSV
    - analytics CSV
    """

    export_completed = Signal(str)
    message = Signal(str, str)

    def __init__(self, config, db):
        super().__init__()

        self.config = config
        self.db = db

        exports_dir = config.get("storage.exports_dir", "exports")

        if not os.path.isabs(exports_dir):
            exports_dir = os.path.join(BASE_DIR, exports_dir)

        self.dir = exports_dir
        os.makedirs(self.dir, exist_ok=True)

        log.info("ExportService started: %s", self.dir)

    def _timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- events ----------------
    def export_events_csv(self, events=None, path=None):
        try:
            if events is None:
                events = self.db.get_events(limit=10000)

            if path is None:
                path = os.path.join(self.dir, f"events_{self._timestamp()}.csv")

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "Time",
                    "Camera",
                    "Person ID",
                    "Person",
                    "Type",
                    "Level",
                    "Confidence",
                    "Snapshot",
                    "Ack",
                ])

                for e in events:
                    t = e.get("time")

                    if hasattr(t, "isoformat"):
                        t = t.isoformat()

                    writer.writerow([
                        t,
                        e.get("camera_id", ""),
                        e.get("person_id", ""),
                        e.get("person_name", ""),
                        e.get("type", ""),
                        e.get("level", ""),
                        e.get("confidence", ""),
                        e.get("snapshot_path", ""),
                        1 if e.get("ack") else 0,
                    ])

            self.export_completed.emit(path)
            self.message.emit(f"Events exported: {os.path.basename(path)}", "success")
            log.info("Events exported: %s", path)

            return path

        except Exception as e:
            log.error("export_events_csv error: %s", e)
            self.message.emit(f"Export failed: {e}", "error")
            return None

    # ---------------- visits ----------------
    def export_visits_csv(self, person_id=None, path=None):
        try:
            visits = self.db.get_visits(person_id=person_id, limit=10000)

            if path is None:
                path = os.path.join(self.dir, f"visits_{self._timestamp()}.csv")

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "Visit ID",
                    "Person ID",
                    "Name",
                    "Camera",
                    "Entered",
                    "Left",
                    "Duration Sec",
                ])

                for v in visits:
                    writer.writerow([
                        v.get("id", ""),
                        v.get("person_id", ""),
                        v.get("name", ""),
                        v.get("camera_id", ""),
                        v.get("entered_at", ""),
                        v.get("left_at", ""),
                        v.get("duration_sec", ""),
                    ])

            self.export_completed.emit(path)
            self.message.emit(f"Visits exported: {os.path.basename(path)}", "success")
            log.info("Visits exported: %s", path)

            return path

        except Exception as e:
            log.error("export_visits_csv error: %s", e)
            self.message.emit(f"Export failed: {e}", "error")
            return None

    # ---------------- analytics ----------------
    def export_analytics_csv(self, date_str=None, path=None):
        try:
            if date_str is None:
                date_str = datetime.now().strftime("%Y-%m-%d")

            rows = self.db.get_analytics_hourly(date_str)

            if path is None:
                path = os.path.join(self.dir, f"analytics_{date_str}_{self._timestamp()}.csv")

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow([
                    "Date",
                    "Hour",
                    "Camera",
                    "Occupancy Sum",
                    "Known Count",
                    "Unknown Count",
                    "Detection Count",
                    "Recognition Count",
                ])

                for r in rows:
                    writer.writerow([
                        r.get("date", ""),
                        r.get("hour", ""),
                        r.get("camera_id", ""),
                        r.get("occupancy_sum", ""),
                        r.get("known_count", ""),
                        r.get("unknown_count", ""),
                        r.get("detection_count", ""),
                        r.get("recognition_count", ""),
                    ])

            self.export_completed.emit(path)
            self.message.emit(f"Analytics exported: {os.path.basename(path)}", "success")
            log.info("Analytics exported: %s", path)

            return path

        except Exception as e:
            log.error("export_analytics_csv error: %s", e)
            self.message.emit(f"Export failed: {e}", "error")
            return None