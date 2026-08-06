import time
from datetime import datetime

from PySide6.QtCore import QObject, Signal, QTimer

from backend.core.logger import get_logger

log = get_logger("features.events")


class EventsService(QObject):
    """
    Central events service.

    Event types:
        person_detected
        person_recognized
        unknown_detected
        camera_offline
        camera_online
        enrollment_completed
        snapshot
        error
        intrusion
        overstay
        system
    """

    event_added = Signal(dict)
    event_acked = Signal(dict)
    events_loaded = Signal()

    def __init__(self, config, db, db_writer=None, event_bus=None):
        super().__init__()

        self.config = config
        self.db = db
        self.db_writer = db_writer

        self.ui_limit = int(config.get("events.ui_limit", 300))
        self.retention_days = int(config.get("events.retention_days", 30))

        self.events = []

        self.snapshot_service = None

        self._cooldowns = {}

        if event_bus is not None:
            try:
                event_bus.subscribe(self.on_bus_event)
            except Exception as e:
                log.error("event_bus subscribe error: %s", e)

        # periodic cleanup
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.setInterval(60 * 60 * 1000)  # 1 hour
        self._cleanup_timer.timeout.connect(self.cleanup_old_events)
        self._cleanup_timer.start()

        self.load_from_db()

        log.info("EventsService started")

    # ---------------- date filter ----------------
    def load_by_date(self, date_str: str):
        """Load events for specific date. Format: YYYY-MM-DD"""
        self.load_from_db(date_str=date_str)
        log.info("Events loaded for date: %s (%d events)", date_str, len(self.events))

    def get_available_dates(self):
        """Get all dates that have events."""
        try:
            return self.db.get_event_dates()
        except Exception:
            return []

    # ---------------- setup ----------------
    def set_snapshot_service(self, snapshot_service):
        self.snapshot_service = snapshot_service

        try:
            snapshot_service.snapshot_taken.connect(self._on_snapshot_taken)
        except Exception as e:
            log.error("snapshot_service connect error: %s", e)

    # ---------------- load ----------------
    def load_from_db(self, limit: int = None, date_str: str = None):
        """
        Load events from DB.
        date_str: 'YYYY-MM-DD' format. None = bugungi kun.
        """
        try:
            if date_str is None:
                from datetime import datetime as _dt
                date_str = _dt.now().strftime("%Y-%m-%d")
            
            rows = self.db.get_events_by_date(date_str, limit or self.ui_limit)
            self.events = [self._row_to_event(r) for r in rows]
            self.current_date = date_str
            self.events_loaded.emit()
        except Exception as e:
            log.error("load_from_db error: %s", e)

    # ---------------- normalize ----------------
    def _normalize(self, event: dict) -> dict:
        e = dict(event)

        if "time" not in e or e["time"] is None:
            e["time"] = datetime.now()

        if isinstance(e["time"], str):
            try:
                e["time"] = datetime.fromisoformat(e["time"])
            except Exception:
                e["time"] = datetime.now()

        # aliases
        e["camera_id"] = e.get("camera_id") or e.get("cam") or "SYS"
        e["person_name"] = e.get("person_name") or e.get("person") or ""
        e["confidence"] = float(e.get("confidence", e.get("conf", 0.0)) or 0.0)

        # ✅ UI aliaslari (cam/person/conf) — barcha consumerlar uchun
        e["cam"] = e["camera_id"]
        e["person"] = e["person_name"]
        e["conf"] = e["confidence"]
        e["level"] = e.get("level", "info")
        e["type"] = e.get("type", "system")
        e["ack"] = bool(e.get("ack", False))
        e["snapshot_path"] = e.get("snapshot_path")

        return e

    def _to_db_dict(self, e: dict) -> dict:
        return {
            "time": e["time"].isoformat() if hasattr(e["time"], "isoformat") else str(e["time"]),
            "camera_id": e.get("camera_id"),
            "person_id": e.get("person_id"),
            "person_name": e.get("person_name"),
            "type": e.get("type"),
            "level": e.get("level", "info"),
            "confidence": float(e.get("confidence", 0.0)),
            "snapshot_path": e.get("snapshot_path"),
            "ack": e.get("ack", False),
            "extra": e.get("extra"),
        }

    def _row_to_event(self, row: dict) -> dict:
        e = dict(row)

        t = e.get("time")

        if isinstance(t, str):
            try:
                e["time"] = datetime.fromisoformat(t)
            except Exception:
                e["time"] = datetime.now()

        e["camera_id"] = e.get("camera_id") or "SYS"
        e["person_name"] = e.get("person_name") or ""
        e["confidence"] = float(e.get("confidence", 0.0) or 0.0)
        e["ack"] = bool(e.get("ack", False))

        return e

    # ---------------- publish ----------------
    def publish_event(self, event: dict, frame=None):
        print(f"[EventsService] 📨 RECEIVED: type={event.get('type','?')} person={event.get('person_name','?')} cam={event.get('camera_id','?')}", flush=True)
        e = self._normalize(event)

        # cooldown for very frequent events
        now = time.time()

        if e["type"] == "person_detected":
            key = ("person_detected", e["camera_id"], e.get("person_name"))
            if now - self._cooldowns.get(key, 0) < 300.0:
                return e
            self._cooldowns[key] = now

        # ✅ GLOBAL person cooldown: bir odam 30 sek ichida 1 marta
        # (barcha kameralar bo'ylab, camera_id dan qat'i nazar)
        if e["type"] in ("person_recognized", "recognized") and e.get("person_id"):
            key = ("person_recognized", e["camera_id"], e["person_id"])
            if now - self._cooldowns.get(key, 0) < 5.0:
                return e
            self._cooldowns[key] = now

        # ✅ Unknown cooldown: bir xil unknown 15 sek ichida 1 marta
        if e["type"] in ("unknown_detected", "unknown", "unknown_person"):
            key = ("unknown_detected", e.get("camera_id"), e.get("person_name"))
            if now - self._cooldowns.get(key, 0) < 5.0:
                return e
            self._cooldowns[key] = now

        # attach snapshot if frame provided
        if frame is not None and self.snapshot_service is not None:
            try:
                path = self.snapshot_service.take_snapshot_qimage(
                    camera_id=e["camera_id"],
                    qimage=frame,
                    person_name=e.get("person_name"),
                    emit_signal=False,
                )
                e["snapshot_path"] = path
            except Exception as ex:
                log.error("publish_event snapshot error: %s", ex)

        # database
        try:
            if self.db_writer is not None:
                self.db_writer.submit("add_event", self._to_db_dict(e))
            else:
                self.db.add_event(self._to_db_dict(e))
        except Exception as ex:
            log.error("publish_event db error: %s", ex)

        # memory
        self.events.insert(0, e)

        if len(self.events) > self.ui_limit:
            self.events = self.events[: self.ui_limit]

        print(f"[EventsService] 📢 EMIT event_added: type={e.get('type')} person={e.get('person_name')}", flush=True)
        self.event_added.emit(e)

        return e

    # ---------------- camera status ----------------
    def camera_status_changed(self, camera_id: str, online: bool):
        if online:
            self.publish_event({
                "type": "camera_online",
                "level": "ok",
                "camera_id": camera_id,
                "person_name": "Camera back online",
                "confidence": 1.0,
            })
        else:
            self.publish_event({
                "type": "camera_offline",
                "level": "err",
                "camera_id": camera_id,
                "person_name": "Camera offline",
                "confidence": 1.0,
            })

    # ---------------- enrollment ----------------
    def enrollment_completed(self, person_name: str, camera_id: str = "CAM-EN"):
        self.publish_event({
            "type": "enrollment_completed",
            "level": "ok",
            "camera_id": camera_id,
            "person_name": person_name,
            "confidence": 1.0,
        })

    # ---------------- error ----------------
    def error_event(self, message: str, camera_id: str = "SYS"):
        self.publish_event({
            "type": "error",
            "level": "err",
            "camera_id": camera_id,
            "person_name": message,
            "confidence": 1.0,
        })

    # ---------------- snapshot event ----------------
    def _on_snapshot_taken(self, info: dict):
        self.publish_event({
            "type": "snapshot",
            "level": "info",
            "camera_id": info.get("camera_id", "SYS"),
            "person_name": info.get("person_name") or "Snapshot captured",
            "confidence": 1.0,
            "snapshot_path": info.get("path"),
        })

    # ---------------- ack ----------------
    def ack_event(self, event: dict):
        if event is None:
            return

        event["ack"] = True

        try:
            if self.db_writer is not None:
                self.db_writer.submit(
                    "ack_event_by_key",
                    time_val=event.get("time"),
                    camera_id=event.get("camera_id"),
                    person_name=event.get("person_name"),
                )
            else:
                self.db.ack_event_by_key(
                    event.get("time"),
                    event.get("camera_id"),
                    event.get("person_name"),
                )
        except Exception as e:
            log.error("ack_event error: %s", e)

        self.event_acked.emit(event)

    # ---------------- query ----------------
    def get_events(self, event_type=None, level=None, search=None, limit=None):
        out = []

        for e in self.events:
            if event_type and e.get("type") != event_type:
                continue

            if level and e.get("level") != level:
                continue

            if search:
                q = search.lower()
                text = f"{e.get('camera_id','')} {e.get('person_name','')} {e.get('type','')}".lower()

                if q not in text:
                    continue

            out.append(e)

            if limit and len(out) >= limit:
                break

        return out

    # ---------------- event bus ----------------
    def on_bus_event(self, data: dict):
        try:
            topic = data.get("topic", "")

            if topic == "camera.status":
                self.camera_status_changed(
                    data.get("camera_id", "SYS"),
                    bool(data.get("online", False)),
                )
                return

            if topic == "enrollment.completed":
                self.enrollment_completed(
                    data.get("person_name", ""),
                    data.get("camera_id", "CAM-EN"),
                )
                return

            if topic == "system.error":
                self.error_event(
                    data.get("message", "System error"),
                    data.get("camera_id", "SYS"),
                )
                return

            # generic ai event
            if "type" in data:
                self.publish_event(data)

        except Exception as e:
            log.error("on_bus_event error: %s", e)

    # ---------------- cleanup ----------------
    def cleanup_old_events(self):
        try:
            self.db.delete_old_events(self.retention_days)
            log.info("Old events cleaned (%s days)", self.retention_days)
        except Exception as e:
            log.error("cleanup_old_events error: %s", e)