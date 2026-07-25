from PySide6.QtCore import QObject, Signal

from backend.core.logger import get_logger

log = get_logger("features.alerts")


class AlertsService(QObject):
    """
    Alerts service.

    - watches EventsService
    - triggers sound for configured event types
    - emits desktop notification signal
    - keeps unread badge count
    """

    alert_added = Signal(dict)
    sound_requested = Signal(str)              # event type
    notification_requested = Signal(dict)      # event
    badge_updated = Signal(int)

    def __init__(self, config, events_service):
        super().__init__()

        self.config = config
        self.events_service = events_service

        self.sound_enabled = bool(config.get("alerts.sound_enabled", True))

        sound_events = config.get("alerts.sound_events", [])
        self.sound_events = set(sound_events)

        self.alerts = []
        self.max_alerts = 100

        self.unseen = 0

        try:
            self.events_service.event_added.connect(self.on_event)
        except Exception as e:
            log.error("AlertsService connect error: %s", e)

        log.info("AlertsService started")

    # ---------------- config ----------------
    def set_sound_enabled(self, enabled: bool):
        self.sound_enabled = bool(enabled)

    def set_sound_events(self, events_list):
        self.sound_events = set(events_list or [])

    # ---------------- main ----------------
    def on_event(self, event: dict):
        level = event.get("level", "info")

        if level not in ("warn", "err"):
            return

        self.alerts.insert(0, event)

        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[: self.max_alerts]

        self.unseen += 1
        self.badge_updated.emit(self.unseen)

        self.alert_added.emit(event)
        self.notification_requested.emit(event)

        event_type = event.get("type", "")

        if self.sound_enabled and event_type in self.sound_events:
            self.sound_requested.emit(event_type)

        log.info("Alert: %s | %s | %s", level, event_type, event.get("person_name"))

    # ---------------- unread ----------------
    def mark_all_read(self):
        self.unseen = 0
        self.badge_updated.emit(0)

    def get_alerts(self, limit: int = 50):
        return self.alerts[:limit]