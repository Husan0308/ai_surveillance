from datetime import datetime
from PySide6.QtCore import QObject, Signal


class EventBus(QObject):
    """
    Thread-safe event bus.
    Backend modullar event yuboradi.
    UI bridge signal orqali oladi.
    """

    event = Signal(dict)

    def publish(self, topic: str, **payload):
        data = {
            "topic": topic,
            "time": datetime.now().isoformat(),
        }
        data.update(payload)
        self.event.emit(data)

    def subscribe(self, callback):
        self.event.connect(callback)

    def unsubscribe(self, callback):
        try:
            self.event.disconnect(callback)
        except Exception:
            pass