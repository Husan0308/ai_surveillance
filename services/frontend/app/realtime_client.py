from __future__ import annotations

import json
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket


def websocket_url(api_base_url: str) -> str:
    parsed = urlsplit(api_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, "/api/v1/ws/monitoring", "", ""))


class MonitoringSocket(QObject):
    message_received = Signal(dict)
    state_changed = Signal(str)

    def __init__(
        self,
        api_base_url: str,
        reconnect_ms: int = 1000,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.url = websocket_url(api_base_url)
        self.socket = QWebSocket()
        self.socket.setParent(self)
        self.stopping = False
        self.socket.connected.connect(lambda: self.state_changed.emit("CONNECTED"))
        self.socket.disconnected.connect(self._on_disconnected)
        self.socket.textMessageReceived.connect(self._on_text_message)

        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.setSingleShot(True)
        self.reconnect_timer.setInterval(max(250, int(reconnect_ms)))
        self.reconnect_timer.timeout.connect(self.start)

    def start(self) -> None:
        if self.stopping:
            return
        self.state_changed.emit("CONNECTING")
        self.socket.open(QUrl(self.url))

    def stop(self) -> None:
        self.stopping = True
        self.reconnect_timer.stop()
        self.socket.close()

    def _on_disconnected(self) -> None:
        if self.stopping:
            return
        self.state_changed.emit("RECONNECTING")
        self.reconnect_timer.start()

    def _on_text_message(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            self.message_received.emit(payload)
