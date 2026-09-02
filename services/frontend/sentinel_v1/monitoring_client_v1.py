from __future__ import annotations

import json
import time
from typing import Any, Iterable

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtWebSockets import QWebSocket


class MonitoringMessageError(ValueError):
    pass


def validate_monitoring_snapshot(
    payload: Any, expected_camera_ids: Iterable[str]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise MonitoringMessageError("unsupported monitoring schema")
    sequence = payload.get("sequence")
    if not isinstance(sequence, int) or sequence < 0:
        raise MonitoringMessageError("invalid monitoring sequence")
    if not isinstance(payload.get("runtime"), dict) or not isinstance(payload.get("cameras"), list):
        raise MonitoringMessageError("runtime/cameras missing")
    camera_ids = [row.get("camera_id") for row in payload["cameras"] if isinstance(row, dict)]
    if camera_ids != list(expected_camera_ids):
        raise MonitoringMessageError("unexpected monitoring camera IDs")
    if payload.get("telemetry_status") == "fresh":
        for key in ("generated_monotonic_ns", "generated_epoch_ms"):
            if not isinstance(payload.get(key), int):
                raise MonitoringMessageError(f"invalid {key}")
    return payload


def parse_monitoring_message(text: str, expected_camera_ids: Iterable[str]) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MonitoringMessageError("invalid monitoring JSON") from exc
    return validate_monitoring_snapshot(payload, expected_camera_ids)


class MonitoringTelemetryClient(QObject):
    snapshotChanged = Signal(dict)
    statusChanged = Signal(str)

    def __init__(self, url: str, camera_ids: Iterable[str], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.url = QUrl(url)
        self.camera_ids = tuple(camera_ids)
        self.socket = QWebSocket()
        self.socket.setParent(self)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.setSingleShot(True)
        self.reconnect_timer.timeout.connect(self._open)
        self.freshness_timer = QTimer(self)
        self.freshness_timer.setInterval(500)
        self.freshness_timer.timeout.connect(self._check_freshness)
        self.socket.connected.connect(self._connected)
        self.socket.disconnected.connect(self._disconnected)
        self.socket.textMessageReceived.connect(self._message)
        self.socket.errorOccurred.connect(self._socket_error)
        self._backoff_ms = (500, 1000, 2000, 5000)
        self._backoff_index = 0
        self._running = False
        self._last_message_mono = 0.0
        self._last_local_status = "offline"

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.freshness_timer.start()
        self._open()

    def stop(self) -> None:
        self._running = False
        self.reconnect_timer.stop()
        self.freshness_timer.stop()
        self.socket.close()

    def _open(self) -> None:
        if self._running:
            self._emit_status("connecting")
            self.socket.open(self.url)

    def _connected(self) -> None:
        self._backoff_index = 0
        self._emit_status("connected")

    def _disconnected(self) -> None:
        self._emit_status("disconnected")
        self._schedule_reconnect()

    def _socket_error(self, _error) -> None:
        self._emit_status("disconnected")
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        if not self._running or self.reconnect_timer.isActive():
            return
        delay = self._backoff_ms[min(self._backoff_index, len(self._backoff_ms) - 1)]
        self._backoff_index = min(self._backoff_index + 1, len(self._backoff_ms) - 1)
        self.reconnect_timer.start(delay)

    def _message(self, text: str) -> None:
        try:
            payload = parse_monitoring_message(text, self.camera_ids)
        except MonitoringMessageError:
            self._emit_status("invalid")
            return
        self._last_message_mono = time.monotonic()
        self._emit_status("connected")
        self.snapshotChanged.emit(payload)

    def _check_freshness(self) -> None:
        if self._last_message_mono <= 0.0:
            return
        age = time.monotonic() - self._last_message_mono
        if age > 5.0:
            self._emit_status("offline")
        elif age > 2.0:
            self._emit_status("stale")

    def _emit_status(self, status: str) -> None:
        if status == self._last_local_status:
            return
        self._last_local_status = status
        self.statusChanged.emit(status)
