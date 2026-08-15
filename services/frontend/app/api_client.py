from __future__ import annotations

import json

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


class ApiClient(QObject):
    api_health_received = Signal(dict)
    ml_health_received = Signal(dict)
    cameras_received = Signal(dict)
    request_failed = Signal(str, str)

    def __init__(self, base_url: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.manager = QNetworkAccessManager(self)
        self.manager.setTransferTimeout(2500)
        self._inflight: set[str] = set()

    def refresh_all(self) -> None:
        self._get("api_health", "/health")
        self._get("ml_health", "/api/v1/ml/health")
        self._get("cameras", "/api/v1/cameras")

    def _get(self, request_name: str, path: str) -> None:
        if request_name in self._inflight:
            return

        self._inflight.add(request_name)
        request = QNetworkRequest(QUrl(f"{self.base_url}{path}"))
        request.setRawHeader(b"Accept", b"application/json")
        reply = self.manager.get(request)
        reply.setProperty("request_name", request_name)
        reply.finished.connect(lambda r=reply: self._on_finished(r))

    def _on_finished(self, reply: QNetworkReply) -> None:
        request_name = str(reply.property("request_name"))
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.request_failed.emit(request_name, reply.errorString())
                return

            payload = bytes(reply.readAll()).decode("utf-8")
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("API response must be a JSON object")

            if request_name == "api_health":
                self.api_health_received.emit(data)
            elif request_name == "ml_health":
                self.ml_health_received.emit(data)
            elif request_name == "cameras":
                self.cameras_received.emit(data)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.request_failed.emit(request_name, str(exc))
        finally:
            self._inflight.discard(request_name)
            reply.deleteLater()
