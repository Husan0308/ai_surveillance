from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


class MjpegStream(QObject):
    """Qt-native latest-only MJPEG consumer.

    Bytes arrive asynchronously through QNetworkReply. Complete JPEGs are parsed
    from the multipart stream, but only the newest completed JPEG is retained for
    decode. A slow GUI therefore drops old display frames instead of building a
    backlog.
    """

    frame_received = Signal(QImage)
    state_changed = Signal(str)

    def __init__(
        self,
        camera_id: str,
        api_base_url: str,
        stream_url: str,
        manager: QNetworkAccessManager,
        decode_interval_ms: int = 33,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self.api_base_url = api_base_url.rstrip("/")
        self.stream_url = stream_url
        self.manager = manager
        self.reply: QNetworkReply | None = None
        self.buffer = bytearray()
        self.pending_jpeg: bytes | None = None
        self.stopping = False
        self.frames = 0
        self.reconnects = 0
        self.last_error = ""

        self.decode_timer = QTimer(self)
        self.decode_timer.setInterval(max(20, int(decode_interval_ms)))
        self.decode_timer.timeout.connect(self._flush_latest)
        self.decode_timer.start()

        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.setSingleShot(True)
        self.reconnect_timer.setInterval(500)
        self.reconnect_timer.timeout.connect(self.start)

    def _absolute_url(self) -> str:
        if self.stream_url.startswith("http://") or self.stream_url.startswith("https://"):
            return self.stream_url
        path = self.stream_url if self.stream_url.startswith("/") else f"/{self.stream_url}"
        return f"{self.api_base_url}{path}"

    def start(self) -> None:
        if self.stopping or self.reply is not None:
            return
        self.state_changed.emit("CONNECTING")
        request = QNetworkRequest(QUrl(self._absolute_url()))
        request.setRawHeader(b"Accept", b"multipart/x-mixed-replace")
        request.setRawHeader(b"Cache-Control", b"no-cache")
        request.setTransferTimeout(0)
        reply = self.manager.get(request)
        self.reply = reply
        reply.readyRead.connect(self._on_ready_read)
        reply.finished.connect(self._on_finished)
        reply.errorOccurred.connect(self._on_error)

    def stop(self) -> None:
        self.stopping = True
        self.reconnect_timer.stop()
        self.decode_timer.stop()
        reply = self.reply
        self.reply = None
        if reply is not None:
            reply.abort()
            reply.deleteLater()

    def _on_error(self, _error) -> None:
        reply = self.reply
        if reply is not None:
            self.last_error = reply.errorString()

    def _on_finished(self) -> None:
        reply = self.reply
        self.reply = None
        if reply is not None:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.last_error = reply.errorString()
            reply.deleteLater()
        if self.stopping:
            return
        self.reconnects += 1
        self.state_changed.emit("RECONNECTING")
        self.reconnect_timer.start()

    def _on_ready_read(self) -> None:
        reply = self.reply
        if reply is None:
            return
        chunk = bytes(reply.readAll())
        if not chunk:
            return
        self.buffer.extend(chunk)

        # MJPEG JPEG payloads are self-delimiting. Keep parsing complete frames,
        # replacing pending_jpeg each time so only the freshest survives.
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                if len(self.buffer) > 512 * 1024:
                    del self.buffer[:-2]
                return
            if start > 0:
                del self.buffer[:start]
            end = self.buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.buffer) > 8 * 1024 * 1024:
                    self.buffer.clear()
                return
            end += 2
            self.pending_jpeg = bytes(self.buffer[:end])
            del self.buffer[:end]

    def _flush_latest(self) -> None:
        payload = self.pending_jpeg
        self.pending_jpeg = None
        if payload is None:
            return
        image = QImage.fromData(payload, "JPG")
        if image.isNull():
            self.last_error = "QImage JPEG decode failed"
            return
        self.frames += 1
        self.last_error = ""
        self.state_changed.emit("LIVE")
        self.frame_received.emit(image)
