from __future__ import annotations

import http.client
import threading
from urllib.parse import urlsplit

from PySide6.QtGui import QImage


class SmoothMjpegReader:
    """One persistent HTTP MJPEG connection per camera, latest QImage only."""

    def __init__(self, camera_id: str, base_url: str) -> None:
        self.camera_id = camera_id
        parsed = urlsplit(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._image: QImage | None = None
        self._version = 0
        self.frames = 0
        self.reconnects = 0
        self.last_error = ""
        self._thread: threading.Thread | None = None
        self._connection: http.client.HTTPConnection | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"mjpeg-{self.camera_id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def join(self, timeout: float = 2.0) -> None:
        if self._thread:
            self._thread.join(timeout)

    def latest(self) -> tuple[QImage | None, int]:
        with self._lock:
            return self._image, self._version

    @staticmethod
    def _read_headers(response) -> dict[str, str]:
        headers: dict[str, str] = {}
        while True:
            line = response.readline()
            if not line:
                raise EOFError("MJPEG stream ended while reading headers")
            if line in (b"\r\n", b"\n"):
                return headers
            try:
                name, value = line.decode("latin-1").split(":", 1)
            except ValueError:
                continue
            headers[name.strip().lower()] = value.strip()

    def _consume(self, response) -> None:
        while not self._stop.is_set():
            line = response.readline()
            if not line:
                raise EOFError("MJPEG stream ended")
            if not line.startswith(b"--frame"):
                continue
            headers = self._read_headers(response)
            try:
                length = int(headers.get("content-length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 8 * 1024 * 1024:
                raise RuntimeError(f"invalid MJPEG content length: {length}")
            payload = response.read(length)
            if len(payload) != length:
                raise EOFError("short MJPEG frame")
            image = QImage.fromData(payload, "JPG")
            if image.isNull():
                raise RuntimeError("QImage JPEG decode failed")
            with self._lock:
                self._image = image
                self._version += 1
                self.frames += 1
                self.last_error = ""

    def _run(self) -> None:
        while not self._stop.is_set():
            connection = None
            response = None
            try:
                connection = http.client.HTTPConnection(self.host, self.port, timeout=4.0)
                self._connection = connection
                connection.request(
                    "GET",
                    f"/video/{self.camera_id}",
                    headers={"Connection": "keep-alive", "Cache-Control": "no-cache"},
                )
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(f"MJPEG HTTP {response.status}")
                self._consume(response)
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self.reconnects += 1
                        self.last_error = str(exc)
                    self._stop.wait(0.25)
            finally:
                self._connection = None
                if response is not None:
                    try:
                        response.close()
                    except Exception:
                        pass
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
