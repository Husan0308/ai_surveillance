from __future__ import annotations

import http.client
import threading

from PySide6.QtGui import QImage

from .dashboard import ML_HOST, ML_PORT


class SmoothFrameReader:
    """One persistent MJPEG HTTP connection per camera.

    The old reader performed one HTTP request for every JPEG. Six cameras at
    20 FPS meant roughly 120 request/response cycles per second. This reader
    opens `/video/{camera}` once, consumes multipart JPEGs continuously and
    still keeps only the newest decoded QImage for the Qt GUI thread.
    """

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._image: QImage | None = None
        self._version = -1
        self.frames = 0
        self.reconnects = 0
        self.decode_failures = 0
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"ui-mjpeg-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()

    def latest(self):
        with self._lock:
            return self._image, self._version

    @staticmethod
    def _read_headers(response):
        headers = {}
        while True:
            line = response.readline()
            if not line:
                raise EOFError("MJPEG stream ended while reading part headers")
            if line in (b"\r\n", b"\n"):
                return headers
            try:
                name, value = line.decode("latin-1").split(":", 1)
            except ValueError:
                continue
            headers[name.strip().lower()] = value.strip()

    def _consume_stream(self, response):
        version = self._version
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
                self.decode_failures += 1
                continue

            version += 1
            with self._lock:
                # Latest-only UI handoff: if Qt has not rendered the previous
                # image yet, this assignment replaces it rather than queueing it.
                self._image = image
                self._version = version
                self.frames += 1

    def _run(self):
        while not self._stop.is_set():
            connection = None
            response = None
            try:
                connection = http.client.HTTPConnection(
                    ML_HOST,
                    ML_PORT,
                    timeout=3.0,
                )
                connection.request(
                    "GET",
                    f"/video/{self.camera_id}",
                    headers={
                        "Connection": "keep-alive",
                        "Cache-Control": "no-cache",
                    },
                )
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(f"MJPEG HTTP {response.status}")
                self._consume_stream(response)
            except Exception:
                if not self._stop.is_set():
                    self.reconnects += 1
                    self._stop.wait(0.10)
            finally:
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
