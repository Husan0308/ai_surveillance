from __future__ import annotations

import threading
import time

from PySide6.QtGui import QImage

from shared.mmap_frame import MmapFrameReader


class SmoothMmapFrameReader:
    """Decode-free latest-frame reader for the local Qt camera wall."""

    def __init__(self, camera_id: str):
        self.camera_id = str(camera_id)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._image: QImage | None = None
        self._version = -1
        self._thread = None
        self.frames = 0
        self.reconnects = 0
        self.copy_failures = 0
        self.last_frame_age_ms = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"ui-mmap-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(1.0)

    def latest(self):
        with self._lock:
            return self._image, self._version

    def _run(self):
        reader = MmapFrameReader(self.camera_id)
        last_sequence: int | None = None
        last_change = time.monotonic()
        try:
            while not self._stop.is_set():
                if not reader.mapping_is_current():
                    if not reader.attach():
                        self.reconnects += 1
                        self._stop.wait(0.08)
                        continue
                    last_sequence = None
                    last_change = time.monotonic()

                packet = reader.snapshot(last_sequence)
                if packet is None:
                    # Backend restart replaces the mmap inode. Re-attach instead
                    # of sitting forever on the old mapping.
                    if time.monotonic() - last_change > 1.0 and not reader.mapping_is_current():
                        reader.close()
                    self._stop.wait(0.004)
                    continue

                last_sequence = packet.sequence
                last_change = time.monotonic()
                if packet.channels != 3:
                    self.copy_failures += 1
                    continue

                try:
                    image = QImage(
                        packet.payload,
                        packet.width,
                        packet.height,
                        packet.width * packet.channels,
                        QImage.Format.Format_BGR888,
                    ).copy()
                except Exception:
                    self.copy_failures += 1
                    continue
                if image.isNull():
                    self.copy_failures += 1
                    continue

                with self._lock:
                    self._image = image
                    self._version = packet.sequence
                    self.frames += 1
                    self.last_frame_age_ms = packet.age_ms
        finally:
            reader.close()
