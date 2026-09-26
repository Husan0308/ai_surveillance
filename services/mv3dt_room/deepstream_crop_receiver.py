"""Receive event-driven DeepStream object crops over a local metadata socket."""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.mv3dt_room.identity_metrics import IdentityMetrics, monotonic_ns


_PREFIX = struct.Struct("<IQ")
_MAX_PACKET = 2 * 1024 * 1024


@dataclass
class CropRecord:
    header: dict
    image: np.ndarray
    received_monotonic_ns: int


class DeepStreamCropReceiver:
    """Bounded receiver for crops encoded from the active DeepStream buffer.

    The channel contains JPEG bytes only after DeepStream has attached
    ``NVDS_CROP_IMAGE_META`` to the tracked object. It never opens a camera or
    decodes an RTSP stream.
    """

    def __init__(self, socket_path: Path, metrics: IdentityMetrics, max_cache: int = 512) -> None:
        self.socket_path = Path(socket_path)
        self.metrics = metrics
        self.max_cache = max(32, int(max_cache))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="deepstream-crop-receiver", daemon=True)
        self._cache: dict[tuple[str, int, int], CropRecord] = {}
        self._order: list[tuple[str, int, int]] = []
        self._connect_attempts = 0
        self._connected = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _connect(self) -> socket.socket | None:
        while not self._stop.is_set():
            self._connect_attempts += 1
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            try:
                sock.connect(str(self.socket_path))
                sock.settimeout(1.0)
                self._connected = True
                self.metrics.inc("crop_socket_connected")
                return sock
            except OSError:
                sock.close()
                self._connected = False
                self.metrics.inc("crop_socket_connect_failures")
                self._stop.wait(0.1)
        return None

    def _store(self, record: CropRecord) -> None:
        header = record.header
        key = (str(header.get("camera_id", "")), int(header.get("native_track_id", -1)), int(header.get("frame_num", -1)))
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
            self._cache[key] = record
            self._order.append(key)
            while len(self._order) > self.max_cache:
                old = self._order.pop(0)
                self._cache.pop(old, None)

    def _consume(self, packet: bytes) -> None:
        if len(packet) < _PREFIX.size:
            self.metrics.inc("crop_failures")
            return
        header_len, jpeg_len = _PREFIX.unpack_from(packet)
        end_header = _PREFIX.size + int(header_len)
        if end_header > len(packet) or end_header + int(jpeg_len) != len(packet):
            self.metrics.inc("crop_failures")
            return
        try:
            header = json.loads(packet[_PREFIX.size:end_header].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.metrics.inc("crop_failures")
            return
        started = monotonic_ns()
        image = cv2.imdecode(np.frombuffer(packet[end_header:], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            self.metrics.inc("crop_failures", camera_id=header.get("camera_id"))
            return
        camera = str(header.get("camera_id", "UNMAPPED"))
        reason = str(header.get("reason", "unknown"))
        self.metrics.inc("crop_success", camera_id=camera)
        self.metrics.inc("crop_requests", camera_id=camera)
        self.metrics.inc(f"crop_reason_{reason}", camera_id=camera)
        self.metrics.inc(f"crop_camera_{camera}")
        encoded_ns = int(header.get("encoded_monotonic_ns", 0) or 0)
        if encoded_ns > 0:
            self.metrics.observe("crop_delivery_latency_ms", max(0.0, (started - encoded_ns) / 1_000_000.0), camera)
        source_ns = int(header.get("ntp_timestamp", 0) or 0)
        if source_ns > 0:
            now_wall = time.time_ns()
            self.metrics.observe("crop_age_at_receiver_ms", max(0.0, (now_wall - source_ns) / 1_000_000.0), camera)
        self._store(CropRecord(header, image, started))

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = self._connect()
            if sock is None:
                return
            try:
                while not self._stop.is_set():
                    try:
                        packet = sock.recv(_MAX_PACKET)
                    except socket.timeout:
                        continue
                    if not packet:
                        break
                    self._consume(packet)
            except OSError:
                self.metrics.inc("crop_socket_receive_failures")
            finally:
                sock.close()
                self._connected = False
                self.metrics.inc("crop_socket_disconnects")

    def get(self, observation: dict) -> CropRecord | None:
        key = (str(observation["camera_id"]), int(observation["native_track_id"]), int(observation["frame"]))
        with self._lock:
            record = self._cache.pop(key, None)
            if record is not None:
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
            return record

    def get_for_track(
        self,
        camera_id: str,
        native_track_id: int,
        generation: int,
        max_frame: int,
        min_frame: int,
    ) -> CropRecord | None:
        """Pop the newest buffered crop for this live track at/before state.

        The crop keeps its own exact DeepStream frame provenance. This lookup
        only bridges socket-delivery timing; it never crosses camera, native
        track, generation, or future-frame boundaries.
        """
        selected_key = None
        selected_frame = -1
        with self._lock:
            for key in self._order:
                if key[0] != str(camera_id) or key[1] != int(native_track_id):
                    continue
                record = self._cache.get(key)
                if record is None:
                    continue
                try:
                    frame = int(record.header["frame_num"])
                    record_generation = int(record.header["track_generation"])
                except (KeyError, TypeError, ValueError):
                    continue
                if record_generation != int(generation) or frame < int(min_frame) or frame > int(max_frame):
                    continue
                if frame > selected_frame:
                    selected_frame = frame
                    selected_key = key
            if selected_key is None:
                return None
            record = self._cache.pop(selected_key)
            try:
                self._order.remove(selected_key)
            except ValueError:
                pass
            return record

    def metrics_snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "connect_attempts": self._connect_attempts,
                "cached_crops": len(self._cache),
            }
