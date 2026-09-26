#!/usr/bin/env python3
"""Hardware-decoded V11 shared-memory previews for non-Dev-Room cameras.

CAM-01 and CAM-04 are intentionally excluded: their previews come from the
production MV3DT tracker-output probe, so no camera is opened twice.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.capture_dev_room_mv3dt import detect_rtsp_codec
from services.camera_v11.ui_preview_ipc_v1 import PreviewFrameWriter
from services.ml_service.app.config import CameraConfig, load_settings
from services.ml_service.app.deepstream.capture import DeepStreamCapture

DEFAULT_CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


def _preview_path(camera_id: str) -> str:
    key = f"V11_UI_PREVIEW_PATH_{camera_id.replace('-', '')}"
    slug = camera_id.lower().replace("-", "")
    return os.getenv(key, f"/dev/shm/v11_ui_preview_{slug}_v1.bin")


def _safe_error(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"(?i)(rtsps?://)[^@/\s]+@", r"\1***:***@", text)
    text = re.sub(r'(?i)user-(?:id|pw)="[^"]*"', 'credential="***"', text)
    return text[-500:]


def _camera_probe(camera: CameraConfig) -> dict:
    return {
        "id": camera.camera_id,
        "uri": camera.uri,
        "username": camera.username,
        "password": camera.password,
    }


@dataclass
class CameraStats:
    camera_id: str
    state: str = "CONNECTING"
    frames: int = 0
    reconnects: int = 0
    failures: int = 0
    stalled_periods: int = 0
    fps: float = 0.0
    fps_min: float | None = None
    fps_max: float | None = None
    last_frame_mono: float = 0.0
    started_mono: float = field(default_factory=time.monotonic)
    codec: str = ""
    width: int = 0
    height: int = 0
    last_error: str = ""

    def observe_fps(self, value: float) -> None:
        if value <= 0:
            return
        self.fps = value
        self.fps_min = value if self.fps_min is None else min(self.fps_min, value)
        self.fps_max = value if self.fps_max is None else max(self.fps_max, value)

    def snapshot(self) -> dict:
        elapsed = max(time.monotonic() - self.started_mono, 1e-6)
        return {
            "camera_id": self.camera_id,
            "state": self.state,
            "frames": self.frames,
            "average_fps": self.frames / elapsed,
            "current_fps": self.fps,
            "minimum_observed_fps": self.fps_min,
            "maximum_observed_fps": self.fps_max,
            "reconnects": self.reconnects,
            "failures": self.failures,
            "stalled_periods": self.stalled_periods,
            "last_frame_age_sec": None if not self.last_frame_mono else time.monotonic() - self.last_frame_mono,
            "resolution": f"{self.width}x{self.height}" if self.width and self.height else None,
            "codec": self.codec or None,
            "last_error": self.last_error,
        }


class PreviewSource(threading.Thread):
    def __init__(self, camera: CameraConfig, config, stop: threading.Event):
        super().__init__(name=f"preview-{camera.camera_id}", daemon=True)
        self.camera = camera
        self.config = config
        self.stop_event = stop
        self.stats = CameraStats(camera.camera_id)
        self.writer: PreviewFrameWriter | None = None

    def run(self) -> None:
        retry = max(0.5, float(self.config.reconnect_delay_sec))
        retry_max = max(retry, float(self.config.reconnect_delay_max_sec))
        capture: DeepStreamCapture | None = None
        while not self.stop_event.is_set():
            try:
                self.stats.state = "CONNECTING"
                codec = detect_rtsp_codec(
                    _camera_probe(self.camera),
                    self.config.latency_ms,
                    timeout_sec=max(5.0, self.config.startup_grace_sec),
                ).lower()
                self.stats.codec = codec
                capture = DeepStreamCapture(
                    self.camera.camera_id,
                    self.camera.uri,
                    codec,
                    self.config,
                    username=self.camera.username,
                    password=self.camera.password,
                    output_bgrx=True,
                )
                retry = max(0.5, float(self.config.reconnect_delay_sec))
                frame_window = 0
                fps_window_start = time.monotonic()
                last_progress = fps_window_start
                next_publish = fps_window_start
                publish_period = 1.0 / max(1, int(self.config.display_fps))
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    now = time.monotonic()
                    if not ok or frame is None:
                        if now - last_progress > max(3.0, self.config.capture_timeout_ms / 1000.0 * 2.5):
                            self.stats.stalled_periods += 1
                            raise RuntimeError("preview frame progression stalled")
                        continue
                    last_progress = now
                    height, width = frame.shape[:2]
                    if self.writer is None:
                        self.writer = PreviewFrameWriter(
                            _preview_path(self.camera.camera_id),
                            width,
                            height,
                            width * 4,
                        )
                    if now >= next_publish:
                        # nvvideoconvert already produces BGRx. Publishing it
                        # directly avoids another full-frame allocation/copy.
                        bgra = frame
                        self.writer.publish(
                            bgra, object_count=0, timestamp_ns=time.monotonic_ns(), fps=self.stats.fps,
                            decoder_reference_ns=capture.last_timing.decoder_reference_ns,
                            decoder_out_ns=capture.last_timing.decoder_out_ns,
                            pts_ns=capture.last_timing.pts_ns,
                            source_frame_num=self.stats.frames + 1,
                        )
                        next_publish = max(next_publish + publish_period, now)
                    self.stats.frames += 1
                    self.stats.last_frame_mono = now
                    self.stats.width = width
                    self.stats.height = height
                    self.stats.state = "LIVE"
                    self.stats.last_error = ""
                    frame_window += 1
                    interval = now - fps_window_start
                    if interval >= 1.0:
                        self.stats.observe_fps(frame_window / interval)
                        frame_window = 0
                        fps_window_start = now
            except Exception as exc:
                self.stats.state = "DISCONNECTED"
                self.stats.failures += 1
                self.stats.last_error = _safe_error(exc)
            finally:
                if capture is not None:
                    capture.close()
                    capture = None
            if not self.stop_event.is_set():
                self.stats.reconnects += 1
                self.stop_event.wait(retry)
                retry = min(retry_max, retry * 1.5)
        if self.writer is not None:
            self.writer.close(unlink=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS))
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.cameras.split(",") if item.strip())
    settings = load_settings()
    cameras = {camera.camera_id: camera for camera in settings.cameras}
    missing = sorted(set(selected) - set(cameras))
    if missing:
        raise SystemExit("missing configured cameras: " + ",".join(missing))

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    workers = [
        PreviewSource(cameras[camera_id], settings.deepstream, stop)
        for camera_id in selected
    ]
    for worker in workers:
        worker.start()

    started = time.monotonic()
    try:
        while not stop.wait(1.0):
            snapshot = {worker.camera.camera_id: worker.stats.snapshot() for worker in workers}
            if args.stats:
                args.stats.parent.mkdir(parents=True, exist_ok=True)
                temp = args.stats.with_suffix(args.stats.suffix + ".tmp")
                temp.write_text(json.dumps(snapshot, indent=2))
                temp.replace(args.stats)
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10)
        if args.stats:
            snapshot = {worker.camera.camera_id: worker.stats.snapshot() for worker in workers}
            args.stats.write_text(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
