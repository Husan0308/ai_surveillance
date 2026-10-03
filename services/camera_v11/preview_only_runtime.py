#!/usr/bin/env python3
"""Camera-only NVDEC previews. Select only cameras not owned by analytics.

The camera-only foundation can own all six. In the analytics topology explicitly
select CAM-02,03,05,06; CAM-01/04 belong to the room-pair owner instead.
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
from services.camera_v11.source_ownership import SourceOwnership
from services.ml_service.app.config import CameraConfig, load_settings
from services.ml_service.app.deepstream.capture import DeepStreamCapture

DEFAULT_CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


class PreviewPublishBudget:
    """Rate-limit writes while tolerating a two-frame arrival burst.

    Credits never queue frames: each allowed write replaces the single latest
    shared-memory slot immediately.
    """

    def __init__(self, fps: int, now: float):
        self.fps = max(1, int(fps))
        self.last = now
        self.credits = 2.0

    def allow(self, now: float) -> bool:
        if now < self.last:
            self.credits = 2.0
        self.credits = min(2.0, self.credits + max(0.0, now - self.last) * self.fps)
        self.last = now
        if self.credits < 1.0 - 1e-9:
            return False
        self.credits = max(0.0, self.credits - 1.0)
        return True


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


def _decoder_low_latency_enabled(
    camera_id: str, configured: bool = False
) -> bool:
    """Use the validated camera profile, with explicit test overrides."""
    allowlist = os.getenv("MV3DT_TEST_DECODER_LOW_LATENCY_CAMERAS")
    if allowlist is not None:
        selected = {item.strip() for item in allowlist.split(",") if item.strip()}
        return camera_id in selected
    legacy_override = os.getenv("MV3DT_TEST_DECODER_LOW_LATENCY")
    if legacy_override is not None:
        return legacy_override == "1"
    return bool(configured)


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
    last_failure_error: str = ""
    last_failure_monotonic_ns: int = 0
    published_frames: int = 0
    decoder_inputs: int = 0
    decoder_outputs: int = 0
    queue_buffers: int = 0
    queue_high_water: int = 0
    bus_errors: int = 0
    bus_warnings: int = 0
    timing_invalid: int = 0
    controlled_restarts: int = 0
    runtime_graph: dict = field(default_factory=dict)

    def record_failure(self, exc: BaseException) -> None:
        self.state = "DISCONNECTED"
        self.failures += 1
        self.last_error = _safe_error(exc)
        # Recovery clears current health, not the evidence for a reconnect.
        self.last_failure_error = self.last_error
        self.last_failure_monotonic_ns = time.monotonic_ns()

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
            "last_failure_error": self.last_failure_error,
            "last_failure_monotonic_ns": self.last_failure_monotonic_ns,
            "published_frames": self.published_frames,
            "decoder_inputs": self.decoder_inputs,
            "decoder_outputs": self.decoder_outputs,
            "queue_buffers": self.queue_buffers,
            "queue_high_water": self.queue_high_water,
            "bus_errors": self.bus_errors,
            "bus_warnings": self.bus_warnings,
            "timing_invalid": self.timing_invalid,
            "controlled_restarts": self.controlled_restarts,
            "runtime_graph": self.runtime_graph,
        }


class PreviewSource(threading.Thread):
    def __init__(self, camera: CameraConfig, config, stop: threading.Event):
        super().__init__(name=f"preview-{camera.camera_id}", daemon=True)
        self.camera = camera
        self.config = config
        self.stop_event = stop
        self.stats = CameraStats(camera.camera_id)
        self.writer: PreviewFrameWriter | None = None
        self.restart_event = threading.Event()

    def run(self) -> None:
        retry = max(0.5, float(self.config.reconnect_delay_sec))
        retry_max = max(retry, float(self.config.reconnect_delay_max_sec))
        capture: DeepStreamCapture | None = None
        while not self.stop_event.is_set():
            try:
                self.stats.state = "CONNECTING"
                effective_latency_ms = self.camera.effective_latency_ms(self.config.latency_ms)
                codec = detect_rtsp_codec(
                    _camera_probe(self.camera),
                    effective_latency_ms,
                    timeout_sec=max(5.0, self.config.startup_grace_sec),
                ).lower()
                self.stats.codec = codec
                low_latency_mode = _decoder_low_latency_enabled(
                    self.camera.camera_id,
                    self.camera.decoder_low_latency_mode,
                )
                capture = DeepStreamCapture(
                    self.camera.camera_id,
                    self.camera.uri,
                    codec,
                    self.config,
                    username=self.camera.username,
                    password=self.camera.password,
                    output_bgrx=True,
                    latency_ms=effective_latency_ms,
                    low_latency_mode=low_latency_mode,
                    decoder_extra_surfaces=self.camera.decoder_extra_surfaces,
                )
                input_base, output_base = self.stats.decoder_inputs, self.stats.decoder_outputs
                error_base, warning_base = self.stats.bus_errors, self.stats.bus_warnings
                self.stats.runtime_graph = capture.graph_summary()
                retry = max(0.5, float(self.config.reconnect_delay_sec))
                frame_window = 0
                fps_window_start = time.monotonic()
                last_progress = fps_window_start
                publish_budget = None
                while not self.stop_event.is_set():
                    if self.restart_event.is_set():
                        self.restart_event.clear()
                        self.stats.controlled_restarts += 1
                        raise RuntimeError("controlled source teardown/reconnect test")
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
                    pts_ns = capture.last_timing.pts_ns
                    publish_clock = pts_ns / 1e9 if 0 < pts_ns < 2**63 else now
                    if publish_budget is None:
                        publish_budget = PreviewPublishBudget(
                            self.config.display_fps, publish_clock
                        )
                    if publish_budget.allow(publish_clock):
                        # nvvideoconvert already produces BGRx. Publishing it
                        # directly avoids another full-frame allocation/copy.
                        bgra = frame
                        self.writer.publish(
                            bgra, object_count=0, timestamp_ns=time.monotonic_ns(), fps=self.stats.fps,
                            decoder_reference_ns=capture.last_timing.decoder_reference_ns,
                            decoder_out_ns=capture.last_timing.decoder_out_ns,
                            pts_ns=capture.last_timing.pts_ns,
                            decoder_dts_ns=capture.last_timing.dts_ns,
                            source_frame_num=self.stats.frames + 1,
                        )
                        self.stats.published_frames += 1
                    timing = capture.last_timing
                    if not (0 < timing.decoder_reference_ns <= timing.decoder_out_ns <= time.monotonic_ns()):
                        self.stats.timing_invalid += 1
                    self.stats.frames += 1
                    self.stats.last_frame_mono = now
                    self.stats.width = width
                    self.stats.height = height
                    self.stats.state = "LIVE"
                    self.stats.last_error = ""
                    frame_window += 1
                    interval = now - fps_window_start
                    if interval >= 1.0:
                        capture.debug_info()  # Consume and count bus messages, not just timeouts.
                        self.stats.decoder_inputs = input_base + capture.decoder_input_count
                        self.stats.decoder_outputs = output_base + capture.decoder_output_count
                        self.stats.bus_errors = error_base + capture.bus_errors
                        self.stats.bus_warnings = warning_base + capture.bus_warnings
                        self.stats.queue_buffers = capture.current_queue_buffers() or 0
                        self.stats.queue_high_water = max(self.stats.queue_high_water, self.stats.queue_buffers)
                        self.stats.observe_fps(frame_window / interval)
                        frame_window = 0
                        fps_window_start = now
            except Exception as exc:
                self.stats.record_failure(exc)
            finally:
                if capture is not None:
                    self.stats.decoder_inputs = input_base + capture.decoder_input_count
                    self.stats.decoder_outputs = output_base + capture.decoder_output_count
                    self.stats.bus_errors = error_base + capture.bus_errors
                    self.stats.bus_warnings = warning_base + capture.bus_warnings
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
    parser.add_argument("--test-reconnect-camera", choices=DEFAULT_CAMERAS)
    parser.add_argument("--test-reconnect-at", type=float, default=20.0)
    args = parser.parse_args()
    selected = tuple(item.strip() for item in args.cameras.split(",") if item.strip())
    if args.test_reconnect_camera and args.test_reconnect_camera not in selected:
        parser.error("reconnect test camera must be selected")
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
    started = time.monotonic()
    ownership = SourceOwnership(selected, Path(os.getenv("CAMERA_OWNER_LOCK_DIR", ".runtime/camera-owner-locks")))
    ownership.__enter__()  # Fail before any codec probe/RTSP open.
    for worker in workers:
        worker.start()
    reconnect_requested = False
    try:
        while not stop.wait(1.0):
            snapshot = {worker.camera.camera_id: worker.stats.snapshot() for worker in workers}
            if args.stats:
                args.stats.parent.mkdir(parents=True, exist_ok=True)
                temp = args.stats.with_suffix(args.stats.suffix + ".tmp")
                temp.write_text(json.dumps(snapshot, indent=2))
                temp.replace(args.stats)
            if (args.test_reconnect_camera and not reconnect_requested
                    and time.monotonic() - started >= args.test_reconnect_at):
                next(w for w in workers if w.camera.camera_id == args.test_reconnect_camera).restart_event.set()
                reconnect_requested = True
            if args.duration > 0 and time.monotonic() - started >= args.duration:
                break
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10)
        if args.stats:
            snapshot = {worker.camera.camera_id: worker.stats.snapshot() for worker in workers}
            args.stats.write_text(json.dumps(snapshot, indent=2))
        if any(worker.is_alive() for worker in workers):
            # Retain ownership until process exit; never release a camera while
            # a worker could still be streaming into a replacement owner's slot.
            raise RuntimeError("preview worker did not shut down within 10 seconds")
        ownership.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
