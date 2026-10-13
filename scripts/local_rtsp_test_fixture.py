#!/usr/bin/env python3
"""Local synthetic RTSP fixture for network-isolated-from-cameras tests only."""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


class RtpOutputTelemetry:
    """Low-volume timing evidence at the synthetic RTSP server's payloader."""

    def __init__(self, port: int):
        self.port = port
        self._lock = threading.Lock()
        self._last_packet_ns = 0
        self._window_start_ns = time.monotonic_ns()
        self._window_packets = 0
        self._window_first_pts = None
        self._window_last_pts = None
        self._window_max_gap_ns = 0

    def on_rtp_buffer(self, _pad, info, _user_data):
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK
        now_ns = time.monotonic_ns()
        pts = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None
        with self._lock:
            if self._last_packet_ns:
                self._window_max_gap_ns = max(
                    self._window_max_gap_ns, now_ns - self._last_packet_ns)
            self._last_packet_ns = now_ns
            self._window_packets += 1
            if self._window_first_pts is None:
                self._window_first_pts = pts
            self._window_last_pts = pts
        return Gst.PadProbeReturn.OK

    def report_window(self):
        now_ns = time.monotonic_ns()
        with self._lock:
            row = {
                "event": "LOCAL_RTSP_RTP_PAYLOADER_WINDOW",
                "port": self.port,
                "window_start_monotonic_ns": self._window_start_ns,
                "window_end_monotonic_ns": now_ns,
                "packets": self._window_packets,
                "first_pts_ns": self._window_first_pts,
                "last_pts_ns": self._window_last_pts,
                "max_interpacket_gap_ms": round(self._window_max_gap_ns / 1e6, 3),
            }
            self._window_start_ns = now_ns
            self._window_packets = 0
            self._window_first_pts = None
            self._window_last_pts = None
            self._window_max_gap_ns = 0
        print(json.dumps(row, sort_keys=True), flush=True)
        return True


def attach_rtp_output_probe(_factory, media, telemetry: RtpOutputTelemetry):
    pipeline = media.get_element()
    payloader = pipeline.get_by_name("pay0") if pipeline else None
    pad = payloader.get_static_pad("src") if payloader else None
    if pad is None:
        print(f"LOCAL_RTSP_RTP_PROBE_FAIL port={telemetry.port} pay0_src_missing", flush=True)
        return
    if not pad.add_probe(Gst.PadProbeType.BUFFER, telemetry.on_rtp_buffer, None):
        print(f"LOCAL_RTSP_RTP_PROBE_FAIL port={telemetry.port} add_probe_failed", flush=True)
        return
    print(f"LOCAL_RTSP_RTP_PROBE_READY port={telemetry.port}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18554)
    parser.add_argument("--pattern", choices=(
        "smpte", "snow", "red", "green", "blue", "checkers-2", "circular",
        "smpte75", "zone-plate", "gamut", "ball", "bar", "pinwheel",
        "spokes", "gradient", "colors", "smpte-rp-219"), default="ball")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("fixture port must be unprivileged")

    width = int(os.environ.get("LOCAL_RTSP_FIXTURE_WIDTH", "1280"))
    height = int(os.environ.get("LOCAL_RTSP_FIXTURE_HEIGHT", "720"))
    fps = int(os.environ.get("LOCAL_RTSP_FIXTURE_FPS", "20"))
    if not 160 <= width <= 1920 or not 90 <= height <= 1080 or not 1 <= fps <= 30:
        raise ValueError("fixture dimensions/FPS are outside the safe test range")

    Gst.init(None)
    server = GstRtspServer.RTSPServer.new()
    server.set_address("127.0.0.1")
    server.set_service(str(args.port))
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        f"( videotestsrc is-live=true pattern={args.pattern} "
        f"! video/x-raw,width={width},height={height},framerate={fps}/1 "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=800 key-int-max=40 "
        "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )
    factory.set_shared(True)
    rtp_telemetry = RtpOutputTelemetry(args.port)
    factory.connect("media-configure", attach_rtp_output_probe, rtp_telemetry)
    server.get_mount_points().add_factory("/synthetic", factory)
    if not server.attach(None):
        raise RuntimeError("could not attach local RTSP fixture")

    loop = GLib.MainLoop()
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(loop.quit))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(loop.quit))
    print(f"LOCAL_RTSP_FIXTURE_READY port={args.port} pattern={args.pattern} "
          f"width={width} height={height} fps={fps} "
          f"uri=rtsp://127.0.0.1:{args.port}/synthetic",
          flush=True)
    GLib.timeout_add(1000, rtp_telemetry.report_window)
    loop.run()
    rtp_telemetry.report_window()
    print("LOCAL_RTSP_FIXTURE_STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
