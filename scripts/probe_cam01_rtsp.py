#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# When executed as `python scripts/probe_cam01_rtsp.py`, Python puts the scripts/
# directory (not the repository root) at sys.path[0].  Make the probe runnable
# exactly as documented without requiring a global PYTHONPATH tweak.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

from services.ml_service.app.config import load_settings


class RtspProbe:
    def __init__(self, camera, *, seconds: float, latency_ms: int, keepalive: bool) -> None:
        self.camera = camera
        self.seconds = float(seconds)
        self.latency_ms = int(latency_ms)
        self.keepalive = bool(keepalive)
        self.frames = 0
        self.first_frame_at: float | None = None
        self.last_frame_at: float | None = None
        self.error = ""
        self.eos = False

        self.pipeline = Gst.Pipeline.new(f"probe-{camera.camera_id}")
        self.src = Gst.ElementFactory.make("rtspsrc", "src")
        self.decode = Gst.ElementFactory.make("decodebin", "decode")
        self.queue = Gst.ElementFactory.make("queue", "q")
        self.convert = Gst.ElementFactory.make("videoconvert", "convert")
        self.ident = Gst.ElementFactory.make("identity", "counter")
        self.sink = Gst.ElementFactory.make("fakesink", "sink")
        if any(x is None for x in (self.pipeline, self.src, self.decode, self.queue, self.convert, self.ident, self.sink)):
            raise RuntimeError("required GStreamer elements are missing")

        self.src.set_property("location", camera.uri)
        self.src.set_property("protocols", 4)  # GstRTSPLowerTrans TCP
        self.src.set_property("latency", self.latency_ms)
        self.src.set_property("drop-on-latency", True)
        self.src.set_property("do-rtsp-keep-alive", self.keepalive)
        if camera.username:
            self.src.set_property("user-id", camera.username)
            self.src.set_property("user-pw", camera.password)

        self.queue.set_property("max-size-buffers", 2)
        self.queue.set_property("max-size-bytes", 0)
        self.queue.set_property("max-size-time", 0)
        self.sink.set_property("sync", False)
        self.ident.set_property("signal-handoffs", True)

        for element in (self.src, self.decode, self.queue, self.convert, self.ident, self.sink):
            self.pipeline.add(element)
        if not self.queue.link(self.convert) or not self.convert.link(self.ident) or not self.ident.link(self.sink):
            raise RuntimeError("probe downstream link failed")

        self.src.connect("pad-added", self._on_rtsp_pad)
        self.decode.connect("pad-added", self._on_decode_pad)
        self.ident.connect("handoff", self._on_handoff)

    def _on_rtsp_pad(self, _src, pad) -> None:
        sink = self.decode.get_static_pad("sink")
        if sink is None or sink.is_linked():
            return
        result = pad.link(sink)
        if result != Gst.PadLinkReturn.OK:
            self.error = f"rtspsrc->decodebin link failed: {result}"

    def _on_decode_pad(self, _decode, pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return
        media = str(caps.get_structure(0).get_name())
        if not media.startswith("video/x-raw"):
            return
        sink = self.queue.get_static_pad("sink")
        if sink is None or sink.is_linked():
            return
        result = pad.link(sink)
        if result != Gst.PadLinkReturn.OK:
            self.error = f"decodebin->queue link failed: {result}"

    def _on_handoff(self, _identity, _buffer) -> None:
        now = time.monotonic()
        if self.first_frame_at is None:
            self.first_frame_at = now
        self.last_frame_at = now
        self.frames += 1

    def run(self) -> None:
        started = time.monotonic()
        state = self.pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE:
            self.error = "pipeline failed to enter PLAYING"
            return

        bus = self.pipeline.get_bus()
        mask = Gst.MessageType.ERROR | Gst.MessageType.EOS
        try:
            while time.monotonic() - started < self.seconds:
                msg = bus.timed_pop_filtered(200 * Gst.MSECOND, mask)
                if msg is None:
                    continue
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    self.error = f"{err.message} | {debug or ''}".strip()
                    break
                if msg.type == Gst.MessageType.EOS:
                    self.eos = True
                    break
        finally:
            self.pipeline.set_state(Gst.State.NULL)

        elapsed = max(0.001, time.monotonic() - started)
        fps = self.frames / elapsed
        first_ms = -1.0 if self.first_frame_at is None else (self.first_frame_at - started) * 1000.0
        tail_ms = -1.0 if self.last_frame_at is None else (time.monotonic() - self.last_frame_at) * 1000.0
        status = "PASS" if self.frames > 0 and not self.error and not self.eos else "FAIL"
        print(
            "CAM01_RTSP_PROBE "
            f"status={status} keepalive={int(self.keepalive)} frames={self.frames} "
            f"fps={fps:.1f} first_ms={first_ms:.0f} tail_ms={tail_ms:.0f} "
            f"eos={int(self.eos)} error={self.error or '-'}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe one configured RTSP camera outside DeepStream nvurisrcbin")
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--keepalive", choices=("on", "off", "both"), default="both")
    args = parser.parse_args()

    Gst.init(None)
    settings = load_settings(ROOT / "config/cameras.yaml")
    camera = next((c for c in settings.cameras if c.camera_id == args.camera), None)
    if camera is None:
        raise SystemExit(f"camera not found: {args.camera}")

    modes = [True, False] if args.keepalive == "both" else [args.keepalive == "on"]
    print(
        f"CAM01_RTSP_PROBE_START camera={camera.camera_id} uri={camera.uri} "
        f"transport=tcp latency={args.latency_ms}ms seconds={args.seconds:g} auth={'yes' if camera.username else 'no'}",
        flush=True,
    )
    failed = False
    for index, keepalive in enumerate(modes):
        probe = RtspProbe(camera, seconds=args.seconds, latency_ms=args.latency_ms, keepalive=keepalive)
        probe.run()
        failed = failed or probe.frames == 0
        if index + 1 < len(modes):
            time.sleep(2.0)
    return 1 if failed and len(modes) == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
