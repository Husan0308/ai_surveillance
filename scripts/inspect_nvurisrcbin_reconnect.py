#!/usr/bin/env python3
"""Observe pinned nvurisrcbin RTSP retry behavior against a localhost fixture."""
from __future__ import annotations

import argparse
import json
import time
from urllib.parse import urlsplit

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri")
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument("--attempts", type=int, default=0)
    args = parser.parse_args()
    if urlsplit(args.uri).hostname != "127.0.0.1":
        raise ValueError("this diagnostic only accepts a 127.0.0.1 RTSP URL")
    if not 1 <= args.duration_sec <= 120:
        raise ValueError("duration-sec must be in 1..120")

    Gst.init(None)
    source = Gst.ElementFactory.make("nvurisrcbin", "reconnect-probe")
    if source is None:
        raise RuntimeError("pinned nvurisrcbin factory is unavailable")
    source.set_property("uri", args.uri)
    source.set_property("select-rtp-protocol", 4)
    source.set_property("rtsp-reconnect-interval", 0)
    source.set_property("init-rtsp-reconnect-interval", 0)
    source.set_property("rtsp-reconnect-attempts", args.attempts)
    effective = {
        name: source.get_property(name)
        for name in ("rtsp-reconnect-interval", "init-rtsp-reconnect-interval",
                     "rtsp-reconnect-attempts")
    }

    pipeline = Gst.Pipeline.new("reconnect-probe-pipeline")
    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", False)
    pipeline.add(source)
    pipeline.add(sink)
    source.connect("pad-added", lambda _src, pad: pad.link(sink.get_static_pad("sink")))
    bus = pipeline.get_bus()
    state_result = pipeline.set_state(Gst.State.PLAYING)
    start = time.monotonic()
    errors: list[str] = []
    warnings: list[str] = []
    while time.monotonic() - start < args.duration_sec:
        message = bus.timed_pop_filtered(
            100 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.WARNING
        )
        if message is None:
            continue
        error, debug = (message.parse_error() if message.type == Gst.MessageType.ERROR
                        else message.parse_warning())
        row = f"{message.src.get_name()}: {error.message}"
        (errors if message.type == Gst.MessageType.ERROR else warnings).append(row)
        print("BUS_ERROR" if message.type == Gst.MessageType.ERROR else "BUS_WARNING", row,
              flush=True)
    elapsed = time.monotonic() - start
    pipeline.set_state(Gst.State.NULL)
    print("RESULT " + json.dumps({
        "elapsed_sec": round(elapsed, 3),
        "state_change_result": str(state_result),
        "requested_attempts": args.attempts,
        "effective_reconnect_properties": effective,
        "bus_errors": errors,
        "bus_warnings": warnings,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
