#!/usr/bin/env python3
"""Loopback-only finite RTSP media fixture for source-EOS recovery tests."""
from __future__ import annotations

import argparse
import signal

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("fixture port must be unprivileged")
    if not 1 <= args.frames <= 10000:
        raise ValueError("finite fixture frame count must be in 1..10000")

    Gst.init(None)
    server = GstRtspServer.RTSPServer.new()
    server.set_address("127.0.0.1")
    server.set_service(str(args.port))
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        f"( videotestsrc is-live=true num-buffers={args.frames} pattern=ball "
        "! video/x-raw,width=640,height=360,framerate=20/1 "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=500 key-int-max=40 "
        "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/synthetic", factory)
    if not server.attach(None):
        raise RuntimeError("could not attach loopback RTSP EOS fixture")

    loop = GLib.MainLoop()
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(loop.quit))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(loop.quit))
    print(f"LOCAL_RTSP_EOS_FIXTURE_READY port={args.port} bind=127.0.0.1 "
          f"frames={args.frames} fps=20 expected_eos_sec={args.frames / 20:.3f}",
          flush=True)
    print(f"LOCAL_RTSP_FIXTURE_READY port={args.port} pattern=ball "
          "width=640 height=360 fps=20", flush=True)
    loop.run()
    print("LOCAL_RTSP_EOS_FIXTURE_STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
