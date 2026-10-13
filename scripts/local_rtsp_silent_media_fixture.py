#!/usr/bin/env python3
"""Loopback RTSP fixture that can stop RTP while keeping the RTSP owner alive.

This is an offline fault-injection fixture only. It binds 127.0.0.1 and never
uses camera configuration or credentials. SIGUSR1 toggles the shared media
pipeline between PLAYING and PAUSED without stopping the RTSP server.
"""
from __future__ import annotations

import argparse
import signal
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pattern", default="ball")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not (160 <= args.width <= 1920):
        raise ValueError("fixture port/width is outside the safe test range")
    if not 90 <= args.height <= 1080 or not 1 <= args.fps <= 30:
        raise ValueError("fixture height/FPS is outside the safe test range")

    Gst.init(None)
    server = GstRtspServer.RTSPServer.new()
    server.set_address("127.0.0.1")
    server.set_service(str(args.port))
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        f"( videotestsrc is-live=true pattern={args.pattern} "
        f"! video/x-raw,width={args.width},height={args.height},framerate={args.fps}/1 "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=800 key-int-max=40 "
        "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )
    factory.set_shared(True)
    media_lock = threading.Lock()
    media_pipelines = []
    silence = threading.Event()

    def media_configured(_factory, media):
        pipeline = media.get_element()
        if pipeline is not None:
            with media_lock:
                if pipeline not in media_pipelines:
                    media_pipelines.append(pipeline)
            payloader = pipeline.get_by_name("pay0")
            pad = payloader.get_static_pad("src") if payloader is not None else None
            if pad is None:
                print(f"SILENT_MEDIA_PROBE_FAIL port={args.port} pay0_src_missing", flush=True)
            else:
                pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    lambda _pad, _info, _data: (
                        Gst.PadProbeReturn.DROP if silence.is_set()
                        else Gst.PadProbeReturn.OK
                    ),
                    None,
                )

    factory.connect("media-configure", media_configured)
    server.get_mount_points().add_factory("/synthetic", factory)
    if not server.attach(None):
        raise RuntimeError("could not attach loopback RTSP fixture")

    loop = GLib.MainLoop()
    def toggle_media():
        with media_lock:
            pipelines = list(media_pipelines)
        if not pipelines:
            print(f"SILENT_MEDIA_TOGGLE_IGNORED port={args.port} reason=no_client_media",
                  flush=True)
            return GLib.SOURCE_REMOVE
        if silence.is_set():
            silence.clear()
        else:
            silence.set()
        print(
            f"SILENT_MEDIA_{'RESUMED' if not silence.is_set() else 'PAUSED'} port={args.port} "
            f"monotonic_ns={time.monotonic_ns()} sessions={len(pipelines)} "
            f"rtsp_server=running",
            flush=True,
        )
        return GLib.SOURCE_REMOVE

    signal.signal(signal.SIGUSR1, lambda *_: GLib.idle_add(toggle_media))
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(loop.quit))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(loop.quit))
    print(
        f"LOCAL_RTSP_SILENT_MEDIA_READY port={args.port} bind=127.0.0.1 "
        f"width={args.width} height={args.height} fps={args.fps} "
        f"uri=rtsp://127.0.0.1:{args.port}/synthetic",
        flush=True,
    )
    print(f"LOCAL_RTSP_FIXTURE_READY port={args.port} pattern={args.pattern} "
          f"width={args.width} height={args.height} fps={args.fps} "
          f"uri=rtsp://127.0.0.1:{args.port}/synthetic", flush=True)
    loop.run()
    with media_lock:
        pipelines = list(media_pipelines)
    for pipeline in pipelines:
        pipeline.set_state(Gst.State.NULL)
    print(f"LOCAL_RTSP_SILENT_MEDIA_STOPPED port={args.port}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
