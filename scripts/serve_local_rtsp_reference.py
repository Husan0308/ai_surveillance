"""Serve the pinned CAM-01/CAM-04 MP4 pair as local read-only RTSP inputs."""
from __future__ import annotations

import argparse
import signal
from pathlib import Path

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


def serve(cam01: Path, cam04: Path, port: int) -> None:
    for path in (cam01, cam04):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            raise ValueError(f"not an MP4 reference video: {path}")
    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_address("127.0.0.1")
    server.set_service(str(port))
    mounts = server.get_mount_points()
    for name, path in (("cam01", cam01), ("cam04", cam04)):
        # MP4 is H.264; stream-copy through qtdemux/parser/payloader, no re-encode.
        location = str(path.resolve()).replace('"', r'\"')
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_launch(
            f'( filesrc location="{location}" ! qtdemux name=d '
            f'd.video_0 ! queue ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )'
        )
        # Each RTSP client gets an independent file timeline beginning at frame 0.
        factory.set_shared(False)
        mounts.add_factory(f"/{name}", factory)
    if server.attach(None) == 0:
        raise RuntimeError("failed to attach local RTSP reference server")
    print(f"REFERENCE_RTSP_READY address=127.0.0.1 port={port} mounts=/cam01,/cam04", flush=True)
    loop = GLib.MainLoop()
    signal.signal(signal.SIGINT, lambda *_: loop.quit())
    signal.signal(signal.SIGTERM, lambda *_: loop.quit())
    loop.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam01", type=Path, required=True)
    parser.add_argument("--cam04", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8554)
    args = parser.parse_args()
    serve(args.cam01, args.cam04, args.port)


if __name__ == "__main__":
    main()
