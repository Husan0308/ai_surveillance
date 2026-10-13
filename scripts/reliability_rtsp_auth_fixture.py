#!/usr/bin/env python3
"""Loopback-only authenticated RTSP fixture for reconnect ownership tests.

This fixture never accepts non-loopback binds. It starts with credentials that
do not match the test client, then can switch to the requested credentials at
a specified delay to test whether a client performs its own retry.
"""
from __future__ import annotations

import argparse
import signal
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtsp", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtsp, GstRtspServer  # noqa: E402


def make_auth(username: str, password: str):
    auth = GstRtspServer.RTSPAuth.new()
    auth.set_supported_methods(GstRtsp.RTSPAuthMethod.BASIC)
    token = GstRtspServer.RTSPToken.new()
    token.set_string(GstRtspServer.RTSP_TOKEN_MEDIA_FACTORY_ROLE, "reliability-test")
    auth.add_basic(auth.make_basic(username, password), token)
    return auth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18570)
    parser.add_argument("--pattern", default="ball")
    parser.add_argument("--initial-user", default="reject")
    parser.add_argument("--initial-password", default="reject")
    parser.add_argument("--accepted-user", default="r11-test")
    parser.add_argument("--accepted-password", default="r11-secret")
    parser.add_argument("--accept-after-sec", type=float, default=0.0)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("fixture port must be unprivileged")
    if args.accept_after_sec < 0:
        raise ValueError("accept-after-sec must be non-negative")

    Gst.init(None)
    server = GstRtspServer.RTSPServer.new()
    server.set_address("127.0.0.1")
    server.set_service(str(args.port))
    server.set_auth(make_auth(args.initial_user, args.initial_password))

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        f"( videotestsrc is-live=true pattern={args.pattern} "
        "! video/x-raw,width=640,height=360,framerate=20/1 "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=500 key-int-max=40 "
        "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )
    permissions = GstRtspServer.RTSPPermissions.new()
    permissions.add_role("reliability-test")
    permissions.add_permission_for_role("reliability-test", "media.factory.access", True)
    permissions.add_permission_for_role("reliability-test", "media.factory.construct", True)
    factory.set_permissions(permissions)
    factory.set_shared(True)
    server.get_mount_points().add_factory("/synthetic", factory)

    counts = {"clients": 0, "describes": 0}

    def on_describe(_client, _context):
        counts["describes"] += 1
        print(f"RTSP_DESCRIBE count={counts['describes']} monotonic={time.monotonic():.3f}", flush=True)
        return GstRtsp.RTSPStatusCode.OK

    def on_client(_server, client):
        counts["clients"] += 1
        print(f"RTSP_CLIENT_CONNECTED count={counts['clients']} monotonic={time.monotonic():.3f}", flush=True)
        client.connect("pre-describe-request", on_describe)

    server.connect("client-connected", on_client)
    if not server.attach(None):
        raise RuntimeError("could not attach loopback RTSP fixture")

    loop = GLib.MainLoop()
    started = time.monotonic()

    def update_auth():
        if time.monotonic() - started < args.accept_after_sec:
            return GLib.SOURCE_CONTINUE
        server.set_auth(make_auth(args.accepted_user, args.accepted_password))
        print("RTSP_AUTH_ACCEPTED_CREDENTIALS_UPDATED", flush=True)
        return GLib.SOURCE_REMOVE

    if args.accept_after_sec:
        GLib.timeout_add(50, update_auth)
    signal.signal(signal.SIGTERM, lambda *_: GLib.idle_add(loop.quit))
    signal.signal(signal.SIGINT, lambda *_: GLib.idle_add(loop.quit))
    print(f"LOCAL_RTSP_AUTH_FIXTURE_READY port={args.port} bind=127.0.0.1 "
          f"accept_after_sec={args.accept_after_sec}", flush=True)
    # Keep the six-camera fixture supervisor's readiness contract while also
    # making the authentication-specific test mode explicit in the log.
    print(f"LOCAL_RTSP_FIXTURE_READY port={args.port} pattern=auth-test "
          "width=640 height=360 fps=20", flush=True)
    loop.run()
    print(f"LOCAL_RTSP_AUTH_FIXTURE_STOPPED clients={counts['clients']} "
          f"describes={counts['describes']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
