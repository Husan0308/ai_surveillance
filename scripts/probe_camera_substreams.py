#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_service.app.config import load_settings


def substream_uri(main_uri: str) -> str:
    parts = urlsplit(main_uri)
    path = parts.path
    marker = "/Streaming/Channels/"
    if marker not in path:
        marker = "/Streaming/channels/"
    if marker not in path:
        raise ValueError(f"unsupported Hikvision RTSP path: {path}")
    prefix, channel = path.rsplit("/", 1)
    if len(channel) < 2 or channel[-2:] != "01":
        raise ValueError(f"expected Hikvision main-stream channel ending in 01, got {channel}")
    return urlunsplit((parts.scheme, parts.netloc, f"{prefix}/{channel[:-2]}02", parts.query, parts.fragment))


def authenticated_uri(uri: str, username: str, password: str) -> str:
    parts = urlsplit(uri)
    host = parts.hostname or ""
    if not host:
        raise ValueError(f"RTSP URI has no host: {uri}")
    port = f":{parts.port}" if parts.port is not None else ""
    if username:
        auth = quote(username, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        netloc = f"{auth}@{host}{port}"
    else:
        netloc = f"{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def probe(camera_id: str, uri: str, username: str, password: str) -> int:
    sub_uri = substream_uri(uri)
    channel = sub_uri.rsplit("/", 1)[-1]
    if not username:
        print(f"SUBSTREAM_PROBE {camera_id} channel={channel} status=NO_CREDENTIALS")
        return 1

    secured = authenticated_uri(sub_uri, username, password)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        secured,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        print("SUBSTREAM_PROBE_FATAL ffprobe_not_found=1", file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print(f"SUBSTREAM_PROBE {camera_id} channel={channel} status=TIMEOUT")
        return 1

    if result.returncode != 0:
        err = " ".join((result.stderr or "").strip().split())
        if "401 Unauthorized" in err:
            status = "AUTH_FAILED"
        elif "404" in err or "Not Found" in err:
            status = "NOT_FOUND"
        else:
            status = "ERROR"
        print(f"SUBSTREAM_PROBE {camera_id} channel={channel} status={status} detail={err[:220]}")
        return 1

    try:
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}
    except Exception as exc:
        print(f"SUBSTREAM_PROBE {camera_id} channel={channel} status=BAD_JSON detail={type(exc).__name__}:{exc}")
        return 1

    print(
        "SUBSTREAM_PROBE "
        f"{camera_id} channel={channel} status=OK "
        f"codec={stream.get('codec_name', '?')} "
        f"size={stream.get('width', '?')}x{stream.get('height', '?')} "
        f"r_fps={stream.get('r_frame_rate', '?')} avg_fps={stream.get('avg_frame_rate', '?')}"
    )
    return 0


def main() -> int:
    settings = load_settings()
    failures = 0
    for camera in settings.cameras:
        failures += int(probe(camera.camera_id, camera.uri, camera.username, camera.password) != 0)
    if failures:
        print(f"SUBSTREAM_PROBE_RESULT status=FAIL failures={failures} cameras={len(settings.cameras)}")
        return 1
    print(f"SUBSTREAM_PROBE_RESULT status=OK cameras={len(settings.cameras)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
