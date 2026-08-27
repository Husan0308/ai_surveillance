#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import Counter
from urllib.parse import quote, urlsplit, urlunsplit

from services.ml_service.app.config import load_settings

CAMERA_ID = "CAM-02"
PROBE_SECONDS = 8


def _auth_uri(uri: str, username: str, password: str) -> str:
    if not username:
        return uri
    parts = urlsplit(uri)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    userinfo = quote(username, safe="")
    if password:
        userinfo += ":" + quote(password, safe="")
    netloc = f"{userinfo}@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _run_ffprobe(uri: str) -> dict:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe missing; install ffmpeg package")

    cmd = [
        ffprobe,
        "-v", "error",
        "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-read_intervals", f"%+{PROBE_SECONDS}",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,has_b_frames,level:"
        "frame=pict_type,key_frame,best_effort_timestamp_time,pkt_dts_time",
        "-of", "json",
        uri,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=25, check=False)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "ffprobe failed").strip()
        raise RuntimeError(err)
    return json.loads(result.stdout or "{}")


def main() -> int:
    settings = load_settings()
    camera = next((c for c in settings.cameras if c.camera_id == CAMERA_ID), None)
    if camera is None:
        print(f"V11_CAM02_PROBE FAIL camera={CAMERA_ID} reason=not-configured")
        return 2

    uri = _auth_uri(camera.uri, camera.username, camera.password)
    try:
        payload = _run_ffprobe(uri)
    except Exception as exc:
        print(f"V11_CAM02_PROBE FAIL camera={CAMERA_ID} error={type(exc).__name__}:{exc}")
        return 2

    streams = payload.get("streams") or []
    if not streams:
        print(f"V11_CAM02_PROBE FAIL camera={CAMERA_ID} reason=no-video-stream")
        return 2
    stream = streams[0]
    frames = payload.get("frames") or []
    types = Counter(str(f.get("pict_type") or "?") for f in frames)
    b_count = int(types.get("B", 0))
    i_count = int(types.get("I", 0))
    p_count = int(types.get("P", 0))
    ff_has_b = int(stream.get("has_b_frames") or 0)
    observed_b = int(b_count > 0)
    safe_low_latency = int(ff_has_b == 0 and observed_b == 0 and (i_count + p_count) > 0)

    print(
        "V11_CAM02_STREAM "
        f"camera={CAMERA_ID} codec={stream.get('codec_name','?')} profile={stream.get('profile','?')} "
        f"size={stream.get('width',0)}x{stream.get('height',0)} pix_fmt={stream.get('pix_fmt','?')} "
        f"r_fps={stream.get('r_frame_rate','?')} avg_fps={stream.get('avg_frame_rate','?')} "
        f"has_b_frames={ff_has_b} level={stream.get('level','?')}"
    )
    print(
        "V11_CAM02_GOP "
        f"probe_seconds={PROBE_SECONDS} frames={len(frames)} I={i_count} P={p_count} B={b_count} "
        f"observed_b={observed_b} safe_low_latency_mode={safe_low_latency}"
    )
    if safe_low_latency:
        print("V11_CAM02_PROBE RESULT=IPPP_NO_B next=test_nvdec_low_latency_mode_only_on_CAM02")
    else:
        print("V11_CAM02_PROBE RESULT=BFRAME_OR_UNSAFE next=keep_nvdec_native_and_inspect_camera_NVR_encoding_profile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
