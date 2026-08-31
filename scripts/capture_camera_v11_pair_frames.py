#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
root_text = str(ROOT)
if root_text not in sys.path:
    sys.path.insert(0, root_text)

# Keep calibration snapshot capture deterministic and TCP-only like the V11 runtime.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402

from services.ml_service.app.config import CameraConfig, load_settings  # noqa: E402


def _authenticated_uri(camera: CameraConfig) -> str:
    if not camera.username:
        return camera.uri
    parts = urlsplit(camera.uri)
    host = parts.hostname or ""
    if not host:
        raise RuntimeError(f"{camera.camera_id}: invalid_rtsp_host")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port is not None else ""
    username = quote(camera.username, safe="")
    password = quote(camera.password or "", safe="")
    netloc = f"{username}:{password}@{host}{port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _capture(camera: CameraConfig, output: Path, *, timeout_sec: float) -> None:
    uri = _authenticated_uri(camera)
    cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"{camera.camera_id}: cannot_open_rtsp auth={int(bool(camera.username))}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.monotonic() + max(2.0, float(timeout_sec))
    latest = None
    reads = 0
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                latest = frame
                reads += 1
                if reads >= 12:
                    break
            else:
                time.sleep(0.03)
    finally:
        cap.release()
    if latest is None:
        raise RuntimeError(f"{camera.camera_id}: no_frame auth={int(bool(camera.username))}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), latest, [cv2.IMWRITE_JPEG_QUALITY, 97]):
        raise RuntimeError(f"{camera.camera_id}: snapshot_write_failed")
    h, w = latest.shape[:2]
    print(
        "V11_CAM_PAIR_SNAPSHOT "
        f"camera={camera.camera_id} result=PASS auth={int(bool(camera.username))} "
        f"reads={reads} size={w}x{h} output={output}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture credential-aware V11 calibration frames.")
    parser.add_argument("--config", default="config/cameras.yaml")
    parser.add_argument("--camera-a", default="CAM-01")
    parser.add_argument("--camera-b", default="CAM-04")
    parser.add_argument("--output-a", required=True)
    parser.add_argument("--output-b", required=True)
    parser.add_argument("--timeout-sec", type=float, default=6.0)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    settings = load_settings(config_path)
    cameras = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera_a not in cameras or args.camera_b not in cameras:
        print("V11_CAM_PAIR_SNAPSHOT RESULT=FAIL reason=camera_not_in_config", flush=True)
        return 2
    camera_a = cameras[args.camera_a]
    camera_b = cameras[args.camera_b]
    if camera_a.room != camera_b.room:
        print("V11_CAM_PAIR_SNAPSHOT RESULT=FAIL reason=not_same_room", flush=True)
        return 2

    try:
        _capture(camera_a, Path(args.output_a), timeout_sec=args.timeout_sec)
        _capture(camera_b, Path(args.output_b), timeout_sec=args.timeout_sec)
    except Exception as exc:
        # Never include authenticated RTSP URI/password in diagnostics.
        print(f"V11_CAM_PAIR_SNAPSHOT RESULT=FAIL reason={exc}", flush=True)
        return 3

    print(
        f"V11_CAM_PAIR_SNAPSHOT RESULT=PASS room={camera_a.room} pair={camera_a.camera_id},{camera_b.camera_id}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
