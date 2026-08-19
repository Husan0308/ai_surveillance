from __future__ import annotations

import json
import os
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("ML_SMOKE_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT_SEC = float(os.getenv("ML_SMOKE_TIMEOUT_SEC", "40"))
EXPECTED_CAMERAS = int(os.getenv("ML_SMOKE_EXPECTED_CAMERAS", "6"))


def _json(path: str, timeout: float = 3.0) -> dict:
    req = Request(f"{BASE_URL}{path}", headers={"Connection": "close"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_until_all_online() -> dict:
    deadline = time.monotonic() + TIMEOUT_SEC
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = _json("/health")
            online = int(last.get("online_camera_count", 0))
            total = int(last.get("camera_count", 0))
            print(
                f"[WAIT] ml status={last.get('status')} online={online}/{total}",
                flush=True,
            )
            if total == EXPECTED_CAMERAS and online == EXPECTED_CAMERAS:
                return last
        except Exception as exc:
            print(f"[WAIT] ml_service unavailable: {exc}", flush=True)
        time.sleep(1.0)
    raise RuntimeError(
        f"ML service did not reach {EXPECTED_CAMERAS}/{EXPECTED_CAMERAS} online "
        f"within {TIMEOUT_SEC:.0f}s; last={last}"
    )


def _check_camera_metrics() -> list[str]:
    payload = _json("/cameras")
    rows = list(payload.get("cameras") or [])
    if int(payload.get("count", len(rows))) != EXPECTED_CAMERAS:
        raise RuntimeError(
            f"expected {EXPECTED_CAMERAS} cameras, got {payload.get('count', len(rows))}"
        )

    camera_ids: list[str] = []
    for row in rows:
        camera_id = str(row.get("id", ""))
        camera_ids.append(camera_id)
        online = bool(row.get("online"))
        fps = float(row.get("fps") or 0.0)
        age = row.get("last_frame_age_ms")
        backend = str(row.get("backend") or "")
        reconnects = int(row.get("reconnects") or 0)
        error = str(row.get("last_error") or "")
        print(
            f"[CAM] {camera_id} online={online} fps={fps:.1f} "
            f"age_ms={age} backend={backend} reconnects={reconnects}",
            flush=True,
        )
        if not online:
            raise RuntimeError(f"{camera_id} is offline: {error}")
        if fps <= 0.0:
            raise RuntimeError(f"{camera_id} has no measured FPS")
        if age is None or float(age) > 3000.0:
            raise RuntimeError(f"{camera_id} latest frame is stale: age_ms={age}")
        if "nvurisrcbin" not in backend:
            raise RuntimeError(f"{camera_id} unexpected backend: {backend}")
    return camera_ids


def _check_mjpeg(camera_id: str) -> None:
    req = Request(
        f"{BASE_URL}/video/{camera_id}",
        headers={"Connection": "close", "Cache-Control": "no-cache"},
    )
    with urlopen(req, timeout=5.0) as response:
        chunk = response.read(4096)
    if b"Content-Type: image/jpeg" not in chunk:
        raise RuntimeError(f"{camera_id}: MJPEG boundary/header not received")
    if b"\xff\xd8" not in chunk:
        raise RuntimeError(f"{camera_id}: JPEG SOI marker not received")
    print(f"[VIDEO] {camera_id} MJPEG=PASS", flush=True)


def main() -> int:
    print(f"=== ML service concurrent smoke check: {BASE_URL} ===", flush=True)
    try:
        health = _wait_until_all_online()
        print(f"[HEALTH] PASS {health}", flush=True)
        camera_ids = _check_camera_metrics()
        for camera_id in camera_ids:
            _check_mjpeg(camera_id)
    except (RuntimeError, URLError, TimeoutError, OSError) as exc:
        print(f"ML_SMOKE=FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    print("ML_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
