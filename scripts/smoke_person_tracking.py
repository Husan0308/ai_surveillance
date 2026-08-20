from __future__ import annotations

import json
import os
import time
from urllib.request import Request, urlopen

BASE_URL = os.getenv("ML_SMOKE_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT_SEC = float(os.getenv("ML_TRACK_SMOKE_TIMEOUT_SEC", "30"))
EXPECTED_CAMERAS = int(os.getenv("ML_SMOKE_EXPECTED_CAMERAS", "6"))


def _json(path: str, timeout: float = 3.0) -> dict:
    request = Request(f"{BASE_URL}{path}", headers={"Connection": "close"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    print(f"=== Person tracking smoke: {BASE_URL} ===", flush=True)
    deadline = time.monotonic() + TIMEOUT_SEC
    health: dict = {}
    while time.monotonic() < deadline:
        health = _json("/health")
        tracker = dict(health.get("tracker") or {})
        print(
            f"[WAIT] tracker={tracker.get('state')} updates={tracker.get('updates')} "
            f"active={tracker.get('active_tracks')}",
            flush=True,
        )
        if tracker.get("state") == "error":
            print(f"PERSON_TRACK_SMOKE=FAIL {tracker.get('last_error')}", flush=True)
            return 1
        if tracker.get("state") == "ready" and int(tracker.get("updates") or 0) > 0:
            break
        time.sleep(1.0)
    else:
        print(f"PERSON_TRACK_SMOKE=FAIL tracker not ready: {health}", flush=True)
        return 1

    cameras = list(_json("/cameras").get("cameras") or [])
    if len(cameras) != EXPECTED_CAMERAS:
        print(f"PERSON_TRACK_SMOKE=FAIL expected {EXPECTED_CAMERAS} cameras, got {len(cameras)}")
        return 1

    total_tracks = 0
    for row in cameras:
        camera_id = str(row["id"])
        tracking = dict(row.get("tracking") or {})
        payload = _json(f"/tracks/{camera_id}")
        result = payload.get("result") or {}
        tracks = list(result.get("tracks") or [])
        total_tracks += len(tracks)
        ids = [track.get("track_id") for track in tracks]
        print(
            f"[TRACK] {camera_id} state={tracking.get('state')} "
            f"active={tracking.get('active_tracks')} ids={ids}",
            flush=True,
        )
        if tracking.get("state") != "ready":
            print(f"PERSON_TRACK_SMOKE=FAIL {camera_id} tracker={tracking}", flush=True)
            return 1
        if len(ids) != len(set(ids)):
            print(f"PERSON_TRACK_SMOKE=FAIL {camera_id} duplicate local track IDs: {ids}", flush=True)
            return 1

    print(f"PERSON_TRACK_SMOKE=PASS active_tracks_now={total_tracks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
