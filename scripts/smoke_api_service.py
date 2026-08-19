from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("API_SMOKE_URL", "http://127.0.0.1:8000").rstrip("/")
EXPECTED_CAMERAS = int(os.getenv("ML_SMOKE_EXPECTED_CAMERAS", "6"))


def _json(path: str, timeout: float = 4.0) -> dict:
    request = Request(f"{BASE_URL}{path}", headers={"Connection": "close"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path}: expected JSON object")
    return payload


def main() -> int:
    print(f"=== API service integration smoke: {BASE_URL} ===", flush=True)

    health = _json("/health")
    if health.get("service") != "api_service" or health.get("status") != "ok":
        print(f"API_SMOKE=FAIL api health={health}", flush=True)
        return 1
    print(f"[API] PASS {health}", flush=True)

    ml = _json("/api/v1/ml/health")
    if ml.get("status") != "playing" or int(ml.get("online_camera_count") or 0) != EXPECTED_CAMERAS:
        print(f"API_SMOKE=FAIL ml health={ml}", flush=True)
        return 1
    detector = dict(ml.get("detector") or {})
    tracker = dict(ml.get("tracker") or {})
    if detector.get("state") != "ready" or tracker.get("state") != "ready":
        print(f"API_SMOKE=FAIL ml stages detector={detector} tracker={tracker}", flush=True)
        return 1
    print(
        f"[ML] PASS online={ml.get('online_camera_count')}/{ml.get('camera_count')} "
        f"detector={detector.get('state')} tracker={tracker.get('state')}",
        flush=True,
    )

    camera_payload = _json("/api/v1/cameras")
    cameras = list(camera_payload.get("cameras") or [])
    if len(cameras) != EXPECTED_CAMERAS:
        print(f"API_SMOKE=FAIL expected {EXPECTED_CAMERAS} cameras, got {len(cameras)}", flush=True)
        return 1

    for row in cameras:
        camera_id = str(row["id"])
        detections = _json(f"/api/v1/cameras/{camera_id}/detections")
        tracks = _json(f"/api/v1/cameras/{camera_id}/tracks")
        detection_result = dict(detections.get("result") or {})
        track_result = dict(tracks.get("result") or {})
        print(
            f"[CAM] {camera_id} detections={detection_result.get('people', 0)} "
            f"tracks={track_result.get('people', 0)}",
            flush=True,
        )

    try:
        _json("/api/v1/cameras/CAM-DOES-NOT-EXIST/tracks")
    except HTTPError as exc:
        if exc.code != 404:
            print(f"API_SMOKE=FAIL unknown-camera status={exc.code}", flush=True)
            return 1
        print("[404] PASS unknown camera remains 404 through API", flush=True)
    else:
        print("API_SMOKE=FAIL unknown camera unexpectedly returned success", flush=True)
        return 1

    print("API_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
