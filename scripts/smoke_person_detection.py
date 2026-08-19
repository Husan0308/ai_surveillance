from __future__ import annotations

import json
import os
import time
from urllib.request import Request, urlopen

BASE_URL = os.getenv("ML_SMOKE_URL", "http://127.0.0.1:8001").rstrip("/")
TIMEOUT_SEC = float(os.getenv("PERSON_DETECT_SMOKE_TIMEOUT_SEC", "45"))
EXPECTED_CAMERAS = int(os.getenv("ML_SMOKE_EXPECTED_CAMERAS", "6"))


def get_json(path: str, timeout: float = 4.0) -> dict:
    req = Request(f"{BASE_URL}{path}", headers={"Connection": "close"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    print(f"=== Person detection smoke: {BASE_URL} ===", flush=True)
    deadline = time.monotonic() + TIMEOUT_SEC
    last_health: dict = {}

    while time.monotonic() < deadline:
        try:
            last_health = get_json("/health")
            detector = dict(last_health.get("detector") or {})
            state = str(detector.get("state", "unknown"))
            batches = int(detector.get("batches") or 0)
            images = int(detector.get("images") or 0)
            print(
                f"[WAIT] detector={state} batches={batches} images={images} "
                f"last_batch_ms={float(detector.get('last_batch_ms') or 0.0):.1f}",
                flush=True,
            )
            if state == "error":
                print(f"PERSON_DETECT_SMOKE=FAIL {detector.get('last_error')}", flush=True)
                return 1
            if state == "ready" and batches > 0 and images >= EXPECTED_CAMERAS:
                break
        except Exception as exc:
            print(f"[WAIT] {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1.0)
    else:
        print(f"PERSON_DETECT_SMOKE=FAIL timeout last_health={last_health}", flush=True)
        return 1

    cameras = get_json("/cameras")
    rows = list(cameras.get("cameras") or [])
    if len(rows) != EXPECTED_CAMERAS:
        print(f"PERSON_DETECT_SMOKE=FAIL expected {EXPECTED_CAMERAS} cameras, got {len(rows)}", flush=True)
        return 1

    for row in rows:
        camera_id = str(row.get("id"))
        detection = dict(row.get("detection") or {})
        print(
            f"[CAM] {camera_id} people={int(row.get('people') or 0)} "
            f"detect_fps={float(detection.get('fps') or 0.0):.2f} "
            f"age_ms={detection.get('age_ms')} frame_id={detection.get('frame_id')} ",
            flush=True,
        )
        if str(detection.get("state")) != "ready":
            print(f"PERSON_DETECT_SMOKE=FAIL {camera_id} detector state={detection}", flush=True)
            return 1
        if float(detection.get("fps") or 0.0) <= 0.0:
            print(f"PERSON_DETECT_SMOKE=FAIL {camera_id} has no detector FPS", flush=True)
            return 1
        if detection.get("age_ms") is None or float(detection["age_ms"]) > 5000.0:
            print(f"PERSON_DETECT_SMOKE=FAIL {camera_id} detection stale: {detection}", flush=True)
            return 1

        payload = get_json(f"/detections/{camera_id}")
        result = payload.get("result")
        if not isinstance(result, dict):
            print(f"PERSON_DETECT_SMOKE=FAIL {camera_id} has no detection result", flush=True)
            return 1
        if int(result.get("frame_id") or 0) <= 0:
            print(f"PERSON_DETECT_SMOKE=FAIL {camera_id} invalid result={result}", flush=True)
            return 1
        print(
            f"[DET] {camera_id} people={int(result.get('people') or 0)} "
            f"batch_ms={float(result.get('batch_ms') or 0.0):.1f}",
            flush=True,
        )

    print("PERSON_DETECT_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
