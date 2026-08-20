from __future__ import annotations

import http.client
import json
import os
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

API_BASE = os.getenv("FRONTEND_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VIDEO_BASE = os.getenv("FRONTEND_ML_VIDEO_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
EXPECTED_CAMERAS = int(os.getenv("ML_SMOKE_EXPECTED_CAMERAS", "6"))
TIMEOUT_SEC = float(os.getenv("FRONTEND_SMOKE_TIMEOUT_SEC", "30"))


def _json(path: str, timeout: float = 3.0) -> dict:
    request = Request(f"{API_BASE}{path}", headers={"Connection": "close"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object for {path}")
    return payload


def _first_mjpeg(camera_id: str, timeout: float = 4.0) -> int:
    parsed = urlsplit(VIDEO_BASE)
    if parsed.scheme != "http":
        raise RuntimeError("frontend MJPEG smoke currently expects http:// video base URL")
    connection = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 80, timeout=timeout)
    response = None
    try:
        connection.request(
            "GET",
            f"/video/{camera_id}",
            headers={"Connection": "close", "Cache-Control": "no-cache"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{camera_id}: MJPEG HTTP {response.status}")
        content_type = str(response.getheader("Content-Type") or "").lower()
        if "multipart/x-mixed-replace" not in content_type:
            raise RuntimeError(f"{camera_id}: unexpected content type {content_type!r}")

        while True:
            line = response.readline()
            if not line:
                raise EOFError(f"{camera_id}: MJPEG stream ended before boundary")
            if line.startswith(b"--frame"):
                break

        headers: dict[str, str] = {}
        while True:
            line = response.readline()
            if not line:
                raise EOFError(f"{camera_id}: MJPEG stream ended in headers")
            if line in (b"\r\n", b"\n"):
                break
            name, _, value = line.decode("latin-1").partition(":")
            if name and value:
                headers[name.strip().lower()] = value.strip()

        length = int(headers.get("content-length", "0"))
        if length <= 0 or length > 8 * 1024 * 1024:
            raise RuntimeError(f"{camera_id}: invalid JPEG length {length}")
        payload = response.read(length)
        if len(payload) != length or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise RuntimeError(f"{camera_id}: invalid JPEG payload")
        return length
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        connection.close()


def main() -> int:
    print(f"=== Frontend integration smoke: api={API_BASE} video={VIDEO_BASE} ===", flush=True)
    deadline = time.monotonic() + TIMEOUT_SEC
    last_error = ""
    while time.monotonic() < deadline:
        try:
            api = _json("/health")
            ml = _json("/api/v1/ml/health")
            cameras = _json("/api/v1/cameras")
            rows = list(cameras.get("cameras") or [])
            if api.get("status") == "ok" and ml.get("status") == "playing" and len(rows) == EXPECTED_CAMERAS:
                break
            last_error = f"api={api.get('status')} ml={ml.get('status')} cameras={len(rows)}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        print(f"[WAIT] {last_error}", flush=True)
        time.sleep(1.0)
    else:
        print(f"FRONTEND_SMOKE=FAIL services not ready: {last_error}", flush=True)
        return 1

    rows = list(cameras.get("cameras") or [])
    camera_ids = [str(row.get("id")) for row in rows]
    online = sum(1 for row in rows if row.get("online"))
    print(f"[API] PASS cameras={len(camera_ids)} online={online}/{len(camera_ids)}", flush=True)
    if len(camera_ids) != EXPECTED_CAMERAS or online != EXPECTED_CAMERAS:
        print("FRONTEND_SMOKE=FAIL not all cameras online through API", flush=True)
        return 1

    for camera_id in camera_ids:
        try:
            size = _first_mjpeg(camera_id)
        except Exception as exc:
            print(f"FRONTEND_SMOKE=FAIL {camera_id} video {type(exc).__name__}: {exc}", flush=True)
            return 1
        print(f"[VIDEO] {camera_id} direct-ml JPEG=PASS bytes={size}", flush=True)

    print("FRONTEND_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
