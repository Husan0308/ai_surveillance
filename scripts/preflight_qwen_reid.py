from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    base = os.environ.get("CAMERA_V2_QWEN_BASE", "http://127.0.0.1:8080").rstrip("/")
    health = base + "/health"
    print(f"QWEN_REID_PREFLIGHT health={health}")
    try:
        with urllib.request.urlopen(health, timeout=4.0) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(response.status)
    except Exception as exc:
        print(f"QWEN_REID_PREFLIGHT=FAIL error={type(exc).__name__}:{exc}")
        print("Start the local server first: bash scripts/run_qwen_reid_server.sh")
        return 2

    print(f"QWEN_REID_PREFLIGHT http={status} body={body[:300]}")
    if status != 200:
        print("QWEN_REID_PREFLIGHT=FAIL server_not_ready")
        return 3

    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and parsed.get("status") not in {None, "ok"}:
            print(f"QWEN_REID_PREFLIGHT=FAIL status={parsed.get('status')}")
            return 4
    except Exception:
        pass

    print("QWEN_REID_PREFLIGHT=PASS")
    print("runtime: CAMERA_V2_REID=1 CAMERA_V2_REID_BACKEND=external CAMERA_V2_QWEN_VERIFY=1 python -m services.camera_v2.person_tracking_qwen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
