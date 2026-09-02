#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

from websockets.sync.client import connect


CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/v1/monitoring")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    previous = -1
    unique_sequences: set[int] = set()
    with connect(args.url, open_timeout=args.timeout, close_timeout=2.0) as websocket:
        for _ in range(args.count):
            raw = websocket.recv(timeout=args.timeout)
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > 256 * 1024:
                raise RuntimeError("payload is not bounded JSON text")
            payload = json.loads(raw)
            if payload.get("schema_version") != 1:
                raise RuntimeError("unexpected schema_version")
            sequence = payload.get("sequence")
            if not isinstance(sequence, int) or sequence < previous:
                raise RuntimeError("sequence is not monotonic")
            if [row.get("camera_id") for row in payload.get("cameras", [])] != CAMERA_IDS:
                raise RuntimeError("unexpected camera IDs")
            if payload.get("telemetry_status") != "fresh":
                raise RuntimeError(f"telemetry is not fresh: {payload.get('telemetry_status')}")
            generated = payload.get("generated_epoch_ms")
            if not isinstance(generated, int) or abs(time.time() * 1000.0 - generated) > 2500:
                raise RuntimeError("generated timestamp is not fresh")
            lower = raw.lower()
            if any(token in lower for token in ('"image"', 'base64', 'data:image')):
                raise RuntimeError("image data is forbidden in monitoring payload")
            previous = sequence
            unique_sequences.add(sequence)
    if len(unique_sequences) < 2:
        raise RuntimeError("telemetry sequence did not advance")
    print(
        "V11_MONITORING_WEBSOCKET RESULT=PASS "
        f"messages={args.count} unique_sequences={len(unique_sequences)} cameras=6 url={args.url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
