#!/usr/bin/env python3
"""Core-v1 burn-in/soak telemetry collector.

Examples:
  python scripts/core_v1_soak.py --minutes 30
  python scripts/core_v1_soak.py --minutes 180 --interval 10 --output core_v1_soak.jsonl

The script only polls /health and writes JSONL; it does not touch camera state.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def fetch_json(url: str, timeout: float = 3.0):
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def pct(values, p: float):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None
    return values[min(len(values)-1, int((len(values)-1)*p))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8001")
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--output", default="core_v1_soak.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    deadline = time.monotonic() + max(1.0, args.minutes * 60.0)
    samples = []
    errors = 0
    print(f"Collecting Core-v1 health for {args.minutes:g} min -> {output}")

    with output.open("a", encoding="utf-8") as handle:
        while time.monotonic() < deadline:
            started = time.monotonic()
            try:
                health = fetch_json(args.base.rstrip("/") + "/health")
                row = {"wall_time": time.time(), "health": health}
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                samples.append(health)
                detector = health.get("detector") or {}
                finish = detector.get("finish_age_ms") or {}
                resources = detector.get("resources") or {}
                cameras = health.get("cameras") or {}
                worst_lag = max((float(v.get("pipeline_lag_ms") or 0.0) for v in cameras.values()), default=0.0)
                print(
                    f"samples={len(samples)} online={health.get('online')}/{health.get('total')} "
                    f"batch={detector.get('last_batch_ms')}ms finish_p95={finish.get('p95')}ms "
                    f"rss={resources.get('rss_mb')}MB vram={resources.get('gpu_process_memory_mb')}MB "
                    f"worst_pipe_lag={worst_lag:.1f}ms stale={detector.get('stale_result_drops')}"
                )
            except Exception as exc:
                errors += 1
                row = {"wall_time": time.time(), "error": f"{type(exc).__name__}: {exc}"}
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                handle.flush()
                print(row["error"])
            sleep_for = max(0.0, args.interval - (time.monotonic() - started))
            time.sleep(sleep_for)

    if not samples:
        raise SystemExit(f"No valid health samples; errors={errors}")

    detector_samples = [s.get("detector") or {} for s in samples]
    rss = [(d.get("resources") or {}).get("rss_mb") for d in detector_samples]
    vram = [(d.get("resources") or {}).get("gpu_process_memory_mb") for d in detector_samples]
    batch = [d.get("last_batch_ms") for d in detector_samples]
    stale = [int(d.get("stale_result_drops") or 0) for d in detector_samples]
    reconnects = []
    for sample in samples:
        reconnects.append(sum(int(v.get("reconnects") or 0) for v in (sample.get("cameras") or {}).values()))

    summary = {
        "samples": len(samples),
        "poll_errors": errors,
        "batch_ms_p50": pct(batch, .50),
        "batch_ms_p95": pct(batch, .95),
        "batch_ms_max": max((v for v in batch if v is not None), default=None),
        "rss_mb_first": next((v for v in rss if v is not None), None),
        "rss_mb_last": next((v for v in reversed(rss) if v is not None), None),
        "vram_mb_first": next((v for v in vram if v is not None), None),
        "vram_mb_last": next((v for v in reversed(vram) if v is not None), None),
        "stale_result_drops_delta": (stale[-1]-stale[0]) if stale else None,
        "camera_reconnects_delta": (reconnects[-1]-reconnects[0]) if reconnects else None,
    }
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
