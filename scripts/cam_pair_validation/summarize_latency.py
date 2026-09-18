#!/usr/bin/env python3
"""Summarize GStreamer latency-tracer output from a CAM pair pipeline log."""
import argparse
import json
from pathlib import Path
import re
import statistics

EVENT_RE = re.compile(
    r"(?P<kind>element-latency|latency),.*?"
    r"time=\(guint64\)(?P<time>\d+).*?"
    r"ts=\(guint64\)(?P<ts>\d+)"
)
SRC_RE = re.compile(r"src-element=\(string\)(?P<src>[^,;]+)")
SINK_RE = re.compile(r"sink-element=\(string\)(?P<sink>[^,;]+)")
ELEMENT_RE = re.compile(r"element=\(string\)(?P<element>[^,;]+)")


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(values_ns):
    values_ms = [v / 1_000_000.0 for v in values_ns]
    return {
        "samples": len(values_ms),
        "min_ms": min(values_ms),
        "p50_ms": pct(values_ms, 0.50),
        "p95_ms": pct(values_ms, 0.95),
        "p99_ms": pct(values_ms, 0.99),
        "mean_ms": statistics.mean(values_ms),
        "max_ms": max(values_ms),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pipeline_log", type=Path)
    ap.add_argument("--warmup-sec", type=float, default=10.0)
    args = ap.parse_args()

    pipeline = {}
    elements = {}
    total = 0
    invalid = 0

    for line in args.pipeline_log.read_text(errors="replace").splitlines():
        m = EVENT_RE.search(line)
        if not m:
            continue

        # Prefix is normally H:MM:SS.nanoseconds. We use it only to discard warmup.
        prefix = line.split()[0]
        try:
            hh, mm, ss = prefix.split(":")
            elapsed = int(hh) * 3600 + int(mm) * 60 + float(ss)
        except Exception:
            elapsed = None
        if elapsed is not None and elapsed < args.warmup_sec:
            continue

        time_ns = int(m.group("time"))
        kind = m.group("kind")
        total += 1

        # GST_CLOCK_TIME_NONE is UINT64_MAX. Aggregators can also yield
        # nonsensical wraparound values in element tracing; do not let those
        # corrupt percentiles/means. A single element taking >=10 seconds in
        # this 20 FPS graph is invalid telemetry, not a useful latency sample.
        if time_ns <= 0 or time_ns >= 10_000_000_000:
            invalid += 1
            continue

        if kind == "latency":
            src = SRC_RE.search(line)
            sink = SINK_RE.search(line)
            key = f"{src.group('src') if src else '?'} -> {sink.group('sink') if sink else '?'}"
            pipeline.setdefault(key, []).append(time_ns)
        else:
            element = ELEMENT_RE.search(line)
            key = element.group("element") if element else "?"
            elements.setdefault(key, []).append(time_ns)

    report = {
        "status": "PASS_PIPELINE" if pipeline else ("PASS_ELEMENT_ONLY" if elements else "NO_LATENCY_SAMPLES"),
        "warmup_sec": args.warmup_sec,
        "total_latency_events": total,
        "invalid_latency_events_ignored": invalid,
        "pipeline": {k: summarize(v) for k, v in sorted(pipeline.items())},
        "elements": {k: summarize(v) for k, v in sorted(elements.items())},
        "note": (
            "GStreamer tracer latency is pipeline processing latency between traced source/sink "
            "elements. It does not by itself include camera sensor exposure, NVR encoder delay, "
            "or network delay before the traced source."
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if (pipeline or elements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
