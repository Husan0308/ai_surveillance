#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))]


def number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dmon", required=True)
    parser.add_argument("--pidstat", required=True)
    args = parser.parse_args()

    gpu: dict[str, list[float]] = defaultdict(list)
    dmon = Path(args.dmon)
    if dmon.is_file():
        for line in dmon.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = line.split()
            if len(row) < 15:
                continue
            for name, index in (("sm", 4), ("mem", 5), ("enc", 6), ("nvdec", 7), ("fb_mib", 14)):
                value = number(row[index])
                if value is not None:
                    gpu[name].append(value)

    cpu_by_time: dict[str, float] = defaultdict(float)
    rss_by_time: dict[str, float] = defaultdict(float)
    pidstat = Path(args.pidstat)
    if pidstat.is_file():
        for line in pidstat.read_text(encoding="utf-8", errors="replace").splitlines():
            row = line.split()
            if len(row) < 15 or row[0] == "Linux" or row[0].startswith("#"):
                continue
            cpu = number(row[7])
            rss_kib = number(row[12])
            if cpu is None or rss_kib is None:
                continue
            cpu_by_time[row[0]] += cpu
            rss_by_time[row[0]] += rss_kib / 1024.0

    gpu_fields = []
    for name in ("sm", "mem", "enc", "nvdec", "fb_mib"):
        values = gpu[name]
        gpu_fields.append(f"{name}_avg={statistics.fmean(values) if values else 0.0:.1f}")
        gpu_fields.append(f"{name}_p95={pct(values, 0.95):.1f}")
    cpu_values = list(cpu_by_time.values())
    rss_values = list(rss_by_time.values())
    print(
        "CAMERA_V11_STEP2_RESOURCE "
        + " ".join(gpu_fields)
        + f" cpu_avg={statistics.fmean(cpu_values) if cpu_values else 0.0:.1f}%"
        + f" cpu_p95={pct(cpu_values, 0.95):.1f}%"
        + f" rss_avg={statistics.fmean(rss_values) if rss_values else 0.0:.1f}MiB"
        + f" rss_p95={pct(rss_values, 0.95):.1f}MiB"
        + f" gpu_samples={len(gpu['sm'])} cpu_samples={len(cpu_values)}",
        flush=True,
    )
    return 0 if gpu["sm"] and cpu_values else 1


if __name__ == "__main__":
    raise SystemExit(main())
