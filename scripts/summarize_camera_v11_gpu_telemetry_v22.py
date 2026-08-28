#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    index = min(len(rows) - 1, max(0, round((len(rows) - 1) * q)))
    return float(rows[index])


def number(raw: str) -> float | None:
    text = raw.strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NA"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--label", default="gpu")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.is_file():
        raise SystemExit(f"V11_GPU_TELEMETRY FAIL missing={path}")

    fields = {
        "sm_mhz": [],
        "mem_mhz": [],
        "gpu_pct": [],
        "mem_pct": [],
        "temp_c": [],
        "power_w": [],
    }
    pstates: dict[str, int] = {}
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8:
                continue
            rows += 1
            # timestamp,pstate,sm_clock,mem_clock,gpu_util,mem_util,temp,power
            pstate = row[1].strip()
            pstates[pstate] = pstates.get(pstate, 0) + 1
            for key, index in (
                ("sm_mhz", 2),
                ("mem_mhz", 3),
                ("gpu_pct", 4),
                ("mem_pct", 5),
                ("temp_c", 6),
                ("power_w", 7),
            ):
                value = number(row[index])
                if value is not None:
                    fields[key].append(value)

    if rows == 0:
        raise SystemExit(f"V11_GPU_TELEMETRY FAIL no_rows={path}")

    parts = [f"label={args.label}", f"rows={rows}"]
    if pstates:
        parts.append("pstates=" + ",".join(f"{key}:{value}" for key, value in sorted(pstates.items())))
    for key, values in fields.items():
        if values:
            parts.extend(
                [
                    f"{key}_min={min(values):.1f}",
                    f"{key}_p50={pct(values, 0.50):.1f}",
                    f"{key}_p95={pct(values, 0.95):.1f}",
                    f"{key}_max={max(values):.1f}",
                ]
            )
    print("V11_GPU_TELEMETRY_RESULT " + " ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
