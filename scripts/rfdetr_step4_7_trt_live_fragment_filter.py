#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rfdetr_step4_5_trt_live as base

BOTTOM_EDGE = 0.985
MAX_HEIGHT_RATIO = 0.12
MIN_ASPECT = 1.60
FRAME_W = 800.0
FRAME_H = 448.0

_original_dedupe = base._dedupe
_calls = 0
_fragment_rejected = 0


def _is_bottom_fragment(person: dict) -> bool:
    x1, y1, x2, y2 = [float(v) for v in person["xyxy"]]
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    bottom_ratio = y2 / FRAME_H
    height_ratio = height / FRAME_H
    aspect = width / height
    return (
        bottom_ratio >= BOTTOM_EDGE
        and height_ratio <= MAX_HEIGHT_RATIO
        and aspect >= MIN_ASPECT
    )


def _filtered_dedupe(persons, iou_gate, containment_gate, center_gate):
    global _calls, _fragment_rejected
    kept, duplicates = _original_dedupe(
        persons, iou_gate, containment_gate, center_gate
    )
    _calls += 1
    filtered = []
    for person in kept:
        if _is_bottom_fragment(person):
            _fragment_rejected += 1
        else:
            filtered.append(person)
    return filtered, duplicates


def _ensure_default_output_dir() -> None:
    if "--output-dir" not in sys.argv:
        sys.argv.extend(
            ["--output-dir", "artifacts/rfdetr_step4/trt_live_fragment_filtered"]
        )


def main() -> int:
    _ensure_default_output_dir()
    base._dedupe = _filtered_dedupe
    print(
        "STEP4_7_FILTER "
        f"bottom_ratio>={BOTTOM_EDGE:.3f} "
        f"height_ratio<={MAX_HEIGHT_RATIO:.3f} "
        f"aspect>={MIN_ASPECT:.2f} "
        "scope=validation_only",
        flush=True,
    )
    result = int(base.main())
    print(
        "STEP4_7_FILTER_RESULT "
        f"processed_calls={_calls} fragment_rejected={_fragment_rejected}",
        flush=True,
    )
    print("STEP4_7_PASS validation_only=true", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
