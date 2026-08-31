#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PREFIX = "CAMERA_V11_STEP4_REID_GALLERY_V1 "
INTEGER = re.compile(r"\b([a-z][a-z0-9_]+)=(\d+)\b")
MILLISECONDS = re.compile(
    r"\b(reid_queue_p50_ms|reid_queue_p95_ms|reid_infer_p50_ms|"
    r"reid_infer_p95_ms|gallery_update_p50_ms|gallery_update_p95_ms)="
    r"([0-9.]+)ms"
)
REQUIRED_COUNTERS = (
    "reid_submitted",
    "reid_completed",
    "reid_pending",
    "reid_replaced_pending",
    "reid_overflow_drop",
    "reid_stale_drop",
    "reid_worker_errors",
    "gallery_tracks",
    "gallery_samples",
    "gallery_tracks_ge3",
    "gallery_max_samples",
    "gallery_bootstrap_add",
    "gallery_diverse_add",
    "gallery_duplicate_drop",
    "gallery_quality_replace",
    "gallery_full_reject_or_replace",
    "gallery_invalid_reject",
)
REQUIRED_TIMINGS = (
    "reid_queue_p50_ms",
    "reid_queue_p95_ms",
    "reid_infer_p50_ms",
    "reid_infer_p95_ms",
    "gallery_update_p50_ms",
    "gallery_update_p95_ms",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display-log", required=True, type=Path)
    parser.add_argument("--gallery-log", required=True, type=Path)
    parser.add_argument("--warmup-windows", type=int, default=2)
    args = parser.parse_args()
    if not args.display_log.is_file() or not args.gallery_log.is_file():
        print("V11_STEP4_REID_GALLERY_V1 RESULT=FAIL reasons=missing_log")
        return 2

    root = Path(__file__).resolve().parents[1]
    reasons: list[str] = []
    step1 = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_camera_v11_step4_reid_quality_v1_log.py"),
            "--display-log",
            str(args.display_log),
            "--quality-log",
            str(args.gallery_log),
            "--warmup-windows",
            str(args.warmup_windows),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(step1.stdout, end="")
    if step1.returncode != 0:
        reasons.append("step1_quality_or_frozen_regression")

    text = args.gallery_log.read_text(encoding="utf-8", errors="replace")
    markers = (
        "CAMERA_V11_STEP4_REID_GALLERY_V1_ARCH",
        "source=step1-accepted-native-crops",
        "engine=known-good-resnet50-fp32-trt86",
        "embedding_dimension=256 normalized=required worker=one async=1",
        "camera_tracker_wait_for_reid=0",
        "queue=latest-only keyed_by=camera_id+local_track_id",
        "CAMERA_V11_STEP4_REID_GALLERY_V1_POLICY",
        "gallery_capacity=8 bootstrap=3 duplicate_cosine=0.975",
        "full_retention=quality50+nearest_diversity35+recency15 deterministic=1",
        "prototype_average=0",
        "pair_scoring=0 cross_camera_decision=0 reciprocal=0 one_to_one=0",
        "room_id=0 global_id=0 face=0 handoff=0",
    )
    for marker in markers:
        if marker not in text:
            reasons.append(f"missing_marker:{marker.split()[0]}")
    if "Traceback (most recent call last)" in text:
        reasons.append("runtime_traceback")
    if "V11_STEP4_REID_GALLERY_WORKER_FATAL" in text:
        reasons.append("worker_fatal")

    lines = [line for line in text.splitlines() if line.startswith(PREFIX)]
    if not lines:
        reasons.append("gallery_metrics_missing")
        final_line = ""
    else:
        final_line = lines[-1]
    counters = {name: int(value) for name, value in INTEGER.findall(final_line)}
    timings = {name: float(value) for name, value in MILLISECONDS.findall(final_line)}
    for name in REQUIRED_COUNTERS:
        if name not in counters:
            reasons.append(f"missing_counter:{name}")
    for name in REQUIRED_TIMINGS:
        if name not in timings:
            reasons.append(f"missing_timing:{name}")

    submitted = counters.get("reid_submitted", 0)
    completed = counters.get("reid_completed", 0)
    pending = counters.get("reid_pending", -1)
    replaced = counters.get("reid_replaced_pending", 0)
    overflow = counters.get("reid_overflow_drop", -1)
    stale = counters.get("reid_stale_drop", -1)
    errors = counters.get("reid_worker_errors", -1)
    if submitted <= 0:
        reasons.append("no_reid_submissions")
    if completed <= 0:
        reasons.append("no_reid_completions")
    if pending != 0:
        reasons.append(f"final_reid_pending={pending}")
    if overflow != 0:
        reasons.append(f"reid_overflow_drop={overflow}")
    if stale != 0:
        reasons.append(f"reid_stale_drop={stale}")
    if errors != 0:
        reasons.append(f"reid_worker_errors={errors}")
    terminal = completed + replaced + max(0, overflow) + max(0, stale) + max(0, pending)
    if submitted != terminal:
        reasons.append(f"reid_accounting={submitted}!={terminal}")

    tracks = counters.get("gallery_tracks", 0)
    samples = counters.get("gallery_samples", 0)
    tracks_ge3 = counters.get("gallery_tracks_ge3", 0)
    max_samples = counters.get("gallery_max_samples", 99)
    if tracks <= 0 or samples <= 0:
        reasons.append("no_gallery_state")
    if samples > tracks * 8:
        reasons.append(f"gallery_capacity_total={samples}/{tracks}")
    if max_samples > 8:
        reasons.append(f"gallery_capacity_track={max_samples}")
    if counters.get("gallery_bootstrap_add", 0) < 3:
        reasons.append("bootstrap_samples_below_3")
    if tracks_ge3 <= 0:
        reasons.append("no_persistent_track_with_3_samples")
    if counters.get("gallery_invalid_reject", -1) != 0:
        reasons.append(
            f"gallery_invalid_reject={counters.get('gallery_invalid_reject', -1)}"
        )
    if timings.get("reid_infer_p50_ms", 0.0) <= 0.0:
        reasons.append("no_reid_infer_timing")
    if timings.get("gallery_update_p50_ms", 0.0) <= 0.0:
        reasons.append("no_gallery_update_timing")

    pending_history = []
    for line in lines:
        values = {name: int(value) for name, value in INTEGER.findall(line)}
        if "reid_pending" in values:
            pending_history.append(values["reid_pending"])
    if not pending_history or max(pending_history) > 12:
        reasons.append(
            f"reid_queue_bound={max(pending_history) if pending_history else 'missing'}"
        )

    if reasons:
        print("V11_STEP4_REID_GALLERY_V1 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1
    print(
        "V11_STEP4_REID_GALLERY_V1 RESULT=PASS "
        f"reid_submitted={submitted} reid_completed={completed} "
        f"gallery_tracks={tracks} gallery_samples={samples} tracks_ge3={tracks_ge3} "
        f"bootstrap_add={counters['gallery_bootstrap_add']} "
        f"diverse_add={counters['gallery_diverse_add']} "
        f"duplicate_drop={counters['gallery_duplicate_drop']} "
        f"quality_replace={counters['gallery_quality_replace']} "
        f"queue_p95={timings['reid_queue_p95_ms']:.3f}ms "
        f"infer_p95={timings['reid_infer_p95_ms']:.3f}ms "
        f"gallery_p95={timings['gallery_update_p95_ms']:.3f}ms "
        "pending=0 overflow_drop=0 stale_drop=0 worker_errors=0 "
        "capacity=8 bootstrap=3 duplicate_cosine=0.975 pair_scoring=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
