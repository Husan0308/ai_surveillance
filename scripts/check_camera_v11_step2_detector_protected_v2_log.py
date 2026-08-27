#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CAM = re.compile(
    r"CAMERA_V11_STEP2_DETECT_CAMERA camera=(\S+) hz=([0-9.]+) captures=(\d+) "
    r"results=(\d+) conversion_p95=([0-9.]+)ms result_age_p95=([0-9.]+)ms "
    r"q=(\d+) qmax=(\d+) last_boxes=(\d+)"
)
V2 = re.compile(
    r"CAMERA_V11_STEP2V2_STATS ready=(\d+) target=([0-9.]+)Hz actual=([0-9.]+)Hz "
    r"capture_wait_p95=([0-9.]+)ms skew_p95=([0-9.]+)ms "
    r"trt_p95=([0-9.]+)ms roundtrip_p95=([0-9.]+)ms "
    r"result_age_p95=([0-9.]+)ms batches=(\d+) timeouts=(\d+) prefetch=(\d+) error=(\S+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step2_detector_protected_v2_log.py /tmp/CAMERA_V11_STEP2V2.log")
        return 2

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"V11_STEP2V2 FAIL missing_log={log_path}")
        return 2
    text = log_path.read_text(encoding="utf-8", errors="replace")

    for marker in (
        "CAMERA_V11_STEP2V2_PREFLIGHT",
        "CAMERA_V11_STEP2V2_INVARIANT",
        "CAMERA_V11_STEP2V2_TARGET",
        "CAMERA_V11_STEP2V2_POLICY",
        "CAMERA_V11_STEP2V2_DETECT_READY",
    ):
        if marker not in text:
            print(f"V11_STEP2V2 FAIL missing={marker}")
            return 2

    # Non-negotiable: detector may not regress the frozen Step1 V7 display.
    step1_checker = Path(__file__).with_name("check_camera_v11_step1_v7_log.py")
    proc = subprocess.run(
        [sys.executable, str(step1_checker), str(log_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        print("V11_STEP2V2 RESULT diagnosis=FAIL_STEP1_REGRESSION")
        print("V11_STEP2V2 next=do not add tracker; lower detector GPU duty or remove overlap")
        return 1

    latest_cam: dict[str, re.Match[str]] = {}
    for m in CAM.finditer(text):
        latest_cam[m.group(1)] = m
    v2_rows = list(V2.finditer(text))
    if len(latest_cam) != 6 or not v2_rows:
        print(
            f"V11_STEP2V2 FAIL detector_camera_rows={len(latest_cam)} "
            f"v2_stats_rows={len(v2_rows)}"
        )
        return 2

    reasons: list[str] = []
    for cid, m in sorted(latest_cam.items()):
        hz = float(m.group(2))
        captures = int(m.group(3))
        results = int(m.group(4))
        conversion95 = float(m.group(5))
        result_age95 = float(m.group(6))
        q = int(m.group(7))
        qmax = int(m.group(8))
        last_boxes = int(m.group(9))

        if hz < 7.0:
            reasons.append(f"{cid}:detect_hz={hz:.2f}")
        if results < 200:
            reasons.append(f"{cid}:results={results}")
        if captures < results or captures - results > 3:
            reasons.append(f"{cid}:capture_result_delta={captures-results}")
        if conversion95 > 70.0:
            reasons.append(f"{cid}:conversion_p95={conversion95:.1f}ms")
        if result_age95 > 140.0:
            reasons.append(f"{cid}:result_age_p95={result_age95:.1f}ms")
        if q > 1 or qmax > 1:
            reasons.append(f"{cid}:detector_q={q}/qmax={qmax}")

        print(
            "V11_STEP2V2_CAMERA "
            f"camera={cid} detect_hz={hz:.2f} captures={captures} results={results} "
            f"conversion_p95={conversion95:.1f}ms result_age_p95={result_age95:.1f}ms "
            f"qmax={qmax} last_boxes={last_boxes}"
        )

    s = v2_rows[-1]
    ready = int(s.group(1))
    target = float(s.group(2))
    actual = float(s.group(3))
    capture_wait95 = float(s.group(4))
    skew95 = float(s.group(5))
    trt95 = float(s.group(6))
    roundtrip95 = float(s.group(7))
    result_age95 = float(s.group(8))
    batches = int(s.group(9))
    timeouts = int(s.group(10))
    prefetch = int(s.group(11))
    error = s.group(12)

    if ready != 1:
        reasons.append(f"detector:ready={ready}")
    if not 7.5 <= target <= 8.5:
        reasons.append(f"detector:target={target:.2f}Hz")
    if actual < 7.0:
        reasons.append(f"detector:actual={actual:.2f}Hz")
    if capture_wait95 > 100.0:
        reasons.append(f"detector:capture_wait_p95={capture_wait95:.1f}ms")
    if skew95 > 80.0:
        reasons.append(f"detector:skew_p95={skew95:.1f}ms")
    if trt95 > 25.0:
        reasons.append(f"detector:trt_p95={trt95:.1f}ms")
    if roundtrip95 > 45.0:
        reasons.append(f"detector:roundtrip_p95={roundtrip95:.1f}ms")
    if result_age95 > 140.0:
        reasons.append(f"detector:result_age_p95={result_age95:.1f}ms")
    if batches < 200:
        reasons.append(f"detector:batches={batches}")
    if timeouts > 2:
        reasons.append(f"detector:timeouts={timeouts}")
    if prefetch != 0:
        reasons.append(f"detector:prefetch={prefetch}")
    if error != "none":
        reasons.append(f"detector:error={error}")

    print(
        "V11_STEP2V2_DETECTOR "
        f"ready={ready} target={target:.2f}Hz actual={actual:.2f}Hz "
        f"capture_wait_p95={capture_wait95:.1f}ms skew_p95={skew95:.1f}ms "
        f"trt_p95={trt95:.1f}ms roundtrip_p95={roundtrip95:.1f}ms "
        f"result_age_p95={result_age95:.1f}ms batches={batches} "
        f"timeouts={timeouts} prefetch={prefetch} error={error}"
    )

    if reasons:
        print("V11_STEP2V2 RESULT diagnosis=FAIL_PROTECTED_DETECTOR reasons=" + ";".join(reasons))
        print("V11_STEP2V2 next=do not add tracker; tune only detector duty/capture path")
        return 1

    print("V11_STEP2V2 RESULT diagnosis=PASS_PROTECTED_DETECTOR_ONLY")
    print("V11_STEP2V2 next=freeze Step2 V2; then add tracker in Step3 without changing display/detector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
