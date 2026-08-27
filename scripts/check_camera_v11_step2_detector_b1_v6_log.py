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
V5 = re.compile(
    r"CAMERA_V11_STEP2V5_STATS ready=(\d+) target_per_camera=([0-9.]+)Hz "
    r"global_actual=([0-9.]+)Hz rates=(\S+) capture_wait_p95=([0-9.]+)ms "
    r"trt_p50=([0-9.]+)ms trt_p95=([0-9.]+)ms roundtrip_p95=([0-9.]+)ms "
    r"result_age_p95=([0-9.]+)ms inferences=(\d+) timeouts=(\d+) batch=(\d+) "
    r"prefetch=(\d+) error=(\S+)"
)


def parse_rates(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in value.split(","):
        cid, raw = item.split(":", 1)
        out[cid] = float(raw)
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step2_detector_b1_v6_log.py /tmp/CAMERA_V11_STEP2V6.log")
        return 2
    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"V11_STEP2V6 FAIL missing_log={log_path}")
        return 2
    text = log_path.read_text(encoding="utf-8", errors="replace")

    for marker in (
        "CAMERA_V11_STEP2V6_PREFLIGHT",
        "CAMERA_V11_STEP2V6_INVARIANT",
        "CAMERA_V11_STEP2V6_TARGET",
        "CAMERA_V11_STEP2V6_ARCH",
        "CAMERA_V11_STEP2V6_POLICY",
        "CAMERA_V11_STEP2V5_DETECT_READY",
    ):
        if marker not in text:
            print(f"V11_STEP2V6 FAIL missing={marker}")
            return 2

    # Hard gate: the detector is acceptable only if frozen Step1 V7 still passes.
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
        print("V11_STEP2V6 RESULT diagnosis=FAIL_STEP1_REGRESSION")
        print("V11_STEP2V6 next=do not add tracker; reduce conversion/copy cost before more detector load")
        return 1

    latest_cam: dict[str, re.Match[str]] = {}
    for m in CAM.finditer(text):
        latest_cam[m.group(1)] = m
    rows = list(V5.finditer(text))
    if len(latest_cam) != 6 or not rows:
        print(f"V11_STEP2V6 FAIL detector_camera_rows={len(latest_cam)} stats_rows={len(rows)}")
        return 2

    reasons: list[str] = []
    s = rows[-1]
    ready = int(s.group(1))
    target = float(s.group(2))
    global_actual = float(s.group(3))
    rates = parse_rates(s.group(4))
    capture_wait95 = float(s.group(5))
    trt50 = float(s.group(6))
    trt95 = float(s.group(7))
    roundtrip95 = float(s.group(8))
    result_age95 = float(s.group(9))
    inferences = int(s.group(10))
    timeouts = int(s.group(11))
    batch = int(s.group(12))
    prefetch = int(s.group(13))
    error = s.group(14)

    if ready != 1:
        reasons.append(f"detector:ready={ready}")
    if not 1.9 <= target <= 2.1:
        reasons.append(f"detector:target={target:.2f}Hz")
    if global_actual < 10.0:
        reasons.append(f"detector:global_actual={global_actual:.2f}Hz")
    if capture_wait95 > 85.0:
        reasons.append(f"detector:capture_wait_p95={capture_wait95:.1f}ms")
    if trt50 > 42.0:
        reasons.append(f"detector:trt_p50={trt50:.1f}ms")
    if trt95 > 45.0:
        reasons.append(f"detector:trt_p95={trt95:.1f}ms")
    if roundtrip95 > 58.0:
        reasons.append(f"detector:roundtrip_p95={roundtrip95:.1f}ms")
    if result_age95 > 140.0:
        reasons.append(f"detector:result_age_p95={result_age95:.1f}ms")
    if inferences < 300:
        reasons.append(f"detector:inferences={inferences}")
    if timeouts > 6:
        reasons.append(f"detector:timeouts={timeouts}")
    if batch != 1:
        reasons.append(f"detector:batch={batch}")
    if prefetch != 0:
        reasons.append(f"detector:prefetch={prefetch}")
    if error != "none":
        reasons.append(f"detector:error={error}")

    for cid, m in sorted(latest_cam.items()):
        captures = int(m.group(3))
        results = int(m.group(4))
        conversion95 = float(m.group(5))
        camera_age95 = float(m.group(6))
        q = int(m.group(7))
        qmax = int(m.group(8))
        actual_rate = rates.get(cid, 0.0)
        if actual_rate < 1.7:
            reasons.append(f"{cid}:detect_hz={actual_rate:.2f}")
        if results < 40:
            reasons.append(f"{cid}:results={results}")
        if captures < results or captures - results > 3:
            reasons.append(f"{cid}:capture_result_delta={captures-results}")
        if conversion95 > 60.0:
            reasons.append(f"{cid}:conversion_p95={conversion95:.1f}ms")
        if camera_age95 > 140.0:
            reasons.append(f"{cid}:result_age_p95={camera_age95:.1f}ms")
        if q > 1 or qmax > 1:
            reasons.append(f"{cid}:detector_q={q}/qmax={qmax}")
        print(
            "V11_STEP2V6_CAMERA "
            f"camera={cid} rate={actual_rate:.2f}Hz captures={captures} results={results} "
            f"conversion_p95={conversion95:.1f}ms result_age_p95={camera_age95:.1f}ms qmax={qmax}"
        )

    print(
        "V11_STEP2V6_DETECTOR "
        f"ready={ready} target_per_camera={target:.2f}Hz global_actual={global_actual:.2f}Hz "
        f"capture_wait_p95={capture_wait95:.1f}ms trt_p50={trt50:.1f}ms "
        f"trt_p95={trt95:.1f}ms roundtrip_p95={roundtrip95:.1f}ms "
        f"result_age_p95={result_age95:.1f}ms inferences={inferences} "
        f"timeouts={timeouts} batch={batch} prefetch={prefetch} error={error}"
    )

    if reasons:
        print("V11_STEP2V6 RESULT diagnosis=FAIL_B1_2HZ reasons=" + ";".join(reasons))
        print("V11_STEP2V6 next=do not add tracker; optimize detector conversion/copy path")
        return 1

    print("V11_STEP2V6 RESULT diagnosis=PASS_B1_2HZ_DISPLAY_PROTECTED")
    print("V11_STEP2V6 next=freeze 2Hz detector duty; then optimize conversion before testing higher rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
