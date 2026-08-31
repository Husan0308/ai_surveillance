#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51"
FROZEN_PATHS = (
    "services/camera_v11/step1_cam02_lowlat_v7.py",
    "services/camera_v11/step1_independent_egl_v4.py",
    "services/camera_v11/step2_production_fp32.py",
    "services/camera_v11/step2_production_fp32_v12.py",
    "services/camera_v11/step2_production_fp32_v13.py",
    "services/camera_v11/step2_production_fp32_v18.py",
    "services/camera_v11/step2_trt86.py",
    "services/camera_v11/step3_tracker_v2.py",
    "services/camera_v11/step3_tracking_v2.py",
    "scripts/yolo26_trt86_step2_worker.py",
    "scripts/run_camera_v11_step1_v7.sh",
    "scripts/check_camera_v11_step1_v7_log.py",
    "scripts/camera_v11_powermizer_keeper_v25.sh",
    "scripts/run_camera_v11_step2_production_fp32_v25.sh",
    "scripts/check_camera_v11_step1_v25_aggregate_log.py",
    "scripts/check_camera_v11_step2_production_log_v25.py",
    "scripts/run_camera_v11_step3_tracker_v2.sh",
    "scripts/check_camera_v11_step3_tracker_v2_log.py",
    "scripts/test_camera_v11_step3_tracker_v2.py",
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    reasons = []
    if run("git", "cat-file", "-e", f"{FROZEN_SHA}^{{commit}}").returncode != 0:
        reasons.append("frozen_sha_missing")
    elif run("git", "merge-base", "--is-ancestor", FROZEN_SHA, "HEAD").returncode != 0:
        reasons.append("head_not_based_on_frozen_sha")
    changed = run("git", "diff", "--name-only", FROZEN_SHA, "--", *FROZEN_PATHS)
    if changed.returncode != 0:
        reasons.append("git_diff_failed")
    else:
        reasons.extend(f"changed:{path}" for path in changed.stdout.splitlines() if path)
    missing = [path for path in FROZEN_PATHS if not (ROOT / path).is_file()]
    reasons.extend(f"missing:{path}" for path in missing)
    if reasons:
        print("V11_FROZEN_STEP123_GUARD RESULT=FAIL reasons=" + ";".join(reasons))
        return 1
    print(
        "V11_FROZEN_STEP123_GUARD RESULT=PASS "
        f"sha={FROZEN_SHA} files={len(FROZEN_PATHS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
