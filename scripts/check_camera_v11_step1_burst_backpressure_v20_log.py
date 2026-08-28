#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


POLICY = re.compile(
    r"CAMERA_V11_STEP1_V20_QUEUE policy=bounded-backpressure "
    r"max_buffers=1 leaky=0 growing_backlog=0 quality_changed=0 "
    r"effective=([^\s]+)"
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_camera_v11_step1_burst_backpressure_v20_log.py DISPLAY_LOG")
        return 2

    log = Path(sys.argv[1])
    if not log.is_file():
        print(f"V11_STEP1_V20 RESULT=FAIL missing_log={log}")
        return 2

    frozen_checker = Path(__file__).with_name("check_camera_v11_step1_v7_log.py")
    frozen = subprocess.run(
        [sys.executable, str(frozen_checker), str(log)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(frozen.stdout, end="")

    text = log.read_text(encoding="utf-8", errors="replace")
    policies = POLICY.findall(text)
    reasons: list[str] = []
    if frozen.returncode != 0:
        reasons.append("authoritative_v7_checker_failed")
    if not policies:
        reasons.append("missing_effective_queue_policy")
    else:
        effective: dict[str, int] = {}
        for item in policies[-1].split(","):
            try:
                camera, value = item.split(":", 1)
                effective[camera] = int(value)
            except ValueError:
                reasons.append(f"malformed_effective_policy={item}")
        expected = {f"CAM-{number:02d}" for number in range(1, 7)}
        if set(effective) != expected:
            reasons.append(f"effective_cameras={len(effective)}")
        for camera, value in sorted(effective.items()):
            if value != 0:
                reasons.append(f"{camera}:leaky={value}")

    if reasons:
        print("V11_STEP1_V20 RESULT=FAIL reasons=" + ";".join(reasons))
        return 1
    print("V11_STEP1_V20 RESULT=PASS policy=bounded-backpressure depth=1 leaky=0 cameras=6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
