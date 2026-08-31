#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_camera_v11_step4_reid_same_room_matcher_v3.sh"
ACCEPTANCE = ROOT / "scripts/run_camera_v11_step4_reid_same_room_matcher_acceptance_v3.sh"


def fail(reason: str) -> int:
    print(f"V11_STEP4_MATCHER_V3_STATIC RESULT=FAIL reason={reason}")
    return 1


def main() -> int:
    for path in (LAUNCHER, ACCEPTANCE):
        if not path.is_file():
            return fail(f"missing:{path.name}")
        proc = subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=False)
        if proc.returncode != 0:
            return fail(f"bash_n:{path.name}")

    guard = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/check_camera_v11_frozen_step123_guard.py")],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(guard.stdout, end="")
    if guard.returncode != 0:
        return fail("frozen_step123_guard")

    text = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "mode=natural-floating",
        "powermizer_write=0",
        "nvidia_settings_keeper=0",
        "CAMERA_V11_STEP4_REID_MATCH_NATURAL_PRIME result=PASS",
        "natural_prime_memory_clock_",
        "V11_STEP4_MIN_ACTIVE_MEMORY_MHZ:-3000",
        "V11_STEP4_MIN_PRIME_GPU_UTIL:-50",
    )
    for marker in required:
        if marker not in text:
            return fail(f"missing_marker:{marker}")
    forbidden = (
        "v11_powermizer_start",
        "GPUPowerMizerMode=1",
        "source \"$ROOT/scripts/camera_v11_powermizer_keeper_v25.sh\"",
    )
    for marker in forbidden:
        if marker in text:
            return fail(f"forbidden_marker:{marker}")

    print("V11_STEP4_MATCHER_V3_STATIC RESULT=PASS natural_floating=1 hard_active_prime_gate=1 frozen_step123=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
