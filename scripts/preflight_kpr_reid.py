from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.kpr_reid_verifier import KPRPairVerifier


def make_person(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((320, 128, 3), dtype=np.uint8)
    image[:] = (32, 34, 36)
    image[20:75, 43:85] = (80, 105, 145)
    image[75:185, 22:106] = (32, 82, 180)
    image[185:300, 28:62] = (70, 62, 52)
    image[185:300, 67:101] = (72, 64, 54)
    noise = rng.integers(0, 5, image.shape, dtype=np.uint8)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def wait_ready(verifier: KPRPairVerifier, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        verifier.poll()
        snap = verifier.snapshot()
        if snap["ready"]:
            return True
        if verifier.error:
            return False
        time.sleep(0.20)
    return False


def wait_result(verifier: KPRPairVerifier, a, b, timeout: float = 90.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = verifier.authorization(a, b)
        if state in {"approved", "blocked", "unavailable"}:
            return state
        time.sleep(0.15)
    return "timeout"


def main() -> int:
    os.environ.setdefault("CAMERA_V2_KPR", "1")
    os.environ.setdefault("CAMERA_V2_KPR_REQUIRED", "1")
    verifier = KPRPairVerifier()
    try:
        # Production is lazy, but preflight explicitly starts the isolated worker.
        verifier.start()
        if not wait_ready(verifier):
            snap = verifier.snapshot()
            print(f"KPR_PREFLIGHT=FAIL startup={snap}")
            return 2

        snap0 = verifier.snapshot()
        print(
            "KPR_PREFLIGHT_START "
            f"backend={snap0['backend']} pid={snap0['worker_pid']} "
            f"fallbacks={snap0['fallbacks']} error={snap0['error'] or 'none'}"
        )

        a, b = (0, 101), (3, 202)
        image = make_person(7)

        verifier.remember(a, image, 0.95)
        verifier.remember(b, image.copy(), 0.95)
        first_deadline = time.monotonic() + 90.0
        while time.monotonic() < first_deadline:
            verifier.authorization(a, b)
            verifier.poll()
            if verifier.snapshot()["responses"] >= 1:
                break
            time.sleep(0.15)
        snap1 = verifier.snapshot()
        if snap1["responses"] < 1 or snap1["failed"]:
            print(f"KPR_PREFLIGHT=FAIL first={snap1}")
            return 2

        time.sleep(0.55)
        verifier.remember(a, make_person(8), 0.96)
        verifier.remember(b, make_person(8), 0.96)
        state = wait_result(verifier, a, b, timeout=90.0)
        snap = verifier.snapshot()
        print(
            "KPR_PREFLIGHT_DIAG "
            f"backend={snap['backend']} pid={snap['worker_pid']} fallbacks={snap['fallbacks']} "
            f"score={snap['score']:.3f} distance={snap['distance']:.3f} "
            f"parts={snap['visible_parts']} latency_ms={snap['latency_ms']:.0f} "
            f"responses={snap['responses']} same={snap['same']} different={snap['different']} "
            f"worker_exit={snap['worker_exit']} error={snap['error'] or 'none'}"
        )
        if state != "approved":
            print(f"KPR_PREFLIGHT=FAIL state={state}")
            return 2
        print("KPR_PREFLIGHT=PASS")
        return 0
    finally:
        verifier.close()


if __name__ == "__main__":
    raise SystemExit(main())
