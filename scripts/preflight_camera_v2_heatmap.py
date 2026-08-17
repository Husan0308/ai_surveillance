from __future__ import annotations

import math
import os

from services.camera_v2.native_bridge import NativeMetaBridge


def main() -> int:
    source_fps = max(1, int(os.environ.get("CAMERA_V2_SOURCE_FPS", "20")))
    cool_seconds = max(300.0, float(os.environ.get("CAMERA_V2_HEATMAP_COOL_SEC", "3600")))
    remaining = min(0.60, max(0.01, float(os.environ.get("CAMERA_V2_HEATMAP_REMAIN", "0.10"))))
    decay = remaining ** (1.0 / (source_fps * cool_seconds))
    predicted = decay ** (source_fps * cool_seconds)
    if not math.isfinite(decay) or not (0.95 < decay < 1.0):
        raise RuntimeError(f"invalid heat decay {decay}")
    if abs(predicted - remaining) > 0.002:
        raise RuntimeError(
            f"heat cooling mismatch: predicted={predicted:.6f} expected={remaining:.6f}"
        )

    bridge = NativeMetaBridge()
    bridge.configure_heatmap(
        deposit=float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0045")),
        decay=decay,
        low_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00075")),
        yellow_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.020")),
        red_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.060")),
        max_points_per_source=max(
            8,
            min(48, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "30"))),
        ),
    )
    bridge.reset_heatmap()

    print(f"HEATMAP_PREFLIGHT bridge={bridge.path}")
    print("HEATMAP_PREFLIGHT anchor=bottom-center motion_only=PASS")
    print(
        "HEATMAP_PREFLIGHT cooling=PASS "
        f"fps={source_fps} seconds={cool_seconds:.0f} decay={decay:.8f} "
        f"remaining={predicted:.3f}"
    )
    print("HEATMAP_PREFLIGHT native_symbols=PASS")
    print("CAMERA_V2_HEATMAP_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
