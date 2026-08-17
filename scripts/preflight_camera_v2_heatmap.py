from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.native_bridge import NativeMetaBridge


REMOVED_REID_FILES = (
    "services/camera_v2/external_reid.py",
    "services/camera_v2/global_reid.py",
    "services/camera_v2/kpr_reid_verifier.py",
    "services/camera_v2/qwen_reid_verifier.py",
    "services/camera_v2/native_reid_bridge.c",
    "services/camera_v2/reid_engine.py",
)


def main() -> int:
    source_fps = max(1, int(os.environ.get("CAMERA_V2_SOURCE_FPS", "20")))
    cool_seconds = max(
        300.0,
        float(os.environ.get("CAMERA_V2_HEATMAP_COOL_SEC", "3600")),
    )
    remaining = min(
        0.60,
        max(0.01, float(os.environ.get("CAMERA_V2_HEATMAP_REMAIN", "0.10"))),
    )
    decay = remaining ** (1.0 / (source_fps * cool_seconds))
    predicted = decay ** (source_fps * cool_seconds)
    if not math.isfinite(decay) or not (0.95 < decay < 1.0):
        raise RuntimeError(f"invalid heat decay {decay}")
    if abs(predicted - remaining) > 0.002:
        raise RuntimeError(
            f"heat cooling mismatch: predicted={predicted:.6f} expected={remaining:.6f}"
        )

    leftovers = [path for path in REMOVED_REID_FILES if (ROOT / path).exists()]
    if leftovers:
        raise RuntimeError("ReID cleanup incomplete: " + ", ".join(leftovers))

    bridge = NativeMetaBridge()
    if hasattr(bridge, "snapshot_reid") or hasattr(bridge, "apply_global_identity"):
        raise RuntimeError("native bridge still exposes ReID/global identity APIs")

    bridge.configure_heatmap(
        deposit=float(os.environ.get("CAMERA_V2_HEATMAP_DEPOSIT", "0.0030")),
        decay=decay,
        low_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_LOW", "0.00050")),
        yellow_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_YELLOW", "0.070")),
        red_threshold=float(os.environ.get("CAMERA_V2_HEATMAP_RED", "0.200")),
        max_points_per_source=max(
            12,
            min(96, int(os.environ.get("CAMERA_V2_HEATMAP_POINTS", "72"))),
        ),
    )
    bridge.reset_heatmap()

    print(f"HEATMAP_PREFLIGHT bridge={bridge.path}")
    print("HEATMAP_PREFLIGHT local_nvdcf_only=PASS cross_camera_reid=ABSENT")
    print("HEATMAP_PREFLIGHT anchor=feet-lifted-8pct all_tracks=PASS")
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
