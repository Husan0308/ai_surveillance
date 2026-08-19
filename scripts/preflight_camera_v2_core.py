from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.heatmap_filter import ensure_heatmap_filter
from services.camera_v2.native_bridge import ensure_bridge
from services.ml_service.app.config import load_settings


def _require_source(path: str, guards: tuple[str, ...], label: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    for guard in guards:
        if guard not in source:
            raise RuntimeError(f"{label} guard missing: {guard}")


def main() -> int:
    settings = load_settings()
    camera_count = len(settings.cameras)
    if not 1 <= camera_count <= 6:
        raise RuntimeError(f"camera wall requires 1..6 enabled cameras, got {camera_count}")

    # Active Sentinel runtime must keep detector and NvDCF geometry close to the
    # 16:9 source. 512x288 is exact 16:9 and both dimensions are 32-aligned.
    _require_source(
        "services/camera_v2/person_tracking_heatmap.py",
        (
            'CAMERA_V2_DETECT_WIDTH", "736"',
            'CAMERA_V2_DETECT_HEIGHT", "416"',
            'CAMERA_V2_TRACKER_WIDTH", "512"',
            'CAMERA_V2_TRACKER_HEIGHT", "288"',
        ),
        "camera tracking geometry",
    )

    _require_source(
        "services/camera_v2/native_display_smoother.c",
        (
            "Intentionally a no-op",
            "return buffer_ptr ? 0 : -1",
        ),
        "current NvDCF bbox",
    )

    ensure_bridge()
    ensure_heatmap_filter()

    print(
        f"CAMERA_PREFLIGHT cameras={camera_count} core=PASS heatmap=PASS "
        "detector=736x416 tracker=512x288 custom_bbox_hold=OFF"
    )
    print("CAMERA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
