from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v2.heatmap_filter import ensure_heatmap_filter
from services.camera_v2.native_bridge import ensure_bridge
from services.ml_service.app.config import load_settings


def main() -> int:
    settings = load_settings()
    camera_count = len(settings.cameras)
    if not 1 <= camera_count <= 6:
        raise RuntimeError(f"core camera wall requires 1..6 enabled cameras, got {camera_count}")

    native = ensure_bridge()
    heat_filter = ensure_heatmap_filter()

    print(
        "CAMERA_V2_CORE_PREFLIGHT "
        f"cameras={camera_count} native=PASS path={native} "
        f"heatmap_filter=PASS path={heat_filter}"
    )
    print(
        "CAMERA_V2_CORE_PREFLIGHT "
        "runtime=DeepStream/NVDEC+YOLO26m+NvDCF+native-heatmap "
        "pose=OFF reid=OFF qwen=OFF face=OFF"
    )
    print("CAMERA_V2_CORE_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
