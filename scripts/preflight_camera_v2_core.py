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
        raise RuntimeError(f"camera wall requires 1..6 enabled cameras, got {camera_count}")

    ensure_bridge()
    ensure_heatmap_filter()

    print(f"CAMERA_PREFLIGHT cameras={camera_count} core=PASS heatmap=PASS")
    print("CAMERA_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
