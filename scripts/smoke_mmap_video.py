from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.frontend.app.config import load_settings
from shared.mmap_frame import MmapFrameReader, frame_path


def pull(camera_id: str, timeout_sec: float = 4.0):
    reader = MmapFrameReader(camera_id)
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline:
            if not reader.mapping_is_current() and not reader.attach():
                time.sleep(0.05)
                continue
            packet = reader.snapshot()
            if packet is not None:
                return packet
            time.sleep(0.01)
        return None
    finally:
        reader.close()


def main() -> int:
    settings = load_settings()
    camera_ids = [f"CAM-{index:02d}" for index in range(1, 7)]
    print(
        f"=== mmap video smoke: expected={settings.source_width}x{settings.source_height} "
        f"transport={settings.video_transport} ===",
        flush=True,
    )

    for camera_id in camera_ids:
        packet = pull(camera_id)
        if packet is None:
            print(f"MMAP_VIDEO_SMOKE=FAIL {camera_id} no frame path={frame_path(camera_id)}", flush=True)
            return 1
        if packet.channels != 3:
            print(f"MMAP_VIDEO_SMOKE=FAIL {camera_id} channels={packet.channels}", flush=True)
            return 1
        if (packet.width, packet.height) != (settings.source_width, settings.source_height):
            print(
                f"MMAP_VIDEO_SMOKE=FAIL {camera_id} got={packet.width}x{packet.height} "
                f"expected={settings.source_width}x{settings.source_height}",
                flush=True,
            )
            return 1
        print(
            f"[MMAP] {camera_id} frame=PASS {packet.width}x{packet.height} "
            f"seq={packet.sequence} frame_id={packet.frame_id} age_ms={packet.age_ms:.1f} "
            f"bytes={len(packet.payload)}",
            flush=True,
        )

    print("MMAP_VIDEO_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
