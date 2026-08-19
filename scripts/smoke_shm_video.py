from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gi


gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from services.frontend.app.config import load_settings  # noqa: E402

Gst.init(None)


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pull_one(camera_id: str, socket_path: Path, width: int, height: int, fps: int) -> int:
    text = " ".join(
        [
            "shmsrc",
            f"socket-path={_gst_quote(str(socket_path))}",
            "is-live=true",
            "do-timestamp=true",
            "!",
            f"video/x-raw,format=BGRx,width={width},height={height},framerate={fps}/1",
            "!",
            "appsink",
            "name=sink",
            "sync=false",
            "drop=true",
            "max-buffers=1",
            "wait-on-eos=false",
        ]
    )
    pipeline = Gst.parse_launch(text)
    sink = pipeline.get_by_name("sink")
    if sink is None:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError(f"{camera_id}: appsink missing")
    try:
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"{camera_id}: shmsrc pipeline failed to start")
        sample = sink.emit("try-pull-sample", 4 * Gst.SECOND)
        if sample is None:
            bus = pipeline.get_bus()
            message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message is not None and message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                raise RuntimeError(f"{camera_id}: {err.message} | {debug or ''}")
            raise RuntimeError(f"{camera_id}: no raw frame within 4s")
        caps = sample.get_caps()
        buffer = sample.get_buffer()
        if caps is None or buffer is None:
            raise RuntimeError(f"{camera_id}: sample missing caps/buffer")
        structure = caps.get_structure(0)
        got_width = int(structure.get_value("width"))
        got_height = int(structure.get_value("height"))
        got_format = str(structure.get_value("format"))
        if (got_width, got_height, got_format) != (width, height, "BGRx"):
            raise RuntimeError(
                f"{camera_id}: unexpected raw caps {got_width}x{got_height} {got_format}"
            )
        size = int(buffer.get_size())
        expected = width * height * 4
        if size < expected:
            raise RuntimeError(f"{camera_id}: raw buffer too small {size} < {expected}")
        return size
    finally:
        pipeline.set_state(Gst.State.NULL)


def main() -> int:
    settings = load_settings()
    shm_dir = Path(os.getenv("FRONTEND_SHM_VIDEO_DIR", settings.shm_video_dir))
    camera_ids = [f"CAM-{index:02d}" for index in range(1, 7)]
    print(
        f"=== Shared-memory video smoke: dir={shm_dir} "
        f"{settings.source_width}x{settings.source_height}@{settings.source_fps} ===",
        flush=True,
    )

    for camera_id in camera_ids:
        socket_path = shm_dir / f"{camera_id}.sock"
        if not socket_path.exists():
            print(f"SHM_VIDEO_SMOKE=FAIL {camera_id} socket missing: {socket_path}", flush=True)
            return 1
        try:
            size = _pull_one(
                camera_id,
                socket_path,
                settings.source_width,
                settings.source_height,
                settings.source_fps,
            )
        except Exception as exc:
            print(f"SHM_VIDEO_SMOKE=FAIL {type(exc).__name__}: {exc}", flush=True)
            return 1
        print(f"[SHM] {camera_id} raw-frame=PASS bytes={size}", flush=True)

    print("SHM_VIDEO_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
