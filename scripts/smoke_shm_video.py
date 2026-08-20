from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gi


gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from services.frontend.app.config import load_settings  # noqa: E402

Gst.init(None)

ML_BASE = os.getenv("ML_SMOKE_URL", "http://127.0.0.1:8001").rstrip("/")


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _json(path: str) -> dict:
    request = Request(f"{ML_BASE}{path}", headers={"Connection": "close"})
    with urlopen(request, timeout=4.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object for {path}")
    return payload


def _pull_one(
    camera_id: str,
    socket_path: Path,
    width: int,
    height: int,
    fps: int,
    pixel_format: str,
) -> int:
    text = " ".join(
        [
            "shmsrc",
            f"socket-path={_gst_quote(str(socket_path))}",
            "is-live=true",
            "do-timestamp=true",
            "!",
            (
                f"video/x-raw,format={pixel_format},width={width},height={height},"
                f"framerate={fps}/1"
            ),
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
        if (got_width, got_height, got_format) != (width, height, pixel_format):
            raise RuntimeError(
                f"{camera_id}: unexpected raw caps {got_width}x{got_height} {got_format}"
            )
        size = int(buffer.get_size())
        if pixel_format == "NV12":
            expected = width * height * 3 // 2
        else:
            expected = width * height
        if size < expected:
            raise RuntimeError(f"{camera_id}: raw buffer too small {size} < {expected}")
        return size
    finally:
        pipeline.set_state(Gst.State.NULL)


def main() -> int:
    settings = load_settings()
    shm_dir = Path(os.getenv("FRONTEND_SHM_VIDEO_DIR", settings.shm_video_dir))
    try:
        cameras_payload = _json("/cameras")
    except Exception as exc:
        print(f"SHM_VIDEO_SMOKE=FAIL ML cameras unavailable: {type(exc).__name__}: {exc}", flush=True)
        return 1

    rows = [row for row in cameras_payload.get("cameras", []) if isinstance(row, dict)]
    print(f"=== Native shared-memory video smoke: dir={shm_dir} cameras={len(rows)} ===", flush=True)
    low_res: list[str] = []

    for row in rows:
        camera_id = str(row.get("id") or "")
        width = int(row.get("render_width") or 0)
        height = int(row.get("render_height") or 0)
        fps = max(1, int(round(float(row.get("render_fps") or settings.source_fps))))
        pixel_format = str(row.get("render_format") or "NV12").upper()
        analysis_width = int(row.get("width") or settings.source_width)
        analysis_height = int(row.get("height") or settings.source_height)
        if not camera_id or width <= 0 or height <= 0:
            print(
                f"SHM_VIDEO_SMOKE=FAIL {camera_id or '?'} native caps unavailable "
                f"render={width}x{height} format={pixel_format}",
                flush=True,
            )
            return 1

        socket_path = shm_dir / f"{camera_id}.sock"
        if not socket_path.exists():
            print(f"SHM_VIDEO_SMOKE=FAIL {camera_id} socket missing: {socket_path}", flush=True)
            return 1
        try:
            size = _pull_one(camera_id, socket_path, width, height, fps, pixel_format)
        except Exception as exc:
            print(f"SHM_VIDEO_SMOKE=FAIL {type(exc).__name__}: {exc}", flush=True)
            return 1

        if width <= analysis_width and height <= analysis_height:
            low_res.append(camera_id)
        print(
            f"[SHM] {camera_id} native={width}x{height}@{fps} {pixel_format} "
            f"analysis={analysis_width}x{analysis_height} bytes={size} PASS",
            flush=True,
        )

    if low_res:
        print(
            "[QUALITY] native stream is not higher-resolution than analysis for: "
            + ",".join(low_res),
            flush=True,
        )
    print("SHM_VIDEO_SMOKE=PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
