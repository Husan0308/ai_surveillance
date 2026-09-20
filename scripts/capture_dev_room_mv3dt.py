#!/usr/bin/env python3
"""Record the two Dev Room RTSP cameras for AMC/MV3DT calibration.

The project credential loader resolves RTSP usernames/passwords from .env.
GStreamer rtspsrc receives credentials via user-id/user-pw properties, so
camera secrets are not embedded in committed URIs or child-process argv.

Each camera pipeline tees the original H.264 elementary stream to:
  1. MP4 recording (stream copy; no re-encode)
  2. local MPEG-TS/UDP preview consumed by ffplay

The two GStreamer pipelines are moved to PLAYING back-to-back and recorded for
the same requested wall-clock interval.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.shared.camera_config import load_settings  # noqa: E402

try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst  # noqa: E402
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Python GStreamer bindings are missing. "
        "Install python3-gi/python3-gst-1.0 and use a venv with "
        "--system-site-packages."
    ) from exc

Gst.init(None)
CAMERA_CONFIG = ROOT / "config" / "cameras.yaml"


def _gst_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return path


def load_dev_cameras() -> list[dict]:
    raw = yaml.safe_load(CAMERA_CONFIG.read_text(encoding="utf-8")) or {}
    metadata = {
        str(row["id"]).strip(): row
        for row in raw.get("cameras", [])
        if bool(row.get("enabled", True))
    }

    settings = load_settings(CAMERA_CONFIG)
    result = []
    for camera in settings.cameras:
        row = metadata.get(camera.camera_id, {})
        if str(row.get("room", "")).strip() != "Devs":
            continue
        result.append(
            {
                "id": camera.camera_id,
                "name": str(row.get("name", camera.camera_id)),
                "room": "Devs",
                "uri": camera.uri,
                "username": camera.username,
                "password": camera.password,
            }
        )

    result.sort(key=lambda c: c["id"])
    if len(result) != 2:
        raise RuntimeError(
            f'Expected exactly 2 enabled cameras in room "Devs"; found {len(result)}'
        )
    missing = [c["id"] for c in result if not c["username"]]
    if missing:
        raise RuntimeError(
            "RTSP credentials were not resolved for: "
            + ", ".join(missing)
            + ". Check .env SURVEILLANCE_RTSP_USERNAME/PASSWORD or per-camera vars."
        )
    return result


def safe_camera_summary(cameras: list[dict]) -> list[dict]:
    return [
        {
            "index": idx,
            "file": f"cam_{idx:02d}.mp4",
            "camera_id": cam["id"],
            "name": cam["name"],
            "room": cam["room"],
        }
        for idx, cam in enumerate(cameras)
    ]


def check_gstreamer_plugins() -> None:
    required = (
        "rtspsrc",
        "rtph264depay",
        "h264parse",
        "tee",
        "queue",
        "mp4mux",
        "filesink",
        "mpegtsmux",
        "udpsink",
    )
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError(
            "Missing required GStreamer plugin(s): " + ", ".join(missing)
        )


def build_camera_pipeline(
    cam: dict,
    output: Path,
    preview_port: int,
    latency_ms: int,
) -> Gst.Pipeline:
    source_options = [
        f"location={_gst_quote(cam['uri'])}",
        f"latency={latency_ms}",
        "drop-on-latency=true",
        "protocols=tcp",
        f"user-id={_gst_quote(cam['username'])}",
        f"user-pw={_gst_quote(cam['password'])}",
    ]

    pipeline_text = " ".join(
        [
            "rtspsrc", "name=source", *source_options,
            "!", "application/x-rtp,media=video,encoding-name=H264",
            "!", "rtph264depay",
            "!", "h264parse", "config-interval=-1",
            "!", "tee", "name=t",
            "t.", "!", "queue",
            "!", "h264parse",
            "!", "mp4mux", "faststart=true",
            "!", "filesink", f"location={_gst_quote(str(output))}",
            "sync=false",
            "t.", "!", "queue",
            "!", "h264parse",
            "!", "mpegtsmux",
            "!", "udpsink", "host=127.0.0.1",
            f"port={preview_port}",
            "sync=false", "async=false",
        ]
    )

    try:
        pipeline = Gst.parse_launch(pipeline_text)
    except Exception as exc:
        # Never include pipeline_text here because it contains RTSP credentials.
        raise RuntimeError(
            f"{cam['id']}: failed to construct GStreamer capture pipeline: {exc}"
        ) from exc
    if not isinstance(pipeline, Gst.Pipeline):
        raise RuntimeError(f"{cam['id']}: GStreamer did not create a pipeline")
    return pipeline


def launch_preview(ffplay: str, cam: dict, port: int, left: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffplay,
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-framedrop",
            "-an",
            "-window_title", f"{cam['id']} - {cam['name']}",
            "-x", "960",
            "-y", "540",
            "-left", str(left),
            "-top", "80",
            f"udp://127.0.0.1:{port}?fifo_size=65536&overrun_nonfatal=1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def pop_pipeline_error(pipeline: Gst.Pipeline) -> str | None:
    bus = pipeline.get_bus()
    while True:
        message = bus.pop_filtered(Gst.MessageType.ERROR)
        if message is None:
            return None
        err, debug = message.parse_error()
        source = message.src.get_name() if message.src is not None else "unknown"
        return f"{source}: {err.message} | {debug or ''}"


def wait_for_eos(pipeline: Gst.Pipeline, timeout_sec: float = 10.0) -> None:
    bus = pipeline.get_bus()
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        message = bus.timed_pop_filtered(
            250 * Gst.MSECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )
        if message is None:
            continue
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            source = message.src.get_name() if message.src is not None else "unknown"
            raise RuntimeError(
                f"GStreamer error while finalizing {source}: "
                f"{err.message} | {debug or ''}"
            )
        if message.type == Gst.MessageType.EOS:
            return
    raise RuntimeError("Timed out waiting for MP4 pipeline EOS/finalization")


def probe_file(ffprobe: str, path: Path, requested_duration: int) -> dict:
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate:format=duration,size",
            "-of", "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    if duration < requested_duration - 3:
        raise RuntimeError(
            f"{path.name} is too short: {duration:.3f}s "
            f"(expected about {requested_duration}s)"
        )
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--latency-ms", type=int, default=100)
    ap.add_argument("--preview-port-base", type=int, default=5700)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory; default: .runtime/mv3dt/dev-room-<timestamp>",
    )
    ap.add_argument(
        "--no-preview",
        action="store_true",
        help="Record without ffplay preview windows.",
    )
    args = ap.parse_args()

    if args.duration < 60:
        raise RuntimeError("Calibration capture duration must be at least 60 seconds")
    if args.latency_ms < 20:
        raise RuntimeError("RTSP latency must be at least 20 ms")
    if not 1024 <= args.preview_port_base <= 65534:
        raise RuntimeError("preview-port-base must leave room for two UDP ports")

    ffprobe = require_tool("ffprobe")
    ffplay = None if args.no_preview else require_tool("ffplay")
    check_gstreamer_plugins()

    cameras = load_dev_cameras()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (ROOT / ".runtime" / "mv3dt" / f"dev-room-{stamp}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    summary = safe_camera_summary(cameras)
    print("Dev Room calibration capture:")
    for item in summary:
        print(f"  {item['file']}: {item['camera_id']} ({item['name']})")
    print(f"Output: {out}")
    print(f"Duration: {args.duration} seconds")
    print("Credentials resolved from project config; secrets are not printed.")

    outputs = [out / "cam_00.mp4", out / "cam_01.mp4"]
    ports = [args.preview_port_base, args.preview_port_base + 1]
    pipelines = [
        build_camera_pipeline(cameras[i], outputs[i], ports[i], args.latency_ms)
        for i in range(2)
    ]
    previews: list[subprocess.Popen] = []

    stopping = False

    def stop_all(*_args) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for pipeline in pipelines:
            try:
                pipeline.send_event(Gst.Event.new_eos())
            except Exception:
                pass
        for proc in previews:
            stop_process(proc)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    try:
        print("\nStarting authenticated GStreamer streams...")
        launch_started = time.monotonic()
        for idx, pipeline in enumerate(pipelines):
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(
                    f"{cameras[idx]['id']}: failed to enter PLAYING state"
                )
        pipeline_launch_skew_ms = (time.monotonic() - launch_started) * 1000.0

        # Allow RTSP negotiation/authentication and mux caps to settle.
        time.sleep(2)
        for idx, pipeline in enumerate(pipelines):
            detail = pop_pipeline_error(pipeline)
            if detail:
                raise RuntimeError(f"{cameras[idx]['id']}: {detail}")

        if ffplay:
            print("Opening local live previews...")
            previews.append(launch_preview(ffplay, cameras[0], ports[0], 0))
            previews.append(launch_preview(ffplay, cameras[1], ports[1], 970))
            time.sleep(2)

        print("\nRECORDING STARTED — walk now.")
        print("Walk across the whole Dev Room and through both cameras' shared area.")
        started = time.monotonic()

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= args.duration:
                break
            for idx, pipeline in enumerate(pipelines):
                detail = pop_pipeline_error(pipeline)
                if detail:
                    raise RuntimeError(f"{cameras[idx]['id']}: {detail}")
            remaining = max(0, int(args.duration - elapsed))
            print(
                f"  elapsed={int(elapsed):3d}s  remaining={remaining:3d}s",
                end="\r",
                flush=True,
            )
            time.sleep(1)

        print("\nFinalizing MP4 files...")
        for pipeline in pipelines:
            pipeline.send_event(Gst.Event.new_eos())
        for pipeline in pipelines:
            wait_for_eos(pipeline)

        results = []
        for idx, path in enumerate(outputs):
            info = probe_file(ffprobe, path, args.duration)
            results.append(
                {
                    **summary[idx],
                    "path": str(path),
                    "probe": info,
                }
            )

        manifest = {
            "status": "PASS",
            "purpose": "AutoMagicCalib / DeepStream MV3DT Dev Room calibration",
            "duration_requested_sec": args.duration,
            "pipeline_launch_skew_ms": pipeline_launch_skew_ms,
            "camera_mapping": summary,
            "files": results,
        }
        (out / "capture_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

        print("\nCAPTURE PASS")
        for result in results:
            fmt = result["probe"]["format"]
            stream = result["probe"]["streams"][0]
            print(
                f"  {Path(result['path']).name}: "
                f"{float(fmt['duration']):.3f}s, "
                f"{stream['width']}x{stream['height']}, "
                f"{stream['codec_name']}"
            )
        print(f"  pipeline_launch_skew_ms={pipeline_launch_skew_ms:.3f}")
        print(f"  manifest: {out / 'capture_manifest.json'}")
        return 0
    finally:
        for pipeline in pipelines:
            pipeline.set_state(Gst.State.NULL)
        for proc in previews:
            stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
