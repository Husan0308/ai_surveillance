#!/usr/bin/env python3
"""Record a synchronized 120 s Dev Room camera pair for AMC/MV3DT calibration.

Dev Room is resolved from config/cameras.yaml. The current mapping is expected
to contain exactly two enabled cameras in room "Devs". Preview windows are opened with ffplay, then two ffmpeg recorder processes are
started back-to-back so both outputs cover the same 120-second wall-clock
window as closely as possible.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAMERA_CONFIG = ROOT / "config" / "cameras.yaml"


def load_dev_cameras() -> list[dict]:
    data = yaml.safe_load(CAMERA_CONFIG.read_text())
    cameras = [
        c for c in data.get("cameras", [])
        if c.get("enabled", True) and c.get("room") == "Devs"
    ]
    cameras.sort(key=lambda c: c["id"])
    if len(cameras) != 2:
        raise RuntimeError(
            f'Expected exactly 2 enabled cameras in room "Devs"; found {len(cameras)}'
        )
    return cameras


def require_tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required tool not found on PATH: {name}")
    return path


def safe_camera_summary(cameras: list[dict]) -> list[dict]:
    return [
        {
            "index": idx,
            "file": f"cam_{idx:02d}.mp4",
            "camera_id": cam["id"],
            "name": cam.get("name", ""),
            "room": cam.get("room", ""),
        }
        for idx, cam in enumerate(cameras)
    ]


def launch_preview(ffplay: str, cam: dict, x: int) -> subprocess.Popen:
    # Keep the preview command intentionally minimal. The recorder is the source
    # of truth; preview should not depend on codec-tuning AVOptions.
    return subprocess.Popen(
        [
            ffplay,
            "-hide_banner",
            "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-framedrop",
            "-window_title", f'{cam["id"]} - {cam.get("name", "")}',
            "-x", "960",
            "-y", "540",
            "-left", str(x),
            "-top", "80",
            cam["uri"],
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def preview_failure_message(proc: subprocess.Popen, cam: dict) -> str:
    try:
        _stdout, stderr = proc.communicate(timeout=1)
    except subprocess.TimeoutExpired:
        stderr = ""
    # Do not surface a credential-bearing RTSP URL even if ffplay echoed it.
    uri = cam.get("uri", "")
    if uri:
        stderr = stderr.replace(uri, "<RTSP_URI>")
    stderr = stderr.strip()
    return stderr or f'{cam["id"]}: ffplay exited with code {proc.returncode}'


def terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=120)
    ap.add_argument("--countdown", type=int, default=5)
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

    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    ffplay = None if args.no_preview else require_tool("ffplay")

    cameras = load_dev_cameras()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (ROOT / ".runtime" / "mv3dt" / f"dev-room-{stamp}")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    summary = safe_camera_summary(cameras)
    print("Dev Room calibration capture:")
    for item in summary:
        print(
            f'  {item["file"]}: {item["camera_id"]} '
            f'({item["name"]})'
        )
    print(f"Output: {out}")
    print(f"Duration: {args.duration} seconds")
    print("RTSP URLs are intentionally not printed.")

    previews: list[subprocess.Popen] = []
    recorders: list[subprocess.Popen] = []

    def cleanup(*_args) -> None:
        for recorder in recorders:
            if recorder.poll() is None:
                recorder.send_signal(signal.SIGINT)
        for recorder in recorders:
            if recorder.poll() is None:
                try:
                    recorder.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    recorder.kill()
                    recorder.wait(timeout=3)
        for proc in previews:
            terminate_process(proc)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        if ffplay:
            print("\nOpening live previews...")
            previews.append(launch_preview(ffplay, cameras[0], 0))
            previews.append(launch_preview(ffplay, cameras[1], 970))
            time.sleep(3)
            dead = [i for i, p in enumerate(previews) if p.poll() is not None]
            if dead:
                details = [
                    preview_failure_message(previews[i], cameras[i])
                    for i in dead
                ]
                raise RuntimeError(
                    "Preview failed to stay open:\n  " + "\n  ".join(details)
                )

        print("\nWalk through the room during the whole capture.")
        print("Try to visit both cameras' shared/overlap area and room edges.")
        for sec in range(args.countdown, 0, -1):
            print(f"Recording starts in {sec}...", flush=True)
            time.sleep(1)

        output0 = out / "cam_00.mp4"
        output1 = out / "cam_01.mp4"

        def recorder_cmd(cam: dict, output: Path) -> list[str]:
            return [
                ffmpeg,
                "-hide_banner",
                "-loglevel", "warning",
                "-y",
                "-rtsp_transport", "tcp",
                "-use_wallclock_as_timestamps", "1",
                "-i", cam["uri"],
                "-map", "0:v:0",
                "-an",
                "-c:v", "copy",
                "-t", str(args.duration),
                "-movflags", "+faststart",
                str(output),
            ]

        print("\nRECORDING STARTED — walk now.")
        launch_started = time.monotonic()
        recorders.append(subprocess.Popen(recorder_cmd(cameras[0], output0)))
        recorders.append(subprocess.Popen(recorder_cmd(cameras[1], output1)))
        launch_skew_ms = (time.monotonic() - launch_started) * 1000.0
        started = time.monotonic()

        while any(proc.poll() is None for proc in recorders):
            elapsed = int(time.monotonic() - started)
            remaining = max(0, args.duration - elapsed)
            print(
                f"  elapsed={elapsed:3d}s  remaining={remaining:3d}s",
                end="\r",
                flush=True,
            )
            time.sleep(1)

        print()
        bad = [
            (idx, proc.returncode)
            for idx, proc in enumerate(recorders)
            if proc.returncode != 0
        ]
        if bad:
            raise RuntimeError(f"ffmpeg recorder failure(s): {bad}")

        results = []
        for idx, path in enumerate((output0, output1)):
            probe = subprocess.run(
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
            info = json.loads(probe.stdout)
            duration = float(info["format"]["duration"])
            if duration < args.duration - 2:
                raise RuntimeError(
                    f"{path.name} is too short: {duration:.3f}s "
                    f"(expected about {args.duration}s)"
                )
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
            "recorder_launch_skew_ms": launch_skew_ms,
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
                f'  {Path(result["path"]).name}: '
                f'{float(fmt["duration"]):.3f}s, '
                f'{stream["width"]}x{stream["height"]}, '
                f'{stream["codec_name"]}'
            )
        print(f"  manifest: {out / 'capture_manifest.json'}")
        return 0
    finally:
        for proc in previews:
            terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
