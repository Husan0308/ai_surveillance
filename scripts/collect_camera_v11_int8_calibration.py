#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.ml_service.app.config import load_settings  # noqa: E402

OUT_W = 672
CONTENT_H = 378
OUT_H = 384
PAD_Y = 3


def auth_uri(uri: str, username: str, password: str) -> str:
    parts = urlsplit(uri)
    if "@" in parts.netloc or not username:
        return uri
    user = quote(username, safe="")
    pw = quote(password or "", safe="")
    auth = user if not password else f"{user}:{pw}"
    return urlunsplit((parts.scheme, f"{auth}@{parts.netloc}", parts.path, parts.query, parts.fragment))


def safe_uri(uri: str) -> str:
    parts = urlsplit(uri)
    host = parts.hostname or "unknown"
    port = f":{parts.port}" if parts.port else ""
    return urlunsplit((parts.scheme, f"{host}{port}", parts.path, parts.query, parts.fragment))


def count_ppm(path: Path) -> int:
    return sum(1 for _ in path.glob("*.ppm"))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect representative V11 office-camera frames for TRT8.6 INT8 calibration"
    )
    ap.add_argument(
        "--output",
        default="artifacts/yolo26s_trt86/int8_calibration_b1",
        help="Calibration directory (PPM RGB frames are written per camera)",
    )
    ap.add_argument("--per-camera", type=int, default=90)
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("V11_INT8_CALIB FAIL ffmpeg missing")
    if args.per_camera < 1:
        raise SystemExit("V11_INT8_CALIB FAIL --per-camera must be >=1")
    if not (0.05 <= args.sample_fps <= 10.0):
        raise SystemExit("V11_INT8_CALIB FAIL --sample-fps must be 0.05..10")

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    if args.reset and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(ROOT / "config/cameras.yaml")
    cameras = list(settings.cameras)
    if len(cameras) != 6:
        raise SystemExit(f"V11_INT8_CALIB FAIL expected 6 enabled cameras, got {len(cameras)}")

    existing = sum(count_ppm(out / c.camera_id) for c in cameras if (out / c.camera_id).exists())
    if existing and not args.reset:
        raise SystemExit(
            f"V11_INT8_CALIB FAIL output already contains {existing} PPM frames; rerun with --reset"
        )

    print(
        "V11_INT8_CALIB_START "
        f"cameras={len(cameras)} per_camera={args.per_camera} target={len(cameras)*args.per_camera} "
        f"sample_fps={args.sample_fps:.3f} geometry={OUT_W}x{CONTENT_H}+pad{PAD_Y}/{PAD_Y} "
        "format=ppm-rgb sequential=1",
        flush=True,
    )

    manifest: list[str] = []
    for camera in cameras:
        cid = camera.camera_id
        camera_dir = out / cid
        camera_dir.mkdir(parents=True, exist_ok=True)
        pattern = camera_dir / f"{cid}_%04d.ppm"
        uri = auth_uri(camera.uri, camera.username, camera.password)

        vf = (
            f"fps={args.sample_fps:.6f},"
            f"scale={OUT_W}:{CONTENT_H}:flags=bicubic,"
            f"pad={OUT_W}:{OUT_H}:0:{PAD_Y}:color=0x727272"
        )
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-rw_timeout",
            "5000000",
            "-i",
            uri,
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(args.per_camera),
            "-pix_fmt",
            "rgb24",
            "-f",
            "image2",
            "-vcodec",
            "ppm",
            str(pattern),
        ]
        print(
            f"V11_INT8_CALIB_CAMERA camera={cid} status=START uri={safe_uri(camera.uri)}",
            flush=True,
        )
        proc = subprocess.run(cmd, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        count = count_ppm(camera_dir)
        if proc.returncode != 0 or count != args.per_camera:
            err = (proc.stderr or "").strip().replace("\n", " | ")[-500:]
            raise SystemExit(
                f"V11_INT8_CALIB FAIL camera={cid} returncode={proc.returncode} "
                f"frames={count}/{args.per_camera} error={err or 'none'}"
            )
        for frame in sorted(camera_dir.glob("*.ppm")):
            manifest.append(f"{cid}\t{frame.relative_to(out)}")
        print(
            f"V11_INT8_CALIB_CAMERA camera={cid} status=OK frames={count}",
            flush=True,
        )

    manifest_path = out / "manifest.tsv"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")
    total = len(manifest)
    if total < 500:
        raise SystemExit(f"V11_INT8_CALIB FAIL representative_frames={total} expected>=500")

    print(
        "V11_INT8_CALIB_RESULT "
        f"status=PASS frames={total} cameras={len(cameras)} per_camera={args.per_camera} "
        f"manifest={manifest_path} geometry={OUT_W}x{OUT_H} rgb=1 pad114=1",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
