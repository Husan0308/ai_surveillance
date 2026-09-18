#!/usr/bin/env python3
"""Validate CAM-01 + CAM-02 together on DeepStream 9.1 with no AI."""
from pathlib import Path
import argparse
import base64
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.shared.camera_config import load_settings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=660)
    ap.add_argument("--latency-ms", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / ".runtime/cam01-cam02-validation")
    ap.add_argument("--interrupt-camera", choices=["none", "CAM-01", "CAM-02"], default="none")
    ap.add_argument("--interrupt-at", type=int, default=20)
    ap.add_argument("--interrupt-seconds", type=int, default=12)
    ap.add_argument("--trace-latency", action="store_true", help="enable GStreamer pipeline/element latency tracer")
    args = ap.parse_args()
    if args.duration < 20:
        ap.error("duration must be >=20 seconds")
    if args.latency_ms < 100:
        ap.error("latency must be >=100 ms for this validation profile")
    if args.interrupt_camera != "none":
        if args.interrupt_at < 10 or args.interrupt_seconds < 5:
            ap.error("interruption requires --interrupt-at >=10 and --interrupt-seconds >=5")
        if args.interrupt_at + args.interrupt_seconds + 15 >= args.duration:
            ap.error("duration must leave at least 15 seconds after interruption")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    lock = open("/tmp/ai_surveillance_camera_v2_gpu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    cfg = Path(os.getenv("CAMERA_CONFIG", str(ROOT / "config/cameras.yaml")))
    settings = load_settings(cfg)
    wanted = {"CAM-01", "CAM-02"}
    cameras = {c.camera_id: c for c in settings.cameras if c.camera_id in wanted}
    if set(cameras) != wanted:
        raise RuntimeError("CAM-01 and CAM-02 must both exist and be enabled")

    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    )
    from scripts.preflight_deepstream91 import version
    name, driver = [s.strip() for s in gpu.splitlines()[0].split(",")]
    if name != platform["gpu_name"] or version(driver) < version(platform["minimum_driver"]):
        raise RuntimeError("CAM pair GPU/driver platform gate failed")

    image = platform["image"]
    base = [
        "docker", "run", "--rm", "--pull=never",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    build = base + [
        "--network=none",
        "-v", f"{ROOT / 'scripts/cam_pair_validation'}:/src:ro",
        "-v", f"{out}:/out",
        "--entrypoint", "bash",
        image,
        "-c",
        "g++ -O2 -std=c++17 -Wall -Wextra /src/main.cpp -o /out/cam-pair-validator "
        "$(pkg-config --cflags --libs gstreamer-1.0)",
    ]
    subprocess.run(build, check=True)

    secret = out / "camera-pair-secret.ini"
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for cid in ("CAM-01", "CAM-02"):
            camera = cameras[cid]
            f.write(f"[{cid}]\n")
            for key, value in (
                ("uri", camera.uri),
                ("username", camera.username),
                ("password", camera.password),
            ):
                f.write(key + "=" + base64.b64encode(value.encode()).decode() + "\n")
            f.write(f"latency_ms={args.latency_ms}\n")
        f.write("[validation]\n")
        f.write(f"interrupt_camera={args.interrupt_camera}\n")
        f.write(f"interrupt_at={args.interrupt_at}\n")
        f.write(f"interrupt_seconds={args.interrupt_seconds}\n")

    container = "ai-surveillance-cam-pair-validation"
    cmd = base + [
        "--name", container,
        "--hostname", socket.gethostname(),
        "--gpus", "device=0",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", ("GST_DEBUG=GST_TRACER:7" if args.trace_latency else "GST_DEBUG=1"),
        "-e", "GST_DEBUG_NO_COLOR=1",
        "-e", "GST_REGISTRY=/tmp/cam-pair-gst-registry.bin",
        *([] if not args.trace_latency else ["-e", "GST_TRACERS=latency(flags=pipeline+element)"]),
        "-v", f"{out}:/work",
        "--entrypoint", "/work/cam-pair-validator",
        image,
        "/work/camera-pair-secret.ini",
        str(args.duration),
    ]

    done = threading.Event()

    def monitor() -> None:
        with (out / "gpu.csv").open("w") as f:
            f.write("time,name,driver,memory_used_mib,gpu_pct,decoder_pct,encoder_pct\n")
            while not done.is_set():
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,driver_version,memory.used,utilization.gpu,"
                        "utilization.decoder,utilization.encoder",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                f.write(f"{time.time():.3f}," + r.stdout.strip() + "\n")
                f.flush()
                done.wait(5)

    t = threading.Thread(target=monitor, daemon=True)
    p = None

    def stop(_sig, _frame) -> None:
        subprocess.run(
            ["docker", "stop", "--time", "10", container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    secrets = []
    for camera in cameras.values():
        secrets.extend([camera.password, camera.uri])

    try:
        t.start()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert p.stdout is not None
        with (out / "pipeline.log").open("w") as log:
            for line in p.stdout:
                for value in secrets:
                    if value:
                        line = line.replace(value, "<redacted>")
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        return p.wait()
    finally:
        done.set()
        t.join(timeout=6)
        if p is not None and p.poll() is None:
            stop(None, None)
        secret.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
