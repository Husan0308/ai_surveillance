#!/usr/bin/env python3
"""Validate CAM-01 + CAM-02 + CAM-03 + CAM-04 + CAM-05 + CAM-06 together on DeepStream 9.1 with no AI."""
from pathlib import Path
import argparse
import base64
import fcntl
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.shared.camera_config import load_settings


def main(*, person_detection: bool = False) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=int, default=60 if person_detection else 660)
    ap.add_argument("--latency-ms", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / (".runtime/yolo26m-person-nms-short" if person_detection else ".runtime/cam01-cam02-cam03-cam04-cam05-cam06-validation"))
    ap.add_argument("--interrupt-camera", choices=["none", "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"], default="none")
    ap.add_argument("--interrupt-at", type=int, default=20)
    ap.add_argument("--interrupt-seconds", type=int, default=12)
    ap.add_argument("--trace-latency", action="store_true", help="enable GStreamer pipeline/element latency tracer")
    ap.add_argument("--no-preview", action="store_true", help="do not auto-open the live ffplay preview")
    ap.add_argument("--preview-port", type=int, default=5600, help="host UDP port for live preview")
    args = ap.parse_args()
    if args.duration < 20:
        ap.error("duration must be >=20 seconds")
    if args.latency_ms < 100:
        ap.error("latency must be >=100 ms for this validation profile")
    if not 1024 <= args.preview_port <= 65535:
        ap.error("preview port must be between 1024 and 65535")
    if args.interrupt_camera != "none":
        if args.interrupt_at < 10 or args.interrupt_seconds < 5:
            ap.error("interruption requires --interrupt-at >=10 and --interrupt-seconds >=5")
        if args.interrupt_at + args.interrupt_seconds + 15 >= args.duration:
            ap.error("duration must leave at least 15 seconds after interruption")

    if person_detection and args.latency_ms != 100:
        ap.error("the validated detection transport uses exactly 100 ms")
    out = args.out.resolve()
    if person_detection and (out / "pipeline.log").exists():
        raise RuntimeError("Use a new output directory; detector evidence must not be overwritten")
    out.mkdir(parents=True, exist_ok=True)

    preview_enabled = not args.no_preview
    if preview_enabled and not os.environ.get("DISPLAY"):
        print("GROUP PREVIEW disabled: DISPLAY is unavailable", flush=True)
        preview_enabled = False
    if preview_enabled and shutil.which("ffplay") is None:
        print("GROUP PREVIEW disabled: ffplay is not installed", flush=True)
        preview_enabled = False

    if person_detection and not preview_enabled:
        raise RuntimeError("Detection gate requires the existing realtime ffplay preview")

    lock = open("/tmp/ai_surveillance_camera_v2_gpu.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    cfg = Path(os.getenv("CAMERA_CONFIG", str(ROOT / "config/cameras.yaml")))
    settings = load_settings(cfg)
    wanted = {"CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"}
    cameras = {c.camera_id: c for c in settings.cameras if c.camera_id in wanted}
    if set(cameras) != wanted:
        raise RuntimeError("CAM-01 through CAM-06 must all exist and be enabled")

    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
    )
    from scripts.preflight_deepstream91 import version
    name, driver = [s.strip() for s in gpu.splitlines()[0].split(",")]
    if name != platform["gpu_name"] or version(driver) < version(platform["minimum_driver"]):
        raise RuntimeError("CAM group GPU/driver platform gate failed")

    image = platform["image"]
    base = [
        "docker", "run", "--rm", "--pull=never",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]
    detection_build = []
    detection_flags = ""
    if person_detection:
        from scripts.yolo26m_person.runtime import validate_artifacts
        validate_artifacts(ROOT)
        detection_build = ["-v", f"{ROOT / 'scripts/yolo26m_person'}:/detector:ro",
                           "-v", f"{ROOT / '.runtime/build-deps/cuda13.2'}:/cuda-headers:ro"]
        detection_flags = (" -DYOLO26_PERSON -I/detector "
            "-I/opt/nvidia/deepstream/deepstream/sources/includes "
            "-I/cuda-headers/usr/local/cuda-13.2/targets/x86_64-linux/include "
            "-L/opt/nvidia/deepstream/deepstream/lib "
            "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib -lnvdsgst_meta -lnvds_meta -ldl")
    build = base + detection_build + [
        "--network=none",
        "-v", f"{ROOT / 'scripts/cam_six_validation'}:/src:ro",
        "-v", f"{out}:/out",
        "--entrypoint", "bash",
        image,
        "-c",
        "g++ -O2 -std=c++17 -Wall -Wextra /src/main.cpp -o /out/cam-six-validator "
        "$(pkg-config --cflags --libs gstreamer-1.0)" + detection_flags,
    ]
    subprocess.run(build, check=True)

    secret = out / "camera-six-secret.ini"
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for cid in ("CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"):
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
        f.write(f"preview_enabled={str(preview_enabled).lower()}\n")
        f.write(f"preview_port={args.preview_port}\n")

    container = "ai-surveillance-cam-six-validation"
    detection_mounts = [] if not person_detection else [
        "-v", f"{ROOT / '.runtime/models/yolo26m'}:/models:ro",
        "-v", f"{ROOT / 'config/deepstream'}:/config:ro"]
    cmd = base + detection_mounts + [
        "--name", container,
        "--hostname", socket.gethostname(),
        "--gpus", "device=0",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "-e", ("GST_DEBUG=GST_TRACER:7" if args.trace_latency else "GST_DEBUG=1"),
        "-e", "GST_DEBUG_NO_COLOR=1",
        "-e", "GST_REGISTRY=/tmp/cam-six-gst-registry.bin",
        *([] if not args.trace_latency else ["-e", "GST_TRACERS=latency(flags=pipeline+element)"]),
        *([] if not preview_enabled else ["--add-host", "host.docker.internal:host-gateway"]),
        "-v", f"{out}:/work",
        "--entrypoint", "/work/cam-six-validator",
        image,
        "/work/camera-six-secret.ini",
        str(args.duration),
    ]

    done = threading.Event()

    def monitor() -> None:
        with (out / "gpu.csv").open("w") as f:
            f.write(
                "time,name,driver,memory_used_mib,gpu_pct,decoder_pct,encoder_pct,"
                "process_memory_used_mib,container_pid\n"
            )
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

                container_pid = 0
                inspect = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Pid}}", container],
                    capture_output=True,
                    text=True,
                )
                if inspect.returncode == 0:
                    try:
                        container_pid = int(inspect.stdout.strip() or "0")
                    except ValueError:
                        container_pid = 0

                process_memory = ""
                if container_pid > 0:
                    proc = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    for row in proc.stdout.splitlines():
                        parts = [part.strip() for part in row.split(",")]
                        if len(parts) != 2:
                            continue
                        try:
                            pid = int(parts[0])
                        except ValueError:
                            continue
                        if pid == container_pid:
                            process_memory = parts[1]
                            break

                f.write(
                    f"{time.time():.3f}," + r.stdout.strip()
                    + f",{process_memory},{container_pid}\n"
                )
                f.flush()
                done.wait(5)

    t = threading.Thread(target=monitor, daemon=True)
    p = None
    preview = None

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
        if preview_enabled:
            preview_url = (
                f"udp://0.0.0.0:{args.preview_port}"
                "?fifo_size=65536&overrun_nonfatal=1"
            )
            preview = subprocess.Popen(
                [
                    "ffplay",
                    "-hide_banner",
                    "-loglevel", "warning",
                    "-fflags", "nobuffer",
                    "-flags", "low_delay",
                    "-framedrop",
                    "-an",
                    "-sync", "video",
                    "-window_title", "CAM-01 + CAM-02 + CAM-03 + CAM-04 + CAM-05 + CAM-06 LIVE",
                    preview_url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(
                f"GROUP PREVIEW opening ffplay on UDP {args.preview_port}; use --no-preview to disable",
                flush=True,
            )
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
        result = p.wait()
        if person_detection:
            alive = preview is not None and preview.poll() is None
            (out / "preview.json").write_text(json.dumps({"enabled": preview_enabled, "alive_at_end": alive}) + "\n")
            if not alive:
                raise RuntimeError("Realtime ffplay preview exited during detection gate")
        return result
    finally:
        done.set()
        t.join(timeout=6)
        if p is not None and p.poll() is None:
            stop(None, None)
        if preview is not None and preview.poll() is None:
            preview.terminate()
            try:
                preview.wait(timeout=3)
            except subprocess.TimeoutExpired:
                preview.kill()
        secret.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
