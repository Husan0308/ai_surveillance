#!/usr/bin/env python3
"""Run the DeepStream 9.1 camera-only diagnostic with all six cameras or one explicitly selected camera."""
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
from scripts.cam_six_validation.credential_redaction import (
    CredentialRedactor,
    child_environment_without_camera_secrets,
)
from scripts.cam_six_validation.stage_capture_manifest import finalize_stage_capture
from scripts.cam_six_validation.camera_selection import (
    CAMERA_IDS,
    camera_ids_for_run,
    recording_filename,
    validate_camera_mode,
)


def _run_sanitized(args: list[str], redactor: CredentialRedactor, *, check: bool = False):
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        env=child_environment_without_camera_secrets(),
    )
    stdout = redactor.sanitize_child_output(result.stdout or "")
    stderr = redactor.sanitize_child_output(result.stderr or "")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
    if check and result.returncode:
        raise subprocess.CalledProcessError(
            result.returncode, redactor.sanitize_args(args), output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(redactor.sanitize_args(args), result.returncode, stdout, stderr)


class _RedactingArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, redactor: CredentialRedactor, **kwargs):
        super().__init__(*args, **kwargs)
        self._redactor = redactor

    def _print_message(self, message, file=None):
        if message:
            (file or sys.stderr).write(self._redactor.sanitize_child_output(message))


def _main(*, person_detection: bool = False, person_tracking: bool = False,
         redactor: CredentialRedactor) -> int:
    ap = _RedactingArgumentParser(description=__doc__, redactor=redactor)
    if person_tracking and not person_detection:
        raise RuntimeError("person_tracking requires person_detection")
    default_out = (
        ".runtime/yolo26m-nvdcf-short"
        if person_tracking
        else ".runtime/yolo26m-person-nms-short"
        if person_detection
        else ".runtime/cam01-cam02-cam03-cam04-cam05-cam06-validation"
    )
    ap.add_argument("--duration", type=int, default=60 if person_detection else 660)
    ap.add_argument("--latency-ms", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    help="fresh evidence directory (required with --only-camera)")
    ap.add_argument("--only-camera", choices=CAMERA_IDS,
                    help="run the diagnostic pipeline with exactly this camera; default uses all six")
    ap.add_argument("--interrupt-camera", choices=["none", "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06"], default="none")
    ap.add_argument("--interrupt-at", type=int, default=20)
    ap.add_argument("--interrupt-seconds", type=int, default=12)
    ap.add_argument("--trace-latency", action="store_true", help="enable GStreamer pipeline/element latency tracer")
    ap.add_argument(
        "--capture-stage-samples",
        action="store_true",
        help="opt-in bounded raw snapshots at decoded source, pre-NVENC mosaic, and post-MKV decode",
    )
    ap.add_argument("--no-preview", action="store_true", help="do not auto-open the live ffplay preview")
    ap.add_argument("--preview-port", type=int, default=5600, help="host UDP port for live preview")
    args = ap.parse_args()
    try:
        selected_camera_ids = camera_ids_for_run(args.only_camera)
        validate_camera_mode(args.only_camera, person_detection)
    except ValueError as exc:
        ap.error(str(exc))
    if args.out is None:
        if args.only_camera:
            ap.error("--out is required with --only-camera so output is explicit and isolated")
        args.out = ROOT / default_out
    if args.duration < 20:
        ap.error("duration must be >=20 seconds")
    if args.capture_stage_samples and args.duration < 57:
        ap.error("stage sample capture requires duration >=57 seconds for the 5/30/55-second points")
    if args.latency_ms < 100:
        ap.error("latency must be >=100 ms for this validation profile")
    if not 1024 <= args.preview_port <= 65535:
        ap.error("preview port must be between 1024 and 65535")
    if args.interrupt_camera != "none":
        if args.only_camera:
            ap.error("--interrupt-camera is available only in the six-camera mode")
        if args.interrupt_at < 10 or args.interrupt_seconds < 5:
            ap.error("interruption requires --interrupt-at >=10 and --interrupt-seconds >=5")
        if args.interrupt_at + args.interrupt_seconds + 15 >= args.duration:
            ap.error("duration must leave at least 15 seconds after interruption")

    if person_detection and args.latency_ms != 100:
        ap.error("the validated detection transport uses exactly 100 ms")
    out = args.out.resolve()
    if args.only_camera and out.exists():
        raise RuntimeError("single-camera diagnostic output path must not already exist")
    if person_detection and (out / "pipeline.log").exists():
        raise RuntimeError("Use a new output directory; detector evidence must not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    if args.capture_stage_samples:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError("stage capture requires local ffmpeg and ffprobe for offline MKV decode")
        capture_dir = out / "stage-captures"
        if capture_dir.exists():
            raise RuntimeError("stage-captures already exists; use a fresh output directory")

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
    wanted = set(selected_camera_ids)
    cameras = {c.camera_id: c for c in settings.cameras if c.camera_id in wanted}
    if set(cameras) != wanted:
        if args.only_camera:
            raise RuntimeError("selected camera must exist and be enabled")
        raise RuntimeError("CAM-01 through CAM-06 must all exist and be enabled")
    for camera in cameras.values():
        redactor.add_credentials(
            username=camera.username,
            password=camera.password,
            uri=camera.uri,
        )
    if args.interrupt_camera != "none" and args.interrupt_camera not in wanted:
        ap.error("--interrupt-camera must be one of the selected cameras")

    platform = json.loads((ROOT / "config/deepstream-platform.json").read_text())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        text=True,
        env=child_environment_without_camera_secrets(),
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
    stage_capture_flags = ""
    if args.capture_stage_samples:
        stage_capture_flags = (
            " -DSTAGE_CAPTURE -pthread"
            " -I/opt/nvidia/deepstream/deepstream/sources/includes"
            " -L/opt/nvidia/deepstream/deepstream/lib"
            " -Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib"
            " -lnvdsgst_meta -lnvds_meta -ldl"
        )
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
        if person_tracking:
            detection_build += ["-v", f"{ROOT / 'scripts/yolo26m_tracker'}:/tracker:ro"]
            detection_flags += " -DYOLO26_TRACKER -I/tracker"
            tracker_host_cfg = ROOT / "config/deepstream/config_tracker_NvDCF_stable_person.yml"
            tracker_text = tracker_host_cfg.read_text()
            required_tracker = {
                "associationMatcherType: 1": "cascaded association",
                "maxShadowTrackingAge: 90": "extended shadow tracking",
                "useColorNames: 1": "ColorNames visual features",
                "useHog: 1": "HOG visual features",
                "featureImgSizeLevel: 3": "higher-resolution visual features",
                "reidType: 0": "ReID must remain disabled in tracker stage",
            }
            missing_tracker = [
                why for needle, why in required_tracker.items()
                if needle not in tracker_text
            ]
            if missing_tracker:
                raise RuntimeError(
                    "Unsafe NvDCF stable-person config: " + "; ".join(missing_tracker)
                )
            tracker_cfg = (
                "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
                "config_tracker_NvDCF_perf.yml"
            )
            tracker_lib = (
                "/opt/nvidia/deepstream/deepstream/lib/"
                "libnvds_nvmultiobjecttracker.so"
            )
            tracker_preflight = base + [
                "--gpus", "device=0",
                "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
                "-e", "GST_REGISTRY=/tmp/nvtracker-preflight-registry.bin",
                "--network=none",
                "--entrypoint", "bash",
                image,
                "-c",
                f"test -r {tracker_cfg} && test -r {tracker_lib} && "
                f"grep -q 'VisualTracker' {tracker_cfg} && "
                f"! grep -Eq '^[[:space:]]*reidType:[[:space:]]*[1-9]' {tracker_cfg} && "
                "gst-inspect-1.0 nvtracker > /tmp/nvtracker-inspect.txt && "
                "grep -q 'tracker-width' /tmp/nvtracker-inspect.txt && "
                "grep -q 'tracker-height' /tmp/nvtracker-inspect.txt && "
                "grep -q 'll-lib-file' /tmp/nvtracker-inspect.txt && "
                "grep -q 'll-config-file' /tmp/nvtracker-inspect.txt && "
                "grep -q 'compute-hw' /tmp/nvtracker-inspect.txt && "
                "grep -q 'tracking-id-reset-mode' /tmp/nvtracker-inspect.txt && "
                "! grep -q 'enable-batch-process' /tmp/nvtracker-inspect.txt && "
                "! grep -q 'enable-past-frame' /tmp/nvtracker-inspect.txt",
            ]
            _run_sanitized(tracker_preflight, redactor, check=True)
    build = base + detection_build + [
        "--network=none",
        "-v", f"{ROOT / 'scripts/cam_six_validation'}:/src:ro",
        "-v", f"{out}:/out",
        "--entrypoint", "bash",
        image,
        "-c",
        "g++ -O2 -std=c++17 -Wall -Wextra /src/main.cpp -o /out/cam-six-validator "
        "$(pkg-config --cflags --libs gstreamer-1.0)" + detection_flags + stage_capture_flags,
    ]
    _run_sanitized(build, redactor, check=True)

    secret = out / "camera-six-secret.ini"
    fd = os.open(secret, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        for cid in selected_camera_ids:
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

    container = (
        f"ai-surveillance-{args.only_camera.lower()}-diagnostic"
        if args.only_camera
        else "ai-surveillance-cam-six-validation"
    )
    detection_mounts = [] if not person_detection else [
        "-v", f"{ROOT / '.runtime/models/yolo26m'}:/models:ro",
        "-v", f"{ROOT / 'config/deepstream'}:/config:ro"]
    cmd = base + detection_mounts + [
        "--name", container,
        # The parent captures and sanitizes the attached stream before it is
        # shown or persisted. Disable Docker's parallel log-driver copy so
        # unsanitized GStreamer diagnostics cannot bypass that boundary.
        "--log-driver=none",
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
        *(["--capture-stage-samples"] if args.capture_stage_samples else []),
        *(["--only-camera", args.only_camera] if args.only_camera else []),
    ]

    done = threading.Event()

    def monitor() -> None:
        with (out / "gpu.csv").open("w") as f, (out / "gpu-query-events.jsonl").open("w") as query_events:
            f.write(
                "time,monotonic_ns,name,driver,memory_used_mib,gpu_pct,decoder_pct,encoder_pct,"
                "process_memory_used_mib,container_pid\n"
            )

            def record_query_event(query: str, status: str, monotonic_ns: int) -> None:
                query_events.write(json.dumps({
                    "wall_time": time.time(), "monotonic_ns": monotonic_ns,
                    "query": query, "status": status,
                }) + "\n")
                query_events.flush()

            while not done.is_set():
                sample_mono_ns = time.monotonic_ns()
                try:
                    r = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,driver_version,memory.used,utilization.gpu,"
                            "utilization.decoder,utilization.encoder",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        env=child_environment_without_camera_secrets(),
                    )
                    if r.returncode != 0:
                        record_query_event("device_gpu", f"exit_{r.returncode}", sample_mono_ns)
                        done.wait(5)
                        continue
                except subprocess.TimeoutExpired:
                    record_query_event("device_gpu", "timeout_2s", sample_mono_ns)
                    done.wait(5)
                    continue

                container_pid = 0
                try:
                    inspect = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.Pid}}", container],
                        capture_output=True,
                        text=True,
                        timeout=2,
                        env=child_environment_without_camera_secrets(),
                    )
                except subprocess.TimeoutExpired:
                    inspect = None
                    record_query_event("container_pid", "timeout_2s", sample_mono_ns)
                if inspect is not None and inspect.returncode == 0:
                    try:
                        container_pid = int(inspect.stdout.strip() or "0")
                    except ValueError:
                        container_pid = 0

                process_memory = ""
                if container_pid > 0:
                    try:
                        proc = subprocess.run(
                            [
                                "nvidia-smi",
                                "--query-compute-apps=pid,used_gpu_memory",
                                "--format=csv,noheader,nounits",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=2,
                            env=child_environment_without_camera_secrets(),
                        )
                        if proc.returncode == 0:
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
                        else:
                            record_query_event("process_gpu_memory", f"exit_{proc.returncode}", sample_mono_ns)
                    except subprocess.TimeoutExpired:
                        record_query_event("process_gpu_memory", "timeout_2s", sample_mono_ns)

                f.write(
                    f"{time.time():.3f},{sample_mono_ns},{redactor.sanitize(r.stdout.strip())},"
                    f"{process_memory},{container_pid}\n"
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
            env=child_environment_without_camera_secrets(),
        )

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

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
                    "-window_title", " + ".join(selected_camera_ids) + " LIVE",
                    preview_url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_environment_without_camera_secrets(),
            )
            print(
                f"GROUP PREVIEW opening ffplay on UDP {args.preview_port}; use --no-preview to disable",
                flush=True,
            )
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_environment_without_camera_secrets(),
        )
        assert p.stdout is not None
        with (out / "pipeline.log").open("w") as log:
            for line in p.stdout:
                line = redactor.sanitize_child_output(line)
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        result = p.wait()
        if args.capture_stage_samples:
            manifest = finalize_stage_capture(
                out / "stage-captures",
                out / recording_filename(tuple(selected_camera_ids)),
                redact_text=redactor.sanitize,
            )
            print(f"GROUP STAGE_CAPTURE finalized={manifest.relative_to(ROOT)}", flush=True)
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


def main(*, person_detection: bool = False, person_tracking: bool = False) -> int:
    """CLI-safe boundary for errors that may embed child or URI diagnostics."""
    redactor = CredentialRedactor()
    for key, value in os.environ.items():
        if (key.upper().endswith(("_RTSP_USERNAME", "_RTSP_PASSWORD"))
                or key.upper() in {"RTSP_USERNAME", "RTSP_PASSWORD"}):
            redactor.add_secret(value)
    try:
        return _main(
            person_detection=person_detection,
            person_tracking=person_tracking,
            redactor=redactor,
        )
    except Exception as exc:
        print(redactor.sanitize_exception(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
