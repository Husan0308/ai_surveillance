#!/usr/bin/env python3
"""Exercise the real native recorder using six direct-file GPU decode sources."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
CAMERAS = tuple(f"CAM-{index:02d}" for index in range(1, 7))


def make_clip(path: Path, width: int, height: int, color: str, duration_sec: int) -> None:
    source_duration = duration_sec + 3
    source = f"color=c={color}:s={width}x{height}:r=20:d={source_duration},drawbox=x='mod(t*80\\,{width - 160})':y=120:w=160:h=240:color=white:t=fill"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source,
        "-frames:v", str(source_duration * 20), "-an", "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ll",
        "-rc", "constqp", "-qp", "30", "-bf", "0", "-g", "40", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], check=True)


def encoded(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def write_camera_config(path: Path) -> None:
    sources = {
        "CAM-01": "/offline-input/2560x1440.mp4",
        "CAM-02": "/offline-input/3200x1800.mp4",
        "CAM-03": "/offline-input/2560x1440.mp4",
        "CAM-04": "/offline-input/3200x1800.mp4",
        "CAM-05": "/offline-input/2560x1440.mp4",
        "CAM-06": "/offline-input/3200x1800.mp4",
    }
    lines = []
    for camera, source in sources.items():
        lines.extend((f"[{camera}]", f"uri={encoded('file://' + source)}",
                      "username=", "password=", "latency_ms=0", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def docker_command(candidate: Path, output: Path, sources: Path, recordings: Path,
                   duration_sec: int) -> list[str]:
    env = {
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "V11_CALIBRATION_RECORDING_ENABLED": "1",
        "V11_CALIBRATION_RECORDING_CONTROL_PATH": "/work/calibration-recording-command.tsv",
        "V11_CALIBRATION_RECORDING_STATUS_PATH": "/work/calibration-recording-status.json",
        "V11_CALIBRATION_RECORDING_ROOT": "/calibration_recordings",
        "PGIE_ISOLATION_SOURCE_COUNT": "6",
        "PGIE_ISOLATION_MUX_BATCH": "6",
        "PGIE_ISOLATION_MUX_WIDTH": "2560",
        "PGIE_ISOLATION_MUX_HEIGHT": "1440",
        "PGIE_ISOLATION_LIVE_SOURCE": "0",
        "PGIE_ISOLATION_BATCHED_PUSH_TIMEOUT_US": "1000",
        "PGIE_ISOLATION_REALTIME_PACE": "1",
        "V11_AUX_YOLO_ENABLE": "1",
        "V11_AUX_YOLO_CONFIG": "/config/config_infer_aux_yolo26m_768.txt",
        "V11_AUX_YOLO_METAMUX_CONFIG": "/config/config_metamux_yolo26m_aux768.txt",
        "V11_AUX_YOLO_INPUT_SIZE": "768",
        "V11_LOWCONF_VERIFIER_ENABLE": "1",
        "V11_LOWCONF_PRIMARY_CONFIG": "/config/config_infer_primary_yolo26m_raw_otm_lowconf023.txt",
        "V11_LOWCONF_VERIFIER_CONFIG": "/config/yolo26m_context_verifier_448.txt",
        "V11_ACTIVE_TRACK_VERIFIER_ENABLE": "1",
        "V11_CONTEXT_VERIFIER_NAME": "YOLO26m-448-FP16",
        "V11_CONTEXT_VERIFIER_BATCH_SIZE": "2",
    }
    command = ["docker", "run", "--rm", "--init", "--pull=never", "--network=host", "--gpus", "all",
               "--user", f"{os.getuid()}:{os.getgid()}"]
    for key, value in env.items():
        command.extend(("-e", f"{key}={value}"))
    for source, target, mode in ((candidate / "bin", "/candidate/bin", "ro"),
                                 (candidate / "config", "/config", "ro"),
                                 (candidate / "models", "/models", "ro"),
                                 (output, "/work", "rw"),
                                 (sources, "/offline-input", "ro"),
                                 (recordings, "/calibration_recordings", "rw")):
        command.extend(("--mount", f"type=bind,src={source},dst={target},{'readonly' if mode == 'ro' else ''}".rstrip(",")))
    command.extend((IMAGE, "/candidate/bin/cam-six-yolo-local6-direct-gatea",
                    "/work/camera-six-offline.ini", str(duration_sec + 30)))
    return command


def atomic_command(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def run(candidate: Path, output: Path, duration_sec: int = 120) -> dict:
    if duration_sec not in (120, 180):
        raise ValueError("offline duration must be 120 or 180 seconds")
    candidate, output = candidate.resolve(), output.resolve()
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate/test output must remain under .runtime")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite offline recorder test: {output}")
    output.mkdir(parents=True)
    sources = output / "inputs"
    sources.mkdir()
    make_clip(sources / "3200x1800.mp4", 3200, 1800, "0x293746", duration_sec)
    make_clip(sources / "2560x1440.mp4", 2560, 1440, "0x594733", duration_sec)
    write_camera_config(output / "camera-six-offline.ini")
    recordings = output / "recordings"
    recordings.mkdir()
    (recordings / "entrance").mkdir()
    (output / "runtime-contract.json").write_text(json.dumps({
        "test": "actual native recorder on direct-file inputs; no RTSP sockets",
        "input_mapping": {camera: ("3200x1800.mp4" if int(camera[-2:]) % 2 == 0 else "2560x1440.mp4")
                          for camera in CAMERAS},
        "unique_live_sources": False,
        "duplicate_file_inputs": True,
        "purpose": "offline recorder branch validation only",
        "source_fps": 20,
        "source_duration_sec": duration_sec + 3,
        "recording_duration_sec": duration_sec,
    }, indent=2) + "\n", encoding="utf-8")
    control = output / "calibration-recording-command.tsv"
    status_path = output / "calibration-recording-status.json"
    log_path = output / "native.log"
    command = docker_command(candidate, output, sources, recordings, duration_sec)
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        (output / "native-pid.json").write_text(json.dumps({"pid": process.pid}) + "\n")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"native test process exited before recorder IDLE, code={process.returncode}")
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
                continue
            if status.get("state") == "IDLE":
                break
            time.sleep(0.1)
        else:
            raise TimeoutError("native recorder did not announce IDLE")
        session_id = f"offline-smoke-{duration_sec}s-20261009"
        atomic_command(control, f"START\tentrance\t{session_id}\t/calibration_recordings/entrance/{session_id}\t{duration_sec}\n")
        terminal = {"COMPLETE", "REVIEW_REQUIRED", "FAILED"}
        deadline = time.monotonic() + duration_sec + 25
        last_progress = -1
        while time.monotonic() < deadline:
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {}
            elapsed = int(float(status.get("elapsed_sec", 0)))
            if elapsed >= last_progress + 15:
                last_progress = elapsed
                rows = status.get("cameras", [])
                print(f"OFFLINE RECORDER TEST {status.get('state', '?')} elapsed={elapsed}s " +
                      " ".join(f"{r.get('camera_id')} in={r.get('input_frames')} enc={r.get('encoded_frames')} bytes={r.get('video_bytes')}"
                               for r in rows), flush=True)
            if status.get("state") in terminal:
                break
            if process.poll() is not None:
                break
            time.sleep(0.25)
        try:
            return_code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=10)
        final_status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if return_code != 0:
            raise RuntimeError(f"offline direct-file candidate exit code {return_code}; inspect {log_path}")
        if final_status.get("state") not in {"COMPLETE", "REVIEW_REQUIRED"}:
            raise RuntimeError(f"recorder failed to finalize: {final_status.get('state')!r}")
    session_dir = recordings / "entrance" / session_id
    validation = subprocess.run([sys.executable, str(ROOT / "scripts/validate_calibration_recording.py"),
                                 str(session_dir)], cwd=ROOT, capture_output=True, text=True)
    (output / "validation.stdout.json").write_text(validation.stdout, encoding="utf-8")
    (output / "validation.stderr.txt").write_text(validation.stderr, encoding="utf-8")
    if validation.returncode:
        raise RuntimeError(f"recording validation failed; inspect {output / 'validation.stdout.json'}")
    result = json.loads(validation.stdout)
    result["native_process_return_code"] = return_code
    result["test_result"] = "PASS"
    (output / "offline-test-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / ".runtime/sixcam-calibration-recording-20261009-r2")
    parser.add_argument("--output", type=Path, default=ROOT / ".runtime/sixcam-calibration-recording-offline-20261009-r1")
    parser.add_argument("--duration-sec", type=int, choices=(120, 180), default=120)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.candidate, args.output, args.duration_sec), indent=2))
    except Exception as exc:
        print(f"CALIBRATION_RECORDER_OFFLINE_TEST_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
