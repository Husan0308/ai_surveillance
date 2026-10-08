"""Run the integrated six-camera candidate on pinned CAM-01/CAM-04 MP4s via local RTSP."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from scripts.build_six_camera_v13_wall_candidate import CANDIDATE as DEFAULT_CANDIDATE
from scripts.run_six_camera_v13_wall_candidate import CAMERAS, docker_command, validate_candidate
from services.camera_v11.source_ownership import SourceOwnership

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos"
INPUTS = {
    "CAM-01": (PAIR / "cam_00.mp4", "1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1"),
    "CAM-04": (PAIR / "cam_01.mp4", "2b73ecfd4c9361c02442b67966fe78e17aa2352e7b6f39eff0f0258de2ff425f"),
}
APPROVED = ROOT / ".runtime/sixcam-local6-20261004-b/shadow-assisted-detector-ab-v4"
LOCAL_RTSP = {"CAM-01": "rtsp://127.0.0.1:8554/cam01", "CAM-04": "rtsp://127.0.0.1:8554/cam04"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_camera_config() -> str:
    lines: list[str] = []
    for camera in CAMERAS:
        source = "CAM-01" if camera in ("CAM-01", "CAM-02", "CAM-03") else "CAM-04"
        lines.extend((f"[{camera}]", "uri=" + base64.b64encode(LOCAL_RTSP[source].encode()).decode(),
                      "username=", "password=", "latency_ms=20", "drop_on_latency=false",
                      "decoder_low_latency_mode=false", "decoder_extra_surfaces=4", ""))
    lines.extend(("[validation]", "interrupt_camera=none", "interrupt_at=20",
                  "interrupt_seconds=12", "preview_enabled=false", "preview_port=5600", ""))
    return "\n".join(lines)


def established_camera_rtsp() -> list[str]:
    result = subprocess.run(["ss", "-Htnp", "state", "established", "( dport = :554 )"],
                            text=True, capture_output=True, check=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def _frames_seen(path: Path) -> dict[int, set[int]]:
    seen = {index: set() for index in range(6)}
    if not path.exists():
        return seen
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
                source_id, frame = int(row["source_id"]), int(row["frame"])
                if source_id in seen:
                    seen[source_id].add(frame)
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return seen


def run(candidate: Path, output: Path, timeout_sec: int = 160, nan_diagnostics: bool = False,
        decision_diagnostics: bool = False) -> dict:
    candidate, output = candidate.resolve(), output.resolve()
    if not candidate.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate and output must remain under .runtime")
    manifest = validate_candidate(candidate)
    for camera, (path, expected) in INPUTS.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"{camera} exact recorded input hash mismatch: {path}")
    if not (APPROVED / "CAM01-per-frame.csv").is_file() or not (APPROVED / "CAM04-per-frame.csv").is_file():
        raise FileNotFoundError("human-approved v4 per-frame references are missing")
    if output.exists():
        raise FileExistsError(f"refusing to reuse replay evidence: {output}")
    if established_camera_rtsp():
        raise RuntimeError("physical camera RTSP connections exist; refusing overlap with live owners")

    output.mkdir(parents=True)
    camera_config = output / "camera-reference.ini"
    camera_config.write_text(reference_camera_config())
    camera_config.chmod(0o600)
    provenance = {
        "mode": "same integrated six-source YOLO+local-NvDCF candidate, local RTSP replay transport",
        "candidate_binary_sha256": manifest["binary_sha256"],
        "candidate_source_sha256": manifest["source_sha256"],
        "yolo_engine_sha256": manifest["model_sha256"]["yolo26m_raw_otm_b6_fp16.engine"],
        "parser_sha256": manifest["model_sha256"]["libyolo26rawperson.so"],
        "detector_config_sha256": manifest["config_sha256"]["config_infer_primary_yolo26m_raw_otm.txt"],
        "local_tracker_sha256": manifest["config_sha256"]["config_tracker_NvDCF_v13_local6.yml"],
        "frozen_v13_reference_sha256": manifest["frozen_v13_reference_sha256"],
        "calibration_loaded": False,
        "identity_enabled": False,
        "batch_size": 6,
        "source_mapping": {"CAM-01/02/03": "same CAM-01 recording", "CAM-04/05/06": "same CAM-04 recording"},
        "inputs": {camera: {"path": str(path), "sha256": sha256(path),
                             "width": 1920, "height": 1080, "fps": 20, "frames": 2400}
                   for camera, (path, _) in INPUTS.items()},
        "approved_reference": {camera: str(APPROVED / f"{camera.replace('-', '')}-per-frame.csv")
                               for camera in INPUTS},
        "expected_policy": json.loads((APPROVED / "summary.json").read_text())["per_camera"]["CAM-01"]["match_policy"],
        "nan_diagnostics_enabled": nan_diagnostics,
        "decision_diagnostics_enabled": decision_diagnostics,
        "effective_mux_width": int(os.getenv("V11_MUX_WIDTH", "2560")),
        "effective_mux_height": int(os.getenv("V11_MUX_HEIGHT", "1440")),
        "effective_tracker_width": int(os.getenv("V11_TRACKER_WIDTH", "960")),
        "effective_tracker_height": int(os.getenv("V11_TRACKER_HEIGHT", "544")),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    lock_dir = ROOT / ".runtime/camera-owner-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    command = docker_command(candidate, output, camera_config, 86400)
    if nan_diagnostics:
        image_index = command.index("nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f")
        command[image_index:image_index] = ["-e", "V11_TRACKER_NAN_DIAGNOSTICS=1"]
    if decision_diagnostics:
        image_index = command.index("nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f")
        command[image_index:image_index] = ["-e", "V11_TRACKER_DECISION_DIAGNOSTICS=1"]
    # This runner expects the local RTSP test server; the test is still handled
    # by the candidate's normal RTSP/NVDEC path and unmodified pipeline binary.
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    (output / "native.log").touch(exist_ok=False)
    frame_path = output / "frames.jsonl"
    with SourceOwnership(CAMERAS, lock_dir):
        process = subprocess.Popen(command, cwd=ROOT, stdout=(output / "native.log").open("w"),
                                   stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_sec
        completed = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            seen = _frames_seen(frame_path)
            if all(len(seen[index]) >= 2400 and max(seen[index], default=-1) >= 2399
                   for index in range(6)):
                completed = True
                process.send_signal(signal.SIGINT)
                break
            time.sleep(1)
        if process.poll() is None:
            if not completed:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError("integrated reference replay failed to shut down cleanly")
        seen = _frames_seen(frame_path)
    result = {"status": "PASS" if completed and process.returncode in (0, -signal.SIGINT, 130) else "FAIL",
              "return_code": process.returncode, "all_source_frames_complete": completed,
              "frames_per_source": {f"CAM-{i+1:02d}": len(rows) for i, rows in seen.items()},
              "max_frame_per_source": {f"CAM-{i+1:02d}": max(rows, default=-1) for i, rows in seen.items()},
              "candidate_binary_sha256": manifest["binary_sha256"]}
    (output / "run-result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "PASS":
        raise RuntimeError(f"integrated replay did not process all exact frames: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=160)
    parser.add_argument("--nan-diagnostics", action="store_true",
                        help="write invalid NvDCF shadow geometry rows; does not change tracker decisions")
    parser.add_argument("--decision-diagnostics", action="store_true",
                        help="write per-detection active/shadow decision details without changing decisions")
    args = parser.parse_args()
    print(json.dumps(run(args.candidate, args.output, args.timeout_sec,
                         args.nan_diagnostics, args.decision_diagnostics), indent=2))


if __name__ == "__main__":
    main()
