"""Isolated Entrance Room live RTSP → rectified appsrc → DeepStream 9.1 UI.

The host side uses the existing protected camera settings/ownership and tested
RTSP decoder, then the empirical coordinate bridge.  A disposable DeepStream
9.1 container receives only rectified BGR pixels over stdin (no camera URL or
credentials), converts to NVMM, batches through legacy nvstreammux, and runs
PeopleNet Transformer + local NvDCF.  MV3DT/global IDs and map projection are
intentionally not part of this experimental live path.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import shutil
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
APP_DIR = Path(__file__).resolve().parent
DS_IMAGE = "nvcr.io/nvidia/deepstream:9.1-triton-multiarch"
DS_IMAGE_ID = "sha256:fd31f5b44ababdbdee8cd397a375e888191b49e402ac237254a4cdc239130f5b"
# The frozen YOLO engine build manifest records this exact DeepStream image.
# Keep the default PeopleNet path on DS_IMAGE; this pin is experimental-only.
YOLO_DS_IMAGE = "nvcr.io/nvidia/deepstream@sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
YOLO_DS_IMAGE_ID = "sha256:10eca409b3894e91c1bac915c9f1346307e56695e552487cbe8cf2f58a3f998f"
MODEL_DIR = Path("/home/apsidal/nvidia/DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view/models/PeopleNetTransformer")
BRIDGE_DEFAULT = Path("/tmp/entrance-live-rectification-bridge-20261019/rectification_bridge.py")
CAMERAS = ("CAM-02", "CAM-05")
CODECS = {"CAM-02": "h264", "CAM-05": "h265"}
NATIVE_SIZE = {"CAM-02": (3200, 1800), "CAM-05": (2560, 1440)}
FRAME_HEADER = struct.Struct("<IHHIIIQqqqQ")
FRAME_MAGIC = 0x454E5446
FRAME_VERSION = 1
SENTINEL_SOURCE = 0xFFFF
SOURCE_BY_ID = {0: "CAM-02", 1: "CAM-05"}
PNT_CONFIG_SHA256 = "5f3f4e177e4f1e33743bc537bfcb3e331cab87cc3912982e1732f193e4214e79"
PNT_ASSET_SHA256 = {
    "peoplenet_transformer_model_op17.onnx": "974d6a77b4fde5ad83b18bbabc5e3771220e6ee22224e98cdc0cbf733da06038",
    "peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine": "99451ab53e3c4b8f2bc50dbd4e33cb15bfcd28ba592e2fd766adfc08770cb3a8",
    "detector_labels.txt": "5618c52db083cfdf40b81fd57d71fda73240738cefcdee7f97add8e0715737e4",
    "custom_parser/libnvds_infercustomparser_tao.so": "0e01920c57fa6762887112d5c0e0e4239e10a094cd33ed4fd580add91d08e015",
}
PNT_ENGINE_ONLY_CONFIG = "/workspace/pgie-pnt-b2-engine-only.txt"


def redact(text: str) -> str:
    text = re.sub(r"(?i)(rtsps?://)[^@/\s]+@", r"\1***@", text)
    text = re.sub(r'(?i)user-(?:id|pw)="[^"]*"', 'credential="***"', text)
    text = re.sub(r"(?i)(password|passwd|token)=([^\s&]+)", r"\1=***", text)
    return text[-1000:]


def import_bridge(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"validated rectification bridge missing: {path}")
    spec = importlib.util.spec_from_file_location("entrance_live_rectification_bridge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the validated rectification bridge")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def smoke_gate_passed(stats: dict[int, dict[str, Any]], baseline: dict[int, dict[str, int]]) -> bool:
    """Require advancing, correctly attributed pipeline output, not zero drops.

    The live path intentionally uses leaky, bounded queues. A late tracker
    result whose source frame has already left the small preview cache is
    counted, but it must not be drawn against another frame. Such an expected
    drop is not itself a pipeline failure.
    """
    return all(
        stats[sid]["mux_frames"] > baseline[sid]["mux_frames"]
        and stats[sid]["pgie_frames"] > baseline[sid]["pgie_frames"]
        and stats[sid]["tracker_frames"] > baseline[sid]["tracker_frames"]
        and stats[sid]["display_frame_matched"] > baseline[sid]["display_frame_matched"]
        and stats[sid]["source_id_mismatch"] == 0
        and stats[sid]["mux_frame_num_nonmonotonic"] == 0
        for sid in (0, 1)
    )


def safe_for_extended_review(gpu_samples: list[dict[str, Any]], limit_percent: int = 95) -> bool:
    """Fail closed if device-wide load is unavailable or repeatedly saturated."""
    recent = [sample for sample in gpu_samples[-3:] if sample.get("status") == "AVAILABLE"]
    if len(recent) < 3:
        return False
    return sum(int(sample["gpu_utilization_percent"]) >= limit_percent for sample in recent) < 2


def validate_offline_stage_crosswalk(rows: list[dict[str, Any]], frame_count: int, stage: str) -> None:
    """Require exact, ordered per-source frame coverage for offline smoke stages."""
    by_source: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        source_id = row.get("source_id")
        if source_id not in by_source:
            raise RuntimeError(f"{stage} emitted unexpected source_id {source_id}")
        by_source[source_id].append(row)
    expected_indices = list(range(frame_count))
    for source_id, source_rows in by_source.items():
        frame_indices = [row.get("input_frame_index") for row in source_rows]
        if frame_indices != expected_indices:
            raise RuntimeError(
                f"{stage} frame crosswalk mismatch for source {source_id}: "
                f"expected={expected_indices}, actual={frame_indices}"
            )
        source_pts = [row.get("source_pts_ns") for row in source_rows]
        if any(value is None for value in source_pts) or any(
            int(right) <= int(left) for left, right in zip(source_pts, source_pts[1:])
        ):
            raise RuntimeError(f"{stage} source PTS not strictly increasing for source {source_id}")


def stage_peoplenet_engine_only(model_dir: Path, config_path: Path, output_root: Path, *,
                                expected_assets: dict[str, str] | None = None,
                                expected_config_sha256: str = PNT_CONFIG_SHA256) -> dict[str, Any]:
    """Stage the verified PeopleNet B2 plan/parser without an ONNX rebuild path."""
    expected_assets = PNT_ASSET_SHA256 if expected_assets is None else expected_assets
    model_dir, config_path, output_root = Path(model_dir), Path(config_path), Path(output_root).resolve()
    if not output_root.is_relative_to(Path("/tmp").resolve()) and not output_root.is_relative_to((ROOT / ".runtime").resolve()):
        raise ValueError("comparison artifacts must remain under /tmp or this checkout's .runtime")
    if model_dir.is_symlink() or not model_dir.is_dir() or config_path.is_symlink() or not config_path.is_file():
        raise ValueError("PeopleNet source model/config paths must be regular existing paths")

    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    actual = {}
    for relative, expected in expected_assets.items():
        source = model_dir / relative
        if source.is_symlink() or not source.is_file() or digest(source) != expected:
            raise ValueError(f"PeopleNet asset missing/changed: {relative}")
        actual[relative] = expected
    if digest(config_path) != expected_config_sha256:
        raise ValueError("PeopleNet comparison config SHA256 mismatch")

    template = config_path.read_text()
    onnx_line = "onnx-file=/workspace/models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx\n"
    engine_line = "model-engine-file=/workspace/models/PeopleNetTransformer/peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine\n"
    if template.count(onnx_line) != 1 or template.count(engine_line) != 1:
        raise ValueError("PeopleNet engine/config contract changed; refusing comparison staging")
    engine_only = template.replace(onnx_line, "", 1)
    if "onnx-file=" in engine_only:
        raise ValueError("PeopleNet engine-only config unexpectedly retains ONNX fallback")

    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_root, 0o700)
    staged_models = output_root / "peoplenet-engine-only-models"
    staged_models.mkdir(mode=0o700, exist_ok=True)
    os.chmod(staged_models, 0o700)
    staged_names = (
        "peoplenet_transformer_model_op17.onnx_b2_gpu0_fp16.engine",
        "detector_labels.txt",
        "custom_parser/libnvds_infercustomparser_tao.so",
    )
    for relative in staged_names:
        source, target = model_dir / relative, staged_models / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.exists():
            if target.is_symlink() or not target.is_file() or digest(target) != actual[relative]:
                raise ValueError(f"refusing to overwrite changed staged PeopleNet asset: {relative}")
        else:
            shutil.copyfile(source, target)
            target.chmod(0o600)
            if digest(target) != actual[relative]:
                raise ValueError(f"staged PeopleNet asset SHA256 mismatch: {relative}")
    staged_config = output_root / "pgie-pnt-b2-engine-only.txt"
    if staged_config.exists():
        if staged_config.is_symlink() or staged_config.read_text() != engine_only:
            raise ValueError("refusing to overwrite a divergent staged PeopleNet config")
    else:
        staged_config.write_text(engine_only)
    staged_config.chmod(0o600)
    return {
        "model_dir": str(staged_models), "config_file": str(staged_config),
        "config_sha256": hashlib.sha256(engine_only.encode()).hexdigest(),
        "source_config_sha256": expected_config_sha256,
        "model_asset_sha256": actual, "engine_only": True,
    }


def docker_command(*, mux_only: bool, output_root: Path, detector: str = "peoplenet",
                   yolo_artifacts: Path | None = None, lossless_input: bool = False,
                   comparison_runtime: bool = False, pnt_engine_only: bool = False) -> list[str]:
    if detector not in {"peoplenet", "yolo26m"}:
        raise ValueError("unsupported detector")
    if comparison_runtime and (mux_only or detector != "peoplenet"):
        raise ValueError("comparison runtime is restricted to offline PeopleNet inference")
    if pnt_engine_only and (mux_only or detector != "peoplenet"):
        raise ValueError("PeopleNet engine-only staging is restricted to offline PeopleNet inference")
    source = str(APP_DIR.resolve())
    mounts = []
    image = DS_IMAGE
    pull_policy = ["--pull=never"] if not mux_only and (detector == "yolo26m" or comparison_runtime or pnt_engine_only) else []
    pgie_config = "/workspace/app/config_pgie_pnt_batch2.txt"
    if not mux_only and detector == "yolo26m":
        if yolo_artifacts is None:
            raise ValueError("YOLO26m requires explicit --yolo-artifacts under .runtime")
        from services.mv3dt_staging.live_tracking.yolo_candidate import (
            CONTAINER_CONFIG, stage_yolo_batch_two,
        )
        profile = stage_yolo_batch_two(yolo_artifacts, output_root, engine_only=True)
        image = YOLO_DS_IMAGE
        pgie_config = CONTAINER_CONFIG
        mounts = [
            "--mount", f"type=bind,source={profile['model_dir']},target=/models,readonly",
            "--mount", f"type=bind,source={profile['config_file']},target={CONTAINER_CONFIG},readonly",
        ]
    elif not mux_only and (comparison_runtime or pnt_engine_only):
        profile = stage_peoplenet_engine_only(
            MODEL_DIR, APP_DIR / "config_pgie_pnt_batch2.txt", output_root,
        )
        image = YOLO_DS_IMAGE if comparison_runtime else DS_IMAGE
        pgie_config = PNT_ENGINE_ONLY_CONFIG
        mounts = [
            "--mount", f"type=bind,source={profile['model_dir']},target=/workspace/models/PeopleNetTransformer,readonly",
            "--mount", f"type=bind,source={profile['config_file']},target={PNT_ENGINE_ONLY_CONFIG},readonly",
        ]
    elif not mux_only:
        models = str(MODEL_DIR.resolve())
        if not MODEL_DIR.is_dir():
            raise FileNotFoundError(f"existing PeopleNet Transformer model directory missing: {MODEL_DIR}")
        mounts = ["--mount", f"type=bind,source={models},target=/workspace/models/PeopleNetTransformer,readonly"]
    compile_and_run = (
        "set -euo pipefail; "
        "g++ -std=c++17 -O2 -Wall -Wextra "
        "-I/opt/nvidia/deepstream/deepstream/sources/includes "
        "$(pkg-config --cflags gstreamer-app-1.0 gstreamer-video-1.0) "
        "/workspace/app/deepstream_appsrc.cpp "
        "-L/opt/nvidia/deepstream/deepstream/lib "
        "-Wl,-rpath,/opt/nvidia/deepstream/deepstream/lib "
        "-lnvds_meta -lnvdsgst_meta $(pkg-config --libs gstreamer-app-1.0 gstreamer-video-1.0) "
        "-o /tmp/entrance_ds_appsrc; "
        "/tmp/entrance_ds_appsrc "
        + ("--mux-only" if mux_only else f"--pgie-config {pgie_config}")
        + (" --lossless-input" if lossless_input else "")
    )
    return [
        "docker", "run", "--rm", *pull_policy, "--init", "--interactive", "--gpus", "all",
        "--network", "none", "--env", "USE_NEW_NVSTREAMMUX=no",
        "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
        "--mount", f"type=bind,source={source},target=/workspace/app,readonly",
        *mounts,
        "--workdir", "/workspace/app", "--entrypoint", "/bin/bash",
        image, "-lc", compile_and_run,
    ]


class DeepStreamProcess:
    def __init__(self, *, mux_only: bool, output_root: Path,
                 detector: str = "peoplenet", yolo_artifacts: Path | None = None,
                 lossless_input: bool = False, comparison_runtime: bool = False,
                 pnt_engine_only: bool = False):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.output_root, 0o700)
        self.stdout_path = self.output_root / "deepstream.stdout.log"
        self.stderr_path = self.output_root / "deepstream.stderr.log"
        self.image = YOLO_DS_IMAGE if detector == "yolo26m" or comparison_runtime else DS_IMAGE
        self.image_id = YOLO_DS_IMAGE_ID if detector == "yolo26m" or comparison_runtime else DS_IMAGE_ID
        self._stdout_file = self.stdout_path.open("w", encoding="utf-8")
        self._stderr_file = self.stderr_path.open("w", encoding="utf-8")
        os.chmod(self.stdout_path, 0o600)
        os.chmod(self.stderr_path, 0o600)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8192)
        self.ready = threading.Event()
        self.done = threading.Event()
        self.failed = threading.Event()
        self._input_lock = threading.Lock()
        self.process = subprocess.Popen(
            docker_command(mux_only=mux_only, output_root=output_root,
                           detector=detector, yolo_artifacts=yolo_artifacts,
                           lossless_input=lossless_input,
                           comparison_runtime=comparison_runtime,
                           pnt_engine_only=pnt_engine_only),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env={**os.environ, "USE_NEW_NVSTREAMMUX": "no"},
        )
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _event(self, record: dict[str, Any]) -> None:
        try:
            self.events.put_nowait(record)
        except queue.Full:
            self.failed.set()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for raw in iter(self.process.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                self._stdout_file.write(line + "\n")
                if line.startswith("READY "):
                    try: self._event({"type": "READY", **json.loads(line[6:])})
                    except json.JSONDecodeError: self._event({"type": "READY", "raw": line})
                    self.ready.set()
                elif line.startswith("DONE "):
                    try: self._event({"type": "DONE", **json.loads(line[5:])})
                    except json.JSONDecodeError: self._event({"type": "DONE", "raw": line})
                    self.done.set()
                else:
                    prefix, separator, body = line.partition(" ")
                    if separator and prefix in {"MUX", "PGIE", "TRACK", "ERROR"}:
                        try: self._event({"type": prefix, **json.loads(body)})
                        except json.JSONDecodeError: self._event({"type": prefix, "raw": redact(line)})
        finally:
            self._stdout_file.flush()
            self.done.set()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            for raw in iter(self.process.stderr.readline, b""):
                self._stderr_file.write(redact(raw.decode("utf-8", errors="replace").rstrip()) + "\n")
        finally:
            self._stderr_file.flush()

    def wait_ready(self, timeout_sec: float = 30.0) -> None:
        if not self.ready.wait(timeout_sec):
            self.failed.set()
            self.stop(timeout_sec=5.0)
            tail = self.stderr_path.read_text(errors="replace")[-2000:] if self.stderr_path.exists() else ""
            raise RuntimeError(f"DeepStream 9.1 did not signal READY within {timeout_sec}s: {redact(tail)}")

    @staticmethod
    def _write_all(fd, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = fd.write(view)
            if written is None or written <= 0:
                raise BrokenPipeError("DeepStream appsrc stdin closed")
            view = view[written:]

    def send_frame(self, source_id: int, frame_index: int, frame: np.ndarray,
                   source_pts_ns: int, source_dts_ns: int, arrival_mono_ns: int) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("DeepStream container exited before appsrc accepted the frame")
        image = np.ascontiguousarray(frame)
        if image.shape != (1080, 1920, 3) or image.dtype != np.uint8:
            raise ValueError(f"appsrc requires 1920x1080 uint8 BGR, got {image.shape}/{image.dtype}")
        data = memoryview(image).cast("B")
        header = FRAME_HEADER.pack(
            FRAME_MAGIC, FRAME_VERSION, source_id, 1920, 1080, 1920 * 3,
            frame_index, source_pts_ns, source_dts_ns, arrival_mono_ns, data.nbytes,
        )
        with self._input_lock:
            self._write_all(self.process.stdin, header)
            self._write_all(self.process.stdin, data)

    def end_input(self) -> None:
        if self.process.stdin is None or self.process.stdin.closed or self.process.poll() is not None:
            return
        sentinel = FRAME_HEADER.pack(FRAME_MAGIC, FRAME_VERSION, SENTINEL_SOURCE,
                                     0, 0, 0, 0, -1, -1, time.monotonic_ns(), 0)
        with self._input_lock:
            try:
                self._write_all(self.process.stdin, sentinel)
                self.process.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def stop(self, timeout_sec: float = 10.0) -> None:
        self.end_input()
        try:
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try: self.process.wait(timeout=4)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=2)
        self.done.set()
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        self._stdout_file.close()
        self._stderr_file.close()


def run_mux_smoke(args) -> int:
    import cv2

    output = Path(args.output).resolve() if args.output else Path(tempfile.mkdtemp(prefix="entrance-appsrc-mux-"))
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    # AMC staging videos are ALREADY rectified. Applying the bridge a second
    # time would invalidate any detector A/B comparison.
    bridge_mod = import_bridge(Path(args.bridge).expanduser()) if args.input_geometry == "raw" else None
    cameras = [
        (0, "CAM-02", Path(args.cam02_video).expanduser(), "CAM-02"),
        (1, "CAM-05", Path(args.cam05_video).expanduser(), "CAM-05"),
    ]
    captures = []
    run_started_ns = time.monotonic_ns()
    for _, camera_id, path, _ in cameras:
        if not path.is_file(): raise FileNotFoundError(f"recorded input missing: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise RuntimeError(f"OpenCV could not read recorded {camera_id} frames")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_size = NATIVE_SIZE[camera_id] if args.input_geometry == "raw" else (1920, 1080)
        if (w, h) != expected_size:
            cap.release(); raise RuntimeError(
                f"{camera_id} {args.input_geometry} input expected {expected_size}, got {(w, h)}"
            )
        captures.append(cap)
    mux_only = not bool(args.inference)
    pipe = DeepStreamProcess(
        mux_only=mux_only, output_root=output,
        detector=args.detector,
        yolo_artifacts=Path(args.yolo_artifacts).expanduser() if args.yolo_artifacts else None,
        lossless_input=True,
        comparison_runtime=bool(getattr(args, "comparison_runtime", False)),
        pnt_engine_only=bool(getattr(args, "pnt_engine_only", False)),
    )
    sent = {0: 0, 1: 0}
    feeder_start_ns = None
    feeder_end_ns = None
    pipe_end_ns = None
    try:
        pipe.wait_ready()
        ready_ns = time.monotonic_ns()
        transforms = ({"CAM-02": bridge_mod.RectificationBridge("CAM-02"),
                       "CAM-05": bridge_mod.RectificationBridge("CAM-05")}
                      if bridge_mod is not None else {})
        feeder_start_ns = time.monotonic_ns()
        for index in range(args.frames):
            for (source_id, camera_id, _path, _), cap in zip(cameras, captures):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"{camera_id}: recorded decode ended before frame {index}")
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                pts_ns = int(pts_ms * 1_000_000) if np.isfinite(pts_ms) and pts_ms >= 0 else -1
                rectified = transforms[camera_id](frame) if transforms else frame
                pipe.send_frame(source_id, index, rectified, pts_ns, -1, time.monotonic_ns())
                sent[source_id] += 1
            time.sleep(0.05)
        feeder_end_ns = time.monotonic_ns()
        pipe.end_input()
        if not pipe.done.wait(20): raise RuntimeError("recorded appsrc mux smoke did not reach EOS")
        try: pipe.process.wait(timeout=10)
        except subprocess.TimeoutExpired: raise RuntimeError("DeepStream reported EOS but did not exit")
        if pipe.process.returncode != 0: raise RuntimeError(f"DeepStream mux smoke exited {pipe.process.returncode}")
        pipe_end_ns = time.monotonic_ns()
    finally:
        for cap in captures: cap.release()
        pipe.stop()
    mux_rows = []
    pgie_rows = []
    track_rows = []
    event_records = list(pipe.events.queue)
    event_path = output / "mux-smoke-events.jsonl"
    with event_path.open("w", encoding="utf-8") as stream:
        for record in event_records:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    os.chmod(event_path, 0o600)
    for record in event_records:
        if record.get("type") == "MUX": mux_rows.append(record)
        elif record.get("type") == "PGIE": pgie_rows.append(record)
        elif record.get("type") == "TRACK": track_rows.append(record)
    counts = {0: 0, 1: 0}
    frame_sequences = {0: [], 1: []}
    sidecar_ok = True
    for row in mux_rows:
        sid = row.get("source_id")
        if sid in counts:
            counts[sid] += 1
            if row.get("input_frame_index") is None: sidecar_ok = False
            frame_sequences[sid].append((row.get("frame_num"), row.get("input_frame_index"), row.get("buf_pts")))
    if counts != sent or not sidecar_ok:
        raise RuntimeError(f"mux metadata mismatch: sent={sent}, muxed={counts}, sidecar_ok={sidecar_ok}")
    validate_offline_stage_crosswalk(mux_rows, args.frames, "MUX")
    if args.inference:
        pgie_counts = {sid: sum(1 for row in pgie_rows if row.get("source_id") == sid) for sid in (0, 1)}
        track_counts = {sid: sum(1 for row in track_rows if row.get("source_id") == sid) for sid in (0, 1)}
        if any(value == 0 for value in pgie_counts.values()) or any(value == 0 for value in track_counts.values()):
            raise RuntimeError(f"recorded inference did not reach PGIE/NvDCF for both sources: PGIE={pgie_counts}, NvDCF={track_counts}")
        if any(row.get("input_frame_index") is None for row in pgie_rows + track_rows):
            raise RuntimeError("PGIE/NvDCF output lost its input-frame sidecar crosswalk")
        validate_offline_stage_crosswalk(pgie_rows, args.frames, "PGIE")
        validate_offline_stage_crosswalk(track_rows, args.frames, "NvDCF")
    pgie_config_sha256 = None
    if args.inference and (args.detector == "yolo26m" or getattr(args, "pnt_engine_only", False)):
        if getattr(args, "pnt_engine_only", False):
            config_name = "pgie-pnt-b2-engine-only.txt"
        else:
            from services.mv3dt_staging.live_tracking.yolo_candidate import STAGED_CONFIG_FILENAME
            config_name = STAGED_CONFIG_FILENAME
        config_path = output / config_name
        if not config_path.is_file():
            raise RuntimeError("staged engine-only PGIE config missing after inference")
        pgie_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()

    person_class_id = 0 if args.detector == "yolo26m" else 1

    def latency_summary(values_ns):
        values_ms = [int(value) / 1_000_000 for value in values_ns if value is not None]
        if not values_ms:
            return "NOT_MEASURED"
        return {"samples": len(values_ms), "p50_ms": round(float(np.percentile(values_ms, 50)), 3),
                "p95_ms": round(float(np.percentile(values_ms, 95)), 3), "max_ms": round(max(values_ms), 3)}

    def confidence_summary(values):
        values = [float(value) for value in values if value is not None]
        if not values:
            return "NO_PERSON_DETECTIONS"
        return {"samples": len(values), "p50": round(float(np.percentile(values, 50)), 4),
                "p95": round(float(np.percentile(values, 95)), 4),
                "min": round(min(values), 4), "max": round(max(values), 4)}

    performance_by_source = {}
    if args.inference:
        for sid in (0, 1):
            pgie_source = [row for row in pgie_rows if row.get("source_id") == sid]
            track_source = [row for row in track_rows if row.get("source_id") == sid]
            detections = [obj for row in pgie_source for obj in row.get("objects", [])
                          if int(obj.get("class_id", -1)) == person_class_id]
            tracked = [obj for row in track_source for obj in row.get("objects", [])
                       if int(obj.get("class_id", -1)) == person_class_id]
            local_ids = {int(obj["tracker_id"]) for obj in tracked if obj.get("tracker_id") is not None}
            pgie_times = [int(row["stage_mono_ns"]) for row in pgie_source if row.get("stage_mono_ns") is not None]
            tracker_times = [int(row["stage_mono_ns"]) for row in track_source if row.get("stage_mono_ns") is not None]
            performance_by_source[str(sid)] = {
                "person_class_id": person_class_id,
                "pgie_frames": len(pgie_source), "tracker_frames": len(track_source),
                "person_detections": len(detections), "person_track_observations": len(tracked),
                "unique_local_tracker_ids": len(local_ids),
                "confidence": confidence_summary([obj.get("confidence") for obj in detections]),
                "pgie_pad_latency": latency_summary([row.get("pgie_pad_latency_ns") for row in pgie_source]),
                "host_input_to_pgie": latency_summary([row.get("host_input_to_stage_ns") for row in pgie_source]),
                "tracker_pad_latency": latency_summary([row.get("tracker_pad_latency_ns") for row in track_source]),
                "host_input_to_tracker": latency_summary([row.get("host_input_to_stage_ns") for row in track_source]),
                "pgie_output_fps": (round((len(pgie_times)-1) / ((max(pgie_times)-min(pgie_times))/1e9), 3)
                                     if len(pgie_times) > 1 and max(pgie_times) > min(pgie_times) else None),
                "tracker_output_fps": (round((len(tracker_times)-1) / ((max(tracker_times)-min(tracker_times))/1e9), 3)
                                        if len(tracker_times) > 1 and max(tracker_times) > min(tracker_times) else None),
            }
    result = {
        "status": "PASS",
        "image": pipe.image,
        "image_id": pipe.image_id,
        "mux": "legacy via process-scoped USE_NEW_NVSTREAMMUX=no", "batch_size": 2,
        "geometry": "1920x1080 BGR→NV12→NVMM", "sent_by_source": sent,
        "muxed_by_source": counts, "bounded_queue_drops_before_mux": {str(k): sent[k]-counts[k] for k in sent},
        "source_mapping": {"0": "CAM-02", "1": "CAM-05"},
        "mode": (f"{args.detector} + local NvDCF" if args.inference else "mux-only"),
        "detector": args.detector if args.inference else None,
        "input_geometry": args.input_geometry,
        "double_rectification": False,
        "pgie_config_sha256": pgie_config_sha256,
        "engine_only_runtime": bool(args.inference and (args.detector == "yolo26m" or getattr(args, "pnt_engine_only", False))),
        "person_class_id": person_class_id if args.inference else None,
        "performance_by_source": performance_by_source if args.inference else None,
        "timing_scope": "PGIE/tracker pad-crossing wall time; host_input_to_* includes host pipe/appsrc/mux scheduling, not isolated CUDA kernel time",
        "run_timing_sec": {
            "initialization_until_ready": round((ready_ns-run_started_ns)/1e9, 3),
            "host_frame_feeding": round((feeder_end_ns-feeder_start_ns)/1e9, 3),
            "eos_drain_after_feed": round((pipe_end_ns-feeder_end_ns)/1e9, 3),
            "total_wall": round((pipe_end_ns-run_started_ns)/1e9, 3),
        },
        "appsrc_advertised_fps": 20,
        "input_clip_fps": 30,
        "feeder_note": "consecutive 30-FPS source frames are paced through appsrc at 20 FPS; throughput is not reported as 30-FPS real-time",
        "event_log": str(event_path),
        "inference_validation_scope": "PGIE and NvDCF actual frame/metadata flow only; not person recall, cross-camera ID, performance or GPU safety",
        "pgie_frames_by_source": ({str(k): v for k, v in pgie_counts.items()} if args.inference else None),
        "tracker_frames_by_source": ({str(k): v for k, v in track_counts.items()} if args.inference else None),
        "recorded_person_detection_observations": (sum(int(row.get("num_obj_meta", 0)) for row in pgie_rows) if args.inference else None),
        "recorded_local_track_observations": (sum(len(row.get("objects", [])) for row in track_rows) if args.inference else None),
        "frame_meta_to_input_crosswalk": {str(sid): frame_sequences[sid] for sid in frame_sequences},
        "source_pts_scope": "recorded-file presentation timestamps; no cross-camera synchronization claim",
        "deepstream_stderr": str(output / "deepstream.stderr.log"),
        "deepstream_stdout": str(output / "deepstream.stdout.log"),
    }
    (output / "mux-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


class CaptureWorker(threading.Thread):
    def __init__(self, controller, source_id: int, camera, settings, transform):
        super().__init__(name=f"entrance-capture-{camera.camera_id}", daemon=True)
        self.controller = controller
        self.source_id = source_id
        self.camera = camera
        self.settings = settings
        self.transform = transform
        self.stop_event = controller.stop_event
        self.frames = 0
        self.started_ns = 0
        self.started_utc = None
        self.last_pts_ns = -1
        self.last_frame_ns = 0
        self.error = ""
        self.state = "CONNECTING"

    def run(self) -> None:
        from services.ml_service.app.deepstream.capture import DeepStreamCapture
        width, height = NATIVE_SIZE[self.camera.camera_id]
        config = replace(self.settings.deepstream, display_width=width, display_height=height,
                         postdecode_queue_buffers=1, capture_timeout_ms=600)
        capture = None
        try:
            capture = DeepStreamCapture(
                self.camera.camera_id, self.camera.uri, CODECS[self.camera.camera_id], config,
                transport=self.settings.deepstream.rtsp_transport,
                username=self.camera.username, password=self.camera.password,
                output_bgrx=True, latency_ms=config.latency_ms,
                low_latency_mode=self.camera.decoder_low_latency_mode,
                decoder_extra_surfaces=self.camera.decoder_extra_surfaces,
            )
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if capture.last_error(): raise RuntimeError(capture.last_error())
                    continue
                if frame.ndim != 3 or frame.shape != (height, width, 4):
                    raise RuntimeError(f"{self.camera.camera_id}: decoder returned unexpected BGRx geometry")
                arrival_ns = capture.last_timing.appsink_receive_ns or time.monotonic_ns()
                pts_ns = capture.last_timing.pts_ns
                dts_ns = capture.last_timing.dts_ns
                rectified = self.transform(frame[..., :3].copy())
                self.frames += 1
                self.last_pts_ns = pts_ns
                self.last_frame_ns = arrival_ns
                self.state = "ONLINE"
                self.controller.queue_frame(
                    self.source_id, self.frames - 1, rectified, pts_ns, dts_ns, arrival_ns
                )
        except Exception as exc:
            self.error = redact(f"{type(exc).__name__}: {exc}")
            self.state = "ERROR"
            self.controller.error_queue.put(f"{self.camera.camera_id}: {self.error}")
            self.stop_event.set()
        finally:
            if capture is not None:
                try: capture.close()
                except Exception: pass
            if self.state != "ERROR": self.state = "STOPPED"


class LiveController:
    def __init__(self, args, output: Path):
        self.args = args
        self.output = output
        self.stop_event = threading.Event()
        self.error_queue: queue.Queue[str] = queue.Queue()
        self.frame_queues = {0: queue.Queue(maxsize=1), 1: queue.Queue(maxsize=1)}
        self.frame_lock = threading.Lock()
        self.frame_cache = {0: OrderedDict(), 1: OrderedDict()}
        self.latest_frame = {0: None, 1: None}
        self.latest_meta = {0: None, 1: None}
        self.stats = {
            sid: {"frames_captured": 0, "frames_sent": 0, "queue_drops": 0,
                  "mux_frames": 0, "pgie_frames": 0, "pgie_person_detections": 0,
                  "tracker_frames": 0, "tracker_person_observations": 0,
                  "local_tracker_ids": set(), "source_id_mismatch": 0,
                  "mux_frame_num_nonmonotonic": 0, "display_frame_matched": 0,
                  "display_frame_missing": 0, "last_mux_frame_num": -1,
                  "last_meta_mono_ns": 0, "fps_window": deque(maxlen=300),
                  "capture_times": deque(maxlen=300)}
            for sid in (0, 1)
        }
        self.pipe: DeepStreamProcess | None = None
        self.settings = None
        self.ownership = None
        self.capture_workers: list[CaptureWorker] = []
        self.writer_thread: threading.Thread | None = None
        self.started_ns = 0
        self.ready_ns = 0
        self.run_phase = "STARTING"
        self.error = ""
        self._processed_events = 0
        self.stopped_ns = 0
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._gpu_stop = threading.Event()
        self._gpu_thread: threading.Thread | None = None
        self.gpu_samples: list[dict[str, Any]] = []
        self.paint_latency_ms: deque[float] = deque(maxlen=10000)
        self.event_records: list[dict[str, Any]] = []
        self.events_path = self.output / "live-events.jsonl"

    def start(self) -> None:
        from services.camera_v11.source_ownership import SourceOwnership
        from services.ml_service.app.config import load_settings

        self.started_ns = time.monotonic_ns()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        os.chdir(ROOT)
        self.settings = load_settings()
        by_id = {camera.camera_id: camera for camera in self.settings.cameras}
        missing = [camera for camera in CAMERAS if camera not in by_id]
        if missing: raise RuntimeError("authorized camera config lacks required sources: " + ",".join(missing))
        lock_dir = Path(os.getenv("CAMERA_OWNER_LOCK_DIR", ".runtime/camera-owner-locks"))
        if not lock_dir.is_absolute(): lock_dir = ROOT / lock_dir
        self.ownership = SourceOwnership(CAMERAS, lock_dir)
        self.ownership.__enter__()  # refuse to connect if an existing cooperative owner is active
        self.pipe = DeepStreamProcess(mux_only=False, output_root=self.output)
        self.pipe.wait_ready(timeout_sec=30)
        bridge_mod = import_bridge(Path(self.args.bridge).expanduser())
        self.capture_workers = [
            CaptureWorker(self, sid, by_id[camera_id], self.settings,
                          bridge_mod.RectificationBridge(camera_id))
            for sid, camera_id in SOURCE_BY_ID.items()
        ]
        self.writer_thread = threading.Thread(target=self._writer_loop, name="appsrc-writer", daemon=True)
        self.writer_thread.start()
        for worker in self.capture_workers: worker.start()
        self._gpu_thread = threading.Thread(target=self._sample_gpu, name="device-gpu-sampler", daemon=True)
        self._gpu_thread.start()
        self.run_phase = "LIVE_CONNECTING"

    def _sample_gpu(self) -> None:
        """Sample device-wide utilization only; this is not process attribution."""
        while not self._gpu_stop.is_set():
            sample: dict[str, Any] = {"monotonic_ns": time.monotonic_ns(), "scope": "device-total"}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.5, check=True,
                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                         "LC_ALL": "C"},
                )
                fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                if len(fields) == 3:
                    sample.update({"gpu_utilization_percent": int(fields[0]),
                                   "memory_used_mib": int(fields[1]),
                                   "memory_total_mib": int(fields[2]), "status": "AVAILABLE"})
                else:
                    sample.update({"status": "UNAVAILABLE", "reason": "unexpected nvidia-smi output"})
            except Exception as exc:
                sample.update({"status": "UNAVAILABLE", "reason": type(exc).__name__})
            self.gpu_samples.append(sample)
            if len(self.gpu_samples) > 100:
                del self.gpu_samples[:-100]
            self._gpu_stop.wait(2.0)

    def queue_frame(self, source_id: int, frame_index: int, image: np.ndarray,
                    source_pts_ns: int, source_dts_ns: int, arrival_ns: int) -> None:
        with self.frame_lock:
            cache = self.frame_cache[source_id]
            cache[frame_index] = (image, source_pts_ns, source_dts_ns, arrival_ns)
            while len(cache) > 12: cache.popitem(last=False)
            self.latest_frame[source_id] = frame_index
            self.stats[source_id]["frames_captured"] += 1
            self.stats[source_id]["capture_times"].append(arrival_ns / 1e9)
        q = self.frame_queues[source_id]
        try: q.put_nowait((frame_index, image, source_pts_ns, source_dts_ns, arrival_ns))
        except queue.Full:
            try: q.get_nowait()
            except queue.Empty: pass
            self.stats[source_id]["queue_drops"] += 1
            q.put_nowait((frame_index, image, source_pts_ns, source_dts_ns, arrival_ns))

    def _writer_loop(self) -> None:
        assert self.pipe is not None
        source_ids = (0, 1)
        cursor = 0
        while not self.stop_event.is_set():
            source_id = source_ids[cursor % 2]
            cursor += 1
            try: item = self.frame_queues[source_id].get(timeout=0.02)
            except queue.Empty: continue
            frame_index, image, pts, dts, arrival = item
            try:
                self.pipe.send_frame(source_id, frame_index, image, pts, dts, arrival)
                self.stats[source_id]["frames_sent"] += 1
            except Exception as exc:
                self.error = redact(f"DeepStream appsrc writer: {exc}")
                self.error_queue.put(self.error)
                self.stop_event.set()
                break

    def process_events(self) -> list[dict[str, Any]]:
        assert self.pipe is not None
        records = []
        while True:
            try: message = self.error_queue.get_nowait()
            except queue.Empty: break
            if not self.error: self.error = redact(message)
        while True:
            try: record = self.pipe.events.get_nowait()
            except queue.Empty: break
            records.append(record)
            self.event_records.append(record)
            kind = record.get("type")
            if kind == "ERROR":
                self.error = redact(str(record.get("message") or record.get("raw") or "DeepStream error"))
                self.stop_event.set()
                continue
            if kind not in {"MUX", "PGIE", "TRACK"}: continue
            sid = int(record.get("source_id", -1))
            if sid not in self.stats:
                self.error = f"DeepStream emitted unexpected source_id {sid}"
                self.stop_event.set()
                continue
            stat = self.stats[sid]
            if record.get("input_frame_index") is None:
                stat["source_id_mismatch"] += 1
                continue
            if kind == "MUX":
                stat["mux_frames"] += 1
                frame_num = int(record.get("frame_num", -1))
                if frame_num <= stat["last_mux_frame_num"]:
                    stat["mux_frame_num_nonmonotonic"] += 1
                stat["last_mux_frame_num"] = frame_num
            elif kind == "PGIE":
                stat["pgie_frames"] += 1
                stat["pgie_person_detections"] += int(record.get("num_obj_meta", 0))
            elif kind == "TRACK":
                stat["tracker_frames"] += 1
                objects = record.get("objects", [])
                stat["tracker_person_observations"] += len(objects)
                for obj in objects:
                    track_id = obj.get("tracker_id")
                    if track_id is not None: stat["local_tracker_ids"].add(int(track_id))
                index = int(record["input_frame_index"])
                with self.frame_lock:
                    exact = self.frame_cache[sid].get(index)
                    if exact is not None:
                        stat["display_frame_matched"] += 1
                        self.latest_meta[sid] = record
                        stat["last_meta_mono_ns"] = time.monotonic_ns()
                        stat["fps_window"].append(time.monotonic())
                    else:
                        stat["display_frame_missing"] += 1
            self._processed_events += 1
        if self.pipe.process.poll() is not None and not self.stop_event.is_set() and not self.error:
            self.error = f"DeepStream container exited unexpectedly with status {self.pipe.process.returncode}"
            self.stop_event.set()
        return records

    def frame_payload(self, sid: int):
        with self.frame_lock:
            meta = self.latest_meta[sid]
            if meta is None: return None
            frame_index = int(meta["input_frame_index"])
            cached = self.frame_cache[sid].get(frame_index)
            if cached is None: return None
            return cached[0], meta, cached[3]

    def source_state(self, sid: int) -> str:
        if self.error: return "ERROR"
        worker = self.capture_workers[sid] if len(self.capture_workers) > sid else None
        return worker.state if worker else self.run_phase

    def cam_fps(self, sid: int) -> float:
        times = list(self.stats[sid]["fps_window"])
        if len(times) < 2 or times[-1] <= times[0]: return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def source_fps(self, sid: int) -> float:
        times = list(self.stats[sid]["capture_times"])
        if len(times) < 2 or times[-1] <= times[0]: return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def record_paint_latency(self, value_ms: float) -> None:
        self.paint_latency_ms.append(float(value_ms))

    def stop(self) -> None:
        if not self._stop_lock.acquire(blocking=False):
            return
        try:
            if self._stopped: return
            self.stop_event.set()
            for worker in self.capture_workers:
                worker.join(timeout=8)
            if self.writer_thread is not None:
                self.writer_thread.join(timeout=3)
            self._gpu_stop.set()
            if self._gpu_thread is not None:
                self._gpu_thread.join(timeout=2)
            if self.pipe is not None:
                self.pipe.end_input()
                self.pipe.stop(timeout_sec=10)
                if self.pipe.process.returncode not in (0, None) and not self.error:
                    self.error = f"DeepStream shutdown was not clean (exit status {self.pipe.process.returncode})"
                # Drain records emitted immediately before EOS so the final
                # report and raw sidecar include the pipeline tail.
                self.process_events()
            workers_alive = [worker.name for worker in self.capture_workers if worker.is_alive()]
            if self.ownership is not None and not workers_alive:
                self.ownership.__exit__(None, None, None)
                self.ownership = None
            elif workers_alive:
                self.error = self.error or ("capture cleanup timeout; source locks retained until process exit: "
                                            + ",".join(workers_alive))
            with self.events_path.open("w", encoding="utf-8") as event_file:
                os.chmod(self.events_path, 0o600)
                for record in self.event_records:
                    event_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.run_phase = "STOPPED" if not self.error else "ERROR"
            self.stopped_ns = time.monotonic_ns()
            self._stopped = True
        finally:
            self._stop_lock.release()


def run_live(args) -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget

    output = Path(args.output).resolve() if args.output else Path(tempfile.mkdtemp(prefix="entrance-live-track-"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty live evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    layout_path = Path(args.layout).expanduser() if args.layout else Path(
        "/tmp/amc-nvidia-skill-stage.pd7zJs/projects/project_20261018_080650_1744/manual_adjustment/layout.png"
    )
    app = QApplication(sys.argv[:1])

    class VideoPane(QWidget):
        def __init__(self, camera: str, latency_callback):
            super().__init__()
            self.camera = camera
            self.latency_callback = latency_callback
            self.image = QImage()
            self.objects = []
            self.info = "Waiting for an inference-matched frame"
            self.arrival_ns = 0
            self.frame_index = -1
            self.latency_frame_index = -1
            self.setMinimumSize(440, 380)

        def set_payload(self, frame, meta, arrival_ns):
            h, w = frame.shape[:2]
            qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
            self.image = qimg.copy()
            self.objects = list(meta.get("objects", []))
            self.arrival_ns = int(arrival_ns)
            pts = meta.get("source_pts_ns")
            pts_text = "UNAVAILABLE" if pts in (None, -1) else f"{int(pts) / 1e9:.3f}s"
            frame_idx = meta.get("input_frame_index", "?")
            self.frame_index = int(frame_idx) if frame_idx != "?" else -1
            gst_fn = meta.get("frame_num", "?")
            local = ", ".join(str(o.get("tracker_id")) for o in self.objects if o.get("tracker_id") is not None)
            self.info = (f"Source ID {meta.get('source_id')} · input frame {frame_idx} / mux frame {gst_fn} · "
                         f"source PTS {pts_text} / mux buf_pts {int(meta.get('buf_pts', 0))/1e9:.3f}s · "
                         f"people detections {len(self.objects)} · local NvDCF IDs {local or 'NO_PEOPLE'}")
            self.update()

        def paintEvent(self, _event):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor("#101820"))
            margin, header = 8, 32
            p.setPen(QColor("#f3f7fa")); p.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
            p.drawText(margin, 22, f"{self.camera} — LIVE rectified frame")
            area = self.rect().adjusted(margin, header, -margin, -40)
            if not self.image.isNull():
                size = self.image.size(); scale = min(area.width()/size.width(), area.height()/size.height())
                width, height = size.width()*scale, size.height()*scale
                target = (area.x()+(area.width()-width)/2, area.y()+(area.height()-height)/2, width, height)
                p.drawImage(int(target[0]), int(target[1]), self.image.scaled(int(width), int(height),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                sx, sy = width/1920.0, height/1080.0
                p.setPen(QPen(QColor("#20f080"), 2))
                for obj in self.objects:
                    x = target[0] + float(obj.get("left", 0))*sx
                    y = target[1] + float(obj.get("top", 0))*sy
                    w = float(obj.get("width", 0))*sx; h = float(obj.get("height", 0))*sy
                    p.drawRect(int(x), int(y), int(w), int(h))
                    tid = obj.get("tracker_id")
                    conf = float(obj.get("confidence", 0.0))
                    label = f"local NvDCF {tid if tid is not None else 'pending'} · det {conf:.2f}"
                    label_width = min(max(190, len(label)*7), max(1, int(width)))
                    label_x = min(max(int(x), int(target[0])), int(target[0]+width)-label_width)
                    label_y = min(max(int(y)-20, int(target[1])), int(target[1]+height)-20)
                    p.fillRect(label_x, label_y, label_width, 20, QColor(0, 0, 0, 180))
                    p.setPen(QColor("#ffffff")); p.drawText(label_x+4, label_y+15, label)
                    p.setPen(QPen(QColor("#20f080"), 2))
            p.setPen(QColor("#d8e3eb")); p.setFont(QFont("Sans Serif", 9))
            p.drawText(self.rect().adjusted(8, self.height()-36, -8, -4), Qt.AlignmentFlag.AlignLeft, self.info)
            p.end()
            if self.arrival_ns and self.frame_index >= 0 and self.frame_index != self.latency_frame_index:
                latency_ms = max(0.0, (time.monotonic_ns() - self.arrival_ns) / 1e6)
                self.latency_frame_index = self.frame_index
                self.info += f" · decoded→Qt paint {latency_ms:.1f} ms"
                self.latency_callback(latency_ms)
                self.update()

    class MapPane(QWidget):
        def __init__(self):
            super().__init__(); self.pixmap = QPixmap(str(layout_path)) if layout_path.is_file() else QPixmap()
            self.setMinimumSize(360, 380)

        def paintEvent(self, _event):
            p = QPainter(self); p.fillRect(self.rect(), QColor("#151c22"))
            p.setPen(QColor("#ffffff")); p.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
            p.drawText(8, 22, "Entrance Room map — projection disabled")
            area = self.rect().adjusted(8, 34, -8, -86)
            if not self.pixmap.isNull():
                pix = self.pixmap.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
                x = area.x()+(area.width()-pix.width())//2; y = area.y()+(area.height()-pix.height())//2
                p.setOpacity(0.45); p.drawPixmap(x, y, pix); p.setOpacity(1.0)
            p.fillRect(self.rect().adjusted(8, self.height()-78, -8, -8), QColor("#713b12"))
            p.setPen(QColor("#fff0d1")); p.setFont(QFont("Sans Serif", 11, QFont.Weight.Bold))
            p.drawText(self.rect().adjusted(16, self.height()-74, -16, -10),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       "GEOMETRY UNVERIFIED\nPairwise calibration and source-clock synchronization are unverified. No markers are projected.")

    class Window(QMainWindow):
        def __init__(self):
            super().__init__(); self.controller = None; self.run_started = 0.0
            self.smoke_passed = False; self.review_started = 0.0; self.finish_started = 0.0
            self.smoke_baseline = None
            self.shutdown_started = 0.0
            self.review_skipped_reason = ""
            self.setWindowTitle("Entrance Room — Isolated LIVE DeepStream 9.1 / local NvDCF")
            self.resize(1780, 920)
            central = QWidget(); layout = QVBoxLayout(central)
            self.banner = QLabel("STARTING ISOLATED DEEPSTREAM 9.1 — no global IDs / map projection")
            self.banner.setStyleSheet("background:#25435b;color:white;padding:9px;font-weight:700;font-size:16px")
            latency_callback = lambda value: self.controller.record_paint_latency(value) if self.controller else None
            row = QHBoxLayout(); self.panes = {
                "CAM-02": VideoPane("CAM-02", latency_callback),
                "CAM-05": VideoPane("CAM-05", latency_callback)}
            for camera in CAMERAS: row.addWidget(self.panes[camera], 2)
            row.addWidget(MapPane(), 1)
            self.status = QLabel("Starting local DeepStream process…")
            self.status.setWordWrap(True)
            self.status.setStyleSheet("background:#202b33;color:#e3eef4;padding:8px")
            layout.addWidget(self.banner); layout.addLayout(row, 1); layout.addWidget(self.status)
            self.setCentralWidget(central)
            self.timer = QTimer(self); self.timer.setInterval(50); self.timer.timeout.connect(self.refresh)
            self.timer.start()

        def refresh(self):
            if self.controller is None:
                if not self.run_started:
                    self.run_started = time.monotonic()
                    threading.Thread(target=self._start_controller, daemon=True).start()
                return
            records = self.controller.process_events()
            for camera, sid in (("CAM-02", 0), ("CAM-05", 1)):
                payload = self.controller.frame_payload(sid)
                if payload:
                    frame, meta, arrival = payload
                    current = getattr(self.panes[camera], "_shown_index", -1)
                    if int(meta.get("input_frame_index", -1)) > current:
                        self.panes[camera].set_payload(frame, meta, arrival)
                        self.panes[camera]._shown_index = int(meta["input_frame_index"])
            if self.controller.error:
                self.banner.setText("LIVE TEST STOPPING — ERROR")
                self.status.setText(self.controller.error)
                self._stop_async()
                return
            states = [self.controller.source_state(sid) for sid in (0, 1)]
            stats = self.controller.stats
            ready_cameras = all(state == "ONLINE" for state in states)
            if ready_cameras and self.controller.ready_ns == 0:
                self.controller.ready_ns = time.monotonic_ns()
                self.controller.run_phase = "15-SECOND SMOKE"
                self.smoke_baseline = {
                    sid: {key: stats[sid][key] for key in
                          ("mux_frames", "pgie_frames", "tracker_frames", "display_frame_matched")}
                    for sid in (0, 1)}
            if not self.controller.ready_ns:
                self.banner.setText(f"CONNECTING CAMERAS — {states[0]} / {states[1]}")
                if self.controller.started_ns and time.monotonic_ns() - self.controller.started_ns > 30_000_000_000:
                    self.controller.error = "camera readiness timed out after 30 seconds; closing both subscriptions"
                    self._stop_async(); return
            elapsed = ((time.monotonic_ns() - self.controller.ready_ns)/1e9) if self.controller.ready_ns else 0.0
            detections = sum(stats[sid]["pgie_person_detections"] for sid in (0, 1))
            tracked = sum(stats[sid]["tracker_person_observations"] for sid in (0, 1))
            fps = {camera: self.controller.cam_fps(sid) for camera, sid in (("CAM-02", 0), ("CAM-05", 1))}
            decoded_fps = {camera: self.controller.source_fps(sid) for camera, sid in (("CAM-02", 0), ("CAM-05", 1))}
            gpu = next((sample for sample in reversed(self.controller.gpu_samples)
                        if sample.get("status") == "AVAILABLE"), None)
            gpu_text = (f"GPU device-total {gpu['gpu_utilization_percent']}% · VRAM {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB"
                        if gpu else "GPU/VRAM UNAVAILABLE")
            people_text = f"person observations {tracked}" if detections else "NO_PEOPLE (no person detections observed)"
            self.status.setText(
                f"LIVE · CAM-02 {states[0]} decoded {decoded_fps['CAM-02']:.1f} FPS / NvDCF {fps['CAM-02']:.1f} FPS "
                f"({stats[0]['frames_captured']} received, {stats[0]['frames_sent']} sent, {stats[0]['queue_drops']} host queue drops) · "
                f"CAM-05 {states[1]} decoded {decoded_fps['CAM-05']:.1f} FPS / NvDCF {fps['CAM-05']:.1f} FPS "
                f"({stats[1]['frames_captured']} received, {stats[1]['frames_sent']} sent, {stats[1]['queue_drops']} host queue drops) · "
                f"PGIE frames {stats[0]['pgie_frames']}/{stats[1]['pgie_frames']} · NvDCF frames {stats[0]['tracker_frames']}/{stats[1]['tracker_frames']} · "
                f"{people_text} · local IDs CAM-02 {len(stats[0]['local_tracker_ids'])}, CAM-05 {len(stats[1]['local_tracker_ids'])} · "
                f"global IDs NOT RUN · Re-ID NOT ENABLED · {gpu_text} · elapsed {elapsed:.1f}s"
            )
            if self.controller.ready_ns and not self.smoke_passed and elapsed >= args.smoke_seconds:
                smoke_ok = smoke_gate_passed(stats, self.smoke_baseline)
                if not smoke_ok:
                    self.controller.error = "15-second smoke failed: both sources did not reach mux, PGIE, and NvDCF"
                    self._stop_async(); return
                self.smoke_passed = True
                if not safe_for_extended_review(self.controller.gpu_samples):
                    self.review_skipped_reason = "extended review withheld: device-total GPU telemetry unavailable or >=95% in at least two of the last three samples"
                    self.controller.run_phase = "SMOKE_PASS_REVIEW_SKIPPED_RESOURCE_GUARD"
                    self.banner.setText("15-SECOND SMOKE PASS — EXTENDED REVIEW WITHHELD BY GPU SAFETY GUARD")
                    self._stop_async(); return
                self.review_started = time.monotonic()
                self.controller.run_phase = "60-SECOND VISUAL REVIEW"
                self.banner.setText("15-SECOND SMOKE PASS — CONTINUING ONE BOUNDED 60-SECOND LIVE REVIEW")
            elif self.smoke_passed:
                review_elapsed = time.monotonic() - self.review_started
                self.banner.setText(f"LIVE VISUAL REVIEW · {max(0, args.review_seconds-review_elapsed):.1f}s remaining · LOCAL NvDCF ONLY")
                if review_elapsed >= args.review_seconds:
                    self.banner.setText("LIVE SESSION COMPLETE — RTSP AND DEEPSTREAM STOPPING")
                    self._stop_async()

        def _start_controller(self):
            controller = LiveController(args, output)
            try:
                controller.start()
            except Exception as exc:
                controller.error = redact(f"{type(exc).__name__}: {exc}")
                controller.run_phase = "ERROR"
                controller.stop()
            self.controller = controller

        def _stop_async(self):
            if self.finish_started: return
            self.finish_started = time.monotonic()
            if self.controller is not None:
                threading.Thread(target=self.controller.stop, daemon=True).start()
            self._shutdown_timer = QTimer(self)
            self._shutdown_timer.setInterval(200)
            self._shutdown_timer.timeout.connect(self._poll_shutdown)
            self._shutdown_timer.start()

        def _poll_shutdown(self):
            if self.controller is None:
                app.quit(); return
            if self.controller._stopped:
                self._shutdown_timer.stop()
                QTimer.singleShot(1000, app.quit)
            elif self.finish_started and time.monotonic() - self.finish_started > 45:
                self._shutdown_timer.stop()
                app.quit()

        def closeEvent(self, event):  # noqa: N802
            if self.controller is not None:
                self.controller.stop()
            event.accept()

    window = Window(); window.show()
    app.exec()
    if window.controller is not None:
        window.controller.stop()
        window.controller.process_events()
    stats = window.controller.stats if window.controller else {}
    report = {
        "image": DS_IMAGE, "image_id": DS_IMAGE_ID,
        "pipeline": "host RTSP decode + empirical rectification → BGR appsrc → DS9.1 NVMM + legacy nvstreammux → PeopleNet Transformer → local NvDCF",
        "map": "disabled; geometry and source-clock synchronization unverified",
        "global_ids": "NOT_RUN; no MV3DT association enabled",
        "persistent_identity": "NOT ENABLED",
        "smoke_passed": window.smoke_passed,
        "extended_review_skipped_reason": window.review_skipped_reason,
        "started_utc": window.controller.started_utc if window.controller else None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "session_duration_sec": round((window.controller.stopped_ns-window.controller.ready_ns)/1e9, 3)
            if window.controller and window.controller.ready_ns and window.controller.stopped_ns else None,
        "total_runtime_sec": round((window.controller.stopped_ns-window.controller.started_ns)/1e9, 3)
            if window.controller and window.controller.started_ns and window.controller.stopped_ns else None,
        "status": window.controller.error or window.controller.run_phase if window.controller else "START_FAILED",
        "cameras": {}, "deepstream_stdout": str(output / "deepstream.stdout.log"),
        "deepstream_stderr": str(output / "deepstream.stderr.log"), "output_dir": str(output),
    }
    for camera, sid in (("CAM-02", 0), ("CAM-05", 1)):
        if stats:
            stat = stats[sid]
            report["cameras"][camera] = {
                key: (len(value) if isinstance(value, set) else value)
                for key, value in stat.items() if key not in {"fps_window", "capture_times"}
            }
            report["cameras"][camera]["measured_processed_fps"] = round(window.controller.cam_fps(sid), 3)
            report["cameras"][camera]["measured_decoded_fps"] = round(window.controller.source_fps(sid), 3)
            report["cameras"][camera]["sent_not_observed_at_mux"] = max(
                0, stat["frames_sent"] - stat["mux_frames"])
    latency = sorted(window.controller.paint_latency_ms) if window.controller else []
    if latency:
        report["decoded_to_qt_paint_ms"] = {
            "samples": len(latency), "p50": round(float(np.percentile(latency, 50)), 3),
            "p95": round(float(np.percentile(latency, 95)), 3),
            "p99": round(float(np.percentile(latency, 99)), 3), "max": round(latency[-1], 3),
        }
    else:
        report["decoded_to_qt_paint_ms"] = "NOT_MEASURED"
    report["gpu_device_total_samples"] = window.controller.gpu_samples if window.controller else []
    report["camera_mapping"] = {"DeepStream source_id 0": "CAM-02", "DeepStream source_id 1": "CAM-05"}
    report["drop_counter_scope"] = (
        "queue_drops counts host latest-frame queue replacements; sent_not_observed_at_mux is appsrc/queue frames not seen at mux by EOS; "
        "display_frame_missing counts tracker outputs whose exact source frame had already left the 12-frame UI cache. "
        "A missing-cache record is not drawn with mismatched pixels/boxes."
    )
    report["timing_note"] = "Each source receives increasing pipeline-running-time PTS; original RTSP PTS/DTS and host arrival monotonic timestamps are sidecar metadata. No cross-camera exposure alignment is asserted."
    report_path = output / "live-run.json"
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    os.chmod(report_path, 0o600)
    print(json.dumps(report, indent=2, default=str))
    return 0 if window.smoke_passed else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    mux = sub.add_parser("mux-smoke", help="recorded-frame appsrc→legacy nvstreammux metadata test")
    mux.add_argument("--cam02-video", required=True)
    mux.add_argument("--cam05-video", required=True)
    mux.add_argument("--frames", type=int, default=8)
    mux.add_argument("--inference", action="store_true", help="exercise selected detector and stock NvDCF")
    mux.add_argument("--detector", choices=("peoplenet", "yolo26m"), default="peoplenet")
    mux.add_argument("--yolo-artifacts", help="read-only .runtime directory with SHA-pinned YOLO26m engine, ONNX, parser, labels")
    mux.add_argument("--comparison-runtime", action="store_true",
                     help="PeopleNet-only offline comparison with the pinned YOLO DeepStream/TensorRT image; engine-only staging")
    mux.add_argument("--pnt-engine-only", action="store_true",
                     help="PeopleNet offline test using its default pinned image and hash-checked engine-only staging")
    mux.add_argument("--input-geometry", choices=("rectified", "raw"), default="rectified",
                     help="existing AMC rectified MP4s must NOT be rectified again")
    mux.add_argument("--bridge", default=str(BRIDGE_DEFAULT))
    mux.add_argument("--output")
    live = sub.add_parser("live", help="one bounded live inference and Qt review")
    live.add_argument("--bridge", default=str(BRIDGE_DEFAULT))
    live.add_argument("--layout")
    live.add_argument("--output")
    live.add_argument("--smoke-seconds", type=float, default=15.0)
    live.add_argument("--review-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if args.mode == "mux-smoke":
        if args.frames < 2 or args.frames > 600: parser.error("--frames must be 2..600 (30 seconds at the 20 FPS feeder rate)")
        if args.detector == "yolo26m" and not args.inference:
            parser.error("--detector yolo26m requires --inference")
        if args.detector == "yolo26m" and not args.yolo_artifacts:
            parser.error("--detector yolo26m requires --yolo-artifacts")
        if args.comparison_runtime and args.detector != "peoplenet":
            parser.error("--comparison-runtime is restricted to PeopleNet offline comparison")
        if args.pnt_engine_only and args.detector != "peoplenet":
            parser.error("--pnt-engine-only is restricted to PeopleNet offline inference")
        return run_mux_smoke(args)
    if args.smoke_seconds != 15.0 or args.review_seconds != 60.0:
        parser.error("the bounded live run is fixed at 15s smoke plus 60s review")
    return run_live(args)


if __name__ == "__main__":
    sys.exit(main())
