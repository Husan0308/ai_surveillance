#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import site
import statistics
import subprocess
import sys
import threading
import time
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 4.5: RF-DETR-S TensorRT 8.6 FP32 + AnchoredPersonTracker live validation. "
            "Frames are scaled by ffmpeg directly to the fixed 800x448 engine input "
            "and a latest-frame-only mailbox prevents backlog."
        )
    )
    p.add_argument("--camera", default="CAM-03")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--capture-fps", type=float, default=10.0)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--person-threshold", type=float, default=0.18)
    p.add_argument("--person-class-id", type=int, default=1)
    p.add_argument("--iou", type=float, default=0.62)
    p.add_argument("--containment", type=float, default=0.90)
    p.add_argument("--center", type=float, default=0.35)
    p.add_argument("--expected-persons", type=int, default=-1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument(
        "--engine",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt86_fp32/rfdetr-small-800x448-fp32.engine"),
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step4/trt_live")
    )
    return p.parse_args()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[pos])


def _camera_rtsp_url(camera) -> str:
    raw = str(camera.uri)
    parts = urlsplit(raw)
    if parts.username or not camera.username:
        return raw
    host = parts.hostname or ""
    if not host:
        raise RuntimeError(f"invalid RTSP URI for {camera.camera_id}")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    user = quote(str(camera.username), safe="")
    password = quote(str(camera.password or ""), safe="")
    auth = user if not password else f"{user}:{password}"
    return urlunsplit((parts.scheme, f"{auth}@{host}", parts.path, parts.query, parts.fragment))


def _redacted_rtsp_url(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or "camera"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))


def _find_library(patterns: list[str]) -> Path | None:
    roots = [Path(p) for p in site.getsitepackages()]
    candidates: list[Path] = []
    for root in roots:
        for pattern in patterns:
            candidates.extend(root.glob(pattern))
    return sorted({p.resolve() for p in candidates if p.is_file()})[0] if candidates else None


def _preload_cuda_stack() -> dict[str, str]:
    specs = {
        "cudart": ["nvidia/cuda_runtime/lib/libcudart.so.12*"],
        "cublasLt": ["nvidia/cublas/lib/libcublasLt.so.12*"],
        "cublas": ["nvidia/cublas/lib/libcublas.so.12*"],
        "cudnn": ["nvidia/cudnn/lib/libcudnn.so.8*"],
    }
    loaded: dict[str, str] = {}
    for name, patterns in specs.items():
        path = _find_library(patterns)
        if path is None:
            raise SystemExit(f"STEP5_FAIL nvidia_library_missing={name}")
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        loaded[name] = str(path)
    return loaded


def _load_cudart() -> ctypes.CDLL:
    path = _find_library(["nvidia/cuda_runtime/lib/libcudart.so.12*"])
    if path is None:
        raise SystemExit("STEP5_FAIL cudart_missing")
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    return lib


def _cuda_check(code: int, op: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"CUDA {op} failed code={code}")


class TrtRunner:
    def __init__(self, engine_path: Path) -> None:
        _preload_cuda_stack()
        import tensorrt as trt

        if not str(trt.__version__).startswith("8.6.1"):
            raise SystemExit(f"STEP5_FAIL wrong_tensorrt={trt.__version__}")
        self.trt = trt
        self.cudart = _load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise SystemExit("STEP5_FAIL engine_deserialize")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise SystemExit("STEP5_FAIL context_create")

        self.bindings: list[int] = [0] * int(self.engine.num_bindings)
        self.device_ptrs: list[ctypes.c_void_p] = []
        self.outputs: dict[str, np.ndarray] = {}
        self.input_index = -1
        self.input_shape: tuple[int, ...] | None = None
        self.input_dtype: np.dtype | None = None

        for index in range(int(self.engine.num_bindings)):
            name = self.engine.get_binding_name(index)
            shape = tuple(int(v) for v in self.engine.get_binding_shape(index))
            dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(index)))
            host = np.empty(shape, dtype=dtype)
            ptr = ctypes.c_void_p()
            _cuda_check(self.cudart.cudaMalloc(ctypes.byref(ptr), host.nbytes), f"malloc:{name}")
            self.device_ptrs.append(ptr)
            self.bindings[index] = int(ptr.value or 0)
            if self.engine.binding_is_input(index):
                self.input_index = index
                self.input_shape = shape
                self.input_dtype = dtype
            else:
                self.outputs[name] = host

        if self.input_index < 0 or self.input_shape is None or self.input_dtype is None:
            raise SystemExit("STEP5_FAIL input_binding_missing")
        if "dets" not in self.outputs or "labels" not in self.outputs:
            raise SystemExit(f"STEP5_FAIL output_bindings={list(self.outputs)}")

    @property
    def version(self) -> str:
        return str(self.trt.__version__)

    def infer(self, input_array: np.ndarray) -> tuple[dict[str, np.ndarray], float, float]:
        if tuple(input_array.shape) != self.input_shape or input_array.dtype != self.input_dtype:
            raise RuntimeError(
                f"input mismatch engine={self.input_shape}/{self.input_dtype} host={input_array.shape}/{input_array.dtype}"
            )
        input_array = np.ascontiguousarray(input_array)
        input_ptr = ctypes.c_void_p(self.bindings[self.input_index])

        total_start = time.perf_counter()
        _cuda_check(
            self.cudart.cudaMemcpy(
                input_ptr,
                ctypes.c_void_p(input_array.ctypes.data),
                input_array.nbytes,
                1,
            ),
            "H2D:input",
        )
        gpu_start = time.perf_counter()
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("execute_v2 returned false")
        _cuda_check(self.cudart.cudaDeviceSynchronize(), "sync_after_execute")
        gpu_ms = (time.perf_counter() - gpu_start) * 1000.0

        for index in range(int(self.engine.num_bindings)):
            if self.engine.binding_is_input(index):
                continue
            name = self.engine.get_binding_name(index)
            host = self.outputs[name]
            _cuda_check(
                self.cudart.cudaMemcpy(
                    ctypes.c_void_p(host.ctypes.data),
                    ctypes.c_void_p(self.bindings[index]),
                    host.nbytes,
                    2,
                ),
                f"D2H:{name}",
            )
        _cuda_check(self.cudart.cudaDeviceSynchronize(), "sync_after_D2H")
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return self.outputs, gpu_ms, total_ms

    def close(self) -> None:
        for ptr in self.device_ptrs:
            if ptr.value:
                self.cudart.cudaFree(ptr)
        self.device_ptrs.clear()


class LatestFramePipe:
    def __init__(self, process: subprocess.Popen, width: int, height: int) -> None:
        self.process = process
        self.frame_bytes = int(width) * int(height) * 3
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.stop_event = threading.Event()
        self.latest_seq = 0
        self.latest_t = 0.0
        self.latest_frame: bytes | None = None
        self.read_error = ""
        self.thread = threading.Thread(target=self._run, name="rfdetr-trt-live-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read_exact(self) -> bytes | None:
        if self.process.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = self.frame_bytes
        while remaining > 0 and not self.stop_event.is_set():
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if remaining == 0 else None

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                frame = self._read_exact()
                if frame is None:
                    if not self.stop_event.is_set():
                        self.read_error = "ffmpeg rawvideo stream ended"
                    break
                now = time.monotonic()
                with self.lock:
                    self.latest_seq += 1
                    self.latest_t = now
                    self.latest_frame = frame
                self.event.set()
        except BaseException as exc:
            self.read_error = f"{type(exc).__name__}: {exc}"
            self.event.set()

    def snapshot(self, after_seq: int) -> tuple[int, float, bytes] | None:
        with self.lock:
            if self.latest_frame is None or self.latest_seq <= after_seq:
                return None
            return self.latest_seq, self.latest_t, self.latest_frame

    def stop(self) -> None:
        self.stop_event.set()
        self.event.set()


def _start_ffmpeg(args: argparse.Namespace):
    from services.ml_service.app.config import load_settings

    settings = load_settings()
    camera = next((row for row in settings.cameras if row.camera_id == args.camera), None)
    if camera is None:
        available = ",".join(row.camera_id for row in settings.cameras)
        raise SystemExit(f"STEP5_FAIL camera_not_found={args.camera} available=[{available}]")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("STEP5_FAIL ffmpeg_not_found")

    url = _camera_rtsp_url(camera)
    redacted = _redacted_rtsp_url(url)
    vf = f"fps={float(args.capture_fps):.6f},scale={int(args.width)}:{int(args.height)}:flags=bilinear"
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-an",
        "-vf",
        vf,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=max(65536, int(args.width) * int(args.height) * 3),
    )
    return camera, process, url, redacted


def _terminate(process: subprocess.Popen) -> str:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    data = b""
    if process.stderr is not None:
        try:
            data = process.stderr.read() or b""
        except Exception:
            pass
    return data.decode("utf-8", errors="replace").strip()


def _preprocess(raw: bytes, width: int, height: int) -> np.ndarray:
    rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x))


def _person_rows(
    dets: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    class_id: int,
    width: int,
    height: int,
) -> list[dict]:
    boxes = np.asarray(dets[0], dtype=np.float32)
    logits = np.asarray(labels[0, :, :-1], dtype=np.float32)
    if class_id < 0 or class_id >= logits.shape[1]:
        raise RuntimeError(f"person_class_id_out_of_range id={class_id} classes={logits.shape[1]}")
    probs = _sigmoid(logits)
    flat = probs.reshape(-1)
    topk = min(300, flat.size)
    if topk < flat.size:
        order = np.argpartition(flat, -topk)[-topk:]
        order = order[np.argsort(-flat[order], kind="stable")]
    else:
        order = np.argsort(-flat, kind="stable")
    scores = flat[order]
    valid = scores > float(threshold)
    order = order[valid]
    scores = scores[valid]
    class_count = logits.shape[1]
    queries = order // class_count
    classes = order % class_count

    rows: list[dict] = []
    for score, query, cls in zip(scores, queries, classes):
        if int(cls) != int(class_id):
            continue
        cx, cy, bw, bh = [float(v) for v in boxes[int(query)]]
        x1 = max(0.0, min(float(width - 1), (cx - bw * 0.5) * width))
        y1 = max(0.0, min(float(height - 1), (cy - bh * 0.5) * height))
        x2 = max(x1 + 1.0, min(float(width), (cx + bw * 0.5) * width))
        y2 = max(y1 + 1.0, min(float(height), (cy + bh * 0.5) * height))
        rows.append(
            {
                "class_id": int(cls),
                "class_name": "person",
                "confidence": float(score),
                "xyxy": [x1, y1, x2, y2],
                "query": int(query),
            }
        )
    rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return rows


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _iou(a: list[float], b: list[float]) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a: list[float], b: list[float]) -> float:
    inter = _intersection(a, b)
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0.0 else 0.0


def _center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    aw, ah = max(1.0, a[2] - a[0]), max(1.0, a[3] - a[1])
    bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
    scale = max(20.0, math.hypot(aw, ah), math.hypot(bw, bh))
    return math.hypot(acx - bcx, acy - bcy) / scale


def _dedupe(persons: list[dict], iou_gate: float, containment_gate: float, center_gate: float):
    kept: list[dict] = []
    rejected: list[dict] = []
    for candidate in sorted(persons, key=lambda r: float(r["confidence"]), reverse=True):
        cbox = [float(v) for v in candidate["xyxy"]]
        duplicate = None
        for existing in kept:
            ebox = [float(v) for v in existing["xyxy"]]
            pair_iou = _iou(cbox, ebox)
            pair_cont = _containment(cbox, ebox)
            pair_center = _center_distance(cbox, ebox)
            if pair_iou >= iou_gate or (pair_cont >= containment_gate and pair_center <= center_gate):
                duplicate = {
                    **candidate,
                    "duplicate_of_confidence": float(existing["confidence"]),
                    "duplicate_metrics": {
                        "iou": round(pair_iou, 4),
                        "containment": round(pair_cont, 4),
                        "center_distance": round(pair_center, 4),
                    },
                }
                break
        if duplicate is None:
            kept.append(candidate)
        else:
            rejected.append(duplicate)
    return kept, rejected


def _annotate(raw: bytes, width: int, height: int, persons: list[dict], footer: str) -> Image.Image:
    image = Image.frombytes("RGB", (width, height), raw)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    for idx, row in enumerate(persons, start=1):
        x1, y1, x2, y2 = [float(v) for v in row["xyxy"]]
        conf = float(row["confidence"])
        draw.rectangle((x1, y1, x2, y2), outline=(255, 196, 64), width=3)
        track_id = row.get("track_id", idx)
        label = f"ID {track_id} {conf:.2f}"
        draw.rectangle((x1, max(0, y1 - 18), x1 + 115, y1), fill=(255, 196, 64))
        draw.text((x1 + 3, max(0, y1 - 16)), label, fill=(15, 18, 22), font=font)
    draw.rectangle((4, 4, min(width - 4, 390), 27), fill=(15, 18, 22))
    draw.text((10, 9), footer, fill=(255, 255, 255), font=font)
    return image


def main() -> int:
    args = _args()
    if args.seconds < 5.0 or args.seconds > 300.0:
        raise SystemExit("STEP5_FAIL seconds_must_be_5_to_300")
    if args.capture_fps <= 0.0 or args.capture_fps > 20.0:
        raise SystemExit("STEP5_FAIL capture_fps_must_be_0_to_20")
    if (args.width, args.height) != (800, 448):
        raise SystemExit(f"STEP5_FAIL engine_shape_fixed_800x448 got={args.width}x{args.height}")
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise SystemExit(f"STEP5_FAIL engine_not_found={args.engine}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = args.output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for old in sample_dir.glob("sample_*.jpg"):
        old.unlink()

    runner = TrtRunner(args.engine)
    if runner.input_shape != (1, 3, args.height, args.width):
        runner.close()
        raise SystemExit(f"STEP5_FAIL engine_input_shape={runner.input_shape}")

    camera, process, url, redacted = _start_ffmpeg(args)
    mailbox = LatestFramePipe(process, args.width, args.height)
    mailbox.start()

    first = None
    startup_deadline = time.monotonic() + 15.0
    while time.monotonic() < startup_deadline:
        first = mailbox.snapshot(0)
        if first is not None:
            break
        if process.poll() is not None or mailbox.read_error:
            break
        mailbox.event.wait(0.1)
        mailbox.event.clear()

    if first is None:
        mailbox.stop()
        stderr = _terminate(process).replace(url, redacted)
        runner.close()
        error = mailbox.read_error or stderr or "no first frame"
        raise SystemExit(f"STEP5_FAIL live_start camera={args.camera} error={error[-500:]}")

    first_seq, _, first_raw = first
    first_input = _preprocess(first_raw, args.width, args.height)
    for _ in range(max(1, int(args.warmup))):
        runner.infer(first_input)

    print(
        "STEP5_ENV "
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"tensorrt={runner.version} engine={args.engine} precision=FP32 "
        f"model={args.width}x{args.height} capture_fps={args.capture_fps:.1f} "
        "preprocess=ffmpeg_bilinear_rgb+imagenet latest_only=ON tracker=OFF",
        flush=True,
    )
    print(
        f"STEP5_LIVE camera={camera.camera_id} duration={args.seconds:.1f}s "
        f"threshold={args.person_threshold:.2f} person_class_id={args.person_class_id} "
        f"dedupe={args.iou:.2f}/{args.containment:.2f}/{args.center:.2f}",
        flush=True,
    )

    last_seq = first_seq
    started = time.monotonic()
    deadline = started + float(args.seconds)
    processed = 0
    skipped_total = 0
    unique_counts: list[int] = []
    age_ms_values: list[float] = []
    prep_ms_values: list[float] = []
    trt_ms_values: list[float] = []
    trt_total_ms_values: list[float] = []
    frame_reports: list[dict] = []
    stderr = ""

    # STEP5_TRACKER_INTEGRATED
    from services.camera_v2.temporal_tracker import AnchoredPersonTracker

    tracker = AnchoredPersonTracker(args.width, args.height)
    tracked_counts: list[int] = []
    detector_miss_held = 0
    detector_extra_suppressed = 0

    try:
        while time.monotonic() < deadline:
            snap = mailbox.snapshot(last_seq)
            if snap is None:
                if process.poll() is not None or mailbox.read_error:
                    break
                mailbox.event.wait(0.005)
                mailbox.event.clear()
                continue

            seq, captured_t, raw = snap
            skipped = max(0, seq - last_seq - 1)
            skipped_total += skipped
            last_seq = seq
            age_ms = max(0.0, (time.monotonic() - captured_t) * 1000.0)

            prep_start = time.perf_counter()
            input_array = _preprocess(raw, args.width, args.height)
            prep_ms = (time.perf_counter() - prep_start) * 1000.0
            outputs, trt_ms, trt_total_ms = runner.infer(input_array)
            raw_persons = _person_rows(
                outputs["dets"],
                outputs["labels"],
                float(args.person_threshold),
                int(args.person_class_id),
                args.width,
                args.height,
            )
            kept, rejected = _dedupe(
                raw_persons,
                float(args.iou),
                float(args.containment),
                float(args.center),
            )

            tracker_detections = [
                (
                    tuple(float(v) for v in row["xyxy"]),
                    float(row["confidence"]),
                )
                for row in kept
            ]

            tracker.update(
                camera.camera_id,
                captured_t,
                tracker_detections,
            )

            anchors = tracker.anchors(camera.camera_id, captured_t)
            rendered = tracker.render(camera.camera_id, captured_t)

            tracked_persons = []
            for anchor, box in zip(anchors, rendered):
                x1, y1, x2, y2, conf = box
                tracked_persons.append(
                    {
                        "track_id": int(anchor["track_id"]),
                        "confidence": float(conf),
                        "xyxy": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                    }
                )

            track_ids = [int(a["track_id"]) for a in anchors]
            tracked_count = len(track_ids)
            tracked_counts.append(tracked_count)

            if len(kept) < tracked_count:
                detector_miss_held += 1
            elif len(kept) > tracked_count:
                detector_extra_suppressed += 1

            processed += 1
            unique_counts.append(len(kept))
            age_ms_values.append(age_ms)
            prep_ms_values.append(prep_ms)
            trt_ms_values.append(trt_ms)
            trt_total_ms_values.append(trt_total_ms)
            confs = [float(row["confidence"]) for row in kept]
            min_conf = min(confs) if confs else 0.0
            mean_conf = statistics.mean(confs) if confs else 0.0

            print(
                "STEP5_FRAME "
                f"n={processed:03d} seq={seq} "
                f"raw={len(raw_persons)} det={len(kept)} "
                f"tracked={tracked_count} ids={track_ids} "
                f"dup={len(rejected)} skipped={skipped} "
                f"age_ms={age_ms:.1f} prep_ms={prep_ms:.1f} "
                f"trt_ms={trt_ms:.1f} trt_total_ms={trt_total_ms:.1f} "
                f"min_conf={min_conf:.2f} mean_conf={mean_conf:.2f}",
                flush=True,
            )

            frame_reports.append(
                {
                    "n": processed,
                    "capture_seq": seq,
                    "skipped": skipped,
                    "age_ms": round(age_ms, 3),
                    "prep_ms": round(prep_ms, 3),
                    "trt_ms": round(trt_ms, 3),
                    "trt_total_ms": round(trt_total_ms, 3),
                    "raw_persons": len(raw_persons),
                    "unique_persons": len(kept),
                    "tracked_persons": tracked_count,
                    "track_ids": track_ids,
                    "duplicates": len(rejected),
                    "persons": kept,
                    "tracks": tracked_persons,
                }
            )

            if args.save_every > 0 and (processed == 1 or processed % args.save_every == 0):
                footer = (
                    f"{camera.camera_id} n={processed} "
                    f"det={len(kept)} tracked={tracked_count} "
                    f"ids={track_ids} trt={trt_ms:.0f}ms"
                )
                _annotate(
                    raw,
                    args.width,
                    args.height,
                    tracked_persons,
                    footer,
                ).save(
                    sample_dir / f"sample_{processed:04d}.jpg", quality=92, subsampling=0
                )
    finally:
        mailbox.stop()
        stderr = _terminate(process)
        mailbox.thread.join(timeout=1.0)
        runner.close()

    elapsed = max(1e-6, time.monotonic() - started)
    if processed == 0:
        error = (mailbox.read_error or stderr or "no processed frames").replace(url, redacted)
        raise SystemExit(f"STEP5_FAIL no_results error={error[-500:]}")

    mode_count, mode_frames = Counter(unique_counts).most_common(1)[0]
    stable_ratio = mode_frames / processed
    expected_matches = (
        sum(1 for value in unique_counts if value == args.expected_persons)
        if args.expected_persons >= 0
        else None
    )

    summary = {
        "camera": camera.camera_id,
        "elapsed_sec": elapsed,
        "captured": mailbox.latest_seq,
        "processed": processed,
        "skipped": skipped_total,
        "processed_fps": processed / elapsed,
        "mode_unique": mode_count,
        "stable_frames": mode_frames,
        "stable_ratio": stable_ratio,
        "count_min": min(unique_counts),
        "count_max": max(unique_counts),
        "prep_mean_ms": statistics.mean(prep_ms_values),
        "trt_mean_ms": statistics.mean(trt_ms_values),
        "trt_p95_ms": _percentile(trt_ms_values, 0.95),
        "trt_total_mean_ms": statistics.mean(trt_total_ms_values),
        "age_mean_ms": statistics.mean(age_ms_values),
        "age_p95_ms": _percentile(age_ms_values, 0.95),
        "expected_persons": args.expected_persons,
        "expected_matches": expected_matches,
        "tracker_mode": (
            Counter(tracked_counts).most_common(1)[0][0]
            if tracked_counts else 0
        ),
        "tracker_count_min": min(tracked_counts) if tracked_counts else 0,
        "tracker_count_max": max(tracked_counts) if tracked_counts else 0,
        "detector_miss_held_frames": detector_miss_held,
        "detector_extra_suppressed_frames": detector_extra_suppressed,
    }
    report = {
        "stage": "4.5",
        "backend": "RF-DETR-S TensorRT 8.6 FP32 live latest-frame-only",
        "engine": str(args.engine),
        "threshold": float(args.person_threshold),
        "person_class_id": int(args.person_class_id),
        "dedupe": {
            "iou": float(args.iou),
            "containment": float(args.containment),
            "center": float(args.center),
        },
        "summary": summary,
        "frames": frame_reports,
    }
    report_path = args.output_dir / "live_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    expected_text = "disabled"
    if expected_matches is not None:
        expected_text = f"{args.expected_persons}:{expected_matches}/{processed}({expected_matches/processed*100.0:.1f}%)"

    print(
        "STEP5_RESULT "
        f"camera={camera.camera_id} elapsed={elapsed:.1f}s captured={mailbox.latest_seq} processed={processed} "
        f"skipped={skipped_total} processed_fps={processed/elapsed:.2f} "
        f"mode_unique={mode_count} stable={mode_frames}/{processed}({stable_ratio*100.0:.1f}%) "
        f"count_range={min(unique_counts)}-{max(unique_counts)} "
        f"prep_mean_ms={statistics.mean(prep_ms_values):.1f} "
        f"trt_mean_ms={statistics.mean(trt_ms_values):.1f} trt_p95_ms={_percentile(trt_ms_values,0.95):.1f} "
        f"trt_total_mean_ms={statistics.mean(trt_total_ms_values):.1f} "
        f"age_mean_ms={statistics.mean(age_ms_values):.1f} age_p95_ms={_percentile(age_ms_values,0.95):.1f} "
        f"expected={expected_text}",
        flush=True,
    )
    tracked_mode = (
        Counter(tracked_counts).most_common(1)[0][0]
        if tracked_counts else 0
    )
    tracked_mode_frames = (
        Counter(tracked_counts).most_common(1)[0][1]
        if tracked_counts else 0
    )

    print(
        "STEP5_RESULT "
        f"camera={camera.camera_id} "
        f"processed={processed} "
        f"tracker_mode={tracked_mode} "
        f"tracker_stable={tracked_mode_frames}/{processed}"
        f"({tracked_mode_frames/max(1,processed)*100.0:.1f}%) "
        f"tracker_range={min(tracked_counts)}-{max(tracked_counts)} "
        f"miss_held={detector_miss_held} "
        f"extra_suppressed={detector_extra_suppressed}",
        flush=True,
    )

    print(f"STEP5_JSON={report_path}", flush=True)
    print(f"STEP5_SAMPLES={sample_dir}", flush=True)
    print("STEP5_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
