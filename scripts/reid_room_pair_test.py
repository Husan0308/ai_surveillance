#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

import rfdetr_step4_5_trt_live as det
import rfdetr_step4_7_trt_live_fragment_filter as det_filter

from services.ml_service.app.config import load_settings

DET_ENGINE = ROOT / "artifacts/rfdetr_step4/trt86_fp32/rfdetr-small-800x448-fp32.engine"
REID_ENGINE = ROOT / "artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine"

OUT = ROOT / "artifacts/reid/room1_pair_test"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH = 800
HEIGHT = 448

CAMERAS = ["CAM-01", "CAM-04"]

OFFSETS = np.asarray(
    [123.6750, 116.2800, 103.5300],
    dtype=np.float32,
).reshape(1, 1, 3)

SCALE = np.float32(0.01735207)


def capture(camera):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg missing")

    url = det._camera_rtsp_url(camera)

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-an",
        "-frames:v", "6",
        "-vf", f"fps=5,scale={WIDTH}:{HEIGHT}:flags=bilinear",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )

    frame_bytes = WIDTH * HEIGHT * 3
    expected = frame_bytes * 6

    if len(p.stdout) < expected:
        raise RuntimeError(
            f"{camera.camera_id}: capture failed "
            f"bytes={len(p.stdout)}/{expected} "
            f"error={p.stderr.decode(errors='replace')[-300:]}"
        )

    # Ignore RTSP startup frames. Use the newest clean frame.
    raw = p.stdout[-frame_bytes:]

    return np.frombuffer(
        raw,
        dtype=np.uint8,
    ).reshape(HEIGHT, WIDTH, 3).copy()


def detect_people(runner, frame):
    inp = det._preprocess(
        frame.tobytes(),
        WIDTH,
        HEIGHT,
    )

    outputs, trt_ms, _ = runner.infer(inp)

    raw = det._person_rows(
        outputs["dets"],
        outputs["labels"],
        0.32,
        1,
        WIDTH,
        HEIGHT,
    )

    kept, rejected = det_filter._filtered_dedupe(
        raw,
        0.62,
        0.90,
        0.35,
    )

    kept.sort(
        key=lambda x: float(x["confidence"]),
        reverse=True,
    )

    # GTX 1050 Ti ReID engine max batch=8.
    # Max 4 people per camera in this first test.
    return kept[:4], trt_ms, rejected


def crop_person(frame, row):
    x1, y1, x2, y2 = row["xyxy"]

    # Small safety margin around the detector bbox.
    w = x2 - x1
    h = y2 - y1

    x1 -= w * 0.04
    x2 += w * 0.04
    y1 -= h * 0.02
    y2 += h * 0.03

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(WIDTH, int(round(x2)))
    y2 = min(HEIGHT, int(round(y2)))

    return frame[y1:y2, x1:x2]


def reid_preprocess(crop):
    if crop.size == 0:
        raise RuntimeError("empty crop")

    image = Image.fromarray(crop, mode="RGB")

    target_w = 128
    target_h = 256

    src_w, src_h = image.size

    scale = min(
        target_w / max(1, src_w),
        target_h / max(1, src_h),
    )

    new_w = max(1, min(target_w, round(src_w * scale)))
    new_h = max(1, min(target_h, round(src_h * scale)))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    # keepAspc=1; black padding for unused region.
    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8,
    )

    canvas[:new_h, :new_w] = np.asarray(resized)

    x = canvas.astype(np.float32)
    x = (x - OFFSETS) * SCALE

    x = x.transpose(2, 0, 1)

    return np.ascontiguousarray(x, dtype=np.float32)


def load_cudart():
    path = ctypes.util.find_library("cudart")
    if not path:
        raise RuntimeError("libcudart missing")

    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    lib.cudaMalloc.restype = ctypes.c_int

    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int

    lib.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.cudaMemcpy.restype = ctypes.c_int

    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int

    return lib


def cuda_check(code, name):
    if int(code) != 0:
        raise RuntimeError(
            f"{name}: cuda error={code}"
        )


class ReIDRunner:
    def __init__(self):
        import tensorrt as trt

        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(
                f"wrong TensorRT={trt.__version__}"
            )

        self.trt = trt
        self.cudart = load_cudart()

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")

        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            REID_ENGINE.read_bytes()
        )

        if self.engine is None:
            raise RuntimeError("ReID engine deserialize failed")

        self.context = self.engine.create_execution_context()

        self.input_index = None
        self.output_index = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)

            if self.engine.binding_is_input(i):
                self.input_index = i
            elif name == "fc_pred":
                self.output_index = i

        if self.input_index is None or self.output_index is None:
            raise RuntimeError("ReID bindings missing")

    def infer(self, batch):
        batch = np.ascontiguousarray(
            batch,
            dtype=np.float32,
        )

        n = len(batch)

        if not 1 <= n <= 8:
            raise RuntimeError(f"bad batch={n}")

        self.context.set_binding_shape(
            self.input_index,
            (n, 3, 256, 128),
        )

        out_shape = tuple(
            self.context.get_binding_shape(
                self.output_index
            )
        )

        if out_shape != (n, 256):
            raise RuntimeError(
                f"bad output={out_shape}"
            )

        out = np.empty(
            out_shape,
            dtype=np.float32,
        )

        inp_dev = ctypes.c_void_p()
        out_dev = ctypes.c_void_p()

        cuda_check(
            self.cudart.cudaMalloc(
                ctypes.byref(inp_dev),
                batch.nbytes,
            ),
            "malloc input",
        )

        cuda_check(
            self.cudart.cudaMalloc(
                ctypes.byref(out_dev),
                out.nbytes,
            ),
            "malloc output",
        )

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_index] = int(inp_dev.value)
        bindings[self.output_index] = int(out_dev.value)

        try:
            cuda_check(
                self.cudart.cudaMemcpy(
                    inp_dev,
                    ctypes.c_void_p(batch.ctypes.data),
                    batch.nbytes,
                    1,
                ),
                "H2D",
            )

            if not self.context.execute_v2(bindings):
                raise RuntimeError("ReID execute failed")

            cuda_check(
                self.cudart.cudaDeviceSynchronize(),
                "sync",
            )

            cuda_check(
                self.cudart.cudaMemcpy(
                    ctypes.c_void_p(out.ctypes.data),
                    out_dev,
                    out.nbytes,
                    2,
                ),
                "D2H",
            )

        finally:
            self.cudart.cudaFree(inp_dev)
            self.cudart.cudaFree(out_dev)

        # Required for cosine matching.
        norms = np.linalg.norm(
            out,
            axis=1,
            keepdims=True,
        )

        out = out / np.maximum(norms, 1e-12)

        return out


settings = load_settings()

camera_map = {
    c.camera_id: c
    for c in settings.cameras
}

for cid in CAMERAS:
    if cid not in camera_map:
        raise SystemExit(f"missing camera {cid}")

detector = det.TrtRunner(DET_ENGINE)
reid = ReIDRunner()

records = []

try:
    for cid in CAMERAS:
        print(f"\nCAPTURE {cid}", flush=True)

        frame = capture(camera_map[cid])

        Image.fromarray(frame).save(
            OUT / f"{cid}_frame.jpg",
            quality=95,
        )

        people, det_ms, rejected = detect_people(
            detector,
            frame,
        )

        print(
            f"DETECT {cid} people={len(people)} "
            f"rejected={len(rejected)} "
            f"trt_ms={det_ms:.1f}"
        )

        for i, row in enumerate(people, 1):
            crop = crop_person(frame, row)

            crop_path = OUT / f"{cid}_person_{i}.jpg"

            Image.fromarray(crop).save(
                crop_path,
                quality=95,
            )

            records.append(
                {
                    "camera": cid,
                    "index": i,
                    "confidence": float(row["confidence"]),
                    "crop": crop,
                    "path": crop_path,
                }
            )

    if not records:
        raise RuntimeError("no people detected")

    inputs = np.stack([
        reid_preprocess(r["crop"])
        for r in records
    ])

    embeddings = reid.infer(inputs)

    for r, emb in zip(records, embeddings):
        r["embedding"] = emb

    a = [
        r for r in records
        if r["camera"] == "CAM-01"
    ]

    b = [
        r for r in records
        if r["camera"] == "CAM-04"
    ]

    if not a or not b:
        raise RuntimeError(
            f"need people in both cameras "
            f"CAM01={len(a)} CAM04={len(b)}"
        )

    print("\n=== CROPS ===")

    for r in records:
        print(
            f"{r['camera']} P{r['index']} "
            f"det_conf={r['confidence']:.3f} "
            f"file={r['path']}"
        )

    print("\n=== CROSS-CAMERA COSINE ===")

    matrix = np.zeros(
        (len(a), len(b)),
        dtype=np.float32,
    )

    for i, ra in enumerate(a):
        for j, rb in enumerate(b):
            matrix[i, j] = float(
                np.dot(
                    ra["embedding"],
                    rb["embedding"],
                )
            )

    header = "           " + " ".join(
        f"CAM04-P{r['index']:>2}"
        for r in b
    )

    print(header)

    for i, ra in enumerate(a):
        values = " ".join(
            f"{matrix[i,j]:8.4f}"
            for j in range(len(b))
        )

        print(
            f"CAM01-P{ra['index']:<2} {values}"
        )

    print("\n=== BEST MATCHES ===")

    for i, ra in enumerate(a):
        j = int(np.argmax(matrix[i]))

        print(
            f"CAM01-P{ra['index']} -> "
            f"CAM04-P{b[j]['index']} "
            f"score={matrix[i,j]:.4f}"
        )

    print("\nREID_ROOM_PAIR_TEST=PASS")

finally:
    detector.close()
