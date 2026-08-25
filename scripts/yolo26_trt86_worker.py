#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorrt as trt


def emit(obj):
    print(json.dumps(obj, separators=(",", ":")), flush=True)


def cuda_check(code, name):
    if int(code) != 0:
        raise RuntimeError(f"{name}: cuda={code}")


def load_cudart():
    path = ctypes.util.find_library("cudart")
    if not path:
        raise RuntimeError("libcudart not found")

    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
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


class Runner:
    def __init__(self, engine_path: Path):
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")

        self.cuda = load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("engine deserialize failed")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("execution context failed")

        self.input_index = None
        self.output_index = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_index = i
            else:
                self.output_index = i
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("input/output binding missing")

        self.input_shape = tuple(
            int(x) for x in self.context.get_binding_shape(self.input_index)
        )
        self.output_shape = tuple(
            int(x) for x in self.context.get_binding_shape(self.output_index)
        )
        if self.input_shape != (1, 3, 384, 672):
            raise RuntimeError(f"unexpected input shape={self.input_shape}")
        if self.output_shape != (1, 300, 6):
            raise RuntimeError(f"unexpected output shape={self.output_shape}")

        self.x = np.empty(self.input_shape, dtype=np.float32)
        self.y = np.empty(self.output_shape, dtype=np.float32)
        self.in_dev = ctypes.c_void_p()
        self.out_dev = ctypes.c_void_p()
        cuda_check(
            self.cuda.cudaMalloc(ctypes.byref(self.in_dev), self.x.nbytes),
            "cudaMalloc input",
        )
        cuda_check(
            self.cuda.cudaMalloc(ctypes.byref(self.out_dev), self.y.nbytes),
            "cudaMalloc output",
        )
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.in_dev.value)
        self.bindings[self.output_index] = int(self.out_dev.value)

    def preprocess_rgb(self, rgb: np.ndarray):
        if rgb.shape[:2] != (384, 672):
            image = Image.fromarray(rgb, mode="RGB")
            image = image.resize((672, 384), Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.uint8)
        arr = rgb.astype(np.float32, copy=False)
        arr *= (1.0 / 255.0)
        self.x[0] = arr.transpose(2, 0, 1)

    def preprocess_jpeg(self, jpeg_bytes: bytes):
        rgb = np.asarray(
            Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"),
            dtype=np.uint8,
        )
        self.preprocess_rgb(rgb)

    def preprocess_raw_bgr(self, raw: bytes, width: int, height: int):
        expected = int(width) * int(height) * 3
        if len(raw) != expected:
            raise RuntimeError(
                f"raw BGR size mismatch got={len(raw)} expected={expected}"
            )
        bgr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
        # Exported YOLO engine was benchmarked with RGB preprocessing. Camera probe
        # delivers BGR from BGRx, so reverse channels without JPEG encode/decode.
        rgb = bgr[..., ::-1]
        self.preprocess_rgb(rgb)

    def infer_preprocessed(self, conf: float):
        started_total = time.perf_counter()
        cuda_check(
            self.cuda.cudaMemcpy(
                self.in_dev,
                ctypes.c_void_p(self.x.ctypes.data),
                self.x.nbytes,
                1,
            ),
            "H2D",
        )
        started_trt = time.perf_counter()
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("execute_v2=false")
        cuda_check(self.cuda.cudaDeviceSynchronize(), "infer sync")
        trt_ms = (time.perf_counter() - started_trt) * 1000.0
        cuda_check(
            self.cuda.cudaMemcpy(
                ctypes.c_void_p(self.y.ctypes.data),
                self.out_dev,
                self.y.nbytes,
                2,
            ),
            "D2H",
        )

        rows = []
        for row in self.y[0]:
            x1, y1, x2, y2, score, cls = (float(v) for v in row)
            if score < conf:
                continue
            if int(round(cls)) != 0:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2, y2, score])

        total_ms = (time.perf_counter() - started_total) * 1000.0
        return rows, trt_ms, total_ms

    def infer_jpeg(self, jpeg_bytes: bytes, conf: float):
        started = time.perf_counter()
        self.preprocess_jpeg(jpeg_bytes)
        prep_ms = (time.perf_counter() - started) * 1000.0
        rows, trt_ms, core_ms = self.infer_preprocessed(conf)
        return rows, trt_ms, prep_ms + core_ms, prep_ms

    def infer_raw_bgr(self, raw: bytes, width: int, height: int, conf: float):
        started = time.perf_counter()
        self.preprocess_raw_bgr(raw, width, height)
        prep_ms = (time.perf_counter() - started) * 1000.0
        rows, trt_ms, core_ms = self.infer_preprocessed(conf)
        return rows, trt_ms, prep_ms + core_ms, prep_ms

    def close(self):
        if self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
        if self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()

    runner = Runner(Path(args.engine))
    emit({
        "type": "ready",
        "tensorrt": trt.__version__,
        "engine": str(Path(args.engine).resolve()),
        "input_shape": runner.input_shape,
        "output_shape": runner.output_shape,
    })

    try:
        for line in sys.stdin:
            req = None
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    break

                request_id = req.get("id")
                conf = float(req.get("conf", 0.05))
                if "raw_bgr_b64" in req:
                    raw = base64.b64decode(req["raw_bgr_b64"])
                    boxes, trt_ms, total_ms, prep_ms = runner.infer_raw_bgr(
                        raw,
                        int(req.get("width", 672)),
                        int(req.get("height", 384)),
                        conf,
                    )
                    input_mode = "raw-bgr"
                else:
                    jpeg = base64.b64decode(req["jpeg_b64"])
                    boxes, trt_ms, total_ms, prep_ms = runner.infer_jpeg(jpeg, conf)
                    input_mode = "jpeg"

                emit({
                    "id": request_id,
                    "ok": True,
                    "boxes": boxes,
                    "trt_ms": trt_ms,
                    "prep_ms": prep_ms,
                    "total_ms": total_ms,
                    "input_mode": input_mode,
                })
            except Exception as exc:
                emit({
                    "id": req.get("id") if req else None,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        runner.close()


if __name__ == "__main__":
    main()
