#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import io
import json
from multiprocessing import shared_memory
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

    lib.cudaDeviceGetStreamPriorityRange.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.cudaDeviceGetStreamPriorityRange.restype = ctypes.c_int
    lib.cudaStreamCreateWithPriority.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    lib.cudaStreamCreateWithPriority.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
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

        # Run detector kernels on a dedicated highest-priority non-blocking CUDA
        # stream. The six-camera EGL/DeepStream wall uses other/default streams;
        # this gives pending detector kernels preference without reducing display
        # resolution. CUDA priorities are hints, not hard preemption guarantees.
        least_priority = ctypes.c_int()
        greatest_priority = ctypes.c_int()
        cuda_check(
            self.cuda.cudaDeviceGetStreamPriorityRange(
                ctypes.byref(least_priority),
                ctypes.byref(greatest_priority),
            ),
            "cudaDeviceGetStreamPriorityRange",
        )
        self.stream = ctypes.c_void_p()
        CUDA_STREAM_NON_BLOCKING = 1
        cuda_check(
            self.cuda.cudaStreamCreateWithPriority(
                ctypes.byref(self.stream),
                CUDA_STREAM_NON_BLOCKING,
                greatest_priority.value,
            ),
            "cudaStreamCreateWithPriority",
        )
        self.stream_priority = int(greatest_priority.value)
        self.stream_priority_range = (
            int(greatest_priority.value),
            int(least_priority.value),
        )

        # Reuse all host/device buffers for the life of the worker. The fast raw
        # path writes BGR channels directly into this CHW float32 tensor, avoiding
        # rgb.astype(...), transpose temporaries and one full-frame allocation.
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
        self._shm_cache: dict[str, shared_memory.SharedMemory] = {}

    def preprocess_rgb(self, rgb: np.ndarray):
        if rgb.shape[:2] != (384, 672):
            image = Image.fromarray(rgb, mode="RGB")
            image = image.resize((672, 384), Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.uint8)
        scale = np.float32(1.0 / 255.0)
        np.multiply(rgb[..., 0], scale, out=self.x[0, 0], casting="unsafe")
        np.multiply(rgb[..., 1], scale, out=self.x[0, 1], casting="unsafe")
        np.multiply(rgb[..., 2], scale, out=self.x[0, 2], casting="unsafe")

    def preprocess_bgr_array(self, bgr: np.ndarray):
        if bgr.shape != (384, 672, 3):
            raise RuntimeError(f"unexpected BGR array shape={bgr.shape}")
        scale = np.float32(1.0 / 255.0)
        # BGR camera frame -> RGB model tensor, directly into preallocated CHW.
        np.multiply(bgr[..., 2], scale, out=self.x[0, 0], casting="unsafe")
        np.multiply(bgr[..., 1], scale, out=self.x[0, 1], casting="unsafe")
        np.multiply(bgr[..., 0], scale, out=self.x[0, 2], casting="unsafe")

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
        self.preprocess_bgr_array(bgr)

    def shared_bgr(self, name: str, width: int, height: int) -> np.ndarray:
        if (height, width) != (384, 672):
            raise RuntimeError(f"unexpected shared frame shape={width}x{height}")
        shm = self._shm_cache.get(name)
        if shm is None:
            shm = shared_memory.SharedMemory(name=name)
            self._shm_cache[name] = shm
        expected = height * width * 3
        if shm.size < expected:
            raise RuntimeError(f"shared memory too small got={shm.size} expected={expected}")
        return np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

    def infer_preprocessed(self, conf: float):
        started_total = time.perf_counter()
        # H2D/D2H are tiny compared with the live TRT delay and stream priority
        # does not affect memcpy scheduling, so keep the copies simple/synchronous.
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
        if not self.context.execute_async_v2(
            self.bindings,
            int(self.stream.value),
        ):
            raise RuntimeError("execute_async_v2=false")
        cuda_check(
            self.cuda.cudaStreamSynchronize(self.stream),
            "cudaStreamSynchronize detector",
        )
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

    def infer_shared_bgr(self, name: str, width: int, height: int, conf: float):
        started = time.perf_counter()
        bgr = self.shared_bgr(name, width, height)
        self.preprocess_bgr_array(bgr)
        prep_ms = (time.perf_counter() - started) * 1000.0
        rows, trt_ms, core_ms = self.infer_preprocessed(conf)
        return rows, trt_ms, prep_ms + core_ms, prep_ms

    def close(self):
        for shm in self._shm_cache.values():
            try:
                shm.close()
            except Exception:
                pass
        self._shm_cache.clear()
        if self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
        if self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)
        if self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)


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
        "stream_priority": runner.stream_priority,
        "stream_priority_range": runner.stream_priority_range,
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
                if "shm_name" in req:
                    boxes, trt_ms, total_ms, prep_ms = runner.infer_shared_bgr(
                        str(req["shm_name"]),
                        int(req.get("width", 672)),
                        int(req.get("height", 384)),
                        conf,
                    )
                    input_mode = "shared-bgr"
                elif "raw_bgr_b64" in req:
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
