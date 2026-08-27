#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np
import tensorrt as trt


INPUT_W = 672
INPUT_H = 384
EXPECTED_INPUT = (1, 3, INPUT_H, INPUT_W)
EXPECTED_OUTPUT = (1, 300, 6)


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def cuda_check(code, operation: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{operation}: cuda={int(code)}")


def load_cudart():
    path = ctypes.util.find_library("cudart")
    if not path:
        raise RuntimeError("libcudart not found")
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostAlloc.restype = ctypes.c_int
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaFreeHost.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
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
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cudaEventCreateWithFlags.restype = ctypes.c_int
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int
    lib.cudaEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cudaEventElapsedTime.restype = ctypes.c_int
    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int
    return lib


class Runner:
    """TensorRT 8.6 batch-1 worker with no hot-path CUDA allocations.

    The only synchronization is on this worker's stream after D2H is enqueued and
    immediately before the CPU consumes the output. There is no device-wide sync.
    """

    def __init__(self, engine_path: Path) -> None:
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
        if not engine_path.is_file():
            raise FileNotFoundError(f"engine missing: {engine_path}")

        self.cuda = load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"engine deserialize failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution context creation failed")

        inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"expected one input/output, got {inputs}/{outputs}")
        self.input_index = inputs[0]
        self.output_index = outputs[0]
        self.input_shape = tuple(int(v) for v in self.context.get_binding_shape(self.input_index))
        self.output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_index))
        if self.input_shape != EXPECTED_INPUT or self.output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(f"unexpected bindings {self.input_shape}/{self.output_shape}")

        input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index)))
        output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_index)))
        if input_dtype != np.float32 or output_dtype != np.float32:
            raise RuntimeError(f"expected float32 I/O, got {input_dtype}/{output_dtype}")

        self.input_host = ctypes.c_void_p()
        self.output_host = ctypes.c_void_p()
        self.input_device = ctypes.c_void_p()
        self.output_device = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self.events = [ctypes.c_void_p() for _ in range(4)]
        self._shm = None
        self._shm_name = ""
        self._shm_frame = None

        input_count = int(np.prod(self.input_shape))
        output_count = int(np.prod(self.output_shape))
        input_bytes = input_count * np.dtype(np.float32).itemsize
        output_bytes = output_count * np.dtype(np.float32).itemsize
        cuda_check(self.cuda.cudaHostAlloc(ctypes.byref(self.input_host), input_bytes, 0), "cudaHostAlloc input")
        cuda_check(self.cuda.cudaHostAlloc(ctypes.byref(self.output_host), output_bytes, 0), "cudaHostAlloc output")
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.input_device), input_bytes), "cudaMalloc input")
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.output_device), output_bytes), "cudaMalloc output")

        input_buffer_type = ctypes.c_float * input_count
        output_buffer_type = ctypes.c_float * output_count
        self._input_buffer = input_buffer_type.from_address(int(self.input_host.value))
        self._output_buffer = output_buffer_type.from_address(int(self.output_host.value))
        self.x = np.ctypeslib.as_array(self._input_buffer).reshape(self.input_shape)
        self.y = np.ctypeslib.as_array(self._output_buffer).reshape(self.output_shape)

        least = ctypes.c_int()
        greatest = ctypes.c_int()
        cuda_check(
            self.cuda.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)),
            "cudaDeviceGetStreamPriorityRange",
        )
        self.priority_least = int(least.value)
        self.priority_greatest = int(greatest.value)
        cuda_check(
            self.cuda.cudaStreamCreateWithPriority(
                ctypes.byref(self.stream),
                1,  # cudaStreamNonBlocking
                self.priority_least,
            ),
            "cudaStreamCreateWithPriority",
        )
        for event in self.events:
            cuda_check(self.cuda.cudaEventCreateWithFlags(ctypes.byref(event), 0), "cudaEventCreate")

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.input_device.value)
        self.bindings[self.output_index] = int(self.output_device.value)
        self._input_bytes = input_bytes
        self._output_bytes = output_bytes

    def _attach(self, name: str) -> np.ndarray:
        if self._shm is not None and self._shm_name == name and self._shm_frame is not None:
            return self._shm_frame
        if self._shm is not None:
            self._shm.close()
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        required = INPUT_H * INPUT_W * 3
        if shm.size < required:
            shm.close()
            raise RuntimeError(f"shared memory too small: {shm.size} < {required}")
        self._shm = shm
        self._shm_name = name
        self._shm_frame = np.ndarray((INPUT_H, INPUT_W, 3), dtype=np.uint8, buffer=shm.buf)
        return self._shm_frame

    def _elapsed(self, start, end) -> float:
        value = ctypes.c_float()
        cuda_check(self.cuda.cudaEventElapsedTime(ctypes.byref(value), start, end), "cudaEventElapsedTime")
        return float(value.value)

    def infer(self, frame: np.ndarray, conf: float, max_det: int) -> tuple[list[list[float]], dict]:
        if frame.shape != (INPUT_H, INPUT_W, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"unexpected input frame {frame.shape}/{frame.dtype}")
        conf = min(1.0, max(0.0, float(conf)))
        max_det = max(1, min(300, int(max_det)))
        total_started = time.perf_counter()

        prep_started = time.perf_counter()
        scale = 1.0 / 255.0
        np.multiply(frame[:, :, 2], scale, out=self.x[0, 0], casting="unsafe")
        np.multiply(frame[:, :, 1], scale, out=self.x[0, 1], casting="unsafe")
        np.multiply(frame[:, :, 0], scale, out=self.x[0, 2], casting="unsafe")
        preprocess_ms = (time.perf_counter() - prep_started) * 1000.0

        h2d_start, h2d_end, infer_end, d2h_end = self.events
        enqueue_started = time.perf_counter()
        cuda_check(self.cuda.cudaEventRecord(h2d_start, self.stream), "event H2D start")
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                self.input_device,
                self.input_host,
                self._input_bytes,
                1,
                self.stream,
            ),
            "H2D async",
        )
        cuda_check(self.cuda.cudaEventRecord(h2d_end, self.stream), "event H2D end")
        if not self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self.stream.value),
        ):
            raise RuntimeError("execute_async_v2=false")
        cuda_check(self.cuda.cudaEventRecord(infer_end, self.stream), "event inference end")
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                self.output_host,
                self.output_device,
                self._output_bytes,
                2,
                self.stream,
            ),
            "D2H async",
        )
        cuda_check(self.cuda.cudaEventRecord(d2h_end, self.stream), "event D2H end")
        enqueue_ms = (time.perf_counter() - enqueue_started) * 1000.0

        sync_started = time.perf_counter()
        cuda_check(self.cuda.cudaStreamSynchronize(self.stream), "detector stream sync")
        sync_wait_ms = (time.perf_counter() - sync_started) * 1000.0
        h2d_ms = self._elapsed(h2d_start, h2d_end)
        inference_ms = self._elapsed(h2d_end, infer_end)
        d2h_ms = self._elapsed(infer_end, d2h_end)

        post_started = time.perf_counter()
        rows: list[list[float]] = []
        for raw in self.y[0]:
            x1, y1, x2, y2, score, cls = (float(v) for v in raw)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, cls)):
                continue
            if score < conf or int(round(cls)) != 0:
                continue
            x1 = min(671.0, max(0.0, x1))
            x2 = min(671.0, max(0.0, x2))
            y1 = min(383.0, max(0.0, y1))
            y2 = min(383.0, max(0.0, y2))
            if x2 > x1 and y2 > y1:
                rows.append([x1, y1, x2, y2, score])
        rows.sort(key=lambda item: item[4], reverse=True)
        if len(rows) > max_det:
            del rows[max_det:]
        postprocess_ms = (time.perf_counter() - post_started) * 1000.0

        return rows, {
            "preprocess_ms": preprocess_ms,
            "enqueue_ms": enqueue_ms,
            "sync_wait_ms": sync_wait_ms,
            "h2d_ms": h2d_ms,
            "inference_ms": inference_ms,
            "d2h_ms": d2h_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": (time.perf_counter() - total_started) * 1000.0,
        }

    def infer_from_shm(self, name: str, conf: float, max_det: int):
        return self.infer(self._attach(name), conf, max_det)

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._shm_frame = None
        if self.stream.value:
            try:
                self.cuda.cudaStreamSynchronize(self.stream)
            except Exception:
                pass
        for event in self.events:
            if event.value:
                self.cuda.cudaEventDestroy(event)
        if self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
        if self.input_device.value:
            self.cuda.cudaFree(self.input_device)
        if self.output_device.value:
            self.cuda.cudaFree(self.output_device)
        if self.input_host.value:
            self.cuda.cudaFreeHost(self.input_host)
        if self.output_host.value:
            self.cuda.cudaFreeHost(self.output_host)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    runner = Runner(Path(args.engine))
    emit(
        {
            "type": "ready",
            "tensorrt": trt.__version__,
            "engine": str(Path(args.engine).resolve()),
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "transport": "shm-preallocated-pinned-async",
            "stream": "nonblocking-low-priority",
            "priority_least": runner.priority_least,
            "priority_greatest": runner.priority_greatest,
        }
    )
    try:
        for line in sys.stdin:
            request: dict = {}
            try:
                request = json.loads(line)
                if request.get("cmd") == "stop":
                    break
                boxes, stages = runner.infer_from_shm(
                    str(request["shm_name"]),
                    float(request.get("conf", 0.18)),
                    int(request.get("max_det", 20)),
                )
                emit({"id": request.get("id"), "ok": True, "boxes": boxes, "stages": stages})
            except Exception as exc:
                emit(
                    {
                        "id": request.get("id"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
