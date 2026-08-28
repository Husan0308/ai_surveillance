#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np
import tensorrt as trt

MAX_BATCH = 8
INPUT_SHAPE = (MAX_BATCH, 3, 256, 128)
OUTPUT_SHAPE = (MAX_BATCH, 256)


def emit(payload: dict) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def cuda_check(code: int, name: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{name}: cuda={int(code)}")


def attach_nonowning_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach to parent-owned POSIX SHM without letting this subprocess unlink it.

    Python 3.13 added SharedMemory(track=False) for exactly this subprocess case.
    The TRT86 worker is pinned to Python 3.10, so unregister the attached handle
    from this worker's independent resource_tracker while the parent remains the
    sole owner responsible for unlink().
    """
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except BaseException:
        shm.close()
        raise
    return shm


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
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    lib.cudaDeviceGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.cudaDeviceGetStreamPriorityRange.restype = ctypes.c_int
    lib.cudaStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    lib.cudaStreamCreateWithPriority.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaEventCreate.restype = ctypes.c_int
    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int
    lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventElapsedTime.restype = ctypes.c_int
    return lib


class Runner:
    def __init__(self, engine_path: Path, input_shm_name: str, output_shm_name: str) -> None:
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
        self.cudart = load_cudart()
        self.input_shm = attach_nonowning_shared_memory(input_shm_name)
        self.output_shm = attach_nonowning_shared_memory(output_shm_name)
        self.input = np.ndarray(INPUT_SHAPE, dtype=np.float32, buffer=self.input_shm.buf)
        self.output = np.ndarray(OUTPUT_SHAPE, dtype=np.float32, buffer=self.output_shm.buf)
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
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if self.engine.binding_is_input(index):
                self.input_index = index
            elif name == "fc_pred":
                self.output_index = index
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("input/fc_pred binding missing")
        self.max_input_bytes = int(np.prod(INPUT_SHAPE) * 4)
        self.max_output_bytes = int(np.prod(OUTPUT_SHAPE) * 4)
        self.host_input = ctypes.c_void_p()
        self.host_output = ctypes.c_void_p()
        self.dev_input = ctypes.c_void_p()
        self.dev_output = ctypes.c_void_p()
        cuda_check(self.cudart.cudaHostAlloc(ctypes.byref(self.host_input), self.max_input_bytes, 0), "cudaHostAlloc input")
        cuda_check(self.cudart.cudaHostAlloc(ctypes.byref(self.host_output), self.max_output_bytes, 0), "cudaHostAlloc output")
        cuda_check(self.cudart.cudaMalloc(ctypes.byref(self.dev_input), self.max_input_bytes), "cudaMalloc input")
        cuda_check(self.cudart.cudaMalloc(ctypes.byref(self.dev_output), self.max_output_bytes), "cudaMalloc output")
        least = ctypes.c_int()
        greatest = ctypes.c_int()
        cuda_check(self.cudart.cudaDeviceGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)), "cudaDeviceGetStreamPriorityRange")
        self.priority_least = int(least.value)
        self.priority_greatest = int(greatest.value)
        self.stream = ctypes.c_void_p()
        cuda_check(self.cudart.cudaStreamCreateWithPriority(ctypes.byref(self.stream), 1, self.priority_least), "cudaStreamCreateWithPriority")
        self.events = [ctypes.c_void_p() for _ in range(4)]
        for event in self.events:
            cuda_check(self.cudart.cudaEventCreate(ctypes.byref(event)), "cudaEventCreate")
        input_type = ctypes.c_float * (self.max_input_bytes // 4)
        output_type = ctypes.c_float * (self.max_output_bytes // 4)
        self.pinned_input = np.ctypeslib.as_array(input_type.from_address(int(self.host_input.value))).reshape(INPUT_SHAPE)
        self.pinned_output = np.ctypeslib.as_array(output_type.from_address(int(self.host_output.value))).reshape(OUTPUT_SHAPE)

    def _elapsed(self, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
        value = ctypes.c_float()
        cuda_check(self.cudart.cudaEventElapsedTime(ctypes.byref(value), start, end), "cudaEventElapsedTime")
        return float(value.value)

    def infer(self, n: int) -> dict:
        if not 1 <= n <= MAX_BATCH:
            raise ValueError(f"batch must be 1..{MAX_BATCH}, got {n}")
        started = time.perf_counter()
        np.copyto(self.pinned_input[:n], self.input[:n], casting="no")
        host_copy_ms = (time.perf_counter() - started) * 1000.0
        self.context.set_binding_shape(self.input_index, (n, 3, 256, 128))
        out_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_index))
        if out_shape != (n, 256):
            raise RuntimeError(f"unexpected output shape={out_shape}")
        in_bytes = n * 3 * 256 * 128 * 4
        out_bytes = n * 256 * 4
        bindings = [0] * self.engine.num_bindings
        bindings[self.input_index] = int(self.dev_input.value)
        bindings[self.output_index] = int(self.dev_output.value)
        e0, e1, e2, e3 = self.events
        cuda_check(self.cudart.cudaEventRecord(e0, self.stream), "event h2d start")
        cuda_check(self.cudart.cudaMemcpyAsync(self.dev_input, self.host_input, in_bytes, 1, self.stream), "H2D async")
        cuda_check(self.cudart.cudaEventRecord(e1, self.stream), "event infer start")
        if not self.context.execute_async_v2(bindings=bindings, stream_handle=int(self.stream.value)):
            raise RuntimeError("execute_async_v2=false")
        cuda_check(self.cudart.cudaEventRecord(e2, self.stream), "event d2h start")
        cuda_check(self.cudart.cudaMemcpyAsync(self.host_output, self.dev_output, out_bytes, 2, self.stream), "D2H async")
        cuda_check(self.cudart.cudaEventRecord(e3, self.stream), "event done")
        cuda_check(self.cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        values = self.pinned_output[:n].copy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values /= np.maximum(norms, 1e-12)
        np.copyto(self.output[:n], values, casting="no")
        total_ms = (time.perf_counter() - started) * 1000.0
        return {"h2d_ms": round(self._elapsed(e0, e1), 4), "inference_ms": round(self._elapsed(e1, e2), 4), "d2h_ms": round(self._elapsed(e2, e3), 4), "host_copy_ms": round(host_copy_ms, 4), "total_ms": round(total_ms, 4)}

    def close(self) -> None:
        for event in getattr(self, "events", []):
            if event.value:
                self.cudart.cudaEventDestroy(event)
        if getattr(self, "stream", None) is not None and self.stream.value:
            self.cudart.cudaStreamDestroy(self.stream)
        for pointer in (getattr(self, "dev_input", None), getattr(self, "dev_output", None)):
            if pointer is not None and pointer.value:
                self.cudart.cudaFree(pointer)
        for pointer in (getattr(self, "host_input", None), getattr(self, "host_output", None)):
            if pointer is not None and pointer.value:
                self.cudart.cudaFreeHost(pointer)
        for shm in (getattr(self, "input_shm", None), getattr(self, "output_shm", None)):
            if shm is not None:
                shm.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--input-shm", required=True)
    parser.add_argument("--output-shm", required=True)
    args = parser.parse_args()
    runner = Runner(args.engine.resolve(), args.input_shm, args.output_shm)
    try:
        emit({"type": "ready", "tensorrt": trt.__version__, "max_batch": MAX_BATCH, "embedding_size": 256, "input_shape": [3, 256, 128], "priority_least": runner.priority_least, "priority_greatest": runner.priority_greatest, "transport": "shm-f32", "jpeg": 0, "base64": 0, "preallocated": 1, "execute_async_v2": 1, "device_sync": 0})
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                if request.get("cmd") == "stop":
                    emit({"type": "stopped"})
                    return 0
                request_id = int(request["id"])
                stages = runner.infer(int(request["n"]))
                emit({"id": request_id, "ok": True, "stages": stages})
            except Exception as exc:
                emit({"id": request.get("id") if isinstance(request, dict) else None, "ok": False, "error": f"{type(exc).__name__}:{exc}"})
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
