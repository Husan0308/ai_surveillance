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


INPUT_W = 672
INPUT_H = 384
BATCH = 6
EXPECTED_INPUT = (BATCH, 3, INPUT_H, INPUT_W)
EXPECTED_OUTPUT = (BATCH, 300, 6)


def emit(obj) -> None:
    print(json.dumps(obj, separators=(",", ":")), flush=True)


def cuda_check(code, name: str) -> None:
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
    lib.cudaHostAlloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.cudaHostAlloc.restype = ctypes.c_int
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaFreeHost.restype = ctypes.c_int
    lib.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    return lib


def _pinned_float32(cuda, shape: tuple[int, ...]):
    count = int(np.prod(shape))
    nbytes = count * np.dtype(np.float32).itemsize
    ptr = ctypes.c_void_p()
    # cudaHostAllocDefault = 0. Pinned buffers are intentional: async H2D/D2H from
    # pageable NumPy memory can stage/synchronize and was part of the old latency path.
    cuda_check(cuda.cudaHostAlloc(ctypes.byref(ptr), nbytes, 0), "cudaHostAlloc")
    c_array = (ctypes.c_float * count).from_address(int(ptr.value))
    array = np.ctypeslib.as_array(c_array).reshape(shape)
    return ptr, array


class Batch6Runner:
    def __init__(self, engine_path: Path):
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
            raise RuntimeError("engine deserialize failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("execution context failed")

        inputs: list[int] = []
        outputs: list[int] = []
        for i in range(self.engine.num_bindings):
            (inputs if self.engine.binding_is_input(i) else outputs).append(i)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one input/one output, got inputs={inputs} outputs={outputs}"
            )
        self.input_index = inputs[0]
        self.output_index = outputs[0]
        self.input_shape = tuple(int(v) for v in self.context.get_binding_shape(self.input_index))
        self.output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_index))
        if self.input_shape != EXPECTED_INPUT:
            raise RuntimeError(
                f"unexpected input shape={self.input_shape}; expected={EXPECTED_INPUT}"
            )
        if self.output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(
                f"unexpected output shape={self.output_shape}; expected={EXPECTED_OUTPUT}"
            )
        if np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index))) != np.float32:
            raise RuntimeError("V8 batch engine input must be float32")
        if np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_index))) != np.float32:
            raise RuntimeError("V8 batch engine output must be float32")

        self.x_host_ptr, self.x = _pinned_float32(self.cuda, self.input_shape)
        self.y_host_ptr, self.y = _pinned_float32(self.cuda, self.output_shape)
        self.in_dev = ctypes.c_void_p()
        self.out_dev = ctypes.c_void_p()
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.in_dev), self.x.nbytes), "cudaMalloc input")
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.out_dev), self.y.nbytes), "cudaMalloc output")

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.in_dev.value)
        self.bindings[self.output_index] = int(self.out_dev.value)
        self.stream = ctypes.c_void_p()
        # cudaStreamNonBlocking = 1.
        cuda_check(
            self.cuda.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1),
            "cudaStreamCreateWithFlags",
        )
        self._shm = None
        self._shm_name = ""
        self._infer_n = 0

    def _attach(self, name: str):
        if self._shm is not None and self._shm_name == name:
            return self._shm
        if self._shm is not None:
            self._shm.close()
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        self._shm = shm
        self._shm_name = name
        return shm

    def preprocess(self, batch: np.ndarray) -> float:
        if batch.shape != (BATCH, INPUT_H, INPUT_W, 3) or batch.dtype != np.uint8:
            raise RuntimeError(f"unexpected batch shape/dtype: {batch.shape}/{batch.dtype}")
        started = time.perf_counter()
        scale = 1.0 / 255.0
        # Existing camera conversion yields a 672x378 content region in a 384-high
        # tensor. Keep the exact YOLO letterbox convention used by the proven b1 worker.
        batch[:, :3, :, :] = 114
        batch[:, 381:, :, :] = 114
        for i in range(BATCH):
            frame = batch[i]
            np.multiply(frame[:, :, 2], scale, out=self.x[i, 0], casting="unsafe")
            np.multiply(frame[:, :, 1], scale, out=self.x[i, 1], casting="unsafe")
            np.multiply(frame[:, :, 0], scale, out=self.x[i, 2], casting="unsafe")
        return (time.perf_counter() - started) * 1000.0

    def infer_from_shm(self, name: str, conf: float, max_det: int):
        shm = self._attach(name)
        needed = BATCH * INPUT_H * INPUT_W * 3
        if shm.size < needed:
            raise RuntimeError(f"shared memory too small: {shm.size} < {needed}")
        batch = np.ndarray(
            (BATCH, INPUT_H, INPUT_W, 3),
            dtype=np.uint8,
            buffer=shm.buf,
        )
        conf = min(1.0, max(0.0, float(conf)))
        max_det = max(1, min(300, int(max_det)))

        total_started = time.perf_counter()
        prep_ms = self.preprocess(batch)

        gpu_started = time.perf_counter()
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                self.in_dev,
                self.x_host_ptr,
                self.x.nbytes,
                1,
                self.stream,
            ),
            "batch H2D async",
        )
        if not self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self.stream.value),
        ):
            raise RuntimeError("execute_async_v2=false")
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                self.y_host_ptr,
                self.out_dev,
                self.y.nbytes,
                2,
                self.stream,
            ),
            "batch D2H async",
        )
        cuda_check(self.cuda.cudaStreamSynchronize(self.stream), "batch stream sync")
        trt_ms = (time.perf_counter() - gpu_started) * 1000.0

        all_rows: list[list[list[float]]] = []
        for sample in self.y:
            finite = np.isfinite(sample).all(axis=1)
            rows: list[list[float]] = []
            for row in sample[finite]:
                x1, y1, x2, y2, score, cls = (float(v) for v in row)
                if score < conf or int(round(cls)) != 0:
                    continue
                x1 = min(float(INPUT_W - 1), max(0.0, x1))
                x2 = min(float(INPUT_W - 1), max(0.0, x2))
                y1 = min(float(INPUT_H - 1), max(0.0, y1))
                y2 = min(float(INPUT_H - 1), max(0.0, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append([x1, y1, x2, y2, score])
            rows.sort(key=lambda item: item[4], reverse=True)
            all_rows.append(rows[:max_det])

        self._infer_n += 1
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return all_rows, prep_ms, trt_ms, total_ms

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if getattr(self, "stream", None) is not None and self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
            self.stream = ctypes.c_void_p()
        if getattr(self, "in_dev", None) is not None and self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
            self.in_dev = ctypes.c_void_p()
        if getattr(self, "out_dev", None) is not None and self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)
            self.out_dev = ctypes.c_void_p()
        if getattr(self, "x_host_ptr", None) is not None and self.x_host_ptr.value:
            self.cuda.cudaFreeHost(self.x_host_ptr)
            self.x_host_ptr = ctypes.c_void_p()
        if getattr(self, "y_host_ptr", None) is not None and self.y_host_ptr.value:
            self.cuda.cudaFreeHost(self.y_host_ptr)
            self.y_host_ptr = ctypes.c_void_p()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    runner = Batch6Runner(Path(args.engine))
    emit(
        {
            "type": "ready",
            "tensorrt": trt.__version__,
            "engine": str(Path(args.engine).resolve()),
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "transport": "shm-bgr-batch6-v8-pinned",
        }
    )
    try:
        for line in sys.stdin:
            req = {}
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    break
                boxes, prep_ms, trt_ms, total_ms = runner.infer_from_shm(
                    str(req["shm_name"]),
                    float(req.get("conf", 0.18)),
                    int(req.get("max_det", 20)),
                )
                emit(
                    {
                        "id": req.get("id"),
                        "ok": True,
                        "boxes": boxes,
                        "prep_ms": prep_ms,
                        "trt_ms": trt_ms,
                        "total_ms": total_ms,
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "id": req.get("id"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
