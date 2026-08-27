#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT = (1, 3, 384, 672)
EXPECTED_OUTPUT = (1, 300, 6)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((len(s) - 1) * q))))
    return float(s[idx])


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
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    return lib


class Runner:
    def __init__(self, engine_path: Path) -> None:
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
        self.path = engine_path
        self.cuda = load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"deserialize failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"bindings inputs={inputs} outputs={outputs}")
        self.input_index = inputs[0]
        self.output_index = outputs[0]
        self.input_shape = tuple(int(v) for v in self.context.get_binding_shape(self.input_index))
        self.output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_index))
        if self.input_shape != EXPECTED_INPUT or self.output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(
                f"shapes={self.input_shape}/{self.output_shape} expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
            )
        self.input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index)))
        self.output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_index)))
        self.x = np.full(self.input_shape, 0.45, dtype=self.input_dtype)
        self.y = np.empty(self.output_shape, dtype=self.output_dtype)
        self.in_dev = ctypes.c_void_p()
        self.out_dev = ctypes.c_void_p()
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.in_dev), self.x.nbytes), "cudaMalloc input")
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.out_dev), self.y.nbytes), "cudaMalloc output")
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.in_dev.value)
        self.bindings[self.output_index] = int(self.out_dev.value)

    def one(self) -> tuple[float, float]:
        total_started = time.perf_counter()
        cuda_check(
            self.cuda.cudaMemcpy(
                self.in_dev,
                ctypes.c_void_p(self.x.ctypes.data),
                self.x.nbytes,
                1,
            ),
            "H2D",
        )
        trt_started = time.perf_counter()
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("execute_v2=false")
        cuda_check(self.cuda.cudaDeviceSynchronize(), "infer sync")
        trt_ms = (time.perf_counter() - trt_started) * 1000.0
        cuda_check(
            self.cuda.cudaMemcpy(
                ctypes.c_void_p(self.y.ctypes.data),
                self.out_dev,
                self.y.nbytes,
                2,
            ),
            "D2H",
        )
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return trt_ms, total_ms

    def close(self) -> None:
        if self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
        if self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare TRT8.6 batch1 engine latency on the same harness")
    ap.add_argument("--engine", action="append", required=True, help="Repeat for each engine")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=200)
    args = ap.parse_args()

    for raw in args.engine:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise SystemExit(f"V11_TRT86_B1_PRECISION FAIL missing={path}")
        runner = Runner(path)
        try:
            for _ in range(max(1, args.warmup)):
                runner.one()
            trt_times: list[float] = []
            total_times: list[float] = []
            for _ in range(max(1, args.iterations)):
                trt_ms, total_ms = runner.one()
                trt_times.append(trt_ms)
                total_times.append(total_ms)
        finally:
            runner.close()

        p50 = pct(trt_times, 0.50)
        p95 = pct(trt_times, 0.95)
        p99 = pct(trt_times, 0.99)
        max_ms = max(trt_times)
        mean_ms = statistics.fmean(trt_times)
        total95 = pct(total_times, 0.95)
        img_s = 1000.0 / p50 if p50 > 0 else 0.0
        per_cam = img_s / 6.0
        print(
            "V11_TRT86_B1_PRECISION_RESULT "
            f"engine={path.name} input_dtype={runner.input_dtype.name} output_dtype={runner.output_dtype.name} "
            f"mean={mean_ms:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms "
            f"max={max_ms:.1f}ms roundtrip_p95={total95:.1f}ms img_s_p50={img_s:.2f} "
            f"per_cam_hz_p50={per_cam:.2f} iterations={len(trt_times)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
