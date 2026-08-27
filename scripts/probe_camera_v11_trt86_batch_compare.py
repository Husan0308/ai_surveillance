#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import statistics
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]


def pct(values: list[float], q: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    idx = min(len(rows) - 1, int(round((len(rows) - 1) * q)))
    return float(rows[idx])


def cuda_check(code: int, name: str) -> None:
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
    lib.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    lib.cudaHostAlloc.restype = ctypes.c_int
    lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
    lib.cudaFreeHost.restype = ctypes.c_int
    lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    return lib


def pinned_array(cuda, shape: tuple[int, ...], dtype: np.dtype):
    count = int(np.prod(shape))
    itemsize = np.dtype(dtype).itemsize
    nbytes = count * itemsize
    ptr = ctypes.c_void_p()
    cuda_check(cuda.cudaHostAlloc(ctypes.byref(ptr), nbytes, 0), "cudaHostAlloc")
    if np.dtype(dtype) == np.float32:
        c_type = ctypes.c_float
    elif np.dtype(dtype) == np.float16:
        c_type = ctypes.c_uint16
    else:
        raise RuntimeError(f"unsupported host dtype {dtype}")
    c_array = (c_type * count).from_address(int(ptr.value))
    arr = np.ctypeslib.as_array(c_array).view(dtype).reshape(shape)
    return ptr, arr


def bench_engine(engine_path: Path, warmup: int, iterations: int) -> dict[str, float | int | str]:
    if not engine_path.is_file():
        raise FileNotFoundError(engine_path)
    if not str(trt.__version__).startswith("8.6.1"):
        raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")

    cuda = load_cudart()
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"deserialize failed: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"execution context failed: {engine_path}")

    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"expected 1 input/1 output, got {inputs}/{outputs}")
    i_idx, o_idx = inputs[0], outputs[0]
    i_shape = tuple(int(v) for v in context.get_binding_shape(i_idx))
    o_shape = tuple(int(v) for v in context.get_binding_shape(o_idx))
    if not i_shape or i_shape[0] <= 0:
        raise RuntimeError(f"fixed batch required, input_shape={i_shape}")
    batch = int(i_shape[0])

    i_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(i_idx)))
    o_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(o_idx)))
    if i_dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        raise RuntimeError(f"unsupported input dtype={i_dtype}")
    if o_dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
        raise RuntimeError(f"unsupported output dtype={o_dtype}")

    x_ptr, x = pinned_array(cuda, i_shape, i_dtype)
    y_ptr, y = pinned_array(cuda, o_shape, o_dtype)
    x.fill(0.5)
    y.fill(0)

    in_dev = ctypes.c_void_p()
    out_dev = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    cuda_check(cuda.cudaMalloc(ctypes.byref(in_dev), x.nbytes), "cudaMalloc input")
    cuda_check(cuda.cudaMalloc(ctypes.byref(out_dev), y.nbytes), "cudaMalloc output")
    cuda_check(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream), 1), "cudaStreamCreateWithFlags")

    bindings = [0] * engine.num_bindings
    bindings[i_idx] = int(in_dev.value)
    bindings[o_idx] = int(out_dev.value)

    def once() -> float:
        import time
        started = time.perf_counter()
        cuda_check(cuda.cudaMemcpyAsync(in_dev, x_ptr, x.nbytes, 1, stream), "H2D")
        ok = context.execute_async_v2(bindings=bindings, stream_handle=int(stream.value))
        if not ok:
            raise RuntimeError("execute_async_v2=false")
        cuda_check(cuda.cudaMemcpyAsync(y_ptr, out_dev, y.nbytes, 2, stream), "D2H")
        cuda_check(cuda.cudaStreamSynchronize(stream), "stream sync")
        return (time.perf_counter() - started) * 1000.0

    try:
        for _ in range(warmup):
            once()
        samples = [once() for _ in range(iterations)]
    finally:
        cuda.cudaStreamDestroy(stream)
        cuda.cudaFree(in_dev)
        cuda.cudaFree(out_dev)
        cuda.cudaFreeHost(x_ptr)
        cuda.cudaFreeHost(y_ptr)

    p50 = pct(samples, 0.50)
    p95 = pct(samples, 0.95)
    mean = statistics.fmean(samples)
    img_s_p50 = batch * 1000.0 / p50
    img_s_p95 = batch * 1000.0 / p95
    per_cam_hz_p50 = img_s_p50 / 6.0
    return {
        "engine": str(engine_path),
        "batch": batch,
        "input_shape": str(i_shape),
        "output_shape": str(o_shape),
        "dtype": str(i_dtype),
        "mean": mean,
        "p50": p50,
        "p95": p95,
        "p99": pct(samples, 0.99),
        "max": max(samples),
        "img_s_p50": img_s_p50,
        "img_s_p95": img_s_p95,
        "per_cam_hz_p50": per_cam_hz_p50,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--engines",
        nargs="*",
        default=[
            "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
            "artifacts/yolo26s_trt86/yolo26s-672x384-b6-fp32-trt86.engine",
        ],
    )
    args = parser.parse_args()
    warmup = max(10, int(args.warmup))
    iterations = max(100, int(args.iterations))

    rows = []
    for raw in args.engines:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        print(f"V11_TRT86_BATCH_COMPARE_START engine={path}", flush=True)
        row = bench_engine(path, warmup, iterations)
        rows.append(row)
        print(
            "V11_TRT86_BATCH_COMPARE_RESULT "
            f"engine={Path(str(row['engine'])).name} batch={row['batch']} dtype={row['dtype']} "
            f"mean={row['mean']:.1f}ms p50={row['p50']:.1f}ms p95={row['p95']:.1f}ms "
            f"p99={row['p99']:.1f}ms max={row['max']:.1f}ms "
            f"img_s_p50={row['img_s_p50']:.2f} img_s_p95={row['img_s_p95']:.2f} "
            f"per_cam_hz_p50={row['per_cam_hz_p50']:.2f}",
            flush=True,
        )

    best_latency = min(rows, key=lambda r: float(r["p50"]))
    best_throughput = max(rows, key=lambda r: float(r["img_s_p50"]))
    print(
        "V11_TRT86_BATCH_COMPARE_SUMMARY "
        f"best_latency={Path(str(best_latency['engine'])).name}:b{best_latency['batch']}:{best_latency['p50']:.1f}ms "
        f"best_throughput={Path(str(best_throughput['engine'])).name}:b{best_throughput['batch']}:{best_throughput['img_s_p50']:.2f}img_s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
