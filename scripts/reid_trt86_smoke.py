#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import ctypes.util
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

ENGINE = Path(
    "artifacts/reid/"
    "resnet50_market1501_aicity156_b1-8_fp32_trt86.engine"
)

def load_cudart():
    path = ctypes.util.find_library("cudart")
    if not path:
        raise RuntimeError("libcudart not found")

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


def ck(code, name):
    if int(code) != 0:
        raise RuntimeError(f"{name} failed cuda={code}")


logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())

if engine is None:
    raise SystemExit("REID_RUNTIME_FAIL deserialize")

context = engine.create_execution_context()
if context is None:
    raise SystemExit("REID_RUNTIME_FAIL context")

cudart = load_cudart()

input_index = None
output_index = None

for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)

    if engine.binding_is_input(i):
        input_index = i
    elif name == "fc_pred":
        output_index = i

if input_index is None or output_index is None:
    raise SystemExit(
        f"REID_RUNTIME_FAIL bindings input={input_index} "
        f"output={output_index}"
    )

print(
    f"REID_RUNTIME_ENV tensorrt={trt.__version__} "
    f"engine={ENGINE}"
)


def run(batch: int):
    if not context.set_binding_shape(
        input_index,
        (batch, 3, 256, 128),
    ):
        raise RuntimeError(f"set_binding_shape batch={batch}")

    in_shape = tuple(context.get_binding_shape(input_index))
    out_shape = tuple(context.get_binding_shape(output_index))

    if in_shape != (batch, 3, 256, 128):
        raise RuntimeError(f"bad input shape {in_shape}")

    if out_shape != (batch, 256):
        raise RuntimeError(f"bad output shape {out_shape}")

    # Real TAO preprocessing range is roughly centered around zero.
    # Random data is enough for engine execution validation here.
    host_in = np.random.default_rng(1234).normal(
        0.0,
        1.0,
        size=in_shape,
    ).astype(np.float32)

    host_out = np.empty(out_shape, dtype=np.float32)

    in_dev = ctypes.c_void_p()
    out_dev = ctypes.c_void_p()

    ck(
        cudart.cudaMalloc(
            ctypes.byref(in_dev),
            host_in.nbytes,
        ),
        "cudaMalloc input",
    )

    ck(
        cudart.cudaMalloc(
            ctypes.byref(out_dev),
            host_out.nbytes,
        ),
        "cudaMalloc output",
    )

    bindings = [0] * engine.num_bindings
    bindings[input_index] = int(in_dev.value)
    bindings[output_index] = int(out_dev.value)

    try:
        ck(
            cudart.cudaMemcpy(
                in_dev,
                ctypes.c_void_p(host_in.ctypes.data),
                host_in.nbytes,
                1,  # H2D
            ),
            "H2D",
        )

        # Warmup
        if not context.execute_v2(bindings):
            raise RuntimeError("warmup execute_v2=false")

        ck(
            cudart.cudaDeviceSynchronize(),
            "warmup sync",
        )

        times = []

        for _ in range(10):
            start = time.perf_counter()

            if not context.execute_v2(bindings):
                raise RuntimeError("execute_v2=false")

            ck(
                cudart.cudaDeviceSynchronize(),
                "infer sync",
            )

            times.append(
                (time.perf_counter() - start) * 1000.0
            )

        ck(
            cudart.cudaMemcpy(
                ctypes.c_void_p(host_out.ctypes.data),
                out_dev,
                host_out.nbytes,
                2,  # D2H
            ),
            "D2H",
        )

        ck(
            cudart.cudaDeviceSynchronize(),
            "D2H sync",
        )

    finally:
        cudart.cudaFree(in_dev)
        cudart.cudaFree(out_dev)

    finite = bool(np.isfinite(host_out).all())

    norms = np.linalg.norm(
        host_out,
        axis=1,
    )

    print(
        "REID_RUNTIME "
        f"batch={batch} "
        f"input={in_shape} "
        f"output={out_shape} "
        f"mean_ms={np.mean(times):.2f} "
        f"p95_ms={np.percentile(times,95):.2f} "
        f"finite={int(finite)} "
        f"norm_min={norms.min():.3f} "
        f"norm_max={norms.max():.3f}"
    )

    if not finite:
        raise RuntimeError("non-finite embedding")


for batch in (1, 4, 8):
    run(batch)

print("REID_RUNTIME=PASS")
