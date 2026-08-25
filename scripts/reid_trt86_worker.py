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


OFFSETS = np.asarray(
    [123.675, 116.280, 103.530],
    dtype=np.float32,
).reshape(1, 1, 3)

SCALE = np.float32(0.01735207)


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


def preprocess(jpeg_bytes: bytes) -> np.ndarray:
    image = Image.open(
        io.BytesIO(jpeg_bytes)
    ).convert("RGB")

    target_w = 128
    target_h = 256

    src_w, src_h = image.size

    scale = min(
        target_w / max(1, src_w),
        target_h / max(1, src_h),
    )

    new_w = max(
        1,
        min(target_w, round(src_w * scale)),
    )
    new_h = max(
        1,
        min(target_h, round(src_h * scale)),
    )

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR,
    )

    canvas = np.zeros(
        (target_h, target_w, 3),
        dtype=np.uint8,
    )

    # Same keepAspc behaviour used by our validated pair test.
    canvas[:new_h, :new_w] = np.asarray(resized)

    x = canvas.astype(np.float32)
    x = (x - OFFSETS) * SCALE
    x = x.transpose(2, 0, 1)

    return np.ascontiguousarray(x, dtype=np.float32)


class Runner:
    def __init__(self, engine_path: Path):
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(
                f"TensorRT 8.6.1 required, got {trt.__version__}"
            )

        self.cudart = load_cudart()

        self.logger = trt.Logger(
            trt.Logger.WARNING
        )

        trt.init_libnvinfer_plugins(
            self.logger,
            "",
        )

        self.runtime = trt.Runtime(self.logger)

        self.engine = (
            self.runtime.deserialize_cuda_engine(
                engine_path.read_bytes()
            )
        )

        if self.engine is None:
            raise RuntimeError(
                "engine deserialize failed"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        if self.context is None:
            raise RuntimeError(
                "execution context failed"
            )

        self.input_index = None
        self.output_index = None

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)

            if self.engine.binding_is_input(i):
                self.input_index = i
            elif name == "fc_pred":
                self.output_index = i

        if (
            self.input_index is None
            or self.output_index is None
        ):
            raise RuntimeError(
                "input/fc_pred binding missing"
            )

    def infer(self, batch: np.ndarray):
        batch = np.ascontiguousarray(
            batch,
            dtype=np.float32,
        )

        n = int(batch.shape[0])

        if not 1 <= n <= 8:
            raise ValueError(
                f"batch must be 1..8, got {n}"
            )

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
                f"unexpected output {out_shape}"
            )

        output = np.empty(
            out_shape,
            dtype=np.float32,
        )

        in_dev = ctypes.c_void_p()
        out_dev = ctypes.c_void_p()

        cuda_check(
            self.cudart.cudaMalloc(
                ctypes.byref(in_dev),
                batch.nbytes,
            ),
            "cudaMalloc input",
        )

        cuda_check(
            self.cudart.cudaMalloc(
                ctypes.byref(out_dev),
                output.nbytes,
            ),
            "cudaMalloc output",
        )

        bindings = [0] * self.engine.num_bindings
        bindings[self.input_index] = int(
            in_dev.value
        )
        bindings[self.output_index] = int(
            out_dev.value
        )

        try:
            cuda_check(
                self.cudart.cudaMemcpy(
                    in_dev,
                    ctypes.c_void_p(
                        batch.ctypes.data
                    ),
                    batch.nbytes,
                    1,
                ),
                "H2D",
            )

            started = time.perf_counter()

            if not self.context.execute_v2(
                bindings
            ):
                raise RuntimeError(
                    "execute_v2=false"
                )

            cuda_check(
                self.cudart.cudaDeviceSynchronize(),
                "infer sync",
            )

            infer_ms = (
                time.perf_counter() - started
            ) * 1000.0

            cuda_check(
                self.cudart.cudaMemcpy(
                    ctypes.c_void_p(
                        output.ctypes.data
                    ),
                    out_dev,
                    output.nbytes,
                    2,
                ),
                "D2H",
            )

            cuda_check(
                self.cudart.cudaDeviceSynchronize(),
                "D2H sync",
            )

        finally:
            self.cudart.cudaFree(in_dev)
            self.cudart.cudaFree(out_dev)

        # addFeatureNormalization: 1
        norms = np.linalg.norm(
            output,
            axis=1,
            keepdims=True,
        )

        output /= np.maximum(
            norms,
            1e-12,
        )

        return output, infer_ms


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    runner = Runner(args.engine)

    emit({
        "type": "ready",
        "python": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}"
        ),
        "tensorrt": trt.__version__,
        "max_batch": 8,
        "embedding_size": 256,
    })

    for line in sys.stdin:
        try:
            request = json.loads(line)

            if request.get("cmd") == "stop":
                emit({"type": "stopped"})
                return 0

            request_id = request.get("id")

            rows = request.get("jpeg_b64") or []

            if not 1 <= len(rows) <= 8:
                raise ValueError(
                    f"invalid image count={len(rows)}"
                )

            inputs = np.stack([
                preprocess(
                    base64.b64decode(x)
                )
                for x in rows
            ])

            embeddings, infer_ms = (
                runner.infer(inputs)
            )

            raw = embeddings.astype(
                np.float32,
                copy=False,
            ).tobytes()

            emit({
                "id": request_id,
                "ok": True,
                "shape": list(
                    embeddings.shape
                ),
                "dtype": "float32",
                "embedding_b64":
                    base64.b64encode(
                        raw
                    ).decode("ascii"),
                "infer_ms": round(
                    infer_ms,
                    3,
                ),
            })

        except Exception as exc:
            emit({
                "id": (
                    request.get("id")
                    if "request" in locals()
                    else None
                ),
                "ok": False,
                "error":
                    f"{type(exc).__name__}:{exc}",
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
