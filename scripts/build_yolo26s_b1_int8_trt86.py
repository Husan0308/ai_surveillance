#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT = (1, 3, 384, 672)
EXPECTED_OUTPUT = (1, 300, 6)


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
    return lib


def dims_tuple(dims) -> tuple[int, ...]:
    return tuple(int(v) for v in dims)


def _read_token(f) -> bytes:
    token = bytearray()
    while True:
        ch = f.read(1)
        if not ch:
            return bytes(token)
        if ch == b"#":
            f.readline()
            continue
        if ch.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(ch)


def load_ppm_rgb(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = _read_token(f)
        if magic != b"P6":
            raise RuntimeError(f"{path}: expected P6 PPM, got {magic!r}")
        width = int(_read_token(f))
        height = int(_read_token(f))
        maxval = int(_read_token(f))
        if (width, height, maxval) != (672, 384, 255):
            raise RuntimeError(
                f"{path}: geometry/maxval={(width, height, maxval)} expected=(672,384,255)"
            )
        raw = f.read(width * height * 3)
        if len(raw) != width * height * 3:
            raise RuntimeError(f"{path}: short pixel payload {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths: list[Path], cache_path: Path, seed: int = 26) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.paths = list(image_paths)
        random.Random(seed).shuffle(self.paths)
        self.cache_path = cache_path
        self.index = 0
        self.host = np.empty(EXPECTED_INPUT, dtype=np.float32)
        self.cuda = load_cudart()
        self.device = ctypes.c_void_p()
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.device), self.host.nbytes), "cudaMalloc calib")

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.paths):
            return None
        path = self.paths[self.index]
        rgb = load_ppm_rgb(path)
        chw = rgb.transpose(2, 0, 1)
        np.multiply(chw, 1.0 / 255.0, out=self.host[0], casting="unsafe")
        cuda_check(
            self.cuda.cudaMemcpy(
                self.device,
                ctypes.c_void_p(self.host.ctypes.data),
                self.host.nbytes,
                1,
            ),
            "cudaMemcpy H2D calibration",
        )
        self.index += 1
        if self.index <= 5 or self.index % 50 == 0 or self.index == len(self.paths):
            print(
                f"V11_INT8_BUILD_CALIB frame={self.index}/{len(self.paths)} file={path.name}",
                flush=True,
            )
        return [int(self.device.value)]

    def read_calibration_cache(self):
        if self.cache_path.is_file():
            data = self.cache_path.read_bytes()
            print(
                f"V11_INT8_BUILD_CACHE status=HIT path={self.cache_path} bytes={len(data)}",
                flush=True,
            )
            return data
        print(f"V11_INT8_BUILD_CACHE status=MISS path={self.cache_path}", flush=True)
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(bytes(cache))
        print(
            f"V11_INT8_BUILD_CACHE status=SAVED path={self.cache_path} bytes={self.cache_path.stat().st_size}",
            flush=True,
        )

    def close(self) -> None:
        if self.device.value:
            self.cuda.cudaFree(self.device)
            self.device = ctypes.c_void_p()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build YOLO26s batch1 INT8 engine with TensorRT 8.6.1")
    ap.add_argument(
        "--onnx",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-e2e.onnx",
    )
    ap.add_argument(
        "--calibration-dir",
        default="artifacts/yolo26s_trt86/int8_calibration_b1",
    )
    ap.add_argument(
        "--engine",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.engine",
    )
    ap.add_argument(
        "--cache",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.calib",
    )
    ap.add_argument("--workspace-gib", type=float, default=1.0)
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_INT8_BUILD FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx_path = Path(args.onnx)
    calib_dir = Path(args.calibration_dir)
    engine_path = Path(args.engine)
    cache_path = Path(args.cache)
    for p in (onnx_path, calib_dir, engine_path, cache_path):
        if not p.is_absolute():
            p = ROOT / p
        if p is onnx_path:
            onnx_path = p
        elif p is calib_dir:
            calib_dir = p
        elif p is engine_path:
            engine_path = p
        else:
            cache_path = p

    if not onnx_path.is_file():
        raise SystemExit(f"V11_INT8_BUILD FAIL ONNX missing: {onnx_path}")
    images = sorted(calib_dir.glob("CAM-*/*.ppm"))
    if len(images) < 500:
        raise SystemExit(f"V11_INT8_BUILD FAIL calibration_images={len(images)} expected>=500")

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("V11_INT8_BUILD FAIL ONNX parse:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise SystemExit(
            f"V11_INT8_BUILD FAIL network inputs={network.num_inputs} outputs={network.num_outputs}"
        )
    input_shape = dims_tuple(network.get_input(0).shape)
    output_shape = dims_tuple(network.get_output(0).shape)
    if input_shape != EXPECTED_INPUT or output_shape != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_INT8_BUILD FAIL shapes input={input_shape} output={output_shape} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )

    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.builder_optimization_level = max(0, min(5, int(args.optimization_level)))
    config.avg_timing_iterations = 1
    config.set_flag(trt.BuilderFlag.INT8)

    calibrator = EntropyCalibrator(images, cache_path, seed=int(args.seed))
    config.int8_calibrator = calibrator

    print(
        "V11_INT8_BUILD_START "
        f"trt={trt.__version__} onnx={onnx_path} input={input_shape} output={output_shape} "
        f"calibration_images={len(images)} calibrator=IInt8EntropyCalibrator2 batch=1 "
        f"workspace={workspace/(1<<20):.0f}MiB optimization_level={config.builder_optimization_level}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        calibrator.close()
    build_s = time.perf_counter() - started
    if serialized is None:
        raise SystemExit("V11_INT8_BUILD FAIL build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("V11_INT8_BUILD FAIL deserialize after build")
    context = engine.create_execution_context()
    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"V11_INT8_BUILD FAIL engine bindings inputs={inputs} outputs={outputs}")
    actual_input = dims_tuple(context.get_binding_shape(inputs[0]))
    actual_output = dims_tuple(context.get_binding_shape(outputs[0]))
    if actual_input != EXPECTED_INPUT or actual_output != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_INT8_BUILD FAIL engine shapes={actual_input}/{actual_output} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )

    print(
        "V11_INT8_BUILD_RESULT "
        f"status=PASS engine={engine_path} bytes={engine_path.stat().st_size} "
        f"cache={cache_path} input={actual_input} output={actual_output} build_seconds={build_s:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
