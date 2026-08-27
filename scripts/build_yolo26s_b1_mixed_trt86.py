#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import numpy as np
import tensorrt as trt

from build_yolo26s_b1_int8_trt86 import (
    EntropyCalibrator,
    EXPECTED_INPUT,
    EXPECTED_OUTPUT,
    dims_tuple,
    resolve_path,
)

HEAD_START_DEFAULT = 308

# Layers where TensorRT exposes a meaningful floating-point compute precision.
COMPUTE_TYPES = {
    trt.LayerType.CONVOLUTION,
    trt.LayerType.ACTIVATION,
    trt.LayerType.ELEMENTWISE,
    trt.LayerType.REDUCE,
    trt.LayerType.TOPK,
    trt.LayerType.GATHER,
    trt.LayerType.MATRIX_MULTIPLY,
    trt.LayerType.FULLY_CONNECTED,
    trt.LayerType.DECONVOLUTION,
    trt.LayerType.SCALE,
    trt.LayerType.UNARY,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build YOLO26s TRT8.6 mixed engine: INT8 backbone + FP32 Detect head"
    )
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
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-mixed-headfp32-trt86.engine",
    )
    ap.add_argument(
        "--cache",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.calib",
    )
    ap.add_argument("--head-start", type=int, default=HEAD_START_DEFAULT)
    ap.add_argument("--workspace-gib", type=float, default=1.0)
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_MIXED_BUILD FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx_path = resolve_path(args.onnx)
    calib_dir = resolve_path(args.calibration_dir)
    engine_path = resolve_path(args.engine)
    cache_path = resolve_path(args.cache)

    if not onnx_path.is_file():
        raise SystemExit(f"V11_MIXED_BUILD FAIL ONNX missing: {onnx_path}")
    images = sorted(calib_dir.rglob("*.ppm"))
    if len(images) < 500:
        raise SystemExit(f"V11_MIXED_BUILD FAIL calibration_images={len(images)} expected>=500")

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("V11_MIXED_BUILD FAIL ONNX parse:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise SystemExit(
            f"V11_MIXED_BUILD FAIL network inputs={network.num_inputs} outputs={network.num_outputs}"
        )
    input_shape = dims_tuple(network.get_input(0).shape)
    output_shape = dims_tuple(network.get_output(0).shape)
    if input_shape != EXPECTED_INPUT or output_shape != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_MIXED_BUILD FAIL shapes input={input_shape} output={output_shape} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )

    head_start = int(args.head_start)
    if head_start < 0 or head_start >= network.num_layers:
        raise SystemExit(
            f"V11_MIXED_BUILD FAIL head_start={head_start} layers={network.num_layers}"
        )

    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.builder_optimization_level = max(0, min(5, int(args.optimization_level)))
    config.avg_timing_iterations = 1
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    compute_constraints = 0
    output_constraints = 0
    first_name = None
    last_name = None
    for idx in range(head_start, network.num_layers):
        layer = network.get_layer(idx)
        name = layer.name or f"layer_{idx}"
        float_outputs = []
        for out_idx in range(layer.num_outputs):
            tensor = layer.get_output(out_idx)
            if tensor is not None and tensor.dtype == trt.float32:
                float_outputs.append(out_idx)

        if not float_outputs:
            continue

        # Preserve FP32 tensors throughout Detect/postprocess. Shape/index outputs
        # (for example TopK indices) remain INT32 and are deliberately untouched.
        for out_idx in float_outputs:
            layer.set_output_type(out_idx, trt.float32)
            output_constraints += 1

        if layer.type in COMPUTE_TYPES:
            layer.precision = trt.float32
            compute_constraints += 1

        first_name = first_name or name
        last_name = name

    if compute_constraints == 0 or output_constraints == 0:
        raise SystemExit(
            "V11_MIXED_BUILD FAIL no FP32 head constraints applied "
            f"compute={compute_constraints} outputs={output_constraints}"
        )

    # Network binding remains float32, matching the existing FP32/INT8 runners.
    network.get_output(0).dtype = trt.float32

    calibrator = EntropyCalibrator(images, cache_path, seed=int(args.seed))
    config.int8_calibrator = calibrator

    print(
        "V11_MIXED_BUILD_START "
        f"trt={trt.__version__} layers={network.num_layers} head_start={head_start} "
        f"backbone=INT8 head=FP32 calibration_images={len(images)} cache={cache_path} "
        f"input={input_shape} output={output_shape} workspace={workspace/(1<<20):.0f}MiB "
        f"optimization_level={config.builder_optimization_level} obey_precision=1",
        flush=True,
    )
    print(
        "V11_MIXED_BUILD_CONSTRAINTS "
        f"compute_fp32={compute_constraints} output_fp32={output_constraints} "
        f"first={first_name!r} last={last_name!r}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        calibrator.close()
    build_s = time.perf_counter() - started
    if serialized is None:
        raise SystemExit("V11_MIXED_BUILD FAIL build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("V11_MIXED_BUILD FAIL deserialize after build")
    context = engine.create_execution_context()
    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"V11_MIXED_BUILD FAIL bindings inputs={inputs} outputs={outputs}")
    actual_input = dims_tuple(context.get_binding_shape(inputs[0]))
    actual_output = dims_tuple(context.get_binding_shape(outputs[0]))
    input_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(inputs[0])))
    output_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(outputs[0])))
    if actual_input != EXPECTED_INPUT or actual_output != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_MIXED_BUILD FAIL engine shapes={actual_input}/{actual_output} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )
    if input_dtype != np.float32 or output_dtype != np.float32:
        raise SystemExit(
            f"V11_MIXED_BUILD FAIL binding dtypes input={input_dtype} output={output_dtype}"
        )

    print(
        "V11_MIXED_BUILD_RESULT "
        f"status=PASS engine={engine_path} bytes={engine_path.stat().st_size} "
        f"head_start={head_start} compute_fp32={compute_constraints} output_fp32={output_constraints} "
        f"input={actual_input}/{input_dtype} output={actual_output}/{output_dtype} "
        f"build_seconds={build_s:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
