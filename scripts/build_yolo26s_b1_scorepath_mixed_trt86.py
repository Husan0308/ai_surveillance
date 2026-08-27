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

DEFAULT_FP32_MODULES = ("/model.16", "/model.19", "/model.22", "/model.23")

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


def in_module(name: str, module: str) -> bool:
    return name == module or name.startswith(module + "/")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build YOLO26s TRT8.6 selective mixed engine: INT8 backbone + FP32 score-source modules/head"
    )
    ap.add_argument("--onnx", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-e2e.onnx")
    ap.add_argument("--calibration-dir", default="artifacts/yolo26s_trt86/int8_calibration_b1")
    ap.add_argument(
        "--engine",
        default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-mixed-scorepathfp32-trt86.engine",
    )
    ap.add_argument("--cache", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.calib")
    ap.add_argument(
        "--fp32-module",
        action="append",
        dest="fp32_modules",
        help="Module prefix to force FP32; repeatable. Defaults to model.16,19,22,23.",
    )
    ap.add_argument("--workspace-gib", type=float, default=1.0)
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_SCOREPATH_BUILD FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    fp32_modules = tuple(args.fp32_modules or DEFAULT_FP32_MODULES)
    if not fp32_modules:
        raise SystemExit("V11_SCOREPATH_BUILD FAIL no fp32 modules")

    onnx_path = resolve_path(args.onnx)
    calib_dir = resolve_path(args.calibration_dir)
    engine_path = resolve_path(args.engine)
    cache_path = resolve_path(args.cache)

    if not onnx_path.is_file():
        raise SystemExit(f"V11_SCOREPATH_BUILD FAIL ONNX missing: {onnx_path}")
    images = sorted(calib_dir.rglob("*.ppm"))
    if len(images) < 500:
        raise SystemExit(f"V11_SCOREPATH_BUILD FAIL calibration_images={len(images)} expected>=500")

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("V11_SCOREPATH_BUILD FAIL ONNX parse:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise SystemExit(
            f"V11_SCOREPATH_BUILD FAIL network inputs={network.num_inputs} outputs={network.num_outputs}"
        )
    input_shape = dims_tuple(network.get_input(0).shape)
    output_shape = dims_tuple(network.get_output(0).shape)
    if input_shape != EXPECTED_INPUT or output_shape != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_SCOREPATH_BUILD FAIL shapes input={input_shape} output={output_shape} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )

    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    config.builder_optimization_level = max(0, min(5, int(args.optimization_level)))
    config.avg_timing_iterations = 1
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    module_layer_counts = {module: 0 for module in fp32_modules}
    compute_constraints = 0
    output_constraints = 0
    constrained_layers = 0
    first_name = None
    last_name = None

    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        name = layer.name or f"layer_{idx}"
        matched = [module for module in fp32_modules if in_module(name, module)]
        if not matched:
            continue

        constrained_layers += 1
        for module in matched:
            module_layer_counts[module] += 1

        float_outputs: list[int] = []
        for out_idx in range(layer.num_outputs):
            tensor = layer.get_output(out_idx)
            if tensor is not None and tensor.dtype == trt.float32:
                float_outputs.append(out_idx)

        for out_idx in float_outputs:
            layer.set_output_type(out_idx, trt.float32)
            output_constraints += 1

        if float_outputs and layer.type in COMPUTE_TYPES:
            layer.precision = trt.float32
            compute_constraints += 1

        first_name = first_name or name
        last_name = name

    missing = [module for module, count in module_layer_counts.items() if count == 0]
    if missing:
        raise SystemExit(f"V11_SCOREPATH_BUILD FAIL missing_modules={','.join(missing)}")
    if compute_constraints == 0 or output_constraints == 0:
        raise SystemExit(
            "V11_SCOREPATH_BUILD FAIL no FP32 constraints applied "
            f"compute={compute_constraints} outputs={output_constraints}"
        )

    network.get_output(0).dtype = trt.float32
    calibrator = EntropyCalibrator(images, cache_path, seed=int(args.seed))
    config.int8_calibrator = calibrator

    modules_text = ",".join(fp32_modules)
    counts_text = ",".join(f"{m}:{module_layer_counts[m]}" for m in fp32_modules)
    print(
        "V11_SCOREPATH_BUILD_START "
        f"trt={trt.__version__} layers={network.num_layers} fp32_modules={modules_text} "
        f"calibration_images={len(images)} cache={cache_path} input={input_shape} output={output_shape} "
        f"workspace={workspace/(1<<20):.0f}MiB optimization_level={config.builder_optimization_level} obey_precision=1",
        flush=True,
    )
    print(
        "V11_SCOREPATH_BUILD_CONSTRAINTS "
        f"module_layers={counts_text} constrained_layers={constrained_layers} "
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
        raise SystemExit("V11_SCOREPATH_BUILD FAIL build_serialized_network returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("V11_SCOREPATH_BUILD FAIL deserialize after build")
    context = engine.create_execution_context()
    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"V11_SCOREPATH_BUILD FAIL bindings inputs={inputs} outputs={outputs}")

    actual_input = dims_tuple(context.get_binding_shape(inputs[0]))
    actual_output = dims_tuple(context.get_binding_shape(outputs[0]))
    input_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(inputs[0])))
    output_dtype = np.dtype(trt.nptype(engine.get_binding_dtype(outputs[0])))
    if actual_input != EXPECTED_INPUT or actual_output != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V11_SCOREPATH_BUILD FAIL engine shapes={actual_input}/{actual_output} "
            f"expected={EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )
    if input_dtype != np.float32 or output_dtype != np.float32:
        raise SystemExit(
            f"V11_SCOREPATH_BUILD FAIL binding dtypes input={input_dtype} output={output_dtype}"
        )

    print(
        "V11_SCOREPATH_BUILD_RESULT "
        f"status=PASS engine={engine_path} bytes={engine_path.stat().st_size} "
        f"fp32_modules={modules_text} constrained_layers={constrained_layers} "
        f"compute_fp32={compute_constraints} output_fp32={output_constraints} "
        f"input={actual_input}/{input_dtype} output={actual_output}/{output_dtype} build_seconds={build_s:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
