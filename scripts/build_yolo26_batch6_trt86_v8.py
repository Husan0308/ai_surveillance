#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


EXPECTED_INPUT = (6, 3, 384, 672)
EXPECTED_OUTPUT = (6, 300, 6)


def dims_tuple(dims) -> tuple[int, ...]:
    return tuple(int(v) for v in dims)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build YOLO26 batch-6 FP32 engine with TensorRT 8.6.1")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    args = parser.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V8_ENGINE FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    if not onnx_path.is_file():
        raise SystemExit(f"V8_ENGINE FAIL ONNX missing: {onnx_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(onnx_path.read_bytes()):
        errors = [str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)]
        raise SystemExit("V8_ENGINE FAIL ONNX parse:\n" + "\n".join(errors))

    if network.num_inputs != 1:
        raise SystemExit(f"V8_ENGINE FAIL expected one input, got {network.num_inputs}")
    input_tensor = network.get_input(0)
    shape = dims_tuple(input_tensor.shape)
    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

    if shape[0] == -1:
        if len(shape) != 4 or shape[1:] != EXPECTED_INPUT[1:]:
            raise SystemExit(f"V8_ENGINE FAIL unsupported dynamic input shape={shape}")
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, EXPECTED_INPUT, EXPECTED_INPUT, EXPECTED_INPUT)
        config.add_optimization_profile(profile)
    elif shape != EXPECTED_INPUT:
        raise SystemExit(
            f"V8_ENGINE FAIL ONNX input={shape}, expected {EXPECTED_INPUT}. "
            "Export a true batch=6 ONNX; a fixed batch-1 plan cannot be converted in-place."
        )

    print(
        f"V8_ENGINE_BUILD trt={trt.__version__} input={shape} workspace={workspace / (1<<20):.0f}MiB precision=fp32",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("V8_ENGINE FAIL build_serialized_network returned None")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise SystemExit("V8_ENGINE FAIL deserialize after build")
    context = engine.create_execution_context()
    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"V8_ENGINE FAIL bindings inputs={inputs} outputs={outputs}")
    actual_input = dims_tuple(context.get_binding_shape(inputs[0]))
    actual_output = dims_tuple(context.get_binding_shape(outputs[0]))
    if actual_input != EXPECTED_INPUT or actual_output != EXPECTED_OUTPUT:
        raise SystemExit(
            f"V8_ENGINE FAIL engine shapes input={actual_input} output={actual_output}; "
            f"expected {EXPECTED_INPUT}/{EXPECTED_OUTPUT}"
        )
    print(
        f"V8_ENGINE PASS engine={engine_path} input={actual_input} output={actual_output} bytes={engine_path.stat().st_size}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
