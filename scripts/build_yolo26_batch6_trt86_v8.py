#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
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
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--timing-cache", default="")
    args = parser.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V8_ENGINE FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx_path = Path(args.onnx).resolve()
    engine_path = Path(args.engine).resolve()
    cache_path = Path(args.timing_cache).resolve() if args.timing_cache else None
    if not onnx_path.is_file():
        raise SystemExit(f"V8_ENGINE FAIL ONNX missing: {onnx_path}")

    optimization_level = max(0, min(5, int(args.optimization_level)))
    # INFO is deliberate. build_serialized_network() is a blocking call and can spend
    # minutes profiling tactics on Pascal. WARNING made a healthy build look frozen.
    logger = trt.Logger(trt.Logger.INFO)
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
    # TRT 8.6 defaults to level 3. Level 2 materially reduces first-build tactic search
    # while still retaining profiling-based optimization. A shipping rebuild can opt
    # back into level 3 without changing code.
    config.builder_optimization_level = optimization_level
    config.avg_timing_iterations = 1

    timing_cache = None
    cache_bytes = b""
    if cache_path is not None and cache_path.is_file():
        try:
            cache_bytes = cache_path.read_bytes()
        except OSError:
            cache_bytes = b""
    try:
        timing_cache = config.create_timing_cache(cache_bytes)
        if not config.set_timing_cache(timing_cache, False):
            print(
                f"V8_ENGINE_CACHE mismatch=1 path={cache_path}; rebuilding cache from empty",
                flush=True,
            )
            timing_cache = config.create_timing_cache(b"")
            if not config.set_timing_cache(timing_cache, False):
                raise RuntimeError("set_timing_cache(empty) failed")
    except Exception as exc:
        # A cache is an optimization only; never make a valid engine build depend on it.
        timing_cache = None
        print(
            f"V8_ENGINE_CACHE disabled=1 reason={type(exc).__name__}:{exc}",
            flush=True,
        )

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

    cache_state = "hit" if cache_bytes else "empty"
    print(
        f"V8_ENGINE_BUILD trt={trt.__version__} input={shape} "
        f"workspace={workspace / (1<<20):.0f}MiB precision=fp32 "
        f"optimization_level={optimization_level} timing_cache={cache_state}",
        flush=True,
    )
    print(
        "V8_ENGINE_BUILD_PROGRESS stage=tactic_profiling status=START "
        "note='TensorRT may spend several minutes here on Pascal; INFO logs below mean it is working'",
        flush=True,
    )
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - started
    if serialized is None:
        raise SystemExit("V8_ENGINE FAIL build_serialized_network returned None")
    print(f"V8_ENGINE_BUILD_PROGRESS stage=tactic_profiling status=DONE seconds={build_s:.1f}", flush=True)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))

    if timing_cache is not None and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(timing_cache.serialize()))
            print(
                f"V8_ENGINE_CACHE saved={cache_path} bytes={cache_path.stat().st_size}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"V8_ENGINE_CACHE save_warning={type(exc).__name__}:{exc}",
                flush=True,
            )

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
        f"V8_ENGINE PASS engine={engine_path} input={actual_input} output={actual_output} "
        f"bytes={engine_path.stat().st_size} build_seconds={build_s:.1f} optimization_level={optimization_level}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
