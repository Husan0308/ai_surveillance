#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def fail(message: str) -> int:
    print(f"V11_STEP4_REID_BUILD RESULT=FAIL reason={message}", flush=True)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--workspace-gib", type=float, default=1.0)
    parser.add_argument("--optimization-level", type=int, default=3)
    args = parser.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        return fail(f"TensorRT_8.6.1_required_got_{trt.__version__}")
    if not args.onnx.is_file() or args.onnx.stat().st_size < 1_000_000:
        return fail(f"onnx_missing_or_too_small path={args.onnx}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser_onnx = trt.OnnxParser(network, logger)

    model_bytes = args.onnx.read_bytes()
    if not parser_onnx.parse(model_bytes):
        errors = []
        for index in range(parser_onnx.num_errors):
            errors.append(str(parser_onnx.get_error(index)).replace("\n", " "))
        return fail("onnx_parse " + " | ".join(errors[:8]))

    if network.num_inputs != 1:
        return fail(f"expected_one_input got={network.num_inputs}")
    if network.num_outputs != 1:
        return fail(f"expected_one_output got={network.num_outputs}")

    inp = network.get_input(0)
    out = network.get_output(0)
    input_name = str(inp.name)
    output_name = str(out.name)
    input_shape = tuple(int(v) for v in inp.shape)
    output_shape = tuple(int(v) for v in out.shape)

    if len(input_shape) != 4 or tuple(input_shape[1:]) != (3, 256, 128):
        return fail(f"unexpected_input name={input_name} shape={input_shape}")
    if len(output_shape) != 2 or int(output_shape[-1]) != 256:
        return fail(f"unexpected_output name={output_name} shape={output_shape}")
    if output_name != "fc_pred":
        return fail(f"unexpected_output_name expected=fc_pred got={output_name}")

    # NVIDIA's deployable model supports batching. Make the batch dimension dynamic
    # explicitly so the Step4 runtime can use batch sizes 1/2/4/8 with one engine.
    if input_shape[0] != -1:
        inp.shape = (-1, 3, 256, 128)

    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:  # TensorRT 8.x fallback
        config.max_workspace_size = workspace
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = max(0, min(5, int(args.optimization_level)))

    profile = builder.create_optimization_profile()
    if not profile.set_shape(
        input_name,
        min=(1, 3, 256, 128),
        opt=(4, 3, 256, 128),
        max=(8, 3, 256, 128),
    ):
        return fail("profile_set_shape_failed")
    config.add_optimization_profile(profile)

    print(
        "V11_STEP4_REID_BUILD START "
        f"trt={trt.__version__} onnx={args.onnx} input={input_name}:{tuple(inp.shape)} "
        f"output={output_name}:{output_shape} profile=1,4,8 precision=fp32 workspace={workspace}",
        flush=True,
    )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return fail("build_serialized_network_none")

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.engine.with_suffix(args.engine.suffix + ".tmp")
    tmp.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(tmp.read_bytes())
    if engine is None:
        tmp.unlink(missing_ok=True)
        return fail("deserialize_validation_failed")

    input_bindings = []
    output_bindings = []
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        shape = tuple(int(v) for v in engine.get_binding_shape(index))
        row = f"{name}:{shape}"
        if engine.binding_is_input(index):
            input_bindings.append(row)
        else:
            output_bindings.append(row)

    if not any(row.startswith("fc_pred:") for row in output_bindings):
        tmp.unlink(missing_ok=True)
        return fail(f"fc_pred_binding_missing outputs={output_bindings}")

    tmp.replace(args.engine)
    print(
        "V11_STEP4_REID_BUILD RESULT=PASS "
        f"engine={args.engine} bytes={args.engine.stat().st_size} "
        f"inputs={';'.join(input_bindings)} outputs={';'.join(output_bindings)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
