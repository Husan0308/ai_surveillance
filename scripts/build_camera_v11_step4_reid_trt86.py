#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


def fail(message: str) -> int:
    print(f"V11_STEP4_REID_BUILD RESULT=FAIL reason={message}", flush=True)
    return 1


def _shape_tuple(shape) -> tuple[int, ...]:
    return tuple(int(v) for v in shape)


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
    parsed_input_shape = _shape_tuple(inp.shape)
    parsed_output_shape = _shape_tuple(out.shape)

    if len(parsed_input_shape) != 4 or tuple(parsed_input_shape[1:]) != (3, 256, 128):
        return fail(f"unexpected_input name={input_name} shape={parsed_input_shape}")
    if len(parsed_output_shape) != 2 or int(parsed_output_shape[-1]) != 256:
        return fail(f"unexpected_output name={output_name} shape={parsed_output_shape}")
    if output_name != "fc_pred":
        return fail(f"unexpected_output_name expected=fc_pred got={output_name}")

    # NVIDIA's deployable ONNX is published with batch=1. TensorRT explicit-batch
    # networks allow the network input batch dimension to be marked runtime (-1)
    # before the optimization profile is attached. Step4 needs one engine for
    # batches 1/2/4/8, so make only N dynamic and keep C/H/W fixed.
    if parsed_input_shape[0] != -1:
        inp.shape = (-1, 3, 256, 128)

    network_input_shape = _shape_tuple(inp.shape)
    if network_input_shape != (-1, 3, 256, 128):
        return fail(
            f"dynamic_input_shape_not_applied name={input_name} "
            f"parsed={parsed_input_shape} network={network_input_shape}"
        )

    config = builder.create_builder_config()
    workspace = max(256 << 20, int(float(args.workspace_gib) * (1 << 30)))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
    else:  # TensorRT 8.x fallback
        config.max_workspace_size = workspace
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = max(0, min(5, int(args.optimization_level)))

    min_shape = (1, 3, 256, 128)
    opt_shape = (4, 3, 256, 128)
    max_shape = (8, 3, 256, 128)
    profile = builder.create_optimization_profile()

    # TensorRT 8.6.1 Python IOptimizationProfile.set_shape() returns None on
    # success. Do NOT boolean-test the return value; validate by reading the
    # profile back and by checking add_optimization_profile() instead.
    try:
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    except (ValueError, RuntimeError) as exc:
        return fail(f"profile_set_shape_exception type={type(exc).__name__} detail={exc}")

    try:
        profile_shapes = tuple(_shape_tuple(v) for v in profile.get_shape(input_name))
    except Exception as exc:
        return fail(f"profile_get_shape_exception type={type(exc).__name__} detail={exc}")
    expected_profile = (min_shape, opt_shape, max_shape)
    if profile_shapes != expected_profile:
        return fail(f"profile_shape_mismatch expected={expected_profile} got={profile_shapes}")

    profile_index = int(config.add_optimization_profile(profile))
    if profile_index < 0:
        return fail(f"add_optimization_profile_failed profile={profile_shapes}")

    build_output_shape = _shape_tuple(out.shape)
    print(
        "V11_STEP4_REID_BUILD START "
        f"trt={trt.__version__} onnx={args.onnx} "
        f"input={input_name}:{network_input_shape} "
        f"output={output_name}:{build_output_shape} "
        f"parsed_input={parsed_input_shape} parsed_output={parsed_output_shape} "
        f"profile_index={profile_index} profile=1,4,8 precision=fp32 workspace={workspace}",
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
        shape = _shape_tuple(engine.get_binding_shape(index))
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
