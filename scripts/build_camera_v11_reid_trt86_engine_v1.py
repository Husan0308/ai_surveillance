#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import tensorrt as trt


EXPECTED_TRT_PREFIX = "8.6.1"
EXPECTED_INPUT_SHAPE = (-1, 3, 256, 128)
EXPECTED_OUTPUT_SHAPE = (-1, 256)
PROFILE_MIN = (1, 3, 256, 128)
PROFILE_OPT = (4, 3, 256, 128)
PROFILE_MAX = (8, 3, 256, 128)
OUTPUT_NAME = "fc_pred"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"V11_REID_ENGINE_BUILD RESULT=FAIL reason={message}")


def verify_engine(engine_path: Path, logger: trt.Logger) -> None:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        fail("deserialize_failed")
    if engine.num_optimization_profiles != 1:
        fail(f"unexpected_profiles_{engine.num_optimization_profiles}")

    input_idx = None
    output_idx = None
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        if engine.binding_is_input(index):
            if input_idx is not None:
                fail("multiple_inputs")
            input_idx = index
        if name == OUTPUT_NAME:
            output_idx = index

    if input_idx is None:
        fail("input_binding_missing")
    if output_idx is None:
        fail("fc_pred_binding_missing")

    min_shape, opt_shape, max_shape = engine.get_profile_shape(0, input_idx)
    actual_profile = (
        tuple(int(v) for v in min_shape),
        tuple(int(v) for v in opt_shape),
        tuple(int(v) for v in max_shape),
    )
    expected_profile = (PROFILE_MIN, PROFILE_OPT, PROFILE_MAX)
    if actual_profile != expected_profile:
        fail(f"profile_mismatch_actual_{actual_profile}_expected_{expected_profile}")

    context = engine.create_execution_context()
    if context is None:
        fail("execution_context_failed")
    for batch in (1, 4, 8):
        shape = (batch, 3, 256, 128)
        if not context.set_binding_shape(input_idx, shape):
            fail(f"set_binding_shape_failed_batch_{batch}")
        output_shape = tuple(int(v) for v in context.get_binding_shape(output_idx))
        if output_shape != (batch, 256):
            fail(f"output_shape_batch_{batch}_{output_shape}")

    print(
        "V11_REID_ENGINE_VERIFY RESULT=PASS "
        f"profiles=1 min={PROFILE_MIN} opt={PROFILE_OPT} max={PROFILE_MAX} "
        f"output={OUTPUT_NAME}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        type=Path,
        default=root / "artifacts/reid/resnet50_market1501_aicity156.onnx",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=root
        / "artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine",
    )
    parser.add_argument("--workspace-mib", type=int, default=1024)
    args = parser.parse_args()

    onnx_path = args.onnx.expanduser().resolve()
    engine_path = args.engine.expanduser().resolve()
    if not str(trt.__version__).startswith(EXPECTED_TRT_PREFIX):
        fail(f"tensorrt_version_{trt.__version__}_expected_{EXPECTED_TRT_PREFIX}")
    if not onnx_path.is_file() or onnx_path.stat().st_size <= 0:
        fail(f"onnx_missing_{onnx_path}")
    if args.workspace_mib <= 0:
        fail("invalid_workspace_mib")

    print(
        "V11_REID_ENGINE_BUILD READY "
        f"trt={trt.__version__} onnx={onnx_path} engine={engine_path} "
        f"precision=fp32 profile=min1_opt4_max8 workspace_mib={args.workspace_mib}"
    )

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser_onnx = trt.OnnxParser(network, logger)
    if not parser_onnx.parse(onnx_path.read_bytes()):
        for index in range(parser_onnx.num_errors):
            print(f"ONNX_ERROR[{index}] {parser_onnx.get_error(index)}")
        fail("onnx_parse_failed")

    if network.num_inputs != 1:
        fail(f"unexpected_input_count_{network.num_inputs}")
    inp = network.get_input(0)
    input_shape = tuple(int(v) for v in inp.shape)
    if input_shape != EXPECTED_INPUT_SHAPE:
        fail(f"unexpected_input_shape_{input_shape}")

    output = None
    for index in range(network.num_outputs):
        candidate = network.get_output(index)
        if candidate.name == OUTPUT_NAME:
            output = candidate
            break
    if output is None:
        fail("fc_pred_output_missing")
    output_shape = tuple(int(v) for v in output.shape)
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        fail(f"unexpected_output_shape_{output_shape}")

    profile = builder.create_optimization_profile()
    # TensorRT 8.6.1 set_shape() returns None. Validate the profile via bool(profile)
    # and add_optimization_profile() instead of treating set_shape() as a boolean.
    profile.set_shape(inp.name, PROFILE_MIN, PROFILE_OPT, PROFILE_MAX)
    if not bool(profile):
        fail("optimization_profile_invalid")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_mib) * 1024 * 1024
    )
    profile_index = config.add_optimization_profile(profile)
    if profile_index != 0:
        fail(f"optimization_profile_add_index_{profile_index}")

    print(
        "V11_REID_ENGINE_BUILD START "
        f"input={inp.name}{input_shape} output={OUTPUT_NAME}{output_shape} "
        f"min={PROFILE_MIN} opt={PROFILE_OPT} max={PROFILE_MAX}"
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        fail("build_serialized_network_failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = engine_path.with_name(engine_path.name + ".tmp")
    tmp_path.write_bytes(bytes(plan))
    if tmp_path.stat().st_size <= 0:
        tmp_path.unlink(missing_ok=True)
        fail("serialized_engine_empty")
    os.replace(tmp_path, engine_path)

    print(
        "V11_REID_ENGINE_BUILD SERIALIZED "
        f"bytes={engine_path.stat().st_size} path={engine_path}"
    )
    verify_engine(engine_path, logger)
    print(
        "V11_REID_ENGINE_BUILD RESULT=PASS "
        f"engine={engine_path} bytes={engine_path.stat().st_size} precision=fp32"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
