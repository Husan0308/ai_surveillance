#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "artifacts" / "yolo26m_trt86" / "yolo26m-672x384-b1-e2e.onnx"
ENGINE = ROOT / "artifacts" / "yolo26m_trt86" / "yolo26m-672x384-b1-fp32-trt86.engine"
EXPECTED_INPUT = (1, 3, 384, 672)
EXPECTED_OUTPUT = (1, 300, 6)


def main() -> int:
    version = str(trt.__version__)
    print(f"YOLO26M_TRT86_ENV python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} tensorrt={version}")
    if not version.startswith("8.6.1"):
        raise SystemExit(f"YOLO26M_TRT86_FAIL expected_trt=8.6.1 got={version}")
    if not ONNX.is_file():
        raise SystemExit(f"YOLO26M_TRT86_FAIL missing={ONNX}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(ONNX.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("YOLO26M_TRT86_FAIL onnx_parse")
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise SystemExit(f"YOLO26M_TRT86_FAIL io inputs={network.num_inputs} outputs={network.num_outputs}")

    inp = network.get_input(0)
    out = network.get_output(0)
    input_shape = tuple(int(v) for v in inp.shape)
    output_shape = tuple(int(v) for v in out.shape)
    print(f"YOLO26M_TRT86_NETWORK input={inp.name}:{input_shape} output={out.name}:{output_shape}")
    if input_shape != EXPECTED_INPUT:
        raise SystemExit(f"YOLO26M_TRT86_FAIL input={input_shape} expected={EXPECTED_INPUT}")
    if output_shape != EXPECTED_OUTPUT:
        raise SystemExit(f"YOLO26M_TRT86_FAIL output={output_shape} expected={EXPECTED_OUTPUT}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)
    try:
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.INT8)
    except Exception:
        pass
    if hasattr(config, "builder_optimization_level"):
        config.builder_optimization_level = 3

    ENGINE.parent.mkdir(parents=True, exist_ok=True)
    ENGINE.unlink(missing_ok=True)
    print("YOLO26M_TRT86_BUILD precision=fp32 batch=1 input=672x384 workspace=512MB")
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if serialized is None:
        raise SystemExit("YOLO26M_TRT86_FAIL build_returned_none")
    ENGINE.write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())
    if engine is None:
        raise SystemExit("YOLO26M_TRT86_FAIL deserialize")
    inputs = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    outputs = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit(f"YOLO26M_TRT86_FAIL bindings inputs={inputs} outputs={outputs}")
    if tuple(int(v) for v in engine.get_binding_shape(inputs[0])) != EXPECTED_INPUT:
        raise SystemExit("YOLO26M_TRT86_FAIL engine_input_shape")
    if tuple(int(v) for v in engine.get_binding_shape(outputs[0])) != EXPECTED_OUTPUT:
        raise SystemExit("YOLO26M_TRT86_FAIL engine_output_shape")

    print(f"YOLO26M_TRT86_ENGINE path={ENGINE} mb={ENGINE.stat().st_size / 1024 / 1024:.1f} build_sec={elapsed:.1f}")
    print("YOLO26M_TRT86_BUILD=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
