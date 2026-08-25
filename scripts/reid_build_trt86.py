#!/usr/bin/env python3
from pathlib import Path
import sys
import time

ONNX = Path("models/reid/resnet50_market1501_aicity156.onnx")
ENGINE = Path("artifacts/reid/resnet50_market1501_aicity156_b1-8_fp32_trt86.engine")

try:
    import tensorrt as trt
except Exception as exc:
    raise SystemExit(f"REID_TRT_FAIL import={type(exc).__name__}:{exc}")

version = str(trt.__version__)

print(
    f"REID_TRT_ENV python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
    f"tensorrt={version}",
    flush=True,
)

if sys.version_info[:2] != (3, 10):
    raise SystemExit("REID_TRT_FAIL expected_python=3.10")

if not version.startswith("8.6.1"):
    raise SystemExit(f"REID_TRT_FAIL expected_trt=8.6.1 got={version}")

if not ONNX.is_file():
    raise SystemExit(f"REID_TRT_FAIL missing={ONNX}")

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")

builder = trt.Builder(logger)

flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(flags)
parser = trt.OnnxParser(network, logger)

if not parser.parse(ONNX.read_bytes()):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    raise SystemExit("REID_TRT_FAIL onnx_parse")

print(f"REID_TRT_NETWORK inputs={network.num_inputs} outputs={network.num_outputs}")

for i in range(network.num_inputs):
    x = network.get_input(i)
    print(
        f"INPUT index={i} name={x.name} "
        f"shape={tuple(x.shape)} dtype={x.dtype}"
    )

for i in range(network.num_outputs):
    x = network.get_output(i)
    print(
        f"OUTPUT index={i} name={x.name} "
        f"shape={tuple(x.shape)} dtype={x.dtype}"
    )

if network.num_inputs != 1:
    raise SystemExit(f"REID_TRT_FAIL inputs={network.num_inputs}")

inp = network.get_input(0)

if tuple(inp.shape) != (-1, 3, 256, 128):
    raise SystemExit(
        f"REID_TRT_FAIL unexpected_input={tuple(inp.shape)}"
    )

config = builder.create_builder_config()

config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    512 * 1024 * 1024,
)

# First validation = FP32.
try:
    config.clear_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.INT8)
except Exception:
    pass

profile = builder.create_optimization_profile()

profile.set_shape(
    inp.name,
    min=(1, 3, 256, 128),
    opt=(4, 3, 256, 128),
    max=(8, 3, 256, 128),
)

config.add_optimization_profile(profile)

ENGINE.parent.mkdir(parents=True, exist_ok=True)
ENGINE.unlink(missing_ok=True)

print(
    "REID_TRT_BUILD "
    "precision=fp32 "
    "profile=min1,opt4,max8 "
    "workspace=512MB",
    flush=True,
)

started = time.perf_counter()

serialized = builder.build_serialized_network(network, config)

elapsed = time.perf_counter() - started

if serialized is None:
    raise SystemExit("REID_TRT_FAIL build_returned_none")

ENGINE.write_bytes(bytes(serialized))

runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(ENGINE.read_bytes())

if engine is None:
    raise SystemExit("REID_TRT_FAIL deserialize")

print(f"REID_TRT_ENGINE path={ENGINE}")
print(f"REID_TRT_ENGINE_MB={ENGINE.stat().st_size / 1024 / 1024:.1f}")
print(f"REID_TRT_BUILD_SEC={elapsed:.1f}")

for i in range(engine.num_bindings):
    print(
        "BINDING "
        f"index={i} "
        f"name={engine.get_binding_name(i)} "
        f"input={int(engine.binding_is_input(i))} "
        f"shape={tuple(engine.get_binding_shape(i))} "
        f"dtype={engine.get_binding_dtype(i)}"
    )

print("REID_TRT_BUILD=PASS")
