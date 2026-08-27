#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-e2e.onnx"
DEFAULT_ENGINE = ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp16-trt86.engine"


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the one permitted YOLO26s FP16 TRT8.6 experiment")
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--output", default=str(DEFAULT_ENGINE))
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_FP16_BUILD FAIL TensorRT 8.6.1 required, got {trt.__version__}")
    onnx = resolve(args.onnx)
    output = resolve(args.output)
    if not onnx.is_file():
        raise SystemExit(f"V11_FP16_BUILD FAIL ONNX missing: {onnx}")
    if output.exists() and not args.force:
        raise SystemExit(f"V11_FP16_BUILD FAIL output exists (use --force): {output}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, logger)
    if not parser_trt.parse(onnx.read_bytes()):
        errors = [str(parser_trt.get_error(index)) for index in range(parser_trt.num_errors)]
        raise SystemExit("V11_FP16_BUILD FAIL parse=" + " | ".join(errors))
    if network.num_inputs != 1 or network.num_outputs != 1:
        raise SystemExit(
            f"V11_FP16_BUILD FAIL expected one input/output, got {network.num_inputs}/{network.num_outputs}"
        )
    shape = tuple(int(value) for value in network.get_input(0).shape)
    if shape != (1, 3, 384, 672):
        raise SystemExit(f"V11_FP16_BUILD FAIL input shape={shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max(128, int(args.workspace_mib)) * 1024 * 1024,
    )
    config.set_flag(trt.BuilderFlag.FP16)
    print(
        "V11_FP16_BUILD_START "
        f"onnx={onnx} output={output} workspace_mib={max(128, int(args.workspace_mib))} "
        f"platform_fast_fp16={int(builder.platform_has_fast_fp16)} explicit_batch=1",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("V11_FP16_BUILD FAIL build_serialized_network returned None")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(bytes(serialized))
    temporary.replace(output)
    print(
        f"V11_FP16_BUILD_RESULT status=OK output={output} bytes={output.stat().st_size} "
        f"platform_fast_fp16={int(builder.platform_has_fast_fp16)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
