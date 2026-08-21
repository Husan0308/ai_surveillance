#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4.2: verify isolated TensorRT 8.6.1 and parse the fixed RF-DETR ONNX graph without building an engine."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/rfdetr_step4/onnx_800x448/rfdetr-small.onnx"),
    )
    return parser.parse_args()


def _loaded_nvinfer_paths() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths: set[str] = set()
    for line in maps.read_text(encoding="utf-8", errors="replace").splitlines():
        if "libnvinfer" not in line:
            continue
        path = line.rsplit(" ", 1)[-1].strip()
        if path.startswith("/"):
            paths.add(path)
    return sorted(paths)


def main() -> int:
    args = _args()
    if not args.onnx.is_file() or args.onnx.stat().st_size == 0:
        raise SystemExit(f"STEP4_2_FAIL onnx_not_found={args.onnx}")

    try:
        import tensorrt as trt
    except Exception as exc:
        raise SystemExit(
            f"STEP4_2_FAIL tensorrt_import error={type(exc).__name__}:{exc}"
        )

    version = str(getattr(trt, "__version__", "unknown"))
    print(
        "STEP4_2_ENV "
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"executable={sys.executable} tensorrt={version}",
        flush=True,
    )

    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"STEP4_2_FAIL wrong_python expected=3.10 got={sys.version_info.major}.{sys.version_info.minor}"
        )
    if not version.startswith("8.6.1"):
        raise SystemExit(f"STEP4_2_FAIL wrong_tensorrt expected=8.6.1 got={version}")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    blob = args.onnx.read_bytes()
    parsed = bool(parser.parse(blob))
    if not parsed:
        errors = []
        for index in range(int(parser.num_errors)):
            errors.append(str(parser.get_error(index)).replace("\n", " | "))
        raise SystemExit(
            "STEP4_2_FAIL onnx_parse " + " || ".join(errors[-10:])
        )

    loaded = _loaded_nvinfer_paths()
    wrong = [path for path in loaded if "libnvinfer.so.8" not in path]
    print(
        f"STEP4_2_LIBS loaded_nvinfer={loaded or ['not_visible']} wrong_major={wrong}",
        flush=True,
    )
    if wrong:
        raise SystemExit(
            "STEP4_2_FAIL mixed_tensorrt_runtime expected_only_libnvinfer_so_8"
        )

    inputs = []
    for index in range(int(network.num_inputs)):
        tensor = network.get_input(index)
        inputs.append(
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )

    outputs = []
    for index in range(int(network.num_outputs)):
        tensor = network.get_output(index)
        outputs.append(
            {
                "name": tensor.name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )

    print(
        "STEP4_2_RESULT "
        f"onnx={args.onnx} parse=pass inputs={inputs} outputs={outputs} "
        f"fast_fp16={bool(builder.platform_has_fast_fp16)} "
        f"fast_int8={bool(builder.platform_has_fast_int8)}",
        flush=True,
    )
    print("STEP4_2_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
