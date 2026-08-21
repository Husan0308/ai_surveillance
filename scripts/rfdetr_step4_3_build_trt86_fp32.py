#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4.3: build a fixed-shape RF-DETR-S TensorRT 8.6 FP32 engine on the target Pascal GPU."
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/rfdetr_step4/onnx_800x448/rfdetr-small.onnx"),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt86_fp32/rfdetr-small-800x448-fp32.engine"),
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="TensorRT builder workspace limit in MiB.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_name() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        value = (result.stdout or "").strip().splitlines()
        return value[0].strip() if value else "unknown"
    except Exception:
        return "unknown"


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


def _is_trt8_library(path: str) -> bool:
    name = Path(path).name
    if name.startswith("libnvinfer.so.8"):
        return True
    if name.startswith("libnvinfer_plugin.so.8"):
        return True
    if name.startswith("libnvinfer_builder_resource.so.8"):
        return True
    if name.startswith("libnvinfer_lean.so.8"):
        return True
    if name.startswith("libnvinfer_dispatch.so.8"):
        return True
    return False


def main() -> int:
    args = _args()
    if not args.onnx.is_file() or args.onnx.stat().st_size == 0:
        raise SystemExit(f"STEP4_3_FAIL onnx_not_found={args.onnx}")
    if args.workspace_mb < 256 or args.workspace_mb > 3072:
        raise SystemExit("STEP4_3_FAIL workspace_mb_must_be_256_to_3072")

    try:
        import tensorrt as trt
    except Exception as exc:
        raise SystemExit(
            f"STEP4_3_FAIL tensorrt_import error={type(exc).__name__}:{exc}"
        )

    version = str(getattr(trt, "__version__", "unknown"))
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"STEP4_3_FAIL wrong_python expected=3.10 got={sys.version_info.major}.{sys.version_info.minor}"
        )
    if not version.startswith("8.6.1"):
        raise SystemExit(f"STEP4_3_FAIL wrong_tensorrt expected=8.6.1 got={version}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(args.onnx.read_bytes()):
        errors = [
            str(parser.get_error(index)).replace("\n", " | ")
            for index in range(int(parser.num_errors))
        ]
        raise SystemExit("STEP4_3_FAIL onnx_parse " + " || ".join(errors[-10:]))

    if int(network.num_inputs) != 1:
        raise SystemExit(f"STEP4_3_FAIL expected_one_input got={network.num_inputs}")
    input_tensor = network.get_input(0)
    input_shape = [int(v) for v in input_tensor.shape]
    if input_shape != [1, 3, 448, 800]:
        raise SystemExit(
            f"STEP4_3_FAIL unexpected_input_shape expected=[1,3,448,800] got={input_shape}"
        )

    loaded = _loaded_nvinfer_paths()
    wrong = [path for path in loaded if not _is_trt8_library(path)]
    if wrong:
        raise SystemExit(
            f"STEP4_3_FAIL mixed_tensorrt_runtime wrong_major={wrong}"
        )

    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mb) * 1024 * 1024
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    # This stage is deliberately FP32-only. Do not enable FP16/INT8 until
    # output parity against the proven PyTorch path has passed.
    try:
        config.clear_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.INT8)
    except Exception:
        pass

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if args.engine.exists():
        args.engine.unlink()

    print(
        "STEP4_3_ENV "
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"tensorrt={version} gpu={_gpu_name()!r} input={input_shape} "
        f"workspace_mb={args.workspace_mb} precision=fp32",
        flush=True,
    )
    print(
        "STEP4_3_BUILD starting=true note='TensorRT tactic search may take several minutes'",
        flush=True,
    )

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_sec = time.perf_counter() - started
    if serialized is None:
        raise SystemExit("STEP4_3_FAIL build_serialized_network_returned_none")

    args.engine.write_bytes(bytes(serialized))
    if not args.engine.is_file() or args.engine.stat().st_size == 0:
        raise SystemExit(f"STEP4_3_FAIL engine_not_written={args.engine}")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise SystemExit("STEP4_3_FAIL engine_deserialize")

    bindings = []
    for index in range(int(engine.num_bindings)):
        bindings.append(
            {
                "index": index,
                "name": engine.get_binding_name(index),
                "is_input": bool(engine.binding_is_input(index)),
                "shape": [int(v) for v in engine.get_binding_shape(index)],
                "dtype": str(engine.get_binding_dtype(index)),
            }
        )

    report = {
        "stage": "4.3",
        "backend": "TensorRT 8.6.1 FP32",
        "onnx": str(args.onnx),
        "engine": str(args.engine),
        "engine_bytes": int(args.engine.stat().st_size),
        "engine_sha256": _sha256(args.engine),
        "build_seconds": round(build_sec, 3),
        "workspace_mb": int(args.workspace_mb),
        "precision": "fp32",
        "tensorrt": version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "gpu": _gpu_name(),
        "loaded_nvinfer": _loaded_nvinfer_paths(),
        "bindings": bindings,
    }
    report_path = args.engine.parent / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    size_mb = args.engine.stat().st_size / (1024.0 * 1024.0)
    print(
        "STEP4_3_RESULT "
        f"engine={args.engine} size_mb={size_mb:.1f} build_sec={build_sec:.1f} "
        f"sha256={report['engine_sha256'][:16]} bindings={bindings}",
        flush=True,
    )
    print(f"STEP4_3_JSON={report_path}", flush=True)
    print("STEP4_3_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
