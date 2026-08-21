#!/usr/bin/env python3
from __future__ import annotations

import ctypes.util
import platform
import subprocess
import sys


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except Exception:
        return "unavailable"


def main() -> int:
    print(
        "STEP4_ENV "
        f"python={platform.python_version()} executable={sys.executable}",
        flush=True,
    )

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(
                "STEP4_GPU "
                f"name={name!r} sm={cap[0]}.{cap[1]} "
                f"torch={torch.__version__} torch_cuda={torch.version.cuda}",
                flush=True,
            )
        else:
            print("STEP4_GPU cuda_unavailable", flush=True)
            cap = (0, 0)
    except Exception as exc:
        print(f"STEP4_GPU torch_error={type(exc).__name__}:{exc}", flush=True)
        cap = (0, 0)

    trt_version = None
    try:
        import tensorrt as trt

        trt_version = str(trt.__version__)
        print(f"STEP4_TRT python_import=ok version={trt_version}", flush=True)
    except Exception as exc:
        print(
            f"STEP4_TRT python_import=missing error={type(exc).__name__}:{exc}",
            flush=True,
        )

    lib = ctypes.util.find_library("nvinfer")
    print(f"STEP4_SYSTEM nvinfer={lib or 'not_found'}", flush=True)
    print(f"STEP4_NVCC {_run(['nvcc', '--version']).splitlines()[-1] if _run(['nvcc', '--version']) != 'unavailable' else 'unavailable'}", flush=True)

    sm = cap[0] * 10 + cap[1]
    if sm and sm <= 61:
        if trt_version:
            try:
                major = int(trt_version.split('.', 1)[0])
            except Exception:
                major = -1
            if major >= 10:
                print(
                    "STEP4_BLOCKED reason=current_tensorrt_does_not_support_pascal_sm61 "
                    "required=isolated_tensorrt_8.6",
                    flush=True,
                )
                return 2
        if sys.version_info >= (3, 12):
            print(
                "STEP4_PLAN gpu=pascal_sm61 target_tensorrt=8.6 "
                "python_target=3.10_or_3.11 isolation=required",
                flush=True,
            )
        else:
            print(
                "STEP4_PLAN gpu=pascal_sm61 target_tensorrt=8.6 "
                "python_target=current_may_be_usable isolation=required",
                flush=True,
            )
    else:
        print("STEP4_PLAN inspect_output_before_install", flush=True)

    print("STEP4_PASS preflight_only=true", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
