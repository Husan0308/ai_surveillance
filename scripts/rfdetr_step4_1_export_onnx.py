#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4.1: export the proven RF-DETR-S detector to fixed-shape ONNX before TensorRT build."
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rfdetr_step4/onnx_800x448"),
    )
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("STEP4_1_FAIL invalid_shape")
    if args.width % 32 or args.height % 32:
        raise SystemExit(
            f"STEP4_1_FAIL shape_must_be_divisible_by_32 got={args.width}x{args.height}"
        )

    import torch
    import rfdetr
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise SystemExit("STEP4_1_FAIL torch_cuda_unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "inference_model.onnx"
    if onnx_path.exists():
        onnx_path.unlink()

    print(
        "STEP4_1_ENV "
        f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)!r} "
        f"sm={'.'.join(map(str, torch.cuda.get_device_capability(0)))} "
        f"rfdetr={getattr(rfdetr, '__version__', 'unknown')} "
        f"shape={args.width}x{args.height}",
        flush=True,
    )

    model = RFDETRSmall(device="cuda:0")
    result = model.export(
        format="onnx",
        output_dir=str(args.output_dir),
        shape=(int(args.height), int(args.width)),
        batch_size=1,
        dynamic_batch=False,
        opset_version=int(args.opset),
        verbose=False,
    )

    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        raise SystemExit(
            f"STEP4_1_FAIL onnx_missing expected={onnx_path} export_result={result!r}"
        )

    onnx_check = "not_checked"
    input_info = []
    output_info = []
    try:
        import onnx

        graph = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(graph)
        onnx_check = "pass"
        for value in graph.graph.input:
            dims = []
            tensor_type = value.type.tensor_type
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(int(dim.dim_value))
                elif dim.dim_param:
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
            input_info.append({"name": value.name, "shape": dims})
        for value in graph.graph.output:
            dims = []
            tensor_type = value.type.tensor_type
            for dim in tensor_type.shape.dim:
                if dim.dim_value:
                    dims.append(int(dim.dim_value))
                elif dim.dim_param:
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
            output_info.append({"name": value.name, "shape": dims})
    except ImportError:
        onnx_check = "onnx_module_missing"
    except Exception as exc:
        raise SystemExit(
            f"STEP4_1_FAIL onnx_validation error={type(exc).__name__}:{exc}"
        )

    report = {
        "stage": "4.1",
        "backend": "RF-DETR-S ONNX fixed-shape export",
        "shape_hw": [int(args.height), int(args.width)],
        "batch_size": 1,
        "dynamic_batch": False,
        "opset": int(args.opset),
        "onnx_path": str(onnx_path),
        "onnx_bytes": int(onnx_path.stat().st_size),
        "onnx_sha256": _sha256(onnx_path),
        "onnx_check": onnx_check,
        "inputs": input_info,
        "outputs": output_info,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "cuda": str(torch.version.cuda),
        },
    }
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    size_mb = onnx_path.stat().st_size / (1024.0 * 1024.0)
    print(
        "STEP4_1_RESULT "
        f"onnx={onnx_path} size_mb={size_mb:.1f} sha256={report['onnx_sha256'][:16]} "
        f"onnx_check={onnx_check} inputs={input_info} outputs={output_info}",
        flush=True,
    )
    print(f"STEP4_1_JSON={report_path}", flush=True)
    print("STEP4_1_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
