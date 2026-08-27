#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def dims_tuple(dims) -> tuple[int, ...]:
    return tuple(int(v) for v in dims)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect YOLO26s TensorRT network layers before mixed-precision build")
    ap.add_argument("--onnx", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-e2e.onnx")
    ap.add_argument("--tail", type=int, default=80)
    args = ap.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_LAYER_INSPECT FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx = resolve(args.onnx)
    if not onnx.is_file():
        raise SystemExit(f"V11_LAYER_INSPECT FAIL ONNX missing={onnx}")

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("V11_LAYER_INSPECT FAIL ONNX parse:\n" + "\n".join(errors))

    total = int(network.num_layers)
    tail = max(1, min(total, int(args.tail)))
    start = total - tail
    print(
        f"V11_LAYER_INSPECT_START trt={trt.__version__} layers={total} tail={tail} "
        f"input={dims_tuple(network.get_input(0).shape)} output={dims_tuple(network.get_output(0).shape)}",
        flush=True,
    )

    sensitive_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.SOFTMAX,
        trt.LayerType.TOPK,
        trt.LayerType.GATHER,
        trt.LayerType.REDUCE,
        trt.LayerType.ELEMENTWISE,
    }

    for i in range(start, total):
        layer = network.get_layer(i)
        outs = []
        for j in range(layer.num_outputs):
            t = layer.get_output(j)
            if t is None:
                continue
            outs.append(f"{t.name}:{dims_tuple(t.shape)}")
        marker = "SENSITIVE" if layer.type in sensitive_types else "-"
        print(
            f"V11_LAYER index={i:04d} type={layer.type.name} marker={marker} "
            f"name={layer.name!r} outputs={';'.join(outs)}",
            flush=True,
        )

    print(f"V11_LAYER_INSPECT_RESULT status=PASS layers={total} printed={tail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
