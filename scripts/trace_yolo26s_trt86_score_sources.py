#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
MODEL23_PREFIX = "/model.23/"
MODULE_RE = re.compile(r"^(/model\.\d+)(?:/|$)")
EXPECTED_SCALES = {(48, 84): 0, (24, 42): 1, (12, 21): 2}


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def shape_tuple(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def module_of(name: str) -> str:
    match = MODULE_RE.match(name or "")
    return match.group(1) if match else "UNSCOPED"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Trace YOLO26 Detect-head feature sources by graph boundary edges"
    )
    ap.add_argument("--onnx", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-e2e.onnx")
    args = ap.parse_args()

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"V11_SCORE_PATH FAIL TensorRT 8.6.1 required, got {trt.__version__}")

    onnx_path = resolve(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"V11_SCORE_PATH FAIL ONNX missing={onnx_path}")

    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("V11_SCORE_PATH FAIL ONNX parse:\n" + "\n".join(errors))

    producer: dict[str, tuple[int, object]] = {}
    module_ranges: dict[str, list[int]] = {}
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        name = layer.name or f"layer_{idx}"
        module = module_of(name)
        if module.startswith("/model."):
            module_ranges.setdefault(module, []).append(idx)
        for out_idx in range(layer.num_outputs):
            tensor = layer.get_output(out_idx)
            if tensor is not None and tensor.name:
                producer[tensor.name] = (idx, layer)

    # Discover the three pyramid features by graph topology, not by fragile layer names:
    # find rank-4 floating tensors that cross from outside /model.23 into /model.23.
    # For fixed 672x384 input the three Detect scales are 48x84, 24x42, 12x21.
    boundary: dict[str, dict] = {}
    model23_layers = 0
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        layer_name = layer.name or f"layer_{idx}"
        if not layer_name.startswith(MODEL23_PREFIX):
            continue
        model23_layers += 1
        for in_idx in range(layer.num_inputs):
            tensor = layer.get_input(in_idx)
            if tensor is None or not tensor.name:
                continue
            shape = shape_tuple(tensor)
            if len(shape) != 4 or tuple(shape[-2:]) not in EXPECTED_SCALES:
                continue
            source = producer.get(tensor.name)
            if source is None:
                producer_idx = -1
                producer_name = "NETWORK_INPUT"
            else:
                producer_idx, producer_layer = source
                producer_name = producer_layer.name or f"layer_{producer_idx}"
                if producer_name.startswith(MODEL23_PREFIX):
                    continue
            row = boundary.setdefault(
                tensor.name,
                {
                    "tensor": tensor,
                    "shape": shape,
                    "producer_idx": producer_idx,
                    "producer_name": producer_name,
                    "consumers": [],
                },
            )
            row["consumers"].append((idx, layer_name, in_idx))

    entries = sorted(
        boundary.values(),
        key=lambda row: EXPECTED_SCALES.get(tuple(row["shape"][-2:]), 99),
    )
    print(
        f"V11_SCORE_PATH_START trt={trt.__version__} layers={network.num_layers} "
        f"model23_layers={model23_layers} boundary_entries={len(entries)} discovery=graph_edges",
        flush=True,
    )
    if len(entries) != 3:
        for row in entries:
            print(
                "V11_SCORE_PATH_CANDIDATE "
                f"input={row['tensor'].name!r} shape={row['shape']} "
                f"producer_index={row['producer_idx']} producer={row['producer_name']!r} "
                f"consumers={len(row['consumers'])}",
                flush=True,
            )
        raise SystemExit(f"V11_SCORE_PATH FAIL expected_entries=3 got={len(entries)}")

    source_modules: set[str] = set()
    seen_scales: set[int] = set()
    for row in entries:
        tensor = row["tensor"]
        shape = row["shape"]
        scale = EXPECTED_SCALES[tuple(shape[-2:])]
        if scale in seen_scales:
            raise SystemExit(f"V11_SCORE_PATH FAIL duplicate_scale={scale} shape={shape}")
        seen_scales.add(scale)

        producer_idx = int(row["producer_idx"])
        producer_name = str(row["producer_name"])
        module = module_of(producer_name)
        indices = module_ranges.get(module, [])
        module_start = min(indices) if indices else producer_idx
        module_end = max(indices) if indices else producer_idx
        if module.startswith("/model."):
            source_modules.add(module)

        consumers = row["consumers"]
        consumer_text = ";".join(
            f"{idx}:{name}:in{in_idx}" for idx, name, in_idx in consumers[:8]
        )
        print(
            "V11_SCORE_PATH_SOURCE "
            f"scale={scale} input={tensor.name!r} input_shape={shape} "
            f"producer_index={producer_idx} producer={producer_name!r} "
            f"producer_module={module} module_range={module_start}:{module_end} "
            f"consumers={len(consumers)} consumer_layers={consumer_text!r}",
            flush=True,
        )

    for module in sorted(source_modules, key=lambda s: int(s.split('.')[-1])):
        indices = module_ranges[module]
        print(
            f"V11_SCORE_PATH_MODULE module={module} start={min(indices)} end={max(indices)} layers={len(indices)}",
            flush=True,
        )

    if seen_scales != {0, 1, 2}:
        raise SystemExit(f"V11_SCORE_PATH FAIL scales={sorted(seen_scales)} expected=[0,1,2]")

    print(
        "V11_SCORE_PATH_RESULT status=PASS "
        f"entries={len(entries)} discovery=graph_edges "
        f"source_modules={','.join(sorted(source_modules, key=lambda s: int(s.split('.')[-1])))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
