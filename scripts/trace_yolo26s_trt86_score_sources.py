#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
ENTRY_RE = re.compile(r"^/model\.23/one2one_cv3\.(\d+)/one2one_cv3\.\1\.0/conv/Conv$")
MODULE_RE = re.compile(r"^(/model\.\d+)(?:/|$)")


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def shape_tuple(tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def main() -> int:
    ap = argparse.ArgumentParser(description="Trace YOLO26 one-to-one classification head feature sources")
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
        match = MODULE_RE.match(name)
        if match:
            module_ranges.setdefault(match.group(1), []).append(idx)
        for out_idx in range(layer.num_outputs):
            tensor = layer.get_output(out_idx)
            if tensor is not None and tensor.name:
                producer[tensor.name] = (idx, layer)

    entries: list[tuple[int, int, object]] = []
    for idx in range(network.num_layers):
        layer = network.get_layer(idx)
        match = ENTRY_RE.match(layer.name or "")
        if match:
            entries.append((int(match.group(1)), idx, layer))

    entries.sort(key=lambda row: row[0])
    print(f"V11_SCORE_PATH_START trt={trt.__version__} layers={network.num_layers} entries={len(entries)}", flush=True)
    if len(entries) != 3:
        raise SystemExit(f"V11_SCORE_PATH FAIL expected_entries=3 got={len(entries)}")

    source_modules: set[str] = set()
    for scale, entry_idx, layer in entries:
        if layer.num_inputs < 1:
            raise SystemExit(f"V11_SCORE_PATH FAIL entry_no_input index={entry_idx} name={layer.name!r}")
        tensor = layer.get_input(0)
        if tensor is None or not tensor.name:
            raise SystemExit(f"V11_SCORE_PATH FAIL entry_input_missing index={entry_idx} name={layer.name!r}")
        source = producer.get(tensor.name)
        if source is None:
            producer_idx = -1
            producer_name = "NETWORK_INPUT"
            module = "NETWORK_INPUT"
            module_start = module_end = -1
        else:
            producer_idx, producer_layer = source
            producer_name = producer_layer.name or f"layer_{producer_idx}"
            mm = MODULE_RE.match(producer_name)
            module = mm.group(1) if mm else "UNSCOPED"
            indices = module_ranges.get(module, [])
            module_start = min(indices) if indices else producer_idx
            module_end = max(indices) if indices else producer_idx
            if module.startswith("/model."):
                source_modules.add(module)
        print(
            "V11_SCORE_PATH_SOURCE "
            f"scale={scale} entry_index={entry_idx} entry={layer.name!r} "
            f"input={tensor.name!r} input_shape={shape_tuple(tensor)} "
            f"producer_index={producer_idx} producer={producer_name!r} "
            f"producer_module={module} module_range={module_start}:{module_end}",
            flush=True,
        )

    for module in sorted(source_modules, key=lambda s: int(s.split('.')[-1])):
        indices = module_ranges[module]
        print(
            f"V11_SCORE_PATH_MODULE module={module} start={min(indices)} end={max(indices)} layers={len(indices)}",
            flush=True,
        )

    print(
        "V11_SCORE_PATH_RESULT status=PASS "
        f"entries={len(entries)} source_modules={','.join(sorted(source_modules, key=lambda s: int(s.split('.')[-1])))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
