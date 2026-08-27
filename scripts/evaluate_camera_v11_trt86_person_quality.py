#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUT = (1, 3, 384, 672)
EXPECTED_OUTPUT = (1, 300, 6)


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def cuda_check(code, name: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{name}: cuda={code}")


def load_cudart():
    path = ctypes.util.find_library("cudart")
    if not path:
        raise RuntimeError("libcudart not found")
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    return lib


def _read_token(f) -> bytes:
    token = bytearray()
    while True:
        ch = f.read(1)
        if not ch:
            return bytes(token)
        if ch == b"#":
            f.readline()
            continue
        if ch.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(ch)


def load_ppm_rgb(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = _read_token(f)
        if magic != b"P6":
            raise RuntimeError(f"{path}: expected P6")
        width = int(_read_token(f))
        height = int(_read_token(f))
        maxval = int(_read_token(f))
        if (width, height, maxval) != (672, 384, 255):
            raise RuntimeError(f"{path}: geometry/maxval={(width,height,maxval)}")
        raw = f.read(width * height * 3)
    if len(raw) != width * height * 3:
        raise RuntimeError(f"{path}: short PPM payload")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = aa + bb - inter
    return inter / den if den > 0.0 else 0.0


def score_image(preds: list[list[float]], gt: list[list[float]], gate: float) -> tuple[int, int, int]:
    used: set[int] = set()
    tp = 0
    for pred in sorted(preds, key=lambda row: row[4], reverse=True):
        best_i = -1
        best = gate
        for idx, box in enumerate(gt):
            if idx in used:
                continue
            value = iou(pred[:4], box)
            if value >= best:
                best = value
                best_i = idx
        if best_i >= 0:
            used.add(best_i)
            tp += 1
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gt) - tp)
    return tp, fp, fn


class Runner:
    def __init__(self, engine_path: Path) -> None:
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
        self.path = engine_path
        self.cuda = load_cudart()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"deserialize failed: {engine_path}")
        self.context = self.engine.create_execution_context()
        inputs = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        outputs = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"bindings inputs={inputs} outputs={outputs}")
        self.input_index = inputs[0]
        self.output_index = outputs[0]
        self.input_shape = tuple(int(v) for v in self.context.get_binding_shape(self.input_index))
        self.output_shape = tuple(int(v) for v in self.context.get_binding_shape(self.output_index))
        if self.input_shape != EXPECTED_INPUT or self.output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(f"shapes={self.input_shape}/{self.output_shape}")
        self.input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index)))
        self.output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_index)))
        self.x = np.empty(self.input_shape, dtype=self.input_dtype)
        self.y = np.empty(self.output_shape, dtype=self.output_dtype)
        self.in_dev = ctypes.c_void_p()
        self.out_dev = ctypes.c_void_p()
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.in_dev), self.x.nbytes), "cudaMalloc input")
        cuda_check(self.cuda.cudaMalloc(ctypes.byref(self.out_dev), self.y.nbytes), "cudaMalloc output")
        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.in_dev.value)
        self.bindings[self.output_index] = int(self.out_dev.value)

    def infer(self, rgb: np.ndarray, conf: float) -> list[list[float]]:
        chw = rgb.transpose(2, 0, 1)
        np.multiply(chw, 1.0 / 255.0, out=self.x[0], casting="unsafe")
        cuda_check(
            self.cuda.cudaMemcpy(self.in_dev, ctypes.c_void_p(self.x.ctypes.data), self.x.nbytes, 1),
            "H2D",
        )
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("execute_v2=false")
        cuda_check(self.cuda.cudaDeviceSynchronize(), "infer sync")
        cuda_check(
            self.cuda.cudaMemcpy(ctypes.c_void_p(self.y.ctypes.data), self.out_dev, self.y.nbytes, 2),
            "D2H",
        )
        rows: list[list[float]] = []
        for row in self.y[0]:
            if not np.isfinite(row).all():
                continue
            x1, y1, x2, y2, score, cls = (float(v) for v in row)
            if int(round(cls)) != 0 or score < conf:
                continue
            x1 = min(671.0, max(0.0, x1))
            x2 = min(671.0, max(0.0, x2))
            y1 = min(383.0, max(0.0, y1))
            y2 = min(383.0, max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2, y2, score])
        return rows

    def close(self) -> None:
        if self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
        if self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)


def metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def evaluate(runner: Runner, entries: list[dict], base: Path, conf: float, gate: float) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for idx, entry in enumerate(entries, 1):
        rgb = load_ppm_rgb(base / entry["file"])
        preds = runner.infer(rgb, conf)
        a, b, c = score_image(preds, entry["person_boxes"], gate)
        tp += a
        fp += b
        fn += c
        if idx <= 5 or idx % 100 == 0 or idx == len(entries):
            print(
                f"V11_PERSON_QUALITY_PROGRESS engine={runner.path.name} image={idx}/{len(entries)} tp={tp} fp={fp} fn={fn}",
                flush=True,
            )
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description="Relative person detection quality gate for FP32 vs INT8 TRT8.6 engines")
    ap.add_argument("--fp32", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine")
    ap.add_argument("--int8", default="artifacts/yolo26s_trt86/yolo26s-672x384-b1-int8-trt86.engine")
    ap.add_argument("--quality-dir", default="artifacts/yolo26s_trt86/person_quality_b1")
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--max-drop", type=float, default=0.03)
    args = ap.parse_args()

    fp32_path = resolve(args.fp32)
    int8_path = resolve(args.int8)
    quality = resolve(args.quality_dir)
    manifest_path = quality / "person_gt.json"
    for path in (fp32_path, int8_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"V11_PERSON_QUALITY FAIL missing={path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("images") or []
    if len(entries) < 100:
        raise SystemExit(f"V11_PERSON_QUALITY FAIL images={len(entries)} expected>=100")

    results: dict[str, tuple[int, int, int, float, float, float]] = {}
    for label, path in (("fp32", fp32_path), ("int8", int8_path)):
        runner = Runner(path)
        try:
            tp, fp, fn = evaluate(runner, entries, quality, float(args.conf), float(args.iou))
        finally:
            runner.close()
        precision, recall, f1 = metrics(tp, fp, fn)
        results[label] = (tp, fp, fn, precision, recall, f1)
        print(
            "V11_PERSON_QUALITY_RESULT "
            f"engine={label} file={path.name} images={len(entries)} conf={args.conf:.2f} iou={args.iou:.2f} "
            f"tp={tp} fp={fp} fn={fn} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}",
            flush=True,
        )

    fp32 = results["fp32"]
    int8 = results["int8"]
    p_drop = fp32[3] - int8[3]
    r_drop = fp32[4] - int8[4]
    f1_drop = fp32[5] - int8[5]
    limit = max(0.0, float(args.max_drop))
    reasons = []
    if p_drop > limit:
        reasons.append(f"precision_drop={p_drop:.4f}")
    if r_drop > limit:
        reasons.append(f"recall_drop={r_drop:.4f}")
    if f1_drop > limit:
        reasons.append(f"f1_drop={f1_drop:.4f}")

    if reasons:
        print(
            "V11_PERSON_QUALITY_GATE status=FAIL "
            f"max_drop={limit:.4f} precision_drop={p_drop:.4f} recall_drop={r_drop:.4f} f1_drop={f1_drop:.4f} "
            f"reasons={';'.join(reasons)}",
            flush=True,
        )
        return 1

    print(
        "V11_PERSON_QUALITY_GATE status=PASS "
        f"max_drop={limit:.4f} precision_drop={p_drop:.4f} recall_drop={r_drop:.4f} f1_drop={f1_drop:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
