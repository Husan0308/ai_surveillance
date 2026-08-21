#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import site
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 4.4b: compare RF-DETR ONNX Runtime FP32 and TensorRT 8.6 FP32 on one identical input tensor."
    )
    p.add_argument(
        "--onnx",
        type=Path,
        default=Path("artifacts/rfdetr_step4/onnx_800x448/rfdetr-small.onnx"),
    )
    p.add_argument(
        "--engine",
        type=Path,
        default=Path("artifacts/rfdetr_step4/trt86_fp32/rfdetr-small-800x448-fp32.engine"),
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/rfdetr_step4/parity/input_800x448.npy"),
    )
    p.add_argument("--threshold", type=float, default=0.18)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/rfdetr_step4/parity")
    )
    return p.parse_args()


def _load_cudart() -> ctypes.CDLL:
    candidates: list[Path] = []
    for root in map(Path, site.getsitepackages()):
        candidates.extend(sorted((root / "nvidia" / "cuda_runtime" / "lib").glob("libcudart.so.12*")))
    for path in candidates:
        try:
            return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    try:
        return ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise SystemExit(f"STEP4_4_FAIL cudart_load error={exc}") from exc


def _configure_cuda(lib: ctypes.CDLL) -> None:
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaDeviceSynchronize.restype = ctypes.c_int


def _cuda_check(code: int, op: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"CUDA {op} failed code={code}")


def _stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).reshape(-1)
    return {
        "max_abs": float(diff.max(initial=0.0)),
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "p99_abs": float(np.quantile(diff, 0.99)) if diff.size else 0.0,
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x))


def _selected(dets: np.ndarray, labels: np.ndarray, threshold: float) -> dict[tuple[int, int], tuple[float, np.ndarray]]:
    # RF-DETR ONNX exports raw normalized cxcywh boxes and unactivated logits.
    # The final column is the implicit no-object slot and is excluded from detection classes.
    boxes = np.asarray(dets[0], dtype=np.float32)
    logits = np.asarray(labels[0, :, :-1], dtype=np.float32)
    probs = _sigmoid(logits)
    flat = probs.reshape(-1)
    order = np.argsort(-flat, kind="stable")[: min(300, flat.size)]
    scores = flat[order]
    order = order[scores > float(threshold)]
    scores = scores[scores > float(threshold)]
    class_count = logits.shape[1]
    queries = order // class_count
    classes = order % class_count
    result: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for score, query, cls in zip(scores, queries, classes):
        result[(int(query), int(cls))] = (float(score), boxes[int(query)].copy())
    return result


def main() -> int:
    args = _args()
    for path, label in ((args.onnx, "onnx"), (args.engine, "engine"), (args.input, "input")):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"STEP4_4_FAIL {label}_not_found={path}")

    try:
        import onnxruntime as ort
        import tensorrt as trt
    except Exception as exc:
        raise SystemExit(f"STEP4_4_FAIL import error={type(exc).__name__}:{exc}")

    if not str(trt.__version__).startswith("8.6.1"):
        raise SystemExit(f"STEP4_4_FAIL wrong_tensorrt={trt.__version__}")

    input_array = np.load(args.input, allow_pickle=False)
    if input_array.dtype != np.float32 or input_array.shape != (1, 3, 448, 800):
        raise SystemExit(
            f"STEP4_4_FAIL bad_input expected=float32[1,3,448,800] got={input_array.dtype}{list(input_array.shape)}"
        )
    input_array = np.ascontiguousarray(input_array)

    # Reference: ONNX Runtime CPU. Same graph, same exact prepared input tensor.
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_outputs_list = session.run(None, {session.get_inputs()[0].name: input_array})
    onnx_outputs = {
        meta.name: np.asarray(value)
        for meta, value in zip(session.get_outputs(), onnx_outputs_list)
    }
    if "dets" not in onnx_outputs or "labels" not in onnx_outputs:
        raise SystemExit(f"STEP4_4_FAIL onnx_outputs={list(onnx_outputs)}")

    cudart = _load_cudart()
    _configure_cuda(cudart)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise SystemExit("STEP4_4_FAIL engine_deserialize")
    context = engine.create_execution_context()
    if context is None:
        raise SystemExit("STEP4_4_FAIL context_create")

    bindings: list[int] = [0] * int(engine.num_bindings)
    device_ptrs: list[ctypes.c_void_p] = []
    output_hosts: dict[str, np.ndarray] = {}

    try:
        for index in range(int(engine.num_bindings)):
            name = engine.get_binding_name(index)
            shape = tuple(int(v) for v in engine.get_binding_shape(index))
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(index)))
            if engine.binding_is_input(index):
                host = input_array
                if tuple(host.shape) != shape or host.dtype != dtype:
                    raise RuntimeError(
                        f"input_binding_mismatch name={name} engine={shape}/{dtype} host={host.shape}/{host.dtype}"
                    )
            else:
                host = np.empty(shape, dtype=dtype)
                output_hosts[name] = host

            ptr = ctypes.c_void_p()
            _cuda_check(cudart.cudaMalloc(ctypes.byref(ptr), host.nbytes), f"malloc:{name}")
            device_ptrs.append(ptr)
            bindings[index] = int(ptr.value or 0)
            if engine.binding_is_input(index):
                _cuda_check(
                    cudart.cudaMemcpy(
                        ptr,
                        ctypes.c_void_p(host.ctypes.data),
                        host.nbytes,
                        1,  # cudaMemcpyHostToDevice
                    ),
                    f"H2D:{name}",
                )

        for _ in range(max(0, int(args.warmup))):
            if not context.execute_v2(bindings):
                raise RuntimeError("execute_v2 returned false during warmup")
        _cuda_check(cudart.cudaDeviceSynchronize(), "sync_after_warmup")

        timings: list[float] = []
        for _ in range(max(1, int(args.runs))):
            started = time.perf_counter()
            if not context.execute_v2(bindings):
                raise RuntimeError("execute_v2 returned false")
            _cuda_check(cudart.cudaDeviceSynchronize(), "sync")
            timings.append((time.perf_counter() - started) * 1000.0)

        for index in range(int(engine.num_bindings)):
            if engine.binding_is_input(index):
                continue
            name = engine.get_binding_name(index)
            host = output_hosts[name]
            ptr = ctypes.c_void_p(bindings[index])
            _cuda_check(
                cudart.cudaMemcpy(
                    ctypes.c_void_p(host.ctypes.data),
                    ptr,
                    host.nbytes,
                    2,  # cudaMemcpyDeviceToHost
                ),
                f"D2H:{name}",
            )
        _cuda_check(cudart.cudaDeviceSynchronize(), "sync_after_copy")
    finally:
        for ptr in device_ptrs:
            if ptr.value:
                cudart.cudaFree(ptr)

    if "dets" not in output_hosts or "labels" not in output_hosts:
        raise SystemExit(f"STEP4_4_FAIL trt_outputs={list(output_hosts)}")
    if not all(np.isfinite(arr).all() for arr in output_hosts.values()):
        raise SystemExit("STEP4_4_FAIL trt_nonfinite")

    det_metrics = _stats(onnx_outputs["dets"], output_hosts["dets"])
    label_metrics = _stats(onnx_outputs["labels"], output_hosts["labels"])

    onnx_sel = _selected(onnx_outputs["dets"], onnx_outputs["labels"], args.threshold)
    trt_sel = _selected(output_hosts["dets"], output_hosts["labels"], args.threshold)
    onnx_keys = set(onnx_sel)
    trt_keys = set(trt_sel)
    shared = sorted(onnx_keys & trt_keys)
    only_onnx = sorted(onnx_keys - trt_keys)
    only_trt = sorted(trt_keys - onnx_keys)

    score_diffs = [abs(onnx_sel[k][0] - trt_sel[k][0]) for k in shared]
    box_diffs = [float(np.max(np.abs(onnx_sel[k][1] - trt_sel[k][1]))) for k in shared]
    semantic = {
        "threshold": float(args.threshold),
        "onnx_selected": len(onnx_keys),
        "trt_selected": len(trt_keys),
        "shared": len(shared),
        "only_onnx": only_onnx[:20],
        "only_trt": only_trt[:20],
        "max_score_abs": max(score_diffs, default=0.0),
        "max_box_abs_normalized": max(box_diffs, default=0.0),
    }

    # FP32 TensorRT kernels need not be bit-identical to ORT CPU. Production parity is
    # defined semantically: same selected query/class pairs and essentially identical boxes/scores.
    parity_pass = (
        not only_onnx
        and not only_trt
        and semantic["max_score_abs"] <= 0.02
        and semantic["max_box_abs_normalized"] <= 0.005
        and det_metrics["max_abs"] <= 0.005
    )

    report = {
        "stage": "4.4b",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "tensorrt": str(trt.__version__),
        "onnxruntime": str(ort.__version__),
        "onnx": str(args.onnx),
        "engine": str(args.engine),
        "input": str(args.input),
        "raw_diff": {"dets": det_metrics, "labels": label_metrics},
        "semantic": semantic,
        "trt_timing_ms": {
            "runs": len(timings),
            "mean": float(statistics.mean(timings)),
            "min": float(min(timings)),
            "max": float(max(timings)),
        },
        "pass": bool(parity_pass),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "trt_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "STEP4_4_RESULT "
        f"dets_max_abs={det_metrics['max_abs']:.6f} dets_mean_abs={det_metrics['mean_abs']:.6f} "
        f"labels_max_abs={label_metrics['max_abs']:.6f} labels_p99_abs={label_metrics['p99_abs']:.6f} "
        f"selected=onnx:{len(onnx_keys)}/trt:{len(trt_keys)}/shared:{len(shared)} "
        f"only_onnx={len(only_onnx)} only_trt={len(only_trt)} "
        f"score_max_abs={semantic['max_score_abs']:.6f} "
        f"box_max_abs={semantic['max_box_abs_normalized']:.6f} "
        f"trt_mean_ms={statistics.mean(timings):.2f}",
        flush=True,
    )
    print(f"STEP4_4_JSON={report_path}", flush=True)
    if not parity_pass:
        raise SystemExit("STEP4_4_FAIL parity_outside_limits")
    print("STEP4_4_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
