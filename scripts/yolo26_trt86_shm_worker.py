#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import tensorrt as trt


INPUT_W = 672
INPUT_H = 384
EXPECTED_INPUT = (1, 3, INPUT_H, INPUT_W)
EXPECTED_OUTPUT = (1, 300, 6)
CUDA_STREAM_NON_BLOCKING = 1


def emit(obj) -> None:
    print(json.dumps(obj, separators=(",", ":")), flush=True)


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
    lib.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaStreamCreateWithFlags.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    return lib


class Runner:
    def __init__(self, engine_path: Path):
        if not str(trt.__version__).startswith("8.6.1"):
            raise RuntimeError(f"TensorRT 8.6.1 required, got {trt.__version__}")
        if not engine_path.is_file():
            raise FileNotFoundError(f"engine missing: {engine_path}")

        self.cuda = load_cudart()
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("engine deserialize failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("execution context failed")

        inputs: list[int] = []
        outputs: list[int] = []
        for i in range(self.engine.num_bindings):
            (inputs if self.engine.binding_is_input(i) else outputs).append(i)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one input/one output, got inputs={inputs} outputs={outputs}"
            )

        self.input_index = inputs[0]
        self.output_index = outputs[0]
        self.input_shape = tuple(
            int(v) for v in self.context.get_binding_shape(self.input_index)
        )
        self.output_shape = tuple(
            int(v) for v in self.context.get_binding_shape(self.output_index)
        )
        if self.input_shape != EXPECTED_INPUT:
            raise RuntimeError(
                f"unexpected input shape={self.input_shape}; expected={EXPECTED_INPUT}"
            )
        if self.output_shape != EXPECTED_OUTPUT:
            raise RuntimeError(
                f"unexpected YOLO26 E2E output shape={self.output_shape}; "
                f"expected={EXPECTED_OUTPUT}"
            )

        input_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.input_index)))
        output_dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(self.output_index)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError(f"unexpected TRT input dtype={input_dtype}; expected=float32")
        if output_dtype != np.dtype(np.float32):
            raise RuntimeError(f"unexpected TRT output dtype={output_dtype}; expected=float32")

        self.x = np.empty(self.input_shape, dtype=np.float32)
        self.y = np.empty(self.output_shape, dtype=np.float32)
        self.in_dev = ctypes.c_void_p()
        self.out_dev = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        cuda_check(
            self.cuda.cudaMalloc(ctypes.byref(self.in_dev), self.x.nbytes),
            "cudaMalloc input",
        )
        cuda_check(
            self.cuda.cudaMalloc(ctypes.byref(self.out_dev), self.y.nbytes),
            "cudaMalloc output",
        )
        cuda_check(
            self.cuda.cudaStreamCreateWithFlags(
                ctypes.byref(self.stream), CUDA_STREAM_NON_BLOCKING
            ),
            "cudaStreamCreateWithFlags",
        )
        if not self.stream.value:
            raise RuntimeError("CUDA nonblocking stream creation returned null")

        self.bindings = [0] * self.engine.num_bindings
        self.bindings[self.input_index] = int(self.in_dev.value)
        self.bindings[self.output_index] = int(self.out_dev.value)
        self._shm = None
        self._shm_name = ""

    def _attach(self, name: str):
        if self._shm is not None and self._shm_name == name:
            return self._shm
        if self._shm is not None:
            self._shm.close()
        self._shm = shared_memory.SharedMemory(name=name, create=False)
        self._shm_name = name
        return self._shm

    def preprocess_bgr(self, frame: np.ndarray) -> dict:
        if frame.shape != (INPUT_H, INPUT_W, 3):
            raise RuntimeError(
                f"unexpected BGR frame shape={frame.shape}; "
                f"expected={(INPUT_H, INPUT_W, 3)}"
            )
        if frame.dtype != np.uint8:
            raise RuntimeError(f"unexpected frame dtype={frame.dtype}")

        rgb = np.ascontiguousarray(frame[..., ::-1])
        chw = rgb.transpose(2, 0, 1)
        np.multiply(chw, 1.0 / 255.0, out=self.x[0], casting="unsafe")

        means = frame.reshape(-1, 3).mean(axis=0)
        return {
            "input_min": int(frame.min()),
            "input_max": int(frame.max()),
            "bgr_mean": [round(float(v), 2) for v in means],
        }

    def infer(self, frame: np.ndarray, conf: float, max_det: int):
        conf = min(1.0, max(0.0, float(conf)))
        max_det = max(1, min(300, int(max_det)))

        started_total = time.perf_counter()
        started_prep = time.perf_counter()
        input_health = self.preprocess_bgr(frame)
        prep_ms = (time.perf_counter() - started_prep) * 1000.0

        # Input is only ~3 MiB and there is no preceding work on our sidecar stream,
        # so a synchronous H2D copy is predictable and avoids pageable-host async
        # staging complexity.  The expensive network execution itself is queued on
        # a dedicated non-default stream below.
        cuda_check(
            self.cuda.cudaMemcpy(
                self.in_dev,
                ctypes.c_void_p(self.x.ctypes.data),
                self.x.nbytes,
                1,
            ),
            "H2D",
        )

        started_trt = time.perf_counter()
        if not self.context.execute_async_v2(
            self.bindings,
            int(self.stream.value),
        ):
            raise RuntimeError("execute_async_v2=false")
        # Synchronize only this TRT stream.  Do not cudaDeviceSynchronize(): that
        # introduces a context-wide barrier and is unnecessary for a single
        # serialized sidecar request.
        cuda_check(
            self.cuda.cudaStreamSynchronize(self.stream),
            "TRT stream sync",
        )
        trt_ms = (time.perf_counter() - started_trt) * 1000.0

        cuda_check(
            self.cuda.cudaMemcpy(
                ctypes.c_void_p(self.y.ctypes.data),
                self.out_dev,
                self.y.nbytes,
                2,
            ),
            "D2H",
        )

        # YOLO26 end-to-end detection contract:
        # (N, 300, 6) -> [x1, y1, x2, y2, confidence, class_id].
        # No external NMS is applied.
        pred = self.y[0]
        finite = np.isfinite(pred).all(axis=1)
        finite_pred = pred[finite]
        nonfinite = int((~finite).sum())

        raw_max_conf = None
        raw_person_max_conf = None
        raw_person_rows = 0
        raw_above_conf = 0
        raw_person_above_conf = 0
        raw_box_max = None

        if finite_pred.size:
            scores = finite_pred[:, 4]
            classes = np.rint(finite_pred[:, 5]).astype(np.int32)
            person = classes == 0
            positive_person = person & (scores > 0.0)
            raw_max_conf = float(np.max(scores))
            raw_above_conf = int(np.sum(scores >= conf))
            raw_person_rows = int(np.sum(positive_person))
            raw_person_above_conf = int(np.sum(person & (scores >= conf)))
            if np.any(person):
                raw_person_max_conf = float(np.max(scores[person]))
            raw_box_max = float(np.max(np.abs(finite_pred[:, :4])))

        rows: list[list[float]] = []
        for row in finite_pred:
            x1, y1, x2, y2, score, cls = (float(v) for v in row)
            if score < conf or int(round(cls)) != 0:
                continue
            x1 = min(float(INPUT_W - 1), max(0.0, x1))
            x2 = min(float(INPUT_W - 1), max(0.0, x2))
            y1 = min(float(INPUT_H - 1), max(0.0, y1))
            y2 = min(float(INPUT_H - 1), max(0.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2, y2, score])

        rows.sort(key=lambda item: item[4], reverse=True)
        if len(rows) > max_det:
            rows = rows[:max_det]

        total_ms = (time.perf_counter() - started_total) * 1000.0
        health = {
            **input_health,
            "raw_max_conf": raw_max_conf,
            "raw_person_max_conf": raw_person_max_conf,
            "raw_person_rows": raw_person_rows,
            "raw_above_conf": raw_above_conf,
            "raw_person_above_conf": raw_person_above_conf,
            "raw_box_max": raw_box_max,
            "nonfinite_rows": nonfinite,
            "execution": "execute_async_v2",
            "stream": "nonblocking",
        }
        return rows, prep_ms, trt_ms, total_ms, health

    def infer_from_shm(self, name: str, conf: float, max_det: int):
        shm = self._attach(name)
        needed = INPUT_H * INPUT_W * 3
        if shm.size < needed:
            raise RuntimeError(f"shared memory too small: {shm.size} < {needed}")
        frame = np.ndarray(
            (INPUT_H, INPUT_W, 3),
            dtype=np.uint8,
            buffer=shm.buf,
        )
        return self.infer(frame, conf, max_det)

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
        if self.stream.value:
            try:
                self.cuda.cudaStreamDestroy(self.stream)
            finally:
                self.stream = ctypes.c_void_p()
        if self.in_dev.value:
            self.cuda.cudaFree(self.in_dev)
        if self.out_dev.value:
            self.cuda.cudaFree(self.out_dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()
    runner = Runner(Path(args.engine))

    emit(
        {
            "type": "ready",
            "tensorrt": trt.__version__,
            "engine": str(Path(args.engine).resolve()),
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "transport": "shm-bgr",
            "execution": "execute_async_v2",
            "stream": "nonblocking",
        }
    )

    try:
        for line in sys.stdin:
            req = {}
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    break
                request_id = req.get("id")
                conf = float(req.get("conf", 0.05))
                max_det = int(req.get("max_det", 40))
                boxes, prep_ms, trt_ms, total_ms, health = runner.infer_from_shm(
                    str(req["shm_name"]),
                    conf,
                    max_det,
                )
                emit(
                    {
                        "id": request_id,
                        "ok": True,
                        "boxes": boxes,
                        "prep_ms": prep_ms,
                        "trt_ms": trt_ms,
                        "total_ms": total_ms,
                        "health": health,
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "id": req.get("id"),
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
