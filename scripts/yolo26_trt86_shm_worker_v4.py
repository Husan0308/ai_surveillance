#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo26_trt86_shm_worker import (  # noqa: E402
    INPUT_H,
    INPUT_W,
    Runner as BaseRunner,
    cuda_check,
    emit,
    trt,
)


class Runner(BaseRunner):
    """Production TRT8.6 runner optimized for low end-to-end latency.

    Differences from the diagnostic runners:
      * no full-frame channel-spread/histogram work on every inference;
      * RGB/NCHW conversion writes directly into the preallocated tensor;
      * TensorRT runs on a non-default CUDA stream;
      * H2D -> inference -> D2H is one async stream chain with one sync point;
      * fixed 672x378 + 3px/3px YOLO letterbox padding remains exact.
    """

    def __init__(self, engine_path: Path):
        super().__init__(engine_path)
        self._infer_n = 0
        self.stream = ctypes.c_void_p()

        self.cuda.cudaStreamCreateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.cuda.cudaStreamCreateWithFlags.restype = ctypes.c_int
        self.cuda.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.cuda.cudaStreamDestroy.restype = ctypes.c_int
        self.cuda.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cuda.cudaStreamSynchronize.restype = ctypes.c_int
        self.cuda.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.cuda.cudaMemcpyAsync.restype = ctypes.c_int

        # cudaStreamNonBlocking = 1. TensorRT explicitly recommends a non-default
        # stream to avoid implicit device-wide synchronization.
        cuda_check(
            self.cuda.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1),
            "cudaStreamCreateWithFlags",
        )

    def _attach(self, name: str):
        if self._shm is not None and self._shm_name == name:
            return self._shm
        if self._shm is not None:
            self._shm.close()
        shm = shared_memory.SharedMemory(name=name, create=False)
        # This process is not the owner. Avoid Python <=3.12 trying to unlink the
        # bridge-owned SHM segment on interpreter shutdown.
        try:
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        self._shm = shm
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

        # Avoid np.ascontiguousarray(frame[..., ::-1]) and the temporary RGB
        # allocation. Convert each BGR plane directly into the persistent NCHW
        # float32 TensorRT input buffer.
        scale = 1.0 / 255.0
        np.multiply(frame[:, :, 2], scale, out=self.x[0, 0], casting="unsafe")
        np.multiply(frame[:, :, 1], scale, out=self.x[0, 1], casting="unsafe")
        np.multiply(frame[:, :, 0], scale, out=self.x[0, 2], casting="unsafe")

        self._infer_n += 1
        diagnostics = self._infer_n <= 3 or self._infer_n % 20 == 0
        if not diagnostics:
            return {
                "input_min": None,
                "input_max": None,
                "bgr_mean": None,
            }

        # Diagnostics are sampled only when the bridge is going to print them.
        means = frame.reshape(-1, 3).mean(axis=0)
        return {
            "input_min": int(frame.min()),
            "input_max": int(frame.max()),
            "bgr_mean": [round(float(v), 2) for v in means],
        }

    def infer(self, frame: np.ndarray, conf: float, max_det: int):
        if frame.shape != (INPUT_H, INPUT_W, 3):
            raise RuntimeError(f"unexpected letterbox frame shape={frame.shape}")

        # nvvideoconvert places the 16:9 image at 672x378. Ultralytics export
        # semantics require value 114 in the 3-pixel top/bottom bars.
        frame[:3, :, :] = 114
        frame[381:, :, :] = 114

        conf = min(1.0, max(0.0, float(conf)))
        max_det = max(1, min(300, int(max_det)))

        started_total = time.perf_counter()
        started_prep = time.perf_counter()
        input_health = self.preprocess_bgr(frame)
        prep_ms = (time.perf_counter() - started_prep) * 1000.0

        started_gpu = time.perf_counter()
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                self.in_dev,
                ctypes.c_void_p(self.x.ctypes.data),
                self.x.nbytes,
                1,  # cudaMemcpyHostToDevice
                self.stream,
            ),
            "H2D async",
        )
        ok = self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self.stream.value),
        )
        if not ok:
            raise RuntimeError("execute_async_v2=false")
        cuda_check(
            self.cuda.cudaMemcpyAsync(
                ctypes.c_void_p(self.y.ctypes.data),
                self.out_dev,
                self.y.nbytes,
                2,  # cudaMemcpyDeviceToHost
                self.stream,
            ),
            "D2H async",
        )
        cuda_check(
            self.cuda.cudaStreamSynchronize(self.stream),
            "TRT86 stream sync",
        )
        trt_ms = (time.perf_counter() - started_gpu) * 1000.0

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
        }
        return rows, prep_ms, trt_ms, total_ms, health

    def close(self):
        if getattr(self, "stream", None) is not None and self.stream.value:
            try:
                self.cuda.cudaStreamDestroy(self.stream)
            finally:
                self.stream = ctypes.c_void_p()
        super().close()


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
            "transport": "shm-bgr-v4-async-production",
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
                    str(req["shm_name"]), conf, max_det
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
