#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

from scripts.yolo26_trt86_shm_worker_v4 import Runner as V4Runner, emit, trt
from scripts.yolo26_trt86_shm_worker import cuda_check


class Runner(V4Runner):
    """TRT8.6 worker with detector compute kept below display/tracker priority.

    CUDA stream priority is only a scheduling hint, but on compute capability >=3.5
    it lets default-priority DeepStream kernels run ahead of detector kernels when
    both are ready. H2D/D2H semantics stay unchanged.
    """

    def __init__(self, engine_path: Path):
        super().__init__(engine_path)

        self.cuda.cudaDeviceGetStreamPriorityRange.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.cuda.cudaDeviceGetStreamPriorityRange.restype = ctypes.c_int
        self.cuda.cudaStreamCreateWithPriority.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.cuda.cudaStreamCreateWithPriority.restype = ctypes.c_int

        least = ctypes.c_int(0)
        greatest = ctypes.c_int(0)
        cuda_check(
            self.cuda.cudaDeviceGetStreamPriorityRange(
                ctypes.byref(least), ctypes.byref(greatest)
            ),
            "cudaDeviceGetStreamPriorityRange",
        )

        # V4 created a non-blocking default-priority stream but has not submitted
        # any work yet. Replace it with the device's least-priority stream.
        if self.stream.value:
            cuda_check(self.cuda.cudaStreamDestroy(self.stream), "destroy v4 stream")
        self.stream = ctypes.c_void_p()
        cuda_check(
            self.cuda.cudaStreamCreateWithPriority(
                ctypes.byref(self.stream),
                1,  # cudaStreamNonBlocking
                int(least.value),
            ),
            "cudaStreamCreateWithPriority",
        )
        self.stream_priority = int(least.value)
        self.stream_priority_range = (int(greatest.value), int(least.value))


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
            "transport": "shm-bgr-v5-lowprio-production",
            "stream_priority": runner.stream_priority,
            "stream_priority_range": list(runner.stream_priority_range),
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
                conf = float(req.get("conf", 0.18))
                max_det = int(req.get("max_det", 20))
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
