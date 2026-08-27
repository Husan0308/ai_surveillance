#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

from scripts.yolo26_trt86_batch6_worker_v8 import Batch6Runner, emit


class WarmBatch6Runner(Batch6Runner):
    """V8.3 keeps the exact V8 engine path but removes cold-start timing feedback.

    The Pascal GPU is allowed to idle/downclock normally. Before the runtime starts
    using detector timing to control cadence, run several back-to-back enqueues in
    the same TRT8.6 process/context. This is intentionally not a fake faster engine:
    it only gives the floating GPU clock a chance to reach its workload state before
    the first production timing sample is fed to the adaptive controller.
    """

    def _device_once(self) -> float:
        started = time.perf_counter()
        cuda = self.cuda
        # cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2.
        code = cuda.cudaMemcpyAsync(
            self.in_dev,
            self.x_host_ptr,
            self.x.nbytes,
            1,
            self.stream,
        )
        if int(code) != 0:
            raise RuntimeError(f"warmup H2D cuda={code}")
        if not self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=int(self.stream.value),
        ):
            raise RuntimeError("warmup execute_async_v2=false")
        code = cuda.cudaMemcpyAsync(
            self.y_host_ptr,
            self.out_dev,
            self.y.nbytes,
            2,
            self.stream,
        )
        if int(code) != 0:
            raise RuntimeError(f"warmup D2H cuda={code}")
        code = cuda.cudaStreamSynchronize(self.stream)
        if int(code) != 0:
            raise RuntimeError(f"warmup sync cuda={code}")
        return (time.perf_counter() - started) * 1000.0

    def warmup(self, iterations: int) -> tuple[list[float], float, float]:
        iterations = max(4, min(40, int(iterations)))
        # Deterministic finite input. The timing path is identical to production
        # H2D -> TRT enqueue -> D2H -> stream sync, only post-processing is skipped.
        self.x.fill(0.0)
        self.y.fill(0.0)
        samples: list[float] = []
        for _ in range(iterations):
            samples.append(self._device_once())
        tail = samples[max(2, iterations // 3):]
        median = float(statistics.median(tail))
        ordered = sorted(tail)
        p95 = float(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))])
        return samples, median, p95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    runner = WarmBatch6Runner(Path(args.engine))
    warmup_iters = int(os.environ.get("CAMERA_V83_TRT_WARMUP_ITERS", "12"))
    samples, warm_med, warm_p95 = runner.warmup(warmup_iters)
    print(
        "CAMERA_V83_TRT_WARMUP "
        f"iters={len(samples)} first={samples[0]:.1f}ms "
        f"last={samples[-1]:.1f}ms median_tail={warm_med:.1f}ms p95_tail={warm_p95:.1f}ms "
        "path=h2d+enqueue+d2h+sync",
        file=sys.stderr,
        flush=True,
    )
    emit(
        {
            "type": "ready",
            "tensorrt": trt.__version__,
            "engine": str(Path(args.engine).resolve()),
            "input_shape": runner.input_shape,
            "output_shape": runner.output_shape,
            "transport": "shm-bgr-batch6-v83-pinned-warm",
            "warmup_median_ms": warm_med,
            "warmup_p95_ms": warm_p95,
        }
    )
    try:
        for line in sys.stdin:
            req = {}
            try:
                req = json.loads(line)
                if req.get("cmd") == "stop":
                    break
                boxes, prep_ms, trt_ms, total_ms = runner.infer_from_shm(
                    str(req["shm_name"]),
                    float(req.get("conf", 0.18)),
                    int(req.get("max_det", 20)),
                )
                emit(
                    {
                        "id": req.get("id"),
                        "ok": True,
                        "boxes": boxes,
                        "prep_ms": prep_ms,
                        "trt_ms": trt_ms,
                        "total_ms": total_ms,
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
