from __future__ import annotations

import statistics
import time

import numpy as np


def main() -> int:
    try:
        import torch
        from rfdetr import RFDETRSmall
    except Exception as exc:
        print(f"RFDETR_GPU_SMOKE=FAIL import {type(exc).__name__}: {exc}", flush=True)
        return 2

    if not torch.cuda.is_available():
        print("RFDETR_GPU_SMOKE=FAIL PyTorch CUDA unavailable", flush=True)
        return 2

    device = "cuda:0"
    shape = (416, 736)
    threshold = 0.18
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    try:
        started = time.perf_counter()
        model = RFDETRSmall(device=device)
        load_s = time.perf_counter() - started

        image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        with torch.inference_mode():
            model.predict(
                image,
                threshold=threshold,
                shape=shape,
                include_source_image=False,
            )
            torch.cuda.synchronize()

            samples_ms: list[float] = []
            for _ in range(5):
                t0 = time.perf_counter()
                model.predict(
                    image,
                    threshold=threshold,
                    shape=shape,
                    include_source_image=False,
                )
                torch.cuda.synchronize()
                samples_ms.append((time.perf_counter() - t0) * 1000.0)

        peak_mb = torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0)
        reserved_mb = torch.cuda.max_memory_reserved(0) / (1024.0 * 1024.0)
        samples_ms.sort()
        p95 = samples_ms[-1]
        avg = statistics.fmean(samples_ms)

        print(
            "RFDETR_GPU_SMOKE=PASS "
            f"device={torch.cuda.get_device_name(0)} shape={shape[1]}x{shape[0]} "
            f"load={load_s:.2f}s avg={avg:.1f}ms p95={p95:.1f}ms "
            f"peak_alloc={peak_mb:.0f}MiB peak_reserved={reserved_mb:.0f}MiB",
            flush=True,
        )
        print(
            "RFDETR_GPU_SAMPLES_MS=" + ",".join(f"{value:.1f}" for value in samples_ms),
            flush=True,
        )
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        print(f"RFDETR_GPU_SMOKE=FAIL CUDA_OOM {exc}", flush=True)
        return 3
    except Exception as exc:
        print(f"RFDETR_GPU_SMOKE=FAIL {type(exc).__name__}: {exc}", flush=True)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
