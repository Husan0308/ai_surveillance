from __future__ import annotations

import ctypes
import ctypes.util
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np

from .runtime_v85_nvdcf_relief import PascalNvDCFReliefRuntime


class PascalInProcessTrt86Runtime(PascalNvDCFReliefRuntime):
    """V9.1: run TRT8.6 in the DeepStream host process on CUDA primary context.

    This removes the multiprocessing + SHM + subprocess sidecar chain used by V8.4.
    One detector thread owns one TensorRT execution context and one non-default CUDA
    stream.  Before constructing TensorRT, that thread explicitly binds device 0's
    CUDA primary context.  The display and NvDCF topology are otherwise V8.5.
    """

    def __init__(self) -> None:
        self.v91_ready = threading.Event()
        self.v91_init_error = ""
        self.v91_runner = None
        self.v91_primary_same = False
        self.v91_primary_ptr = 0
        self.v91_current_ptr = 0
        self.v91_warm_med_ms = 0.0
        self.v91_warm_p95_ms = 0.0
        self.v91_prep_ms_ema = 0.0
        self.v91_post_ms_ema = 0.0
        self.v91_infer_log_n = 0
        self.v91_driver = None
        self.v91_driver_device = None
        super().__init__()
        print(
            "CAMERA_V91_ARCH "
            f"python=inprocess-py310 detector=TRT8.6/batch1 cameras={len(self.cameras)} "
            "multiprocessing=0 shm=0 subprocess=0 primary_context=required "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "display=production-v85 bbox_policy=unchanged",
            flush=True,
        )

    @staticmethod
    def _resolve_engine() -> Path:
        root = Path(__file__).resolve().parents[2]
        raw = os.environ.get(
            "CAMERA_V2_TRT86_ENGINE",
            "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine",
        )
        path = Path(raw)
        return path if path.is_absolute() else root / path

    def _bind_primary_context_v91(self) -> None:
        cudart_path = ctypes.util.find_library("cudart")
        cuda_path = ctypes.util.find_library("cuda")
        if not cudart_path or not cuda_path:
            raise RuntimeError(
                f"CUDA libraries missing cudart={cudart_path!r} driver={cuda_path!r}"
            )

        cudart = ctypes.CDLL(cudart_path, mode=ctypes.RTLD_GLOBAL)
        cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        cudart.cudaSetDevice.restype = ctypes.c_int

        driver = ctypes.CDLL(cuda_path, mode=ctypes.RTLD_GLOBAL)
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        driver.cuDeviceGet.restype = ctypes.c_int
        driver.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        driver.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        driver.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        driver.cuDevicePrimaryCtxRelease.restype = ctypes.c_int
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        driver.cuCtxSetCurrent.restype = ctypes.c_int
        driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        driver.cuCtxGetCurrent.restype = ctypes.c_int

        def check(code: int, label: str) -> None:
            if int(code) != 0:
                raise RuntimeError(f"{label}: CUDA driver/runtime code={code}")

        check(driver.cuInit(0), "cuInit")
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), int(self.gpu_id)), "cuDeviceGet")
        primary = ctypes.c_void_p()
        check(
            driver.cuDevicePrimaryCtxRetain(ctypes.byref(primary), device.value),
            "cuDevicePrimaryCtxRetain",
        )
        check(driver.cuCtxSetCurrent(primary), "cuCtxSetCurrent(primary)")
        check(cudart.cudaSetDevice(int(self.gpu_id)), "cudaSetDevice")
        current = ctypes.c_void_p()
        check(driver.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")

        self.v91_primary_ptr = int(primary.value or 0)
        self.v91_current_ptr = int(current.value or 0)
        self.v91_primary_same = bool(
            self.v91_primary_ptr
            and self.v91_current_ptr
            and self.v91_primary_ptr == self.v91_current_ptr
        )
        self.v91_driver = driver
        self.v91_driver_device = int(device.value)
        print(
            "CAMERA_V91_CONTEXT "
            f"gpu={self.gpu_id} primary=0x{self.v91_primary_ptr:x} "
            f"current=0x{self.v91_current_ptr:x} same={int(self.v91_primary_same)}",
            flush=True,
        )
        if not self.v91_primary_same:
            raise RuntimeError("TRT detector thread is not bound to CUDA primary context")

    def _release_primary_context_v91(self) -> None:
        if self.v91_driver is None or self.v91_driver_device is None:
            return
        try:
            self.v91_driver.cuCtxSetCurrent(ctypes.c_void_p())
        except Exception:
            pass
        try:
            self.v91_driver.cuDevicePrimaryCtxRelease(int(self.v91_driver_device))
        except Exception:
            pass
        self.v91_driver = None
        self.v91_driver_device = None

    def _start_detector(self) -> None:
        if not self.detect_enabled:
            print("CAMERA_V91_DETECT enabled=0", flush=True)
            return
        self.det_process = None
        self.job_q = None
        self.result_q = None
        self.det_thread = threading.Thread(
            target=self._detector_scheduler_v91,
            name="camera-v91-inprocess-trt86",
            daemon=True,
        )
        self.det_thread.start()
        # Important: run() calls _start_detector before pipeline PLAYING.  Waiting
        # here makes TRT engine creation/warmup a clean pre-DeepStream baseline.
        if not self.v91_ready.wait(timeout=45.0):
            raise RuntimeError("V9.1 in-process TRT8.6 startup timeout")
        if self.v91_init_error:
            raise RuntimeError(self.v91_init_error)

    def _detector_scheduler_v91(self) -> None:
        runner = None
        try:
            self._bind_primary_context_v91()
            # Lazy import is deliberate: Python 3.10/GStreamer has already been
            # proven usable before TensorRT's libnvinfer.so.8 is loaded.
            from scripts.yolo26_trt86_shm_worker_v4 import Runner

            engine = self._resolve_engine()
            if not engine.is_file():
                raise FileNotFoundError(f"TRT8.6 engine missing: {engine}")
            runner = Runner(engine)
            self.v91_runner = runner
            if tuple(runner.input_shape) != (1, 3, 384, 672):
                raise RuntimeError(f"unexpected TRT input shape={runner.input_shape}")
            if tuple(runner.output_shape) != (1, 300, 6):
                raise RuntimeError(f"unexpected TRT output shape={runner.output_shape}")

            warm = np.full((384, 672, 3), 114, dtype=np.uint8)
            warm_gpu: list[float] = []
            for _ in range(8):
                _rows, _prep, trt_ms, _total, _health = runner.infer(
                    warm,
                    float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.18")),
                    max(1, min(300, int(os.environ.get("CAMERA_V2_MAX_DET", "20")))),
                )
                warm_gpu.append(float(trt_ms))
            tail = warm_gpu[-5:]
            self.v91_warm_med_ms = statistics.median(tail)
            ordered = sorted(tail)
            self.v91_warm_p95_ms = ordered[-1]

            with self.det_lock:
                self.det_ready = True
                self.det_error = ""
            print(
                "CAMERA_V91_READY "
                f"engine={engine.name} backend=trt86-inprocess-primary batch=1 "
                f"warm_med={self.v91_warm_med_ms:.1f}ms "
                f"warm_p95={self.v91_warm_p95_ms:.1f}ms "
                f"global={self.v84_global_hz:.2f}Hz per_camera={self.detect_hz:.2f}Hz",
                flush=True,
            )
            self.v91_ready.set()

            ids = [camera.camera_id for camera in self.cameras]
            versions = {cid: 0 for cid in ids}
            rr = 0
            conf = float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.18"))
            max_det = max(1, min(300, int(os.environ.get("CAMERA_V2_MAX_DET", "20"))))

            while not self.det_stop.is_set():
                cycle_started = time.monotonic()
                selected = None
                for offset in range(len(ids)):
                    cid = ids[(rr + offset) % len(ids)]
                    if self.stats[cid].frames > 0:
                        selected = cid
                        rr = (rr + offset + 1) % len(ids)
                        break
                if selected is None:
                    self.det_stop.wait(0.03)
                    continue

                cid = selected
                self._request_capture(cid)
                row = self.mailbox.wait_new(
                    cid,
                    versions[cid],
                    timeout=self.v84_capture_timeout,
                )
                self._clear_capture(cid)
                if row is None:
                    self.v84_capture_miss += 1
                    with self.det_lock:
                        self.capture_timeouts += 1
                    self.det_stop.wait(0.003)
                    continue

                version, captured_at, frame = row
                versions[cid] = version

                infer_started = time.perf_counter()
                raw_rows, prep_ms, trt_ms, total_ms, health = runner.infer(
                    frame, conf, max_det
                )
                infer_wall_ms = (time.perf_counter() - infer_started) * 1000.0
                completed = time.monotonic()

                detector_rows = [
                    ([float(x1), float(y1), float(x2), float(y2)], float(score))
                    for x1, y1, x2, y2, score in raw_rows
                ]
                boxes = self._map_detector_rows(detector_rows)
                self._publish_detector(cid, captured_at, boxes)

                gpu_ms = float(trt_ms)
                roundtrip_ms = float(infer_wall_ms)
                post_ms = max(0.0, roundtrip_ms - float(prep_ms) - gpu_ms)
                self.v84_gpu_ms_ema = self._ema_v84(
                    self.v84_gpu_ms_ema, gpu_ms, self.v84_ema_alpha
                )
                self.v84_roundtrip_ms_ema = self._ema_v84(
                    self.v84_roundtrip_ms_ema, roundtrip_ms, self.v84_ema_alpha
                )
                self.v91_prep_ms_ema = self._ema_v84(
                    self.v91_prep_ms_ema, float(prep_ms), self.v84_ema_alpha
                )
                self.v91_post_ms_ema = self._ema_v84(
                    self.v91_post_ms_ema, post_ms, self.v84_ema_alpha
                )
                self._adapt_v84()

                if self.v84_last_complete > 0.0:
                    self.v84_intervals.append(completed - self.v84_last_complete)
                self.v84_last_complete = completed
                self.v84_calls += 1
                self.v84_per_camera_calls[cid] += 1
                self.v91_infer_log_n += 1

                age_ms = max(0.0, (completed - captured_at) * 1000.0)
                self.v84_result_age_ms = age_ms
                with self.det_lock:
                    self.det_counts[cid] = len(boxes)
                    self.detector_times[cid].append(completed)
                    self.det_calls += 1
                    self.det_inputs += 1
                    self.det_batch_ms = gpu_ms
                    self.det_result_age_ms = age_ms
                    self.det_error = ""

                if self.v91_infer_log_n <= 8 or self.v91_infer_log_n % 20 == 0:
                    duty = self.v84_global_hz * max(0.0, self.v84_gpu_ms_ema) / 1000.0
                    print(
                        "CAMERA_V91_TRT "
                        f"n={self.v91_infer_log_n} camera={cid} prep={prep_ms:.1f}ms "
                        f"gpu={gpu_ms:.1f}ms total={roundtrip_ms:.1f}ms "
                        f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
                        f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
                        f"age={age_ms:.0f}ms global={self.v84_global_hz:.2f}Hz "
                        f"duty={duty:.2f} boxes={len(boxes)} "
                        f"person_max={health.get('raw_person_max_conf')}",
                        flush=True,
                    )

                desired = 1.0 / max(0.1, self.v84_global_hz)
                elapsed = time.monotonic() - cycle_started
                self.det_stop.wait(max(0.001, desired - elapsed))

        except BaseException as exc:
            self.v91_init_error = f"V91 {type(exc).__name__}: {exc}"
            with self.det_lock:
                self.det_error = self.v91_init_error
            print(f"CAMERA_V91_ERROR {self.v91_init_error}", flush=True)
            self.v91_ready.set()
        finally:
            if runner is not None:
                try:
                    runner.close()
                except Exception as exc:
                    print(
                        f"CAMERA_V91_CLOSE warning={type(exc).__name__}:{exc}",
                        flush=True,
                    )
            self.v91_runner = None
            self._release_primary_context_v91()

    def _shutdown_detector(self) -> None:
        if self.det_thread is not None:
            self.det_thread.join(timeout=4.0)
        # No process, queue, SHM segment, or subprocess exists in V9.1.

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        warm_speedup = (
            177.3 / self.v84_gpu_ms_ema if self.v84_gpu_ms_ema > 0.0 else 0.0
        )
        print(
            "CAMERA_V91_STATS "
            f"primary_same={int(self.v91_primary_same)} warm_med={self.v91_warm_med_ms:.1f}ms "
            f"gpu_ema={self.v84_gpu_ms_ema:.1f}ms "
            f"roundtrip_ema={self.v84_roundtrip_ms_ema:.1f}ms "
            f"prep_ema={self.v91_prep_ms_ema:.1f}ms post_ema={self.v91_post_ms_ema:.1f}ms "
            f"vs_v85_gpu_speedup={warm_speedup:.2f}x "
            f"tracked_now={self.tracked_now} tracker_batches={self.tracker_batches} "
            "processes=1 shm=0 sidecar=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalInProcessTrt86Runtime().run()


if __name__ == "__main__":
    raise SystemExit(main())
