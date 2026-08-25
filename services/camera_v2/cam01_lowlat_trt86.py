from __future__ import annotations

import json
import os
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from . import cam01_lowlat_gpu as base


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "artifacts/yolo26s_trt86/yolo26s-672x384-b1-fp32-trt86.engine"
TRT_PYTHON = ROOT / ".venv-trt86/bin/python"
TRT_WORKER = ROOT / "scripts/yolo26_trt86_worker.py"
FRAME_SHAPE = (384, 672, 3)
FRAME_BYTES = int(np.prod(FRAME_SHAPE))


def _trt86_worker(job_q, result_q) -> None:
    """Persistent TensorRT 8.6 sidecar adapter for detection.py's worker contract."""

    proc = None
    shm = None
    shm_frame = None
    request_id = 0
    try:
        for required in (ENGINE, TRT_PYTHON, TRT_WORKER):
            if not required.exists():
                raise RuntimeError(f"TRT86 dependency missing: {required}")

        shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
        shm_frame = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

        proc = subprocess.Popen(
            [str(TRT_PYTHON), str(TRT_WORKER), "--engine", str(ENGINE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr = ""
            if proc.stderr is not None:
                stderr = proc.stderr.read()[-2000:]
            raise RuntimeError(f"TRT86 worker exited before ready: {stderr}")
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"TRT86 bad ready message: {ready}")

        result_q.put(
            {
                "type": "ready",
                "device": "NVIDIA/TensorRT86",
                "cuda": f"TRT{ready.get('tensorrt')}",
                "model": str(ENGINE),
                "backend": "trt86-sidecar-shm-bgr-highprio",
                "stream_priority": ready.get("stream_priority"),
                "stream_priority_range": ready.get("stream_priority_range"),
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                break

            started = time.perf_counter()
            try:
                cameras = list(job["cameras"])
                frames = list(job["frames"])
                captured = list(job["captured"])
                if len(cameras) != 1 or cameras[0] != "CAM-01" or len(frames) != 1:
                    raise RuntimeError(
                        f"TRT86 CAM-01 worker requires one CAM-01 frame, got {cameras!r}"
                    )

                frame = np.asarray(frames[0], dtype=np.uint8)
                if frame.shape != FRAME_SHAPE:
                    raise RuntimeError(f"unexpected detector frame shape={frame.shape}")

                copy_started = time.perf_counter()
                np.copyto(shm_frame, frame, casting="no")
                shm_copy_ms = (time.perf_counter() - copy_started) * 1000.0

                request_id += 1
                payload = {
                    "id": request_id,
                    "shm_name": shm.name,
                    "width": 672,
                    "height": 384,
                    "conf": float(os.environ.get("CAMERA_V2_DETECT_CONF", "0.05")),
                }
                proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                proc.stdin.flush()

                line = proc.stdout.readline()
                if not line:
                    stderr = ""
                    if proc.stderr is not None:
                        stderr = proc.stderr.read()[-2000:]
                    raise RuntimeError(f"TRT86 worker stopped during inference: {stderr}")
                reply = json.loads(line)
                if not reply.get("ok"):
                    raise RuntimeError(reply.get("error", "TRT86 inference failed"))
                if int(reply.get("id", -1)) != request_id:
                    raise RuntimeError(
                        f"TRT86 response id mismatch got={reply.get('id')} expected={request_id}"
                    )

                rows = []
                for item in reply.get("boxes", []):
                    x1, y1, x2, y2, score = [float(v) for v in item]
                    rows.append(([x1, y1, x2, y2], score))

                roundtrip_ms = (time.perf_counter() - started) * 1000.0
                sidecar_total_ms = float(reply.get("total_ms") or 0.0)
                trt_ms = float(reply.get("trt_ms") or 0.0)
                prep_ms = float(reply.get("prep_ms") or 0.0)
                ipc_ms = max(0.0, roundtrip_ms - sidecar_total_ms - shm_copy_ms)

                if request_id <= 3 or request_id % 20 == 0:
                    print(
                        "CAM01_TRT86_BREAKDOWN "
                        f"n={request_id} roundtrip={roundtrip_ms:.1f}ms "
                        f"shm_copy={shm_copy_ms:.1f}ms prep={prep_ms:.1f}ms "
                        f"trt={trt_ms:.1f}ms sidecar={sidecar_total_ms:.1f}ms "
                        f"ipc={ipc_ms:.1f}ms boxes={len(rows)}",
                        flush=True,
                    )

                result_q.put(
                    {
                        "type": "result",
                        "cameras": cameras,
                        "captured": captured,
                        "boxes": {"CAM-01": rows},
                        "batch_ms": roundtrip_ms,
                        "trt_ms": trt_ms,
                        "prep_ms": prep_ms,
                        "sidecar_total_ms": sidecar_total_ms,
                        "shm_copy_ms": shm_copy_ms,
                        "ipc_ms": ipc_ms,
                    }
                )
            except BaseException as exc:
                result_q.put(
                    {
                        "type": "batch_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if proc is not None:
            try:
                if proc.stdin is not None and proc.poll() is None:
                    proc.stdin.write('{"cmd":"stop"}\n')
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


base._det._yolo_worker = _trt86_worker


class Cam01LowLatencyTRT86(base.Cam01LowLatencyReID):
    pass


def _force_trt_runtime_profile() -> None:
    # Preserve a sharp HD source for the UI/fullscreen path. The previous
    # 640x360 experiment did not materially improve TRT latency, so do not trade
    # visible camera quality for compute time that was not recovered.
    os.environ["CAMERA_V2_FRAME_WIDTH"] = "1280"
    os.environ["CAMERA_V2_FRAME_HEIGHT"] = "720"
    os.environ["CAMERA_V2_TRACKER_WIDTH"] = "512"
    os.environ["CAMERA_V2_TRACKER_HEIGHT"] = "288"

    # NvDCF owns the frames between detector refreshes. Keep detector cadence
    # moderate while the high-priority CUDA stream shortens each refresh latency.
    os.environ["CAMERA_V2_DETECT_TARGET_HZ"] = "3.0"
    os.environ["CAMERA_V2_DETECT_MIN_HZ"] = "2.7"
    os.environ["CAMERA_V2_DETECT_MAX_HZ"] = "3.3"
    os.environ["CAMERA_V2_DETECT_GPU_DUTY"] = "0.26"
    os.environ["CAMERA_V2_DETECT_GPU_DUTY_MIN"] = "0.20"
    os.environ["CAMERA_V2_DETECT_GPU_DUTY_MAX"] = "0.32"

    # Keep the freshness gate strict. We are reducing actual GPU wait time rather
    # than hiding it by accepting old detector coordinates.
    os.environ["CAMERA_V2_MAX_DETECT_RESULT_AGE_MS"] = "160"


def main() -> int:
    for required in (ENGINE, TRT_PYTHON, TRT_WORKER):
        if not required.exists():
            raise RuntimeError(f"TRT86 dependency missing: {required}")

    _force_trt_runtime_profile()
    base._validate_profile()

    runtime = Cam01LowLatencyTRT86()

    # High-quality scaling stays enabled for the visible wall. Detector capture is
    # an independent 672x384 branch, so UI quality and detector geometry remain
    # separate concerns.
    runtime._set_if(runtime.mux, "interpolation-method", 4)
    runtime._set_if(runtime.tiler, "interpolation-method", 4)

    capsfilter = runtime.pipeline.get_by_name("detect_caps_0")
    appsink = runtime.pipeline.get_by_name("detect_sink_0")
    if capsfilter is None or appsink is None:
        raise RuntimeError("CAM-01 inference branch elements not found")

    appsink.set_property("emit-signals", False)
    appsink.set_property("sync", False)
    appsink.set_property("drop", True)
    appsink.set_property("max-buffers", 1)
    runtime._set_if(appsink, "async", False)
    runtime._set_if(appsink, "qos", False)

    srcpad = capsfilter.get_static_pad("src")
    if srcpad is None:
        raise RuntimeError("CAM-01 detect caps src pad not found")
    srcpad.add_probe(
        runtime.Gst.PadProbeType.BUFFER,
        runtime._capture_converted_probe,
        "CAM-01",
    )

    print(
        "CAM01_TRT86_PROFILE "
        f"engine={ENGINE.name} input=672x384/b1/fp32 active=CAM-01 "
        "capture=postconvert-buffer-probe-latest target=3.0Hz "
        "max_result_age=160ms rtsp=50ms display=1280x720 "
        "tracker=512x288 qwen=0",
        flush=True,
    )
    print(
        "CAM01_TRT86_PIPELINE backend=trt86-sidecar-shm-bgr-highprio "
        "base64=0 jpeg=0 prefetch=0 queue_depth=1 nvdcf=per-frame "
        "display_quality=hd-lanczos detector_independent=672x384",
        flush=True,
    )
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())