from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Prefer a Pascal-friendly detector. Explicit user override always wins.
if not os.environ.get("AI_BATCH6_MODEL"):
    for candidate in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt"):
        if (ROOT / candidate).is_file():
            os.environ["AI_BATCH6_MODEL"] = candidate
            break

# Camera smoothness is the priority. Only a few frames per second need expensive
# detection; the tracker added later will run between detector observations.
os.environ.setdefault("AI_BATCH6_SAMPLE_FPS", "3.0")

from . import deepstream_torch_batch6_wall as base

# Reduce only the detector canvas. The 1280x720 display path remains untouched.
base.INFER_WIDTH = int(os.environ.get("AI_BATCH6_INFER_WIDTH", "640"))
base.INFER_HEIGHT = int(os.environ.get("AI_BATCH6_INFER_HEIGHT", "352"))

from . import deepstream_torch_batch6_fast as fast


class DeepStreamTorchBatch6SmoothWall(fast.DeepStreamTorchBatch6FastWall):
    """Smooth-display-first strict batch-6 profile for GTX 1050 Ti.

    Key invariants:
      * display stays GPU-native and full 1280x720;
      * inference is still exactly six cameras in one model forward;
      * stale inference frames are dropped BEFORE nvvideoconvert;
      * detector GPU duty cycle is bounded so NVDEC/render gets breathing room.
    """

    def __init__(self):
        self.max_batch_fps = max(
            0.5, float(os.environ.get("AI_BATCH6_MAX_BATCH_FPS", "2.0"))
        )
        self.batch_interval = 1.0 / self.max_batch_fps
        self.preprocess_fps = max(
            self.max_batch_fps,
            float(os.environ.get("AI_BATCH6_PREPROCESS_FPS", "3.0")),
        )
        self.preprocess_interval = 1.0 / self.preprocess_fps
        self._last_preprocess: dict[str, float] = {}
        self.preprocess_drops = 0
        self.gpu_idle_sleep_ms = 0.0
        super().__init__()

        # This probe is intentionally upstream of nvvideoconvert. Dropped frames
        # never pay GPU resize/color-conversion or host-copy cost.
        for cid, queue in self.infer_queues.items():
            queue.get_static_pad("src").add_probe(
                self.Gst.PadProbeType.BUFFER,
                self._preprocess_gate_probe,
                cid,
            )

    def _preprocess_gate_probe(self, _pad, _info, cid: str):
        now = time.monotonic()
        last = self._last_preprocess.get(cid, 0.0)
        if now - last < self.preprocess_interval:
            self.preprocess_drops += 1
            return self.Gst.PadProbeReturn.DROP
        self._last_preprocess[cid] = now
        return self.Gst.PadProbeReturn.OK

    def _infer_loop(self) -> None:
        try:
            import numpy as np
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is not available")

            torch.cuda.set_device(0)
            capability = torch.cuda.get_device_capability(0)
            device_name = torch.cuda.get_device_name(0)
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            torch.backends.cudnn.benchmark = True

            print(
                f"TORCH_BATCH6_SMOOTH cuda={torch.version.cuda} device={device_name} "
                f"sm={capability[0]}.{capability[1]} model={base.MODEL_PATH.name} "
                f"infer={base.INFER_WIDTH}x{base.INFER_HEIGHT} "
                f"preprocess={self.preprocess_fps:.1f}fps/cam "
                f"max_batch={self.max_batch_fps:.1f}/s",
                flush=True,
            )
            if base.MODEL_PATH.name == "yolo26m.pt":
                print(
                    "TORCH_BATCH6_SMOOTH note: yolo26m is still heavy for GTX 1050 Ti; "
                    "yolo26n.pt is recommended for smooth camera priority.",
                    flush=True,
                )

            model = YOLO(str(base.MODEL_PATH))
            predict_kwargs = {
                "imgsz": (base.INFER_HEIGHT, base.INFER_WIDTH),
                "classes": [0],
                "conf": float(os.environ.get("AI_BATCH6_CONF", "0.20")),
                "iou": 0.50,
                "max_det": int(os.environ.get("AI_BATCH6_MAX_DET", "15")),
                "device": "cuda:0",
                "verbose": False,
            }
            warm = [
                np.zeros((base.INFER_HEIGHT, base.INFER_WIDTH, 3), dtype=np.uint8)
                for _ in range(base.BATCH_SIZE)
            ]
            model.predict(source=warm, **predict_kwargs)
            print("TORCH_BATCH6_SMOOTH warmup complete: strict batch=6", flush=True)
        except BaseException as exc:
            print(
                f"TORCH_BATCH6_SMOOTH startup error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self.stop_event.set()
            self.loop.quit()
            return

        last_versions = {cid: 0 for cid in self.camera_ids}
        next_batch_allowed = 0.0

        while not self.stop_event.is_set():
            # Reserve part of each second for NVDEC/display instead of running
            # 300 ms YOLO batches back-to-back on Pascal.
            now = time.monotonic()
            if now < next_batch_allowed:
                sleep_s = min(0.05, next_batch_allowed - now)
                self.gpu_idle_sleep_ms += sleep_s * 1000.0
                if self.stop_event.wait(sleep_s):
                    break
                continue

            rows = self.latest.wait_full_new_batch(last_versions, timeout=0.5)
            if rows is None:
                continue

            versions: dict[str, int] = {}
            frames = []
            captured = []
            for cid, row in zip(self.camera_ids, rows):
                version, captured_mono, frame = row
                versions[cid] = int(version)
                frames.append(frame)
                captured.append(float(captured_mono))

            started_mono = time.monotonic()
            started_perf = time.perf_counter()
            try:
                predictions = model.predict(source=frames, **predict_kwargs)
                detections = 0
                for prediction in predictions:
                    boxes = getattr(prediction, "boxes", None)
                    if boxes is not None:
                        detections += len(boxes)
            except BaseException as exc:
                with self._metrics_lock:
                    self.batch_errors += 1
                print(
                    f"TORCH_BATCH6_SMOOTH batch error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "out of memory" in str(exc).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                next_batch_allowed = time.monotonic() + self.batch_interval
                continue

            ended = time.monotonic()
            batch_ms = (time.perf_counter() - started_perf) * 1000.0
            age_ms = max(0.0, (ended - min(captured)) * 1000.0)
            last_versions.update(versions)

            # Start-to-start pacing. If inference itself takes 300 ms at 2 Hz,
            # the GPU gets roughly 200 ms idle before the next batch.
            next_batch_allowed = max(
                ended,
                started_mono + self.batch_interval,
            )

            with self._metrics_lock:
                self.batch_calls += 1
                self.batch_inputs += base.BATCH_SIZE
                self.total_detections += int(detections)
                self.last_batch_ms = batch_ms
                self.last_batch_age_ms = age_ms

    def _print_stats(self) -> bool:
        result = base.DeepStreamTorchBatch6Wall._print_stats(self)
        print(
            f"TORCH_BATCH6_SMOOTH preprocess_drops={self.preprocess_drops} "
            f"preprocess_cap={self.preprocess_fps:.1f}fps/camera "
            f"batch_cap={self.max_batch_fps:.1f}/s "
            f"infer={base.INFER_WIDTH}x{base.INFER_HEIGHT} "
            f"model={base.MODEL_PATH.name} idle_sleep={self.gpu_idle_sleep_ms:.0f}ms",
            flush=True,
        )
        return result


def run() -> int:
    return DeepStreamTorchBatch6SmoothWall().run()


if __name__ == "__main__":
    raise SystemExit(run())
