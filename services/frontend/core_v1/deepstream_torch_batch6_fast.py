from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# Prefer the much lighter YOLO26s on Pascal if it is already present. Keep the
# model override explicit and backwards-compatible for experiments.
if not os.environ.get("AI_BATCH6_MODEL"):
    os.environ["AI_BATCH6_MODEL"] = (
        "yolo26s.pt" if (ROOT / "yolo26s.pt").is_file() else "yolo26m.pt"
    )

from . import deepstream_torch_batch6_wall as base


class DeepStreamTorchBatch6FastWall(base.DeepStreamTorchBatch6Wall):
    """Pascal-friendly strict batch-6 variant.

    Keeps the same six-camera/one-forward architecture, but avoids copying every
    decoded frame to host when the detector cannot consume them that quickly.
    It also removes the deprecated Ultralytics ``half`` argument and uses a
    surveillance-oriented person threshold to reduce useless post-processing.
    """

    def __init__(self):
        self.sample_fps = max(
            1.0, float(os.environ.get("AI_BATCH6_SAMPLE_FPS", "8.0"))
        )
        self.sample_interval = 1.0 / self.sample_fps
        self._last_host_copy: dict[str, float] = {}
        self.host_copy_skips = 0
        super().__init__()

    def _on_new_sample(self, sink, cid: str):
        sample = sink.emit("pull-sample")
        if sample is None:
            return self.Gst.FlowReturn.OK

        now = time.monotonic()
        last = self._last_host_copy.get(cid, 0.0)
        if now - last < self.sample_interval:
            self.host_copy_skips += 1
            return self.Gst.FlowReturn.OK
        self._last_host_copy[cid] = now

        caps = sample.get_caps().get_structure(0)
        width = int(caps.get_value("width"))
        height = int(caps.get_value("height"))
        fmt = str(caps.get_value("format"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            return self.Gst.FlowReturn.OK
        try:
            frame = base.owned_bgr_from_mapped(mapped.data, width, height, fmt)
        finally:
            buffer.unmap(mapped)

        self.latest.put(cid, now, frame)
        return self.Gst.FlowReturn.OK

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
                f"TORCH_BATCH6_FAST cuda={torch.version.cuda} device={device_name} "
                f"sm={capability[0]}.{capability[1]} model={base.MODEL_PATH.name} "
                f"sample_fps={self.sample_fps:.1f}",
                flush=True,
            )
            if base.MODEL_PATH.name == "yolo26m.pt":
                print(
                    "TORCH_BATCH6_FAST note: yolo26s.pt not found; using yolo26m.pt. "
                    "Install yolo26s.pt for the recommended GTX 1050 Ti profile.",
                    flush=True,
                )

            model = YOLO(str(base.MODEL_PATH))
            predict_kwargs = {
                "imgsz": (base.INFER_HEIGHT, base.INFER_WIDTH),
                "classes": [0],
                "conf": float(os.environ.get("AI_BATCH6_CONF", "0.15")),
                "iou": 0.50,
                "max_det": int(os.environ.get("AI_BATCH6_MAX_DET", "20")),
                "device": "cuda:0",
                "verbose": False,
            }

            warm = [
                np.zeros((base.INFER_HEIGHT, base.INFER_WIDTH, 3), dtype=np.uint8)
                for _ in range(base.BATCH_SIZE)
            ]
            # Do not pass deprecated half=False. Default PyTorch precision is
            # FP32; Pascal has no Tensor Cores, so forcing FP16 is not assumed
            # to improve this GPU.
            model.predict(source=warm, **predict_kwargs)
            print("TORCH_BATCH6_FAST warmup complete: strict batch=6", flush=True)
        except BaseException as exc:
            print(
                f"TORCH_BATCH6_FAST startup error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self.stop_event.set()
            self.loop.quit()
            return

        last_versions = {cid: 0 for cid in self.camera_ids}
        while not self.stop_event.is_set():
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

            started = time.perf_counter()
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
                    f"TORCH_BATCH6_FAST batch error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if "out of memory" in str(exc).lower():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            ended = time.monotonic()
            batch_ms = (time.perf_counter() - started) * 1000.0
            age_ms = max(0.0, (ended - min(captured)) * 1000.0)
            last_versions.update(versions)

            with self._metrics_lock:
                self.batch_calls += 1
                self.batch_inputs += base.BATCH_SIZE
                self.total_detections += int(detections)
                self.last_batch_ms = batch_ms
                self.last_batch_age_ms = age_ms

    def _print_stats(self) -> bool:
        result = super()._print_stats()
        print(
            f"TORCH_BATCH6_FAST host_copy_skips={self.host_copy_skips} "
            f"sample_cap={self.sample_fps:.1f}fps/camera model={base.MODEL_PATH.name}",
            flush=True,
        )
        return result


def run() -> int:
    return DeepStreamTorchBatch6FastWall().run()


if __name__ == "__main__":
    raise SystemExit(run())
