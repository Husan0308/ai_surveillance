from __future__ import annotations

import queue
import time
from typing import Any

from services.ml_service.app.detector import (
    Detection,
    DetectionSnapshot,
    PersonDetector,
    _compatible_arches,
    _cuda_device_index,
)


def _legacy_detector_worker(config: dict[str, Any], job_q, result_q) -> None:
    """Old proven detector hot path inside the current isolated CUDA process.

    Inputs are already resized to the network canvas by the parent bridge. This
    keeps the multiprocessing payload bounded and makes the CUDA child see the
    exact 736x416 image used by the old clear-detection baseline. Result boxes
    are mapped back to the 960x540 presentation/source coordinates.
    """
    try:
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

        import numpy as np
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")

        device = str(config["device"])
        device_index = _cuda_device_index(device)
        torch.cuda.set_device(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        required_arch = f"sm_{capability[0]}{capability[1]}"
        compiled_arches = tuple(torch.cuda.get_arch_list())
        compatible_arches = _compatible_arches(capability, compiled_arches)

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        probe = torch.ones((32, 32), device=device)
        probe = probe @ probe
        _ = float(probe.sum().item())
        torch.cuda.synchronize(device_index)
        del probe

        model = YOLO(str(config["model"]))
        input_h = int(config["height"])
        input_w = int(config["width"])
        predict_kwargs: dict[str, Any] = {
            "imgsz": (input_h, input_w),
            "classes": [0],
            "conf": float(config["confidence"]),
            "iou": float(config["iou"]),
            "max_det": int(config["max_detections"]),
            "device": device,
            "verbose": False,
            "stream": False,
            "rect": True,
        }

        warm = [
            np.zeros((input_h, input_w, 3), dtype=np.uint8)
            for _ in range(int(config["batch_size"]))
        ]
        with torch.inference_mode():
            model.predict(source=warm, **predict_kwargs)
        torch.cuda.synchronize(device_index)

        result_q.put(
            {
                "type": "ready",
                "pid": os.getpid(),
                "model": str(config["model"]),
                "device": device,
                "cuda_capability": required_arch,
                "torch_arches": list(compiled_arches),
                "compatible_arches": list(compatible_arches),
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return

            started = time.monotonic()
            items = list(job["items"])
            frames = [item["image"] for item in items]
            with torch.inference_mode():
                predictions = model.predict(source=frames, **predict_kwargs)
            torch.cuda.synchronize(device_index)
            finished = time.monotonic()
            batch_ms = (finished - started) * 1000.0

            output = []
            for item, prediction in zip(items, predictions):
                source_w = max(1, int(item["source_w"]))
                source_h = max(1, int(item["source_h"]))
                sx = float(source_w) / float(input_w)
                sy = float(source_h) / float(input_h)
                detections = []
                boxes = getattr(prediction, "boxes", None)
                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.detach().cpu().tolist()
                    scores = boxes.conf.detach().cpu().tolist()
                    for xyxy, confidence in zip(coords, scores):
                        detections.append(
                            {
                                "xyxy": [
                                    float(xyxy[0]) * sx,
                                    float(xyxy[1]) * sy,
                                    float(xyxy[2]) * sx,
                                    float(xyxy[3]) * sy,
                                ],
                                "confidence": float(confidence),
                            }
                        )
                output.append(
                    {
                        "camera_id": item["camera_id"],
                        "frame_id": int(item["frame_id"]),
                        "captured_monotonic": float(item["captured_monotonic"]),
                        "detections": detections,
                    }
                )

            result_q.put(
                {
                    "type": "result",
                    "batch_ms": batch_ms,
                    "finished_monotonic": finished,
                    "items": output,
                }
            )
    except BaseException as exc:
        try:
            result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


class LegacyLatestPersonDetector(PersonDetector):
    """Proven old detector scheduling with current process isolation.

    The previous good camera/detection branch did not intentionally wait for a
    per-camera 4 FPS deadline. It always selected the newest unprocessed frames,
    submitted one bounded batch, then immediately selected the next latest batch
    after the CUDA result returned. This class restores that behavior while
    keeping the current three-service boundary and spawned CUDA crash isolation.
    """

    max_submit_age_ms = 260.0
    max_result_age_ms = 700.0

    def _start_worker(self) -> None:
        self._job_q = self._ctx.Queue(maxsize=1)
        self._result_q = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_legacy_detector_worker,
            args=(self._worker_config(), self._job_q, self._result_q),
            name="person-detector-cuda",
            daemon=True,
        )
        self._process.start()
        with self._lock:
            self._metrics.worker_pid = int(self._process.pid) if self._process.pid else None
            self._metrics.worker_exitcode = None
        print(
            f"[DETECT] spawned legacy latest-only CUDA worker pid={self._process.pid}",
            flush=True,
        )

    def metrics(self) -> dict:
        payload = super().metrics()
        payload.update(
            {
                "scheduler": "legacy-latest-only-uncapped",
                "pre_resize_before_ipc": True,
                "max_submit_age_ms": self.max_submit_age_ms,
                "max_result_age_ms": self.max_result_age_ms,
            }
        )
        return payload

    def _consume_legacy_message(self, message: dict) -> bool:
        kind = message.get("type")
        if kind == "warning":
            print(f"[DETECT] warning: {message.get('warning')}", flush=True)
            return False
        if kind == "ready":
            with self._lock:
                self._metrics.state = "ready"
                self._metrics.model = str(message.get("model", self.config.model))
                self._metrics.device = str(message.get("device", self.config.device))
                self._metrics.cuda_capability = str(message.get("cuda_capability", ""))
                self._metrics.torch_arches = tuple(message.get("torch_arches") or ())
                self._metrics.last_error = ""
            compatible = ",".join(message.get("compatible_arches") or ()) or "ptx/kernel-probe"
            print(
                f"[DETECT] ready pid={message.get('pid')} model={self.config.model} "
                f"device={self.config.device} batch={self.config.batch_size} "
                f"imgsz={self.config.width}x{self.config.height} "
                f"scheduler=legacy-latest-only compatible={compatible}",
                flush=True,
            )
            return False
        if kind == "fatal":
            error = str(message.get("error") or "detector worker failed")
            self._set_state("error", error)
            print(f"[DETECT] worker error: {error}", flush=True)
            return False
        if kind != "result":
            return False

        finished = float(message.get("finished_monotonic") or time.monotonic())
        batch_ms = float(message.get("batch_ms") or 0.0)
        now = time.monotonic()
        items = list(message.get("items") or ())
        accepted = 0
        for item in items:
            captured = float(item["captured_monotonic"])
            result_age_ms = max(0.0, (now - captured) * 1000.0)
            if result_age_ms > self.max_result_age_ms:
                continue
            camera_id = str(item["camera_id"])
            rows = tuple(
                Detection(
                    xyxy=tuple(float(value) for value in row["xyxy"]),
                    confidence=float(row["confidence"]),
                )
                for row in item.get("detections") or ()
            )
            self.results.put(
                DetectionSnapshot(
                    camera_id=camera_id,
                    frame_id=int(item["frame_id"]),
                    captured_monotonic=captured,
                    inferred_monotonic=finished,
                    batch_ms=batch_ms,
                    detections=rows,
                )
            )
            with self._lock:
                if not self._camera_started[camera_id]:
                    self._camera_started[camera_id] = finished
                self._camera_updates[camera_id] += 1
                self._camera_last_infer[camera_id] = finished
            accepted += 1

        with self._lock:
            self._metrics.batches += 1
            self._metrics.images += accepted
            self._metrics.last_batch_ms = batch_ms
            count = self._metrics.batches
            self._metrics.average_batch_ms += (
                batch_ms - self._metrics.average_batch_ms
            ) / max(1, count)
        return True

    def _run(self) -> None:
        import cv2

        self._set_state("starting")
        camera_ids = list(self.stores)
        last_versions = {camera_id: 0 for camera_id in camera_ids}
        cursor = 0
        in_flight = False

        try:
            self._start_worker()
            while not self._stop.is_set():
                if self._result_q is not None:
                    while True:
                        try:
                            message = self._result_q.get_nowait()
                        except queue.Empty:
                            break
                        except Exception:
                            break
                        if self._consume_legacy_message(message):
                            in_flight = False

                process = self._process
                if process is None:
                    self._set_state("error", "detector worker was not created")
                    return
                if not process.is_alive():
                    process.join(timeout=0.1)
                    exitcode = process.exitcode
                    with self._lock:
                        self._metrics.worker_exitcode = exitcode
                    if self._metrics.state != "error":
                        detail = (
                            f"detector CUDA worker exited unexpectedly with code {exitcode}; "
                            "ml_service cameras remain online"
                        )
                        self._set_state("error", detail)
                        print(f"[DETECT] {detail}", flush=True)
                    return

                with self._lock:
                    ready = self._metrics.state == "ready"
                if not ready or in_flight:
                    self._stop.wait(0.004)
                    continue

                now = time.monotonic()
                selected = []
                checked = 0
                while checked < len(camera_ids) and len(selected) < int(self.config.batch_size):
                    index = (cursor + checked) % len(camera_ids)
                    checked += 1
                    camera_id = camera_ids[index]
                    frame, version = self.stores[camera_id].get()
                    if frame is None or version <= last_versions[camera_id]:
                        continue
                    age_ms = max(0.0, (now - float(frame.captured_monotonic)) * 1000.0)
                    if age_ms > self.max_submit_age_ms:
                        last_versions[camera_id] = version
                        continue
                    selected.append((camera_id, frame, version))

                cursor = (cursor + max(1, checked)) % max(1, len(camera_ids))
                if not selected:
                    self._stop.wait(0.002)
                    continue

                input_w = int(self.config.width)
                input_h = int(self.config.height)
                items = []
                for camera_id, frame, _version in selected:
                    if int(frame.width) == input_w and int(frame.height) == input_h:
                        prepared = frame.image
                    else:
                        prepared = cv2.resize(
                            frame.image,
                            (input_w, input_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    items.append(
                        {
                            "camera_id": camera_id,
                            "frame_id": int(frame.frame_id),
                            "captured_monotonic": float(frame.captured_monotonic),
                            "source_w": int(frame.width),
                            "source_h": int(frame.height),
                            "image": prepared,
                        }
                    )

                if self._job_q is None:
                    self._set_state("error", "detector job queue unavailable")
                    return
                try:
                    self._job_q.put_nowait({"items": items})
                except queue.Full:
                    self._stop.wait(0.002)
                    continue

                for camera_id, _frame, version in selected:
                    last_versions[camera_id] = version
                in_flight = True

        except BaseException as exc:
            self._set_state("error", f"{type(exc).__name__}: {exc}")
            print(f"[DETECT] manager error: {type(exc).__name__}: {exc}", flush=True)
