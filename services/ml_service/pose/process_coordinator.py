from __future__ import annotations

import logging
import math
import multiprocessing as mp
import threading
import time

from .coordinator import PoseKeypoint, PosePerson, PoseResult

log = logging.getLogger(__name__)


def _pose_process_main(connection, config: dict):
    """Own all Ultralytics/PyTorch pose state in an isolated spawned process."""
    import faulthandler

    faulthandler.enable(all_threads=True)
    model_name = str(config.get("model", "yolo26m-pose.pt"))
    # Keep the production detector as the only CUDA analytics process.
    device = "cpu"
    imgsz = max(128, int(config.get("imgsz", 256)))
    conf = float(config.get("conf", 0.25))
    threads = max(1, int(config.get("torch_cpu_threads", 1)))

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        model = YOLO(model_name)
        model.predict(
            source=[np.zeros((imgsz, imgsz, 3), dtype=np.uint8)],
            imgsz=imgsz,
            conf=conf,
            device=device,
            max_det=1,
            verbose=False,
        )
        connection.send(
            (
                "ready",
                {
                    "model": model_name,
                    "device": device,
                    "imgsz": imgsz,
                    "torch_cpu_threads": threads,
                },
            )
        )
    except BaseException as exc:
        try:
            connection.send(("startup_error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        return

    while True:
        try:
            if not connection.poll(0.25):
                continue
            payload = connection.recv()
        except (EOFError, OSError):
            return
        if payload is None:
            return

        job_id, camera_id, frame_id, captured_mono, crops = payload
        started = time.perf_counter()
        try:
            predictions = model.predict(
                source=[entry[0] for entry in crops],
                imgsz=imgsz,
                conf=conf,
                device=device,
                max_det=1,
                verbose=False,
            )
            people = []
            for entry, prediction in zip(crops, predictions):
                _crop, source_bbox, offset = entry
                keypoints = getattr(prediction, "keypoints", None)
                if (
                    keypoints is None
                    or getattr(keypoints, "xy", None) is None
                    or len(keypoints.xy) == 0
                ):
                    continue
                index = 0
                pred_boxes = getattr(prediction, "boxes", None)
                if (
                    pred_boxes is not None
                    and getattr(pred_boxes, "conf", None) is not None
                    and len(pred_boxes.conf)
                ):
                    try:
                        index = int(pred_boxes.conf.argmax().item())
                    except Exception:
                        index = 0
                if index >= len(keypoints.xy):
                    index = 0
                xy = keypoints.xy[index].detach().cpu().tolist()
                conf_tensor = getattr(keypoints, "conf", None)
                if conf_tensor is not None and len(conf_tensor) > index:
                    confs = conf_tensor[index].detach().cpu().tolist()
                else:
                    confs = [1.0] * len(xy)
                ox, oy = offset
                points = [
                    (float(point[0]) + float(ox), float(point[1]) + float(oy), float(score))
                    for point, score in zip(xy, confs)
                ]
                pose_conf = float(source_bbox[4])
                if (
                    pred_boxes is not None
                    and getattr(pred_boxes, "conf", None) is not None
                    and len(pred_boxes.conf) > index
                ):
                    try:
                        pose_conf = float(pred_boxes.conf[index].detach().cpu().item())
                    except Exception:
                        pass
                people.append(
                    {
                        "bbox": tuple(float(value) for value in source_bbox[:4]),
                        "confidence": pose_conf,
                        "keypoints": points,
                    }
                )
            connection.send(
                (
                    "result",
                    {
                        "job_id": int(job_id),
                        "camera_id": str(camera_id),
                        "frame_id": int(frame_id),
                        "captured_mono": float(captured_mono),
                        "produced_mono": time.monotonic(),
                        "wall_ms": (time.perf_counter() - started) * 1000.0,
                        "people": people,
                    },
                )
            )
        except BaseException as exc:
            try:
                connection.send(
                    (
                        "error",
                        {
                            "job_id": int(job_id),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
            except Exception:
                pass


class PoseProcessCoordinator:
    """Latest-only pose side path isolated from the ML/Uvicorn process.

    The parent only crops exact detector frames and transfers one bounded latest
    job at a time. Ultralytics/PyTorch lives exclusively in the spawned child.
    A native child crash therefore cannot abort camera capture, YOLO detection,
    JPEG publication, FastAPI, or the UI stream.
    """

    def __init__(self, frame_stores, detections, config: dict | None = None):
        self.frame_stores = dict(frame_stores)
        self.detections = detections
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.model_name = str(self.config.get("model", "yolo26m-pose.pt"))
        self.device = "cpu"
        self.imgsz = max(128, int(self.config.get("imgsz", 256)))
        self.conf = float(self.config.get("conf", 0.25))
        self.every_n = max(1, int(self.config.get("every_n", 10)))
        self.max_people = max(1, int(self.config.get("max_people", 1)))
        self.max_frame_age_ms = max(
            0.0, float(self.config.get("max_frame_age_ms", 900))
        )
        self.restart_backoff_sec = max(
            1.0, float(self.config.get("restart_backoff_sec", 5.0))
        )
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._connection = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._results: dict[str, PoseResult] = {}
        self._last_frame = {cid: -1 for cid in self.frame_stores}
        self._seen = {cid: 0 for cid in self.frame_stores}
        self._cursor = 0
        self._job_id = 0
        self._inflight_job = None
        self._ready = False
        self._processed = 0
        self._frame_misses = 0
        self._stale_skips = 0
        self._errors = 0
        self._restarts = 0
        self._native_crashes = 0
        self._last_inference_ms = 0.0
        self._last_error = ""
        self._next_restart_mono = 0.0

    def start(self):
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="pose-process-bridge",
            daemon=False,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.send(None)
            except Exception:
                pass

    def join(self, timeout=6):
        deadline = time.monotonic() + max(0.0, float(timeout))
        if self._thread:
            self._thread.join(max(0.0, deadline - time.monotonic()))
        process = self._process
        if process:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(1.0)
        self._close_connection()

    def snapshot(self):
        with self._lock:
            return dict(self._results)

    def metrics(self):
        process = self._process
        with self._lock:
            return {
                "enabled": self.enabled,
                "ready": self._ready,
                "processed": self._processed,
                "frame_misses": self._frame_misses,
                "stale_skips": self._stale_skips,
                "errors": self._errors,
                "restarts": self._restarts,
                "native_crashes": self._native_crashes,
                "last_inference_ms": self._last_inference_ms,
                "last_error": self._last_error,
                "model": self.model_name,
                "device": self.device,
                "process_alive": bool(process and process.is_alive()),
                "process_pid": process.pid if process else None,
                "process_exitcode": process.exitcode if process and not process.is_alive() else None,
                "start_method": "spawn",
                "isolation": "separate_pose_process",
                "detector_gating": False,
            }

    @staticmethod
    def _crop(frame, box):
        image = frame.image
        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, int(float(box.x1))))
        y1 = max(0, min(height - 1, int(float(box.y1))))
        x2 = max(x1 + 1, min(width, int(math.ceil(float(box.x2)))))
        y2 = max(y1 + 1, min(height, int(math.ceil(float(box.y2)))))
        crop = image[y1:y2, x1:x2]
        bbox = (
            float(box.x1),
            float(box.y1),
            float(box.x2),
            float(box.y2),
            float(box.confidence),
        )
        return crop, bbox, (x1, y1)

    def _close_connection(self):
        connection = self._connection
        self._connection = None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def _spawn(self):
        self._close_connection()
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_pose_process_main,
            args=(child, self.config),
            name="core-v1-yolo26m-pose-cpu",
            daemon=False,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        self._ready = False
        self._inflight_job = None
        log.info("CORE_V1_POSE_PROCESS_STARTED pid=%s", process.pid)

    def _process_died(self):
        process = self._process
        if process is None or process.is_alive():
            return False
        exitcode = process.exitcode
        with self._lock:
            self._ready = False
            self._errors += 1
            self._native_crashes += int(exitcode is not None and exitcode < 0 or exitcode == 134)
            self._last_error = f"pose process exited unexpectedly: exitcode={exitcode}"
        self._close_connection()
        self._process = None
        self._inflight_job = None
        self._next_restart_mono = time.monotonic() + self.restart_backoff_sec
        return True

    def _drain(self):
        connection = self._connection
        if connection is None:
            return
        while True:
            try:
                if not connection.poll(0.0):
                    return
                kind, payload = connection.recv()
            except (EOFError, OSError):
                return
            if kind == "ready":
                with self._lock:
                    self._ready = True
                    self._last_error = ""
                continue
            if kind == "startup_error":
                with self._lock:
                    self._ready = False
                    self._errors += 1
                    self._last_error = str(payload)
                self._next_restart_mono = time.monotonic() + self.restart_backoff_sec
                continue
            if kind == "error":
                self._inflight_job = None
                with self._lock:
                    self._errors += 1
                    self._last_error = str(payload.get("error") or payload)
                continue
            if kind != "result":
                continue
            self._inflight_job = None
            people = []
            for item in payload.get("people", ()):
                keypoints = tuple(
                    PoseKeypoint(float(x), float(y), float(confidence))
                    for x, y, confidence in item.get("keypoints", ())
                )
                people.append(
                    PosePerson(
                        tuple(float(value) for value in item["bbox"]),
                        float(item.get("confidence", 0.0)),
                        keypoints,
                    )
                )
            result = PoseResult(
                str(payload["camera_id"]),
                int(payload["frame_id"]),
                float(payload["captured_mono"]),
                float(payload["produced_mono"]),
                tuple(people),
            )
            with self._lock:
                self._results[result.camera_id] = result
                self._processed += 1
                self._last_inference_ms = float(payload.get("wall_ms", 0.0))
                self._last_error = ""

    def _next_job(self):
        snapshot = self.detections.snapshot() if self.detections is not None else {}
        camera_ids = tuple(sorted(self.frame_stores))
        if not camera_ids:
            return None
        for step in range(len(camera_ids)):
            index = (self._cursor + step) % len(camera_ids)
            camera_id = camera_ids[index]
            detection = snapshot.get(camera_id)
            if detection is None:
                continue
            frame_id = int(detection.frame_id)
            if frame_id <= self._last_frame.get(camera_id, -1):
                continue
            self._last_frame[camera_id] = frame_id
            self._seen[camera_id] = self._seen.get(camera_id, 0) + 1
            if self._seen[camera_id] % self.every_n:
                continue
            age_ms = max(
                0.0,
                (time.monotonic() - float(detection.frame_captured_monotonic)) * 1000.0,
            )
            if self.max_frame_age_ms and age_ms > self.max_frame_age_ms:
                self._stale_skips += 1
                continue
            store = self.frame_stores.get(camera_id)
            frame = (
                store.get_frame(frame_id)
                if store is not None and hasattr(store, "get_frame")
                else None
            )
            if frame is None:
                self._frame_misses += 1
                continue
            crops = []
            for box in sorted(
                detection.boxes,
                key=lambda item: float(item.confidence),
                reverse=True,
            )[: self.max_people]:
                crop, bbox, offset = self._crop(frame, box)
                if getattr(crop, "size", 0) == 0:
                    continue
                crops.append((crop.copy(), bbox, offset))
            if not crops:
                continue
            self._cursor = (index + 1) % len(camera_ids)
            self._job_id += 1
            return (
                self._job_id,
                camera_id,
                frame_id,
                float(detection.frame_captured_monotonic),
                crops,
            )
        return None

    def _run(self):
        while not self._stop.is_set():
            self._process_died()
            if self._process is None:
                if time.monotonic() >= self._next_restart_mono:
                    try:
                        if self._next_restart_mono > 0:
                            self._restarts += 1
                        self._spawn()
                    except Exception as exc:
                        with self._lock:
                            self._errors += 1
                            self._last_error = f"{type(exc).__name__}: {exc}"
                        self._next_restart_mono = (
                            time.monotonic() + self.restart_backoff_sec
                        )
                self._stop.wait(0.05)
                continue

            self._drain()
            if not self._ready or self._inflight_job is not None:
                self._stop.wait(0.01)
                continue
            job = self._next_job()
            if job is None:
                self._stop.wait(0.02)
                continue
            try:
                self._connection.send(job)
                self._inflight_job = int(job[0])
            except (BrokenPipeError, EOFError, OSError) as exc:
                with self._lock:
                    self._errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self._process_died()


# Runtime import compatibility: app.py imports PoseCoordinator from the package.
PoseCoordinator = PoseProcessCoordinator
