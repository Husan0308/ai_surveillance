from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    camera_id: str
    frame_id: int
    frame_captured_monotonic: float
    produced_monotonic: float
    boxes: tuple[PersonBox, ...]


class LatestDetectionStore:
    """One detector result per camera. Never queues historical visual state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, DetectionResult] = {}

    def put(self, result: DetectionResult) -> None:
        with self._lock:
            previous = self._results.get(result.camera_id)
            if previous is None or result.frame_id >= previous.frame_id:
                self._results[result.camera_id] = result

    def get(self, camera_id: str) -> DetectionResult | None:
        with self._lock:
            return self._results.get(camera_id)

    def snapshot(self):
        with self._lock:
            return dict(self._results)


def _detector_process_main(input_queue, output_queue, config: dict, model_path: str):
    """CUDA/Ultralytics owner process.

    This function is intentionally module-level so multiprocessing ``spawn`` can
    import it without importing the FastAPI app. GStreamer/NVDEC remains in the
    ML service process; PyTorch CUDA lives in this child process. A native CUDA,
    PyTorch or Ultralytics abort therefore cannot take the six camera streams
    down with it.
    """
    import faulthandler
    faulthandler.enable(all_threads=True)

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        device = str(config.get("device", "cuda:0"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

        raw_imgsz = config.get("imgsz", [384, 640])
        imgsz = tuple(int(v) for v in raw_imgsz) if isinstance(raw_imgsz, (list, tuple)) else int(raw_imgsz)
        conf = float(config.get("conf", 0.15))
        iou = float(config.get("iou", 0.45))
        max_det = max(1, int(config.get("max_det", 40)))
        half = bool(config.get("half", False))

        model = YOLO(model_path)
        warm_h, warm_w = (int(imgsz[0]), int(imgsz[1])) if isinstance(imgsz, tuple) else (int(imgsz), int(imgsz))
        warm = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
        model.predict(
            source=[warm],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            classes=[0],
            max_det=max_det,
            device=device,
            half=half,
            verbose=False,
        )
        output_queue.put(("ready", {"device": device, "model": model_path}))
    except BaseException as exc:
        try:
            output_queue.put(("startup_error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        return

    while True:
        try:
            payload = input_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            return
        if payload is None:
            return

        batch_id, entries = payload
        images = [entry[6] for entry in entries]
        started = time.perf_counter()
        try:
            predictions = model.predict(
                source=images,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                classes=[0],
                max_det=max_det,
                device=device,
                half=half,
                verbose=False,
            )
            if device.startswith("cuda"):
                # Make batch timing meaningful and surface asynchronous CUDA
                # errors inside the isolated detector process.
                torch.cuda.synchronize()
        except BaseException as exc:
            try:
                output_queue.put(("batch_error", {"batch_id": batch_id, "error": f"{type(exc).__name__}: {exc}"}))
            except Exception:
                pass
            continue

        wall_ms = (time.perf_counter() - started) * 1000.0
        produced = time.monotonic()
        result_entries = []
        total_boxes = 0
        for entry, prediction in zip(entries, predictions):
            cid, frame_id, captured_mono, source_w, source_h, resized_shape, _image = entry
            resized_h, resized_w = resized_shape
            sx = float(source_w) / max(1.0, float(resized_w))
            sy = float(source_h) / max(1.0, float(resized_h))
            boxes = []
            pred_boxes = getattr(prediction, "boxes", None)
            if pred_boxes is not None and len(pred_boxes):
                xyxy = pred_boxes.xyxy.detach().cpu().tolist()
                confs = pred_boxes.conf.detach().cpu().tolist()
                for coords, confidence in zip(xyxy, confs):
                    boxes.append((
                        float(coords[0]) * sx,
                        float(coords[1]) * sy,
                        float(coords[2]) * sx,
                        float(coords[3]) * sy,
                        float(confidence),
                    ))
            total_boxes += len(boxes)
            result_entries.append((cid, frame_id, captured_mono, produced, boxes))

        try:
            output_queue.put(("result", {
                "batch_id": batch_id,
                "wall_ms": wall_ms,
                "inputs": len(entries),
                "detections": total_boxes,
                "entries": result_entries,
            }))
        except (BrokenPipeError, EOFError, OSError):
            return


class YoloDetectorWorker:
    """Latest-only detector bridge with CUDA isolated in a spawned process.

    The parent ML service only performs a small CPU resize and IPC copy. It never
    imports/initializes PyTorch CUDA for realtime inference. This preserves the
    already-validated camera/display baseline even if the detector child aborts.
    """

    def __init__(self, frame_stores, config: dict, project_root: Path):
        self.frame_stores = dict(frame_stores)
        self.config = dict(config)
        self.project_root = Path(project_root)
        self.camera_ids = sorted(self.frame_stores)
        self.batch_size = max(1, min(len(self.camera_ids) or 1, int(self.config.get("batch_size", 3))))
        self.batch_interval = max(0.0, float(self.config.get("batch_interval_ms", 20.0)) / 1000.0)
        raw_imgsz = self.config.get("imgsz", [384, 640])
        if isinstance(raw_imgsz, (list, tuple)):
            self.input_h, self.input_w = int(raw_imgsz[0]), int(raw_imgsz[1])
        else:
            self.input_h = self.input_w = int(raw_imgsz)
        model_value = str(self.config.get("model", "models/yolo26m.pt"))
        model_path = Path(model_value).expanduser()
        self.model_path = model_path if model_path.is_absolute() else (self.project_root / model_path)
        self.start_delay_sec = max(0.0, float(self.config.get("start_delay_sec", 2.0)))

        self.results = LatestDetectionStore()
        self._stop = threading.Event()
        self._thread = None
        self._process = None
        self._ctx = mp.get_context("spawn")
        self._input_queue = None
        self._output_queue = None
        self._last_versions = {cid: 0 for cid in self.camera_ids}
        self._cursor = 0
        self._batch_id = 0
        self._lock = threading.Lock()
        self._started_mono = time.monotonic()
        self._ready = False
        self._submitted = 0
        self._batches = 0
        self._inputs = 0
        self._detections = 0
        self._dropped_batches = 0
        self._last_batch_ms = 0.0
        self._last_error = ""
        self._per_camera_inputs = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_frame_id = {cid: 0 for cid in self.camera_ids}
        self._per_camera_last_detection_mono = {cid: 0.0 for cid in self.camera_ids}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_bridge, name="core-v1-yolo-bridge", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        q = self._input_queue
        if q is not None:
            try:
                q.put_nowait(None)
            except Exception:
                pass

    def join(self, timeout=10):
        deadline = time.monotonic() + timeout
        if self._thread:
            self._thread.join(max(0.0, deadline - time.monotonic()))
        process = self._process
        if process is not None and process.is_alive():
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(2.0)
        return (not self._thread or not self._thread.is_alive()) and (process is None or not process.is_alive())

    def _spawn_process(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        self._input_queue = self._ctx.Queue(maxsize=1)
        self._output_queue = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_detector_process_main,
            name="core-v1-yolo-cuda",
            args=(self._input_queue, self._output_queue, self.config, str(self.model_path)),
            daemon=False,
        )
        self._process.start()
        log.info("CORE_V1_YOLO_PROCESS_STARTED pid=%s start_method=spawn", self._process.pid)

    def _select_latest_batch(self):
        if not self.camera_ids:
            return []
        selected = []
        n = len(self.camera_ids)
        scanned = 0
        while scanned < n and len(selected) < self.batch_size:
            cid = self.camera_ids[(self._cursor + scanned) % n]
            frame, version = self.frame_stores[cid].get()
            if frame is not None and version > self._last_versions[cid]:
                newest, newest_version = self.frame_stores[cid].get()
                if newest is not None and newest_version > self._last_versions[cid]:
                    selected.append((cid, newest, newest_version))
            scanned += 1
        self._cursor = (self._cursor + max(1, scanned)) % n
        return selected

    def _prepare_payload(self, selected):
        import cv2
        entries = []
        for cid, frame, version in selected:
            # Resize only frames that are actually being submitted to YOLO. The
            # camera capture and 12 FPS display paths keep their original frames.
            small = cv2.resize(frame.image, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
            entries.append((
                cid,
                int(frame.frame_id),
                float(frame.captured_monotonic),
                int(frame.width),
                int(frame.height),
                (self.input_h, self.input_w),
                small,
            ))
        return entries

    def _submit_latest(self):
        selected = self._select_latest_batch()
        if not selected:
            return
        entries = self._prepare_payload(selected)
        self._batch_id += 1
        payload = (self._batch_id, entries)

        # Strict latest-only IPC. If the CUDA process has not consumed the last
        # batch, discard it and replace it with the newest camera snapshots.
        try:
            self._input_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
                with self._lock:
                    self._dropped_batches += 1
            except Exception:
                pass
            try:
                self._input_queue.put_nowait(payload)
            except queue.Full:
                return

        for cid, _frame, version in selected:
            self._last_versions[cid] = version
        with self._lock:
            self._submitted += len(entries)

    def _drain_outputs(self):
        if self._output_queue is None:
            return
        while True:
            try:
                kind, payload = self._output_queue.get_nowait()
            except queue.Empty:
                return
            except (EOFError, OSError):
                return

            if kind == "ready":
                with self._lock:
                    self._ready = True
                    self._last_error = ""
                log.info("CORE_V1_YOLO_READY process_pid=%s device=%s model=%s", self._process.pid if self._process else None, payload.get("device"), payload.get("model"))
                continue

            if kind in {"startup_error", "batch_error"}:
                error = payload if isinstance(payload, str) else payload.get("error", str(payload))
                with self._lock:
                    self._last_error = error
                log.error("CORE_V1_YOLO_%s %s", kind.upper(), error)
                continue

            if kind != "result":
                continue

            for cid, frame_id, captured_mono, produced_mono, raw_boxes in payload["entries"]:
                boxes = tuple(PersonBox(*map(float, box)) for box in raw_boxes)
                self.results.put(DetectionResult(
                    camera_id=str(cid),
                    frame_id=int(frame_id),
                    frame_captured_monotonic=float(captured_mono),
                    produced_monotonic=float(produced_mono),
                    boxes=boxes,
                ))
                with self._lock:
                    self._per_camera_inputs[str(cid)] += 1
                    self._per_camera_last_frame_id[str(cid)] = int(frame_id)
                    self._per_camera_last_detection_mono[str(cid)] = float(produced_mono)

            with self._lock:
                self._batches += 1
                self._inputs += int(payload["inputs"])
                self._detections += int(payload["detections"])
                self._last_batch_ms = float(payload["wall_ms"])
                self._last_error = ""

    def _run_bridge(self):
        if self.start_delay_sec and self._stop.wait(self.start_delay_sec):
            return
        try:
            self._spawn_process()
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            log.exception("CORE_V1_YOLO_PROCESS_START_FAILED")
            return

        while not self._stop.is_set():
            self._drain_outputs()
            process = self._process
            if process is not None and not process.is_alive():
                exitcode = process.exitcode
                with self._lock:
                    self._ready = False
                    if not self._last_error:
                        self._last_error = f"detector process exited unexpectedly with exitcode={exitcode}"
                log.error("CORE_V1_YOLO_PROCESS_EXITED exitcode=%s camera_core_continues=true", exitcode)
                break
            if self._ready:
                self._submit_latest()
            self._stop.wait(self.batch_interval if self.batch_interval else 0.01)

        self._drain_outputs()

    def metrics(self):
        now = time.monotonic()
        process = self._process
        with self._lock:
            elapsed = max(0.001, now - self._started_mono)
            cameras = {}
            for cid in self.camera_ids:
                last = self._per_camera_last_detection_mono[cid]
                cameras[cid] = {
                    "inputs": self._per_camera_inputs[cid],
                    "input_rate": self._per_camera_inputs[cid] / elapsed,
                    "last_frame_id": self._per_camera_last_frame_id[cid],
                    "observation_age_ms": ((now - last) * 1000.0) if last else None,
                }
            return {
                "ready": self._ready,
                "process_alive": bool(process and process.is_alive()),
                "process_pid": process.pid if process else None,
                "process_exitcode": process.exitcode if process and not process.is_alive() else None,
                "start_method": "spawn",
                "model": str(self.model_path),
                "device": str(self.config.get("device", "cuda:0")),
                "batch_size": self.batch_size,
                "submitted_camera_inputs": self._submitted,
                "dropped_pending_batches": self._dropped_batches,
                "batches": self._batches,
                "batch_rate": self._batches / elapsed,
                "camera_inputs": self._inputs,
                "camera_input_rate": self._inputs / elapsed,
                "detections": self._detections,
                "last_batch_ms": self._last_batch_ms,
                "last_error": self._last_error,
                "cameras": cameras,
            }
