from __future__ import annotations

from dataclasses import asdict, dataclass
import multiprocessing as mp
import queue
import threading
import time
from typing import Any

from services.ml_service.app.latest_frame import LatestFrameStore


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    camera_id: str
    frame_id: int
    captured_monotonic: float
    inferred_monotonic: float
    batch_ms: float
    detections: tuple[Detection, ...]


class DetectionStore:
    """Thread-safe latest detection result per camera."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: dict[str, DetectionSnapshot] = {}

    def put(self, snapshot: DetectionSnapshot) -> None:
        with self._lock:
            self._rows[snapshot.camera_id] = snapshot

    def get(self, camera_id: str) -> DetectionSnapshot | None:
        with self._lock:
            return self._rows.get(camera_id)

    def payload(self, camera_id: str) -> dict | None:
        snapshot = self.get(camera_id)
        if snapshot is None:
            return None
        now = time.monotonic()
        return {
            "camera_id": snapshot.camera_id,
            "frame_id": snapshot.frame_id,
            "people": len(snapshot.detections),
            "age_ms": max(0.0, (now - snapshot.inferred_monotonic) * 1000.0),
            "source_age_ms": max(0.0, (now - snapshot.captured_monotonic) * 1000.0),
            "batch_ms": snapshot.batch_ms,
            "detections": [asdict(row) for row in snapshot.detections],
        }


@dataclass
class DetectorMetrics:
    state: str = "stopped"
    model: str = ""
    device: str = ""
    worker_pid: int | None = None
    worker_exitcode: int | None = None
    cuda_capability: str = ""
    torch_arches: tuple[str, ...] = ()
    batches: int = 0
    images: int = 0
    last_batch_ms: float = 0.0
    average_batch_ms: float = 0.0
    last_error: str = ""


def _cuda_device_index(device: str) -> int:
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    raise RuntimeError(f"person detector requires CUDA device, got {device!r}")


def _parse_sm_arch(value: str) -> tuple[int, int] | None:
    text = str(value).strip().lower()
    if not text.startswith("sm_"):
        return None
    digits = text[3:]
    if len(digits) < 2 or not digits.isdigit():
        return None
    return int(digits[:-1]), int(digits[-1])


def _compatible_arches(
    capability: tuple[int, int], compiled_arches: tuple[str, ...]
) -> tuple[str, ...]:
    """Return cubin targets that can execute on this desktop GPU.

    CUDA desktop cubins are binary-compatible within one compute-capability
    major family when the device minor version is >= the cubin target minor.
    Therefore an sm_60 cubin is compatible with an sm_61 GPU.
    """
    gpu_major, gpu_minor = capability
    compatible: list[str] = []
    for arch in compiled_arches:
        parsed = _parse_sm_arch(arch)
        if parsed is None:
            continue
        arch_major, arch_minor = parsed
        if arch_major == gpu_major and arch_minor <= gpu_minor:
            compatible.append(arch)
    return tuple(compatible)


def _detector_worker(config: dict[str, Any], job_q, result_q) -> None:
    """CUDA-only child process. A native crash here must not kill ml_service."""
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

        # Do not require an exact get_arch_list() match. NVIDIA desktop cubins
        # are compatible forward across minor revisions of the same major
        # compute capability (sm_60 -> sm_61). If no compatible cubin appears
        # in the list, the real kernel below is still authoritative because
        # the wheel may contain PTX that the driver can JIT for this device.
        if compiled_arches and not compatible_arches:
            result_q.put(
                {
                    "type": "warning",
                    "warning": (
                        f"no same-major compatible cubin listed for {required_arch}; "
                        "probing CUDA kernel/PTX JIT"
                    ),
                }
            )

        probe = torch.ones((32, 32), device=device)
        probe = probe @ probe
        _ = float(probe.sum().item())
        torch.cuda.synchronize(device_index)
        del probe

        model = YOLO(str(config["model"]))
        predict_kwargs: dict[str, Any] = {
            "imgsz": (int(config["height"]), int(config["width"])),
            "classes": [0],
            "conf": float(config["confidence"]),
            "iou": float(config["iou"]),
            "max_det": int(config["max_detections"]),
            "device": device,
            "verbose": False,
            "stream": False,
            "rect": True,
            "half": bool(config["half"]),
        }

        warm = [
            np.zeros((int(config["height"]), int(config["width"]), 3), dtype=np.uint8)
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
                detections = []
                boxes = getattr(prediction, "boxes", None)
                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.detach().cpu().tolist()
                    scores = boxes.conf.detach().cpu().tolist()
                    for xyxy, confidence in zip(coords, scores):
                        detections.append(
                            {
                                "xyxy": [float(value) for value in xyxy],
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


class PersonDetector:
    """Latest-only person detection with CUDA isolated in a spawned process.

    Camera ingest, FastAPI and MJPEG live in the ml_service parent process.
    The child owns PyTorch/Ultralytics/CUDA. No tracker, ReID or face logic is
    present here. At most one inference batch is in flight, so a slow detector
    can never create an unbounded frame backlog.
    """

    def __init__(self, config, stores: dict[str, LatestFrameStore]) -> None:
        self.config = config
        self.stores = dict(stores)
        self.results = DetectionStore()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._metrics = DetectorMetrics(state="disabled" if not config.enabled else "stopped")
        self._camera_updates: dict[str, int] = {camera_id: 0 for camera_id in self.stores}
        self._camera_started: dict[str, float] = {camera_id: 0.0 for camera_id in self.stores}
        self._camera_last_infer: dict[str, float] = {camera_id: 0.0 for camera_id in self.stores}
        self._ctx = mp.get_context("spawn")
        self._process = None
        self._job_q = None
        self._result_q = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="person-detector-manager", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        job_q = self._job_q
        if job_q is not None:
            try:
                job_q.put_nowait(None)
            except Exception:
                pass

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout)
        process = self._process
        if process is not None and process.is_alive():
            process.join(1.0)
            if process.is_alive():
                process.terminate()
                process.join(1.0)

    def metrics(self) -> dict:
        with self._lock:
            payload = asdict(self._metrics)
        payload["enabled"] = self.enabled
        payload["batch_size"] = int(self.config.batch_size)
        payload["target_fps_per_camera"] = float(self.config.target_fps_per_camera)
        payload["imgsz"] = [int(self.config.height), int(self.config.width)]
        payload["isolation"] = "spawn-process"
        return payload

    def camera_metrics(self, camera_id: str) -> dict:
        now = time.monotonic()
        snapshot = self.results.get(camera_id)
        with self._lock:
            updates = int(self._camera_updates.get(camera_id, 0))
            started = float(self._camera_started.get(camera_id, 0.0))
            last_infer = float(self._camera_last_infer.get(camera_id, 0.0))
            state = self._metrics.state
            error = self._metrics.last_error
        elapsed = max(0.001, now - started) if started else 0.0
        return {
            "state": state,
            "people": len(snapshot.detections) if snapshot else 0,
            "fps": updates / elapsed if elapsed else 0.0,
            "age_ms": max(0.0, (now - last_infer) * 1000.0) if last_infer else None,
            "frame_id": snapshot.frame_id if snapshot else 0,
            "last_error": error if state == "error" else "",
        }

    def snapshot_payload(self, camera_id: str) -> dict:
        return {
            "detector": self.metrics(),
            "result": self.results.payload(camera_id),
        }

    def _set_state(self, state: str, error: str = "") -> None:
        with self._lock:
            self._metrics.state = state
            self._metrics.last_error = error

    def _worker_config(self) -> dict[str, Any]:
        return {
            "model": str(self.config.model),
            "device": str(self.config.device),
            "width": int(self.config.width),
            "height": int(self.config.height),
            "batch_size": int(self.config.batch_size),
            "confidence": float(self.config.confidence),
            "iou": float(self.config.iou),
            "max_detections": int(self.config.max_detections),
            "half": bool(self.config.half),
        }

    def _start_worker(self) -> None:
        self._job_q = self._ctx.Queue(maxsize=1)
        self._result_q = self._ctx.Queue(maxsize=8)
        self._process = self._ctx.Process(
            target=_detector_worker,
            args=(self._worker_config(), self._job_q, self._result_q),
            name="person-detector-cuda",
            daemon=True,
        )
        self._process.start()
        with self._lock:
            self._metrics.worker_pid = int(self._process.pid) if self._process.pid else None
            self._metrics.worker_exitcode = None
        print(f"[DETECT] spawned CUDA worker pid={self._process.pid}", flush=True)

    def _consume_message(self, message: dict, next_due: dict[str, float]) -> bool:
        kind = message.get("type")
        if kind == "warning":
            warning = str(message.get("warning") or "detector compatibility warning")
            print(f"[DETECT] warning: {warning}", flush=True)
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
                f"imgsz={self.config.width}x{self.config.height} compatible={compatible}",
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
        items = list(message.get("items") or ())
        for item in items:
            camera_id = str(item["camera_id"])
            rows = tuple(
                Detection(
                    xyxy=tuple(float(value) for value in row["xyxy"]),
                    confidence=float(row["confidence"]),
                )
                for row in item.get("detections") or ()
            )
            snapshot = DetectionSnapshot(
                camera_id=camera_id,
                frame_id=int(item["frame_id"]),
                captured_monotonic=float(item["captured_monotonic"]),
                inferred_monotonic=finished,
                batch_ms=batch_ms,
                detections=rows,
            )
            self.results.put(snapshot)
            next_due[camera_id] = finished + 1.0 / max(
                0.1, float(self.config.target_fps_per_camera)
            )
            with self._lock:
                if not self._camera_started[camera_id]:
                    self._camera_started[camera_id] = finished
                self._camera_updates[camera_id] += 1
                self._camera_last_infer[camera_id] = finished

        with self._lock:
            self._metrics.batches += 1
            self._metrics.images += len(items)
            self._metrics.last_batch_ms = batch_ms
            count = self._metrics.batches
            self._metrics.average_batch_ms += (
                batch_ms - self._metrics.average_batch_ms
            ) / max(1, count)
        return True

    def _run(self) -> None:
        self._set_state("starting")
        camera_ids = list(self.stores)
        last_versions = {camera_id: 0 for camera_id in camera_ids}
        next_due = {camera_id: 0.0 for camera_id in camera_ids}
        cursor = 0
        in_flight = False

        try:
            self._start_worker()

            while not self._stop.is_set():
                result_q = self._result_q
                if result_q is not None:
                    while True:
                        try:
                            message = result_q.get_nowait()
                        except queue.Empty:
                            break
                        except Exception:
                            break
                        if self._consume_message(message, next_due):
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
                    self._stop.wait(0.01)
                    continue

                now = time.monotonic()
                selected = []
                checked = 0
                while checked < len(camera_ids) and len(selected) < int(self.config.batch_size):
                    index = (cursor + checked) % len(camera_ids)
                    checked += 1
                    camera_id = camera_ids[index]
                    if now < next_due[camera_id]:
                        continue
                    frame, version = self.stores[camera_id].get()
                    if frame is None or version <= last_versions[camera_id]:
                        continue
                    selected.append((camera_id, frame, version))

                cursor = (cursor + max(1, checked)) % max(1, len(camera_ids))
                if not selected:
                    self._stop.wait(0.005)
                    continue

                payload = {
                    "items": [
                        {
                            "camera_id": camera_id,
                            "frame_id": int(frame.frame_id),
                            "captured_monotonic": float(frame.captured_monotonic),
                            "image": frame.image,
                        }
                        for camera_id, frame, _version in selected
                    ]
                }
                job_q = self._job_q
                if job_q is None:
                    self._set_state("error", "detector job queue unavailable")
                    return
                try:
                    job_q.put_nowait(payload)
                except queue.Full:
                    self._stop.wait(0.005)
                    continue

                for camera_id, _frame, version in selected:
                    last_versions[camera_id] = version
                in_flight = True

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._set_state("error", message)
            print(f"[DETECT] manager error: {message}", flush=True)
        finally:
            process = self._process
            if self._stop.is_set() and process is not None and process.is_alive():
                try:
                    if self._job_q is not None:
                        self._job_q.put_nowait(None)
                except Exception:
                    pass
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1.0)
            if self._stop.is_set():
                with self._lock:
                    if self._metrics.state != "error":
                        self._metrics.state = "stopped"
