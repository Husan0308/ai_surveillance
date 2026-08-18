from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL = os.environ.get("CAMERA_V2_POSE_MODEL", "yolo26n-pose.pt")
POSE_DEVICE = os.environ.get("CAMERA_V2_POSE_DEVICE", "cpu").strip() or "cpu"
POSE_WIDTH = max(320, int(os.environ.get("CAMERA_V2_POSE_WIDTH", "704")))
POSE_HEIGHT = max(192, int(os.environ.get("CAMERA_V2_POSE_HEIGHT", "384")))
POSE_TARGET_HZ = max(0.20, min(2.0, float(os.environ.get("CAMERA_V2_POSE_TARGET_HZ", "0.80"))))
POSE_CONF = max(0.02, min(0.80, float(os.environ.get("CAMERA_V2_POSE_CONF", "0.12"))))
POSE_KPT_CONF = max(0.05, min(0.90, float(os.environ.get("CAMERA_V2_POSE_KPT_CONF", "0.25"))))
POSE_MIN_VISIBLE = max(3, min(12, int(os.environ.get("CAMERA_V2_POSE_MIN_VISIBLE", "4"))))
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _resolve_model(spec: str) -> str:
    path = Path(spec)
    if path.is_file():
        return str(path)
    local = ROOT / spec
    if local.is_file():
        return str(local)
    # Official Ultralytics model names (for example yolo26n-pose.pt) are resolved
    # by Ultralytics. Keeping this here makes first-use download explicit in logs.
    return spec


def _pose_worker(job_q, result_q) -> None:
    try:
        try:
            os.nice(10)
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

        import torch
        from ultralytics import YOLO

        if POSE_DEVICE.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        else:
            device = POSE_DEVICE
        if device == "cpu":
            try:
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

        model_path = _resolve_model(POSE_MODEL)
        model = YOLO(model_path)
        kwargs = {
            "imgsz": (POSE_HEIGHT, POSE_WIDTH),
            "conf": POSE_CONF,
            "iou": 0.65,
            "max_det": 30,
            "device": device,
            "verbose": False,
            "stream": False,
        }
        warm = np.zeros((POSE_HEIGHT, POSE_WIDTH, 3), dtype=np.uint8)
        model.predict(source=[warm], **kwargs)
        result_q.put(
            {
                "type": "ready",
                "model": model_path,
                "device": device,
                "width": POSE_WIDTH,
                "height": POSE_HEIGHT,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            started = time.monotonic()
            try:
                predictions = model.predict(source=job["frames"], **kwargs)
                completed = time.monotonic()
                camera_rows: list[dict] = []
                for cid, captured_t, frame, prediction in zip(
                    job["cameras"], job["captured"], job["frames"], predictions
                ):
                    frame_h, frame_w = frame.shape[:2]
                    boxes = getattr(prediction, "boxes", None)
                    keypoints = getattr(prediction, "keypoints", None)
                    rows: list[dict] = []
                    if boxes is not None and len(boxes):
                        xyxy = boxes.xyxy.detach().cpu().numpy()
                        confs = boxes.conf.detach().cpu().numpy()
                        data = None
                        if keypoints is not None and getattr(keypoints, "data", None) is not None:
                            data = keypoints.data.detach().cpu().numpy()
                        for index, (coords, box_conf) in enumerate(zip(xyxy, confs)):
                            ankle = None
                            visible = 0
                            ankle_count = 0
                            if data is not None and index < len(data):
                                kp = data[index]
                                if kp.ndim == 2 and kp.shape[0] >= 17:
                                    conf_col = kp[:, 2] if kp.shape[1] >= 3 else np.ones((kp.shape[0],), dtype=np.float32)
                                    visible = int(np.count_nonzero(conf_col >= POSE_KPT_CONF))
                                    ankle_points = []
                                    for kidx in (LEFT_ANKLE, RIGHT_ANKLE):
                                        x = float(kp[kidx, 0])
                                        y = float(kp[kidx, 1])
                                        q = float(conf_col[kidx])
                                        if q >= POSE_KPT_CONF and x > 0.0 and y > 0.0:
                                            ankle_points.append((x, y, q))
                                    ankle_count = len(ankle_points)
                                    if ankle_points:
                                        total_q = sum(p[2] for p in ankle_points)
                                        x = sum(p[0] * p[2] for p in ankle_points) / max(1e-6, total_q)
                                        y = sum(p[1] * p[2] for p in ankle_points) / max(1e-6, total_q)
                                        ankle = [x, y, total_q / len(ankle_points)]
                            rows.append(
                                {
                                    "box": [float(v) for v in coords],
                                    "box_conf": float(box_conf),
                                    "visible_keypoints": visible,
                                    "ankle_count": ankle_count,
                                    "ankle": ankle,
                                }
                            )
                    camera_rows.append(
                        {
                            "camera": cid,
                            "captured": float(captured_t),
                            "frame_width": int(frame_w),
                            "frame_height": int(frame_h),
                            "rows": rows,
                        }
                    )
                result_q.put(
                    {
                        "type": "result",
                        "cameras": camera_rows,
                        "batch_ms": (completed - started) * 1000.0,
                    }
                )
            except BaseException as exc:
                result_q.put({"type": "batch_error", "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})


class PoseAnkleSidecar:
    """Low-rate YOLO26n-pose sidecar using the detector's already-decoded frames.

    The main YOLO26m + NvDCF path remains authoritative and real-time. This sidecar
    never opens RTSP itself. It samples the shared latest-frame mailbox at a bounded
    rate, returns ankle keypoints for heatmap anchoring and pose boxes that the
    caller may use only as conservative recovery observations.
    """

    def __init__(
        self,
        *,
        mailbox,
        camera_ids: list[str],
        on_result: Callable[[dict], None],
    ) -> None:
        self.mailbox = mailbox
        self.camera_ids = list(camera_ids)
        self.on_result = on_result
        self.ctx = mp.get_context("spawn")
        self.job_q = self.ctx.Queue(maxsize=1)
        self.result_q = self.ctx.Queue(maxsize=4)
        self.process = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.ready = False
        self.error = ""
        self.calls = 0
        self.inputs = 0
        self.batch_ms = 0.0
        self.ankles = 0
        self.recovery_candidates = 0
        self.last_versions = {cid: 0 for cid in self.camera_ids}
        self.last_submit = {cid: -1e9 for cid in self.camera_ids}
        self.round_robin = 0
        self.inflight = False

    def start(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        self.stop_event.clear()
        self.process = self.ctx.Process(
            target=_pose_worker,
            args=(self.job_q, self.result_q),
            name="camera-v2-pose-ankle",
            daemon=True,
        )
        self.process.start()
        self.thread = threading.Thread(
            target=self._loop,
            name="camera-v2-pose-ankle-dispatch",
            daemon=True,
        )
        self.thread.start()

    def _drain_results(self) -> None:
        while True:
            try:
                result = self.result_q.get_nowait()
            except queue.Empty:
                return
            kind = result.get("type")
            if kind == "ready":
                with self.lock:
                    self.ready = True
                    self.error = ""
                print(
                    "CAMERA_POSE ready "
                    f"model={result.get('model')} device={result.get('device')} "
                    f"input={result.get('width')}x{result.get('height')} "
                    f"target={POSE_TARGET_HZ:.2f}Hz/cam heatmap_anchor=ankle",
                    flush=True,
                )
                continue
            if kind == "fatal":
                with self.lock:
                    self.error = str(result.get("error") or "pose fatal")
                self.inflight = False
                return
            if kind == "batch_error":
                with self.lock:
                    self.error = str(result.get("error") or "pose batch error")
                self.inflight = False
                continue
            if kind != "result":
                continue
            self.inflight = False
            camera_rows = list(result.get("cameras") or [])
            with self.lock:
                self.calls += 1
                self.inputs += len(camera_rows)
                self.batch_ms = float(result.get("batch_ms") or 0.0)
                self.error = ""
                self.ankles += sum(
                    1
                    for camera in camera_rows
                    for row in (camera.get("rows") or [])
                    if row.get("ankle") is not None
                )
                self.recovery_candidates += sum(
                    1
                    for camera in camera_rows
                    for row in (camera.get("rows") or [])
                    if int(row.get("visible_keypoints") or 0) >= POSE_MIN_VISIBLE
                )
            for camera in camera_rows:
                try:
                    self.on_result(camera)
                except Exception as exc:
                    with self.lock:
                        self.error = f"callback:{type(exc).__name__}: {exc}"

    def _submit_due(self) -> None:
        if not self.ready or self.inflight or not self.camera_ids:
            return
        now = time.monotonic()
        interval = 1.0 / POSE_TARGET_HZ
        for _ in range(len(self.camera_ids)):
            cid = self.camera_ids[self.round_robin % len(self.camera_ids)]
            self.round_robin += 1
            with self.mailbox.cv:
                row = self.mailbox.rows.get(cid)
            if row is None:
                continue
            version, captured_t, frame = row
            if int(version) <= self.last_versions.get(cid, 0):
                continue
            if now - self.last_submit.get(cid, -1e9) < interval:
                continue
            try:
                self.job_q.put_nowait(
                    {
                        "cameras": [cid],
                        "captured": [float(captured_t)],
                        "frames": [frame],
                    }
                )
            except queue.Full:
                return
            self.last_versions[cid] = int(version)
            self.last_submit[cid] = now
            self.inflight = True
            return

    def _loop(self) -> None:
        deadline = time.monotonic() + 45.0
        while not self.stop_event.is_set():
            self._drain_results()
            if not self.ready and self.error:
                self.stop_event.wait(0.20)
                continue
            if not self.ready and time.monotonic() > deadline:
                with self.lock:
                    self.error = "pose worker startup timeout"
                self.stop_event.wait(0.50)
                continue
            self._submit_due()
            self.stop_event.wait(0.018 if self.inflight else 0.035)

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.job_q.put_nowait(None)
        except Exception:
            pass
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.process is not None:
            self.process.join(timeout=3.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

    def metrics(self) -> dict:
        with self.lock:
            return {
                "ready": self.ready,
                "error": self.error,
                "calls": self.calls,
                "inputs": self.inputs,
                "batch_ms": self.batch_ms,
                "ankles": self.ankles,
                "recovery_candidates": self.recovery_candidates,
                "model": POSE_MODEL,
                "device": POSE_DEVICE,
                "target_hz": POSE_TARGET_HZ,
                "keypoint_conf": POSE_KPT_CONF,
            }
