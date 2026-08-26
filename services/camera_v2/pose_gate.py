from __future__ import annotations

import multiprocessing as mp
import os
import queue as pyqueue
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _resolve_pose_model() -> str:
    configured = os.environ.get("CAMERA_V2_POSE_GATE_MODEL", "").strip()
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path)
        local = ROOT / configured
        return str(local) if local.is_file() else configured

    # Prefer the nano pose model for a validation gate. A local s-pose checkpoint
    # is still accepted as a fallback for machines that already have it cached.
    for name in ("yolo26n-pose.pt", "yolo26s-pose.pt"):
        local = ROOT / name
        if local.is_file():
            return str(local)
    return "yolo26n-pose.pt"


def _pose_evidence(result: Any, crop_shape: tuple[int, int, int]) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return {
            "accept": False,
            "pose_conf": 0.0,
            "usable": 0,
            "strong": 0,
            "torso": 0,
            "instances": 0,
        }

    try:
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        kpts = keypoints.data.detach().cpu().numpy()
    except Exception:
        return {
            "accept": False,
            "pose_conf": 0.0,
            "usable": 0,
            "strong": 0,
            "torso": 0,
            "instances": int(len(boxes)),
        }

    kp_usable = float(os.environ.get("CAMERA_V2_POSE_GATE_KP_USABLE", "0.25"))
    kp_strong = float(os.environ.get("CAMERA_V2_POSE_GATE_KP_STRONG", "0.50"))
    min_pose_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MIN_POSE_CONF", "0.05"))
    h, w = crop_shape[:2]
    crop_area = max(1.0, float(w * h))

    best = {
        "accept": False,
        "pose_conf": 0.0,
        "usable": 0,
        "strong": 0,
        "torso": 0,
        "instances": int(len(confs)),
    }
    best_quality = -1.0

    # COCO person keypoints: shoulders 5/6 and hips 11/12 are useful torso proof.
    torso_indices = (5, 6, 11, 12)

    for conf, box, kp in zip(confs, xyxy, kpts):
        if kp.ndim != 2 or kp.shape[0] < 13:
            continue
        if kp.shape[1] >= 3:
            visibility = kp[:, 2]
        else:
            visibility = np.ones((kp.shape[0],), dtype=np.float32)

        usable_mask = visibility >= kp_usable
        strong_mask = visibility >= kp_strong
        usable = int(usable_mask.sum())
        strong = int(strong_mask.sum())
        torso = int(sum(bool(usable_mask[i]) for i in torso_indices if i < len(usable_mask)))
        pose_conf = float(conf)

        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area_ratio = (bw * bh) / crop_area
        cx = (x1 + x2) * 0.5 / max(1.0, float(w))
        cy = (y1 + y2) * 0.5 / max(1.0, float(h))
        centered = 0.08 <= cx <= 0.92 and 0.05 <= cy <= 0.95
        large_enough = area_ratio >= 0.08

        # A low detector-confidence candidate must show actual articulated body
        # evidence. The second clause protects partially occluded/sitting people
        # where only two reliable joints survive but pose box confidence is strong.
        accept = bool(
            pose_conf >= min_pose_conf
            and centered
            and large_enough
            and (
                (usable >= 3 and torso >= 1)
                or (pose_conf >= 0.20 and usable >= 2 and strong >= 1)
            )
        )
        quality = pose_conf + 0.03 * usable + 0.02 * strong + 0.04 * torso
        if quality > best_quality:
            best_quality = quality
            best = {
                "accept": accept,
                "pose_conf": pose_conf,
                "usable": usable,
                "strong": strong,
                "torso": torso,
                "instances": int(len(confs)),
            }

    return best


def _pose_gate_worker(job_q, result_q) -> None:
    try:
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        try:
            os.nice(10)
        except Exception:
            pass

        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

        import torch
        from ultralytics import YOLO

        threads = max(1, min(4, int(os.environ.get("CAMERA_V2_POSE_GATE_THREADS", "2"))))
        try:
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        model_spec = _resolve_pose_model()
        device = os.environ.get("CAMERA_V2_POSE_GATE_DEVICE", "cpu").strip() or "cpu"
        imgsz = max(192, int(os.environ.get("CAMERA_V2_POSE_GATE_IMGSZ", "256")))
        model_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MODEL_CONF", "0.03"))
        model = YOLO(model_spec)
        task = str(getattr(model, "task", "") or "")
        if task and task != "pose":
            raise RuntimeError(f"pose-gate model task must be pose, got {task!r}")

        warm = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(
            warm,
            imgsz=imgsz,
            conf=model_conf,
            iou=0.70,
            classes=[0],
            max_det=4,
            device=device,
            verbose=False,
        )
        result_q.put(
            {
                "type": "ready",
                "model": model_spec,
                "device": device,
                "imgsz": imgsz,
                "threads": threads,
            }
        )

        while True:
            job = job_q.get()
            if job is None:
                return
            crops = job.get("crops") or []
            started = time.monotonic()
            try:
                predictions = model.predict(
                    crops,
                    imgsz=imgsz,
                    conf=model_conf,
                    iou=0.70,
                    classes=[0],
                    max_det=4,
                    device=device,
                    verbose=False,
                ) if crops else []
                if not isinstance(predictions, (list, tuple)):
                    predictions = [predictions]
                evidence = [
                    _pose_evidence(result, crop.shape)
                    for result, crop in zip(predictions, crops)
                ]
                result_q.put(
                    {
                        "type": "result",
                        "request_id": job.get("request_id"),
                        "evidence": evidence,
                        "pose_ms": (time.monotonic() - started) * 1000.0,
                    }
                )
            except BaseException as exc:
                result_q.put(
                    {
                        "type": "error",
                        "request_id": job.get("request_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except BaseException as exc:
        try:
            result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


@dataclass
class PoseGateMetrics:
    raw: int = 0
    direct: int = 0
    pose_checked: int = 0
    pose_accept: int = 0
    pose_reject: int = 0
    low_reject: int = 0
    overflow: int = 0
    final: int = 0
    fallback: int = 0
    pose_ms: float = 0.0


class PoseGateClient:
    """Validate only ambiguous detector boxes using pose on cropped ROIs.

    The primary YOLO26 TensorRT detector remains authoritative for strong boxes.
    Pose is intentionally CPU/crop-only by default so the GTX 1050 Ti keeps its
    GPU budget for decode/display/TRT8.6/NvDCF.
    """

    def __init__(self) -> None:
        self.min_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MIN_CONF", "0.08"))
        self.strong_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_STRONG_CONF", "0.35"))
        self.fallback_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_FALLBACK_CONF", "0.25"))
        self.padding = max(0.0, min(0.35, float(os.environ.get("CAMERA_V2_POSE_GATE_PADDING", "0.12"))))
        self.max_candidates = max(1, min(12, int(os.environ.get("CAMERA_V2_POSE_GATE_MAX_CANDIDATES", "6"))))
        self.timeout = max(0.1, float(os.environ.get("CAMERA_V2_POSE_GATE_TIMEOUT_SEC", "1.50")))
        self._request_id = 0
        self._disabled_reason = ""
        self.ready = False

        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=2)
        self.result_q = ctx.Queue(maxsize=2)
        self.process = ctx.Process(
            target=_pose_gate_worker,
            args=(self.job_q, self.result_q),
            name="camera-v2-pose-gate",
            daemon=True,
        )
        self.process.start()
        try:
            ready = self.result_q.get(timeout=90.0)
        except pyqueue.Empty:
            ready = {"type": "fatal", "error": "pose-gate startup timeout"}

        if ready.get("type") == "ready":
            self.ready = True
            print(
                "CAMERA_POSE_GATE_READY "
                f"model={ready.get('model')} device={ready.get('device')} "
                f"imgsz={ready.get('imgsz')} threads={ready.get('threads')} "
                f"primary_min={self.min_conf:.2f} direct={self.strong_conf:.2f} "
                f"fallback={self.fallback_conf:.2f}",
                flush=True,
            )
        else:
            self._disabled_reason = str(ready.get("error") or "pose gate failed")
            print(
                "CAMERA_POSE_GATE_DISABLED "
                f"reason={self._disabled_reason} fallback_conf={self.fallback_conf:.2f}",
                flush=True,
            )

    def _crop(self, frame: np.ndarray, coords) -> np.ndarray | None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in coords]
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)
        px = bw * self.padding
        py = bh * self.padding
        left = max(0, int(np.floor(x1 - px)))
        top = max(0, int(np.floor(y1 - py)))
        right = min(w, int(np.ceil(x2 + px)))
        bottom = min(h, int(np.ceil(y2 + py)))
        if right - left < 12 or bottom - top < 20:
            return None
        return np.ascontiguousarray(frame[top:bottom, left:right])

    def _fallback(self, rows, metrics: PoseGateMetrics):
        accepted = []
        for coords, score in rows:
            if float(score) >= self.fallback_conf:
                accepted.append((coords, float(score)))
            else:
                metrics.pose_reject += 1
        metrics.fallback = 1
        metrics.final = len(accepted)
        return accepted, metrics

    def filter(self, cid: str, frame: np.ndarray, rows) -> tuple[list, PoseGateMetrics]:
        metrics = PoseGateMetrics(raw=len(rows))
        direct_indices: set[int] = set()
        ambiguous: list[tuple[int, Any, float, np.ndarray]] = []

        for index, (coords, score_value) in enumerate(rows):
            score = float(score_value)
            if score < self.min_conf:
                metrics.low_reject += 1
                continue
            if score >= self.strong_conf:
                direct_indices.add(index)
                metrics.direct += 1
                continue
            crop = self._crop(frame, coords)
            if crop is None:
                if score >= self.fallback_conf:
                    direct_indices.add(index)
                    metrics.direct += 1
                else:
                    metrics.pose_reject += 1
                continue
            ambiguous.append((index, coords, score, crop))

        # Pose work is bounded. Overflow candidates use the conservative fallback
        # threshold rather than creating an unbounded CPU queue.
        ambiguous.sort(key=lambda item: item[2], reverse=True)
        checked = ambiguous[: self.max_candidates]
        overflow = ambiguous[self.max_candidates :]
        metrics.overflow = len(overflow)
        for index, _coords, score, _crop in overflow:
            if score >= self.fallback_conf:
                direct_indices.add(index)
            else:
                metrics.pose_reject += 1

        if not checked:
            accepted = [
                (coords, float(score))
                for index, (coords, score) in enumerate(rows)
                if index in direct_indices
            ]
            metrics.final = len(accepted)
            return accepted, metrics

        metrics.pose_checked = len(checked)
        if not self.ready or not self.process.is_alive():
            fallback_rows = [(coords, score) for _index, coords, score, _crop in checked]
            fallback_accepted, fallback_metrics = self._fallback(fallback_rows, metrics)
            accepted_checked = {
                checked[i][0]
                for i, row in enumerate(fallback_rows)
                if row in fallback_accepted
            }
            direct_indices |= accepted_checked
            accepted = [
                (coords, float(score))
                for index, (coords, score) in enumerate(rows)
                if index in direct_indices
            ]
            fallback_metrics.final = len(accepted)
            return accepted, fallback_metrics

        self._request_id += 1
        request_id = self._request_id
        try:
            self.job_q.put(
                {
                    "request_id": request_id,
                    "camera": cid,
                    "crops": [item[3] for item in checked],
                },
                timeout=0.15,
            )
            result = self.result_q.get(timeout=self.timeout)
        except pyqueue.Empty:
            result = {"type": "error", "error": "pose-gate timeout"}

        if result.get("type") != "result" or result.get("request_id") != request_id:
            reason = str(result.get("error") or "pose-gate protocol error")
            print(f"CAMERA_POSE_GATE_FALLBACK cid={cid} reason={reason}", flush=True)
            metrics.fallback = 1
            for index, _coords, score, _crop in checked:
                if score >= self.fallback_conf:
                    direct_indices.add(index)
                else:
                    metrics.pose_reject += 1
        else:
            evidence = result.get("evidence") or []
            metrics.pose_ms = float(result.get("pose_ms") or 0.0)
            for item, proof in zip(checked, evidence):
                index, _coords, _score, _crop = item
                if bool(proof.get("accept")):
                    direct_indices.add(index)
                    metrics.pose_accept += 1
                else:
                    metrics.pose_reject += 1

        accepted = [
            (coords, float(score))
            for index, (coords, score) in enumerate(rows)
            if index in direct_indices
        ]
        metrics.final = len(accepted)
        return accepted, metrics

    def close(self) -> None:
        try:
            self.job_q.put_nowait(None)
        except Exception:
            pass
        try:
            self.process.join(timeout=2.0)
        except Exception:
            pass
        if self.process.is_alive():
            try:
                self.process.terminate()
                self.process.join(timeout=1.0)
            except Exception:
                pass
