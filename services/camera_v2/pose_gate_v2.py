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

    # S is the production default: materially more accurate than N while still
    # reasonable for sparse 224px crop validation. M remains an explicit option.
    for name in ("yolo26s-pose.pt", "yolo26m-pose.pt"):
        local = ROOT / name
        if local.is_file():
            return str(local)
    return "yolo26s-pose.pt"


def _area(box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b) -> float:
    inter = _intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _containment(a, b) -> float:
    inter = _intersection(a, b)
    return inter / max(1.0, min(_area(a), _area(b)))


def _center_distance(a, b) -> float:
    acx = (float(a[0]) + float(a[2])) * 0.5
    acy = (float(a[1]) + float(a[3])) * 0.5
    bcx = (float(b[0]) + float(b[2])) * 0.5
    bcy = (float(b[1]) + float(b[3])) * 0.5
    aw = max(2.0, float(a[2]) - float(a[0]))
    ah = max(2.0, float(a[3]) - float(a[1]))
    bw = max(2.0, float(b[2]) - float(b[0]))
    bh = max(2.0, float(b[3]) - float(b[1]))
    scale = max(12.0, min((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5))
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale


def _same_region(a, b, *, iou_min: float, containment_min: float, center_max: float) -> bool:
    if _iou(a, b) >= iou_min:
        return True
    return _containment(a, b) >= containment_min and _center_distance(a, b) <= center_max


def _pose_evidence(result: Any, crop_shape: tuple[int, int, int]) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return {"accept": False, "pose_conf": 0.0, "usable": 0, "strong": 0, "torso": 0, "instances": 0}

    try:
        confs = boxes.conf.detach().cpu().numpy()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        kpts = keypoints.data.detach().cpu().numpy()
    except Exception:
        return {"accept": False, "pose_conf": 0.0, "usable": 0, "strong": 0, "torso": 0, "instances": int(len(boxes))}

    kp_usable = float(os.environ.get("CAMERA_V2_POSE_GATE_KP_USABLE", "0.25"))
    kp_strong = float(os.environ.get("CAMERA_V2_POSE_GATE_KP_STRONG", "0.50"))
    min_pose_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MIN_POSE_CONF", "0.05"))
    h, w = crop_shape[:2]
    crop_area = max(1.0, float(w * h))
    torso_indices = (5, 6, 11, 12)

    best = {"accept": False, "pose_conf": 0.0, "usable": 0, "strong": 0, "torso": 0, "instances": int(len(confs))}
    best_quality = -1.0

    for conf, box, kp in zip(confs, xyxy, kpts):
        if kp.ndim != 2 or kp.shape[0] < 13:
            continue
        visibility = kp[:, 2] if kp.shape[1] >= 3 else np.ones((kp.shape[0],), dtype=np.float32)
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
        centered = 0.06 <= cx <= 0.94 and 0.04 <= cy <= 0.96
        large_enough = area_ratio >= 0.07

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
        imgsz = max(192, int(os.environ.get("CAMERA_V2_POSE_GATE_IMGSZ", "224")))
        model_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MODEL_CONF", "0.03"))
        model = YOLO(model_spec)
        task = str(getattr(model, "task", "") or "")
        if task and task != "pose":
            raise RuntimeError(f"pose-gate model task must be pose, got {task!r}")

        warm = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(warm, imgsz=imgsz, conf=model_conf, iou=0.70, classes=[0], max_det=4, device=device, verbose=False)
        result_q.put({"type": "ready", "model": model_spec, "device": device, "imgsz": imgsz, "threads": threads})

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
                evidence = [_pose_evidence(result, crop.shape) for result, crop in zip(predictions, crops)]
                result_q.put({
                    "type": "result",
                    "request_id": job.get("request_id"),
                    "evidence": evidence,
                    "pose_ms": (time.monotonic() - started) * 1000.0,
                })
            except BaseException as exc:
                result_q.put({"type": "error", "request_id": job.get("request_id"), "error": f"{type(exc).__name__}: {exc}"})
    except BaseException as exc:
        try:
            result_q.put({"type": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass


@dataclass
class PoseGateMetrics:
    raw: int = 0
    direct: int = 0
    tracker_reuse: int = 0
    cache_accept: int = 0
    cache_reject: int = 0
    pose_checked: int = 0
    pose_accept: int = 0
    pose_reject: int = 0
    low_reject: int = 0
    overflow: int = 0
    final: int = 0
    fallback: int = 0
    pose_ms: float = 0.0


@dataclass
class _CachedDecision:
    box: tuple[float, float, float, float]
    accepted: bool
    expires_at: float


class PoseGateClient:
    """Track-aware sparse pose validator.

    Rules:
    - strong YOLO person boxes bypass pose;
    - ambiguous boxes already covered by a live NvDCF track bypass pose;
    - recent pose decisions are cached spatially;
    - only genuinely new ambiguous candidates reach YOLO26s-pose.
    """

    def __init__(self) -> None:
        self.min_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_MIN_CONF", "0.08"))
        self.strong_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_STRONG_CONF", "0.35"))
        self.fallback_conf = float(os.environ.get("CAMERA_V2_POSE_GATE_FALLBACK_CONF", "0.25"))
        self.padding = max(0.0, min(0.35, float(os.environ.get("CAMERA_V2_POSE_GATE_PADDING", "0.12"))))
        self.max_candidates = max(1, min(8, int(os.environ.get("CAMERA_V2_POSE_GATE_MAX_CANDIDATES", "4"))))
        self.timeout = max(0.1, float(os.environ.get("CAMERA_V2_POSE_GATE_TIMEOUT_SEC", "0.60")))
        self.positive_ttl = max(0.0, float(os.environ.get("CAMERA_V2_POSE_GATE_POSITIVE_TTL_SEC", "12.0")))
        self.negative_ttl = max(0.0, float(os.environ.get("CAMERA_V2_POSE_GATE_NEGATIVE_TTL_SEC", "6.0")))
        self._request_id = 0
        self._disabled_reason = ""
        self._cache: dict[str, list[_CachedDecision]] = {}
        self.ready = False

        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=2)
        self.result_q = ctx.Queue(maxsize=2)
        self.process = ctx.Process(target=_pose_gate_worker, args=(self.job_q, self.result_q), name="camera-v2-pose-gate-v2", daemon=True)
        self.process.start()
        try:
            ready = self.result_q.get(timeout=90.0)
        except pyqueue.Empty:
            ready = {"type": "fatal", "error": "pose-gate startup timeout"}

        if ready.get("type") == "ready":
            self.ready = True
            print(
                "CAMERA_POSE_GATE_READY "
                f"model={ready.get('model')} device={ready.get('device')} imgsz={ready.get('imgsz')} "
                f"threads={ready.get('threads')} primary_min={self.min_conf:.2f} direct={self.strong_conf:.2f} "
                f"cache=+{self.positive_ttl:.0f}s/-{self.negative_ttl:.0f}s tracker_reuse=1",
                flush=True,
            )
        else:
            self._disabled_reason = str(ready.get("error") or "pose gate failed")
            print(f"CAMERA_POSE_GATE_DISABLED reason={self._disabled_reason} fallback_conf={self.fallback_conf:.2f}", flush=True)

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

    def _purge_cache(self, cid: str, now: float) -> list[_CachedDecision]:
        live = [entry for entry in self._cache.get(cid, []) if entry.expires_at > now]
        self._cache[cid] = live[-32:]
        return live

    @staticmethod
    def _matches_tracker(box, trusted_boxes) -> bool:
        for trusted in trusted_boxes or []:
            if _same_region(box, trusted, iou_min=0.34, containment_min=0.68, center_max=0.30):
                return True
        return False

    @staticmethod
    def _matches_cache(box, entry: _CachedDecision) -> bool:
        if entry.accepted:
            return _same_region(box, entry.box, iou_min=0.48, containment_min=0.78, center_max=0.24)
        # Negative decisions are intentionally stricter so a new real person near
        # an old false-positive location is not suppressed by stale cache state.
        return _same_region(box, entry.box, iou_min=0.70, containment_min=0.90, center_max=0.16)

    def _remember(self, cid: str, box, accepted: bool, now: float) -> None:
        ttl = self.positive_ttl if accepted else self.negative_ttl
        if ttl <= 0.0:
            return
        row = _CachedDecision(tuple(float(v) for v in box), bool(accepted), now + ttl)
        live = self._purge_cache(cid, now)
        live.append(row)
        self._cache[cid] = live[-32:]

    def filter(self, cid: str, frame: np.ndarray, rows, trusted_boxes=None) -> tuple[list, PoseGateMetrics]:
        now = time.monotonic()
        metrics = PoseGateMetrics(raw=len(rows))
        direct_indices: set[int] = set()
        ambiguous: list[tuple[int, Any, float, np.ndarray]] = []
        cache = self._purge_cache(cid, now)

        for index, (coords, score_value) in enumerate(rows):
            score = float(score_value)
            if score < self.min_conf:
                metrics.low_reject += 1
                continue
            if score >= self.strong_conf:
                direct_indices.add(index)
                metrics.direct += 1
                continue
            if self._matches_tracker(coords, trusted_boxes):
                direct_indices.add(index)
                metrics.tracker_reuse += 1
                continue

            cached = next((entry for entry in reversed(cache) if self._matches_cache(coords, entry)), None)
            if cached is not None:
                if cached.accepted:
                    direct_indices.add(index)
                    metrics.cache_accept += 1
                else:
                    metrics.cache_reject += 1
                    metrics.pose_reject += 1
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

        ambiguous.sort(key=lambda item: item[2], reverse=True)
        checked = ambiguous[: self.max_candidates]
        overflow = ambiguous[self.max_candidates :]
        metrics.overflow = len(overflow)
        for index, _coords, score, _crop in overflow:
            if score >= self.fallback_conf:
                direct_indices.add(index)
            else:
                metrics.pose_reject += 1

        if checked:
            metrics.pose_checked = len(checked)
            if not self.ready or not self.process.is_alive():
                metrics.fallback = 1
                for index, _coords, score, _crop in checked:
                    if score >= self.fallback_conf:
                        direct_indices.add(index)
                    else:
                        metrics.pose_reject += 1
            else:
                self._request_id += 1
                request_id = self._request_id
                try:
                    self.job_q.put({"request_id": request_id, "camera": cid, "crops": [item[3] for item in checked]}, timeout=0.15)
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
                    evidence = list(result.get("evidence") or [])
                    metrics.pose_ms = float(result.get("pose_ms") or 0.0)
                    # Missing evidence must fail closed for weak candidates.
                    while len(evidence) < len(checked):
                        evidence.append({"accept": False})
                    for item, proof in zip(checked, evidence):
                        index, coords, _score, _crop = item
                        accepted = bool(proof.get("accept"))
                        self._remember(cid, coords, accepted, now)
                        if accepted:
                            direct_indices.add(index)
                            metrics.pose_accept += 1
                        else:
                            metrics.pose_reject += 1

        accepted = [(coords, float(score)) for index, (coords, score) in enumerate(rows) if index in direct_indices]
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
