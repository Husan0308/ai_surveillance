from __future__ import annotations

import os
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LocalKey = tuple[int, int]
PairKey = frozenset[LocalKey]


@dataclass
class VisualSample:
    crop_bgr: np.ndarray
    quality: float
    seen_at: float


@dataclass
class VerifyTask:
    a: LocalKey
    b: LocalKey
    crop_a: np.ndarray
    crop_b: np.ndarray
    seen_a: float
    seen_b: float
    submitted_at: float


@dataclass
class VerifyResult:
    a: LocalKey
    b: LocalKey
    score: float
    distance: float
    visible_parts: int
    seen_a: float
    seen_b: float
    latency_ms: float
    error: str = ""


class KPRPairVerifier:
    """Sparse final authority for cross-camera merges using ECCV'24 KPR.

    The fast TAO embedding remains a cheap candidate generator for every track.
    KPR is intentionally NOT run on every frame/person. It is invoked only after
    the tracklet association layer has already accumulated enough mutual-best
    evidence to attempt a Global-ID merge.

    KPR returns body-part embeddings plus visibility scores. Pair distance is
    computed with the model's own visibility-aware part-distance function, so an
    occluded torso/legs do not contaminate the comparison in the same way as one
    global embedding. Two independent fresh crop-pair verdicts are required before
    a merge is authorized.
    """

    HF_REPO = "trackinglaboratory/keypoint_promptable_reid"
    HF_FILE = "kpr_dancetrack_sportsmot_posetrack21_occludedduke_market_split0.pth.tar"

    def __init__(self) -> None:
        self.enabled = os.environ.get("CAMERA_V2_KPR", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        self.required = os.environ.get("CAMERA_V2_KPR_REQUIRED", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        self.device = os.environ.get("CAMERA_V2_KPR_DEVICE", "cuda").strip().lower()
        self.same_threshold = float(os.environ.get("CAMERA_V2_KPR_SAME", "0.60"))
        self.diff_threshold = float(os.environ.get("CAMERA_V2_KPR_DIFFERENT", "0.43"))
        self.same_votes_required = max(2, int(os.environ.get("CAMERA_V2_KPR_SAME_VOTES", "2")))
        self.diff_votes_required = max(2, int(os.environ.get("CAMERA_V2_KPR_DIFF_VOTES", "2")))
        self.visual_ttl = max(4.0, float(os.environ.get("CAMERA_V2_KPR_VISUAL_TTL", "18")))
        self.block_sec = max(4.0, float(os.environ.get("CAMERA_V2_KPR_BLOCK_SEC", "18")))
        self.min_request_gap = max(0.5, float(os.environ.get("CAMERA_V2_KPR_REQUEST_GAP", "1.0")))
        self.max_queue = 1

        default_model = Path(__file__).resolve().parents[2] / ".runtime" / "kpr" / self.HF_FILE
        self.model_path = Path(os.environ.get("CAMERA_V2_KPR_MODEL", str(default_model))).expanduser()

        self.visuals: dict[LocalKey, deque[VisualSample]] = {}
        self.input_q: queue.Queue[VerifyTask | None] = queue.Queue(maxsize=self.max_queue)
        self.output_q: queue.Queue[VerifyResult] = queue.Queue(maxsize=4)
        self.pending: set[PairKey] = set()
        self.last_request: dict[PairKey, float] = {}
        self.last_used: dict[PairKey, tuple[float, float]] = {}
        self.same_votes: dict[PairKey, int] = defaultdict(int)
        self.diff_votes: dict[PairKey, int] = defaultdict(int)
        self.approved: set[PairKey] = set()
        self.blocked_until: dict[PairKey, float] = {}

        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.fatal_error = ""
        self.last_error = ""
        self.backend = "kpr-uninitialized"
        self.requests = 0
        self.responses = 0
        self.same = 0
        self.different = 0
        self.uncertain = 0
        self.failed = 0
        self.dropped = 0
        self.wait_fresh = 0
        self.no_visual = 0
        self.last_score = -1.0
        self.last_distance = -1.0
        self.last_visible_parts = 0
        self.last_latency_ms = 0.0

        if not self.enabled:
            self.ready_event.set()
            self.backend = "off"
            return
        self.thread = threading.Thread(target=self._run, name="camera-v2-kpr-reid", daemon=True)
        self.thread.start()

    @staticmethod
    def _pair(a: LocalKey, b: LocalKey) -> PairKey:
        return frozenset((a, b))

    @property
    def error(self) -> str:
        return self.fatal_error

    def remember(self, key: LocalKey, crop_bgr: np.ndarray, quality: float) -> None:
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return
        h, w = crop_bgr.shape[:2]
        if h < 40 or w < 14:
            return
        # KPR internally resizes to 256x128. Bounding the cached source crop keeps
        # memory/copy cost low while preserving substantially more detail than its
        # final network input.
        try:
            import cv2

            scale = min(1.0, 420.0 / max(1, h), 210.0 / max(1, w))
            if scale < 0.999:
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (max(14, int(round(w * scale))), max(40, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
        except Exception:
            pass
        sample = VisualSample(np.ascontiguousarray(crop_bgr).copy(), float(quality), time.monotonic())
        rows = self.visuals.get(key)
        if rows is None:
            rows = deque(maxlen=3)
            self.visuals[key] = rows
        if rows and sample.seen_at - rows[-1].seen_at < 0.45:
            if sample.quality > rows[-1].quality:
                rows[-1] = sample
            return
        rows.append(sample)

    def _best_visual(self, key: LocalKey, now: float) -> VisualSample | None:
        rows = [row for row in self.visuals.get(key, ()) if now - row.seen_at <= self.visual_ttl]
        if not rows:
            return None
        # Prefer quality but keep recency material, which gives independent votes
        # when a seated worker slightly changes pose/head orientation.
        return max(rows, key=lambda row: row.quality + 0.015 * max(0.0, self.visual_ttl - (now - row.seen_at)))

    def poll(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        while True:
            try:
                result = self.output_q.get_nowait()
            except queue.Empty:
                break
            pair = self._pair(result.a, result.b)
            self.pending.discard(pair)
            self.responses += 1
            self.last_latency_ms = result.latency_ms
            self.last_score = result.score
            self.last_distance = result.distance
            self.last_visible_parts = result.visible_parts
            self.last_error = result.error
            if result.error:
                self.failed += 1
                continue

            # The task may have waited in a queue while NvDCF IDs changed. Freshness
            # is checked again by the identity controller before an actual merge.
            self.last_used[pair] = (result.seen_a, result.seen_b)
            if result.score >= self.same_threshold and result.visible_parts >= 1:
                self.same += 1
                self.same_votes[pair] += 1
                self.diff_votes[pair] = 0
                if self.same_votes[pair] >= self.same_votes_required:
                    self.approved.add(pair)
                    self.blocked_until.pop(pair, None)
            elif result.score <= self.diff_threshold:
                self.different += 1
                self.diff_votes[pair] += 1
                self.same_votes[pair] = 0
                if self.diff_votes[pair] >= self.diff_votes_required:
                    self.blocked_until[pair] = now + self.block_sec
                    self.approved.discard(pair)
            else:
                self.uncertain += 1
                # Ambiguous evidence must not erase previous support, but it also
                # cannot advance it.

        stale_keys = [
            key for key, rows in self.visuals.items()
            if not rows or now - rows[-1].seen_at > self.visual_ttl * 1.8
        ]
        for key in stale_keys:
            self.visuals.pop(key, None)

    def authorization(self, a: LocalKey, b: LocalKey, now: float | None = None) -> str:
        """Return approved/blocked/pending/waiting/unavailable and schedule if needed."""
        now = time.monotonic() if now is None else float(now)
        self.poll(now)
        pair = self._pair(a, b)
        if not self.enabled:
            return "approved" if not self.required else "unavailable"
        if self.fatal_error:
            return "unavailable"
        if pair in self.approved:
            return "approved"
        if self.blocked_until.get(pair, 0.0) > now:
            return "blocked"
        if pair in self.pending:
            return "pending"
        if not self.ready_event.is_set():
            return "pending"
        if now - self.last_request.get(pair, 0.0) < self.min_request_gap:
            return "waiting"

        va = self._best_visual(a, now)
        vb = self._best_visual(b, now)
        if va is None or vb is None:
            self.no_visual += 1
            return "waiting"
        prev_a, prev_b = self.last_used.get(pair, (0.0, 0.0))
        if va.seen_at <= prev_a + 1e-6 or vb.seen_at <= prev_b + 1e-6:
            self.wait_fresh += 1
            return "waiting"
        if self.input_q.full():
            self.dropped += 1
            return "pending"

        task = VerifyTask(
            a=a,
            b=b,
            crop_a=va.crop_bgr,
            crop_b=vb.crop_bgr,
            seen_a=va.seen_at,
            seen_b=vb.seen_at,
            submitted_at=now,
        )
        try:
            self.input_q.put_nowait(task)
            self.pending.add(pair)
            self.last_request[pair] = now
            self.requests += 1
            return "pending"
        except queue.Full:
            self.dropped += 1
            return "pending"

    def forget_pair(self, a: LocalKey, b: LocalKey) -> None:
        pair = self._pair(a, b)
        self.approved.discard(pair)
        self.blocked_until.pop(pair, None)
        self.same_votes.pop(pair, None)
        self.diff_votes.pop(pair, None)
        self.last_used.pop(pair, None)
        self.last_request.pop(pair, None)

    def _load_engine(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"KPR checkpoint missing: {self.model_path}. Run scripts/setup_kpr_reid.sh"
            )

        import torch
        from torchreid.scripts.builder import build_config
        from torchreid.scripts.default_config import get_default_config
        from torchreid.tools.feature_extractor import KPRFeatureExtractor
        from torchreid.metrics.distance import compute_distance_matrix_using_bp_features

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        base = get_default_config()
        base.use_gpu = bool(use_cuda)
        base.model.load_weights = str(self.model_path)
        base.model.load_config = True
        base.model.pretrained = False
        base.model.discard_test_params = False
        base.model.kpr.dim_reduce_output = 512
        base.model.kpr.masks.preprocess = "five_v"
        base.model.kpr.keypoints.enabled = False
        base.data.sources = ["market1501"]
        base.data.targets = ["market1501"]
        base.data.height = 256
        base.data.width = 128
        base.data.workers = 0
        base.test.normalize_feature = True

        cfg = build_config(config=base, training_enabled=False)
        cfg.use_gpu = bool(use_cuda)
        cfg.model.pretrained = False
        extractor = KPRFeatureExtractor(cfg, image_size=(256, 128), verbose=False)
        self.backend = "kpr-cuda" if use_cuda else "kpr-cpu"
        return torch, extractor, compute_distance_matrix_using_bp_features

    @staticmethod
    def _infer_pair(torch, extractor, distance_fn, task: VerifyTask) -> tuple[float, float, int]:
        with torch.inference_mode():
            _samples, embeddings, visibility, _masks = extractor(
                [{"image": task.crop_a}, {"image": task.crop_b}]
            )
            dist, _part_dist = distance_fn(
                embeddings[0:1],
                embeddings[1:2],
                visibility[0:1],
                visibility[1:2],
                dist_combine_strat="mean",
                use_gpu=False,
                metric="euclidean",
                use_logger=False,
            )
            # KPR's normalized Euclidean part distance is in [0, 2]. The official
            # demo divides by two; convert that normalized distance to similarity.
            distance = max(0.0, min(1.0, float(dist.detach().cpu().reshape(-1)[0]) / 2.0))
            score = 1.0 - distance
            va = visibility[0].detach().float().cpu().reshape(-1)
            vb = visibility[1].detach().float().cpu().reshape(-1)
            visible_parts = int(((va * vb) > 1e-4).sum().item())
            return score, distance, visible_parts

    def _run(self) -> None:
        try:
            torch, extractor, distance_fn = self._load_engine()
            # Warmup with a realistic person-shaped crop so the first live merge
            # does not absorb model initialization latency.
            dummy = np.full((300, 120, 3), 127, dtype=np.uint8)
            warm = VerifyTask((0, 0), (1, 0), dummy, dummy, 0.0, 0.0, time.monotonic())
            self._infer_pair(torch, extractor, distance_fn, warm)
            self.ready_event.set()
            print(
                "CAMERA_KPR_REID ready "
                f"backend={self.backend} checkpoint={self.model_path.name} "
                "role=final-cross-camera-merge-gate part_visibility=1 batch=2",
                flush=True,
            )
        except Exception as exc:
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.last_error = self.fatal_error
            self.ready_event.set()
            print(f"CAMERA_KPR_REID unavailable: {self.fatal_error}", flush=True)
            return

        while not self.stop_event.is_set():
            try:
                task = self.input_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if task is None:
                return
            started = time.monotonic()
            try:
                score, distance, visible_parts = self._infer_pair(torch, extractor, distance_fn, task)
                result = VerifyResult(
                    task.a, task.b, score, distance, visible_parts,
                    task.seen_a, task.seen_b,
                    (time.monotonic() - started) * 1000.0,
                )
            except Exception as exc:
                result = VerifyResult(
                    task.a, task.b, -1.0, 1.0, 0,
                    task.seen_a, task.seen_b,
                    (time.monotonic() - started) * 1000.0,
                    f"{type(exc).__name__}: {exc}",
                )
            try:
                self.output_q.put_nowait(result)
            except queue.Full:
                self.pending.discard(self._pair(task.a, task.b))
                self.dropped += 1

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "required": self.required,
            "ready": self.ready_event.is_set() and not self.fatal_error,
            "backend": self.backend,
            "visual_tracks": len(self.visuals),
            "requests": self.requests,
            "responses": self.responses,
            "pending": len(self.pending),
            "approved": len(self.approved),
            "blocked": sum(1 for until in self.blocked_until.values() if until > time.monotonic()),
            "same": self.same,
            "different": self.different,
            "uncertain": self.uncertain,
            "failed": self.failed,
            "dropped": self.dropped,
            "wait_fresh": self.wait_fresh,
            "no_visual": self.no_visual,
            "score": self.last_score,
            "distance": self.last_distance,
            "visible_parts": self.last_visible_parts,
            "latency_ms": self.last_latency_ms,
            "error": self.fatal_error or self.last_error,
            "model": str(self.model_path),
        }

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.input_q.put_nowait(None)
            except queue.Full:
                pass
