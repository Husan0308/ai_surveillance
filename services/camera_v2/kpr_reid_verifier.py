from __future__ import annotations

import multiprocessing as mp
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


def _load_kpr_engine(model_path: str, requested_device: str, gpu_fraction: float):
    """Load KPR entirely inside the isolated worker process."""
    import torch
    from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
    from torchreid.scripts.builder import build_config
    from torchreid.scripts.default_config import get_default_config
    from torchreid.tools.feature_extractor import KPRFeatureExtractor

    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        # Keep a hard PyTorch allocation ceiling so the sparse KPR verifier cannot
        # starve DeepStream/YOLO on the 4 GB GTX 1050 Ti.  If the model cannot fit,
        # this worker exits and the parent transparently retries KPR on CPU.
        try:
            torch.cuda.set_per_process_memory_fraction(
                max(0.15, min(0.60, float(gpu_fraction))), device=0
            )
        except Exception:
            pass

    base = get_default_config()
    base.use_gpu = bool(use_cuda)
    base.model.load_weights = str(model_path)
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
    backend = "kpr-cuda-process" if use_cuda else "kpr-cpu-process"
    return torch, extractor, compute_distance_matrix_using_bp_features, backend


def _infer_pair(torch, extractor, distance_fn, task: VerifyTask) -> tuple[float, float, int]:
    model = getattr(extractor, "model", None)
    is_cuda = False
    try:
        parameter = next(model.parameters())
        is_cuda = parameter.is_cuda
    except Exception:
        pass

    with torch.inference_mode():
        # Autocast lowers activation memory on Pascal while keeping the official
        # KPR body-part distance logic unchanged. If the local PyTorch build does
        # not support CUDA autocast here, inference automatically uses FP32.
        if is_cuda:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _samples, embeddings, visibility, _masks = extractor(
                        [{"image": task.crop_a}, {"image": task.crop_b}]
                    )
            except Exception:
                _samples, embeddings, visibility, _masks = extractor(
                    [{"image": task.crop_a}, {"image": task.crop_b}]
                )
        else:
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
        # Official KPR demo normalizes its [0, 2] distance by /2.
        distance = max(
            0.0,
            min(1.0, float(dist.detach().cpu().reshape(-1)[0]) / 2.0),
        )
        score = 1.0 - distance
        va = visibility[0].detach().float().cpu().reshape(-1)
        vb = visibility[1].detach().float().cpu().reshape(-1)
        visible_parts = int(((va * vb) > 1e-4).sum().item())
        return score, distance, visible_parts


def _kpr_worker_main(
    input_q,
    output_q,
    control_q,
    model_path: str,
    requested_device: str,
    gpu_fraction: float,
) -> None:
    """Native-model worker. A segfault/core dump is contained to this process."""
    try:
        torch, extractor, distance_fn, backend = _load_kpr_engine(
            model_path, requested_device, gpu_fraction
        )
        dummy = np.full((300, 120, 3), 127, dtype=np.uint8)
        warm = VerifyTask((0, 0), (1, 0), dummy, dummy, 0.0, 0.0, time.monotonic())
        _infer_pair(torch, extractor, distance_fn, warm)
        control_q.put(
            {
                "type": "ready",
                "backend": backend,
                "pid": os.getpid(),
            }
        )
    except BaseException as exc:
        try:
            control_q.put(
                {
                    "type": "fatal",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pid": os.getpid(),
                }
            )
        except Exception:
            pass
        return

    while True:
        try:
            task = input_q.get(timeout=0.25)
        except queue.Empty:
            continue
        if task is None:
            return
        started = time.monotonic()
        try:
            score, distance, visible_parts = _infer_pair(
                torch, extractor, distance_fn, task
            )
            result = VerifyResult(
                task.a,
                task.b,
                score,
                distance,
                visible_parts,
                task.seen_a,
                task.seen_b,
                (time.monotonic() - started) * 1000.0,
            )
        except BaseException as exc:
            result = VerifyResult(
                task.a,
                task.b,
                -1.0,
                1.0,
                0,
                task.seen_a,
                task.seen_b,
                (time.monotonic() - started) * 1000.0,
                f"{type(exc).__name__}: {exc}",
            )
        try:
            output_q.put(result, timeout=0.1)
        except Exception:
            pass


class KPRPairVerifier:
    """Sparse final authority for peer-camera merges using ECCV'24 KPR.

    The fast TAO embedding remains the cheap candidate generator. KPR is loaded
    lazily only when a real mutual-best merge candidate exists. It runs in a
    separate spawn process so CUDA/PyTorch native failures cannot abort the
    DeepStream camera wall. On this 4 GB machine, CUDA failure automatically
    falls back to CPU rather than killing the surveillance pipeline.
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
        self.device = os.environ.get("CAMERA_V2_KPR_DEVICE", "auto").strip().lower()
        if self.device == "auto":
            self.device = "cuda"
        self.gpu_fraction = float(os.environ.get("CAMERA_V2_KPR_GPU_FRACTION", "0.32"))
        self.allow_cpu_fallback = os.environ.get(
            "CAMERA_V2_KPR_CPU_FALLBACK", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}

        self.same_threshold = float(os.environ.get("CAMERA_V2_KPR_SAME", "0.60"))
        self.diff_threshold = float(os.environ.get("CAMERA_V2_KPR_DIFFERENT", "0.43"))
        self.same_votes_required = max(
            2, int(os.environ.get("CAMERA_V2_KPR_SAME_VOTES", "2"))
        )
        self.diff_votes_required = max(
            2, int(os.environ.get("CAMERA_V2_KPR_DIFF_VOTES", "2"))
        )
        self.visual_ttl = max(
            4.0, float(os.environ.get("CAMERA_V2_KPR_VISUAL_TTL", "18"))
        )
        self.block_sec = max(
            4.0, float(os.environ.get("CAMERA_V2_KPR_BLOCK_SEC", "18"))
        )
        self.min_request_gap = max(
            0.5, float(os.environ.get("CAMERA_V2_KPR_REQUEST_GAP", "1.0"))
        )

        default_model = (
            Path(__file__).resolve().parents[2]
            / ".runtime"
            / "kpr"
            / self.HF_FILE
        )
        self.model_path = Path(
            os.environ.get("CAMERA_V2_KPR_MODEL", str(default_model))
        ).expanduser()

        self.visuals: dict[LocalKey, deque[VisualSample]] = {}
        self.pending: set[PairKey] = set()
        self.last_request: dict[PairKey, float] = {}
        self.last_used: dict[PairKey, tuple[float, float]] = {}
        self.same_votes: dict[PairKey, int] = defaultdict(int)
        self.diff_votes: dict[PairKey, int] = defaultdict(int)
        self.approved: set[PairKey] = set()
        self.blocked_until: dict[PairKey, float] = {}

        self.ctx = mp.get_context("spawn")
        self.process = None
        self.input_q = None
        self.output_q = None
        self.control_q = None
        self.active_device = "none"
        self.fallback_used = False
        self.fallbacks = 0
        self.worker_exit = None

        self.ready_event = threading.Event()
        self.fatal_error = ""
        self.last_error = ""
        self.backend = "off" if not self.enabled else "lazy"

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

    @staticmethod
    def _pair(a: LocalKey, b: LocalKey) -> PairKey:
        return frozenset((a, b))

    @property
    def error(self) -> str:
        return self.fatal_error

    def _new_queues(self) -> None:
        self.input_q = self.ctx.Queue(maxsize=1)
        self.output_q = self.ctx.Queue(maxsize=4)
        self.control_q = self.ctx.Queue(maxsize=4)

    def _spawn(self, device: str) -> None:
        if not self.enabled:
            return
        if not self.model_path.exists():
            self.fatal_error = (
                f"FileNotFoundError: KPR checkpoint missing: {self.model_path}. "
                "Run scripts/setup_kpr_reid.sh"
            )
            self.last_error = self.fatal_error
            return

        self._new_queues()
        self.ready_event.clear()
        self.active_device = device
        self.backend = f"starting-{device}-process"
        self.worker_exit = None
        self.process = self.ctx.Process(
            target=_kpr_worker_main,
            args=(
                self.input_q,
                self.output_q,
                self.control_q,
                str(self.model_path),
                device,
                self.gpu_fraction,
            ),
            name=f"camera-v2-kpr-{device}",
            daemon=True,
        )
        self.process.start()
        print(
            "CAMERA_KPR_REID worker_start "
            f"pid={self.process.pid} device={device} isolated=1 lazy=1 "
            f"gpu_fraction={self.gpu_fraction:.2f}",
            flush=True,
        )

    def start(self) -> None:
        if not self.enabled or self.process is not None or self.fatal_error:
            return
        self._spawn(self.device)

    def _restart_cpu(self, reason: str) -> None:
        if not self.allow_cpu_fallback or self.fallback_used:
            self.fatal_error = reason
            self.last_error = reason
            return
        self.fallback_used = True
        self.fallbacks += 1
        self.pending.clear()
        old = self.process
        if old is not None:
            try:
                if old.is_alive():
                    old.terminate()
                old.join(timeout=1.0)
            except Exception:
                pass
        self.process = None
        self.last_error = f"{reason}; fallback=cpu"
        print(f"CAMERA_KPR_REID cuda_failed {self.last_error}", flush=True)
        self._spawn("cpu")

    def _check_worker(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            return
        exitcode = process.exitcode
        self.worker_exit = exitcode
        self.process = None
        if self.ready_event.is_set() and exitcode == 0:
            return
        reason = f"KPR worker exited device={self.active_device} code={exitcode}"
        if self.active_device.startswith("cuda"):
            self._restart_cpu(reason)
        else:
            self.fatal_error = reason
            self.last_error = reason

    def _drain_control(self) -> None:
        control_q = self.control_q
        if control_q is None:
            return
        while True:
            try:
                row = control_q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            kind = str(row.get("type", ""))
            if kind == "ready":
                self.backend = str(row.get("backend", self.backend))
                self.ready_event.set()
                self.last_error = ""
                print(
                    "CAMERA_KPR_REID ready "
                    f"backend={self.backend} pid={row.get('pid', 0)} "
                    f"checkpoint={self.model_path.name} role=final-cross-camera-merge-gate "
                    "part_visibility=1 batch=2 isolated=1",
                    flush=True,
                )
            elif kind == "fatal":
                reason = str(row.get("error", "KPR worker fatal"))
                if self.active_device.startswith("cuda"):
                    self._restart_cpu(reason)
                else:
                    self.fatal_error = reason
                    self.last_error = reason

    def remember(self, key: LocalKey, crop_bgr: np.ndarray, quality: float) -> None:
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return
        h, w = crop_bgr.shape[:2]
        if h < 40 or w < 14:
            return
        try:
            import cv2

            scale = min(1.0, 420.0 / max(1, h), 210.0 / max(1, w))
            if scale < 0.999:
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (
                        max(14, int(round(w * scale))),
                        max(40, int(round(h * scale))),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
        except Exception:
            pass
        sample = VisualSample(
            np.ascontiguousarray(crop_bgr).copy(), float(quality), time.monotonic()
        )
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
        rows = [
            row
            for row in self.visuals.get(key, ())
            if now - row.seen_at <= self.visual_ttl
        ]
        if not rows:
            return None
        return max(
            rows,
            key=lambda row: row.quality
            + 0.015 * max(0.0, self.visual_ttl - (now - row.seen_at)),
        )

    def poll(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._drain_control()
        self._check_worker()
        self._drain_control()

        output_q = self.output_q
        if output_q is not None:
            while True:
                try:
                    result = output_q.get_nowait()
                except queue.Empty:
                    break
                except Exception:
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

                self.last_used[pair] = (result.seen_a, result.seen_b)
                if (
                    result.score >= self.same_threshold
                    and result.visible_parts >= 1
                ):
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

        stale_keys = [
            key
            for key, rows in self.visuals.items()
            if not rows or now - rows[-1].seen_at > self.visual_ttl * 1.8
        ]
        for key in stale_keys:
            self.visuals.pop(key, None)

    def authorization(
        self, a: LocalKey, b: LocalKey, now: float | None = None
    ) -> str:
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
        if now - self.last_request.get(pair, 0.0) < self.min_request_gap:
            return "waiting"

        va = self._best_visual(a, now)
        vb = self._best_visual(b, now)
        if va is None or vb is None:
            self.no_visual += 1
            return "waiting"

        # Critical ordering: start the heavyweight verifier only after the camera
        # wall is already alive and a real merge candidate exists.
        if self.process is None:
            self.start()
            return "pending" if not self.fatal_error else "unavailable"
        if not self.ready_event.is_set():
            return "pending"

        prev_a, prev_b = self.last_used.get(pair, (0.0, 0.0))
        if va.seen_at <= prev_a + 1e-6 or vb.seen_at <= prev_b + 1e-6:
            self.wait_fresh += 1
            return "waiting"
        if self.input_q is None:
            return "pending"
        try:
            if self.input_q.full():
                self.dropped += 1
                return "pending"
        except Exception:
            pass

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
        except Exception as exc:
            self.last_error = f"queue:{type(exc).__name__}: {exc}"
            return "pending"

    def forget_pair(self, a: LocalKey, b: LocalKey) -> None:
        pair = self._pair(a, b)
        self.approved.discard(pair)
        self.blocked_until.pop(pair, None)
        self.same_votes.pop(pair, None)
        self.diff_votes.pop(pair, None)
        self.last_used.pop(pair, None)
        self.last_request.pop(pair, None)

    def snapshot(self) -> dict:
        self.poll(time.monotonic())
        process = self.process
        return {
            "enabled": self.enabled,
            "required": self.required,
            "ready": self.ready_event.is_set() and not self.fatal_error,
            "backend": self.backend,
            "worker_pid": int(process.pid or 0) if process is not None else 0,
            "worker_exit": self.worker_exit,
            "fallbacks": self.fallbacks,
            "visual_tracks": len(self.visuals),
            "requests": self.requests,
            "responses": self.responses,
            "pending": len(self.pending),
            "approved": len(self.approved),
            "blocked": sum(
                1 for until in self.blocked_until.values() if until > time.monotonic()
            ),
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
        process = self.process
        if process is None:
            return
        try:
            if self.input_q is not None:
                self.input_q.put_nowait(None)
        except Exception:
            pass
        try:
            process.join(timeout=2.0)
        except Exception:
            pass
        try:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        except Exception:
            pass
        self.process = None
