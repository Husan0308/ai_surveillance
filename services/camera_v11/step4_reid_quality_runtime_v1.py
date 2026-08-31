from __future__ import annotations

import os
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .step2_production_fp32 import _pct
from .step3_tracking_v2 import V11Step3TrackingV2
from .step4_reid_quality_v1 import ReIDCropQualityDecision, evaluate_reid_crop_quality


FROZEN_PRODUCTION_SHA = "d2c9e62f9ed2b5f80dc9a4d496e0fda94afddc51"
TERMINAL_REASONS = (
    "accepted",
    "reject_predicted",
    "reject_score",
    "reject_size",
    "reject_edge",
    "reject_aspect",
    "reject_blur",
    "reject_invalid",
)


@dataclass(frozen=True)
class GateCandidate:
    camera_id: str
    track_id: str
    captured_ns: int
    bbox_xyxy: tuple[float, float, float, float]
    detector_score: float
    metadata_gate_ms: float


@dataclass(frozen=True)
class GateFrameJob:
    camera_id: str
    captured_ns: int
    sink: object
    expected_pts: int
    candidates: tuple[GateCandidate, ...]


class ReIDQualityMetricsV1:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.counters = {name: 0 for name in ("submitted", *TERMINAL_REASONS)}
        self.gate_ms: deque[float] = deque(maxlen=4096)
        self.latest_replaced = 0
        self.source_shapes: dict[str, tuple[int, int]] = {}
        self.quality_scores: deque[float] = deque(maxlen=4096)

    def submitted(self) -> None:
        with self.lock:
            self.counters["submitted"] += 1

    def terminal(self, reason: str, gate_ms: float, quality_score: float = 0.0) -> None:
        key = reason if reason.startswith("reject_") or reason == "accepted" else f"reject_{reason}"
        if key not in TERMINAL_REASONS:
            key = "reject_invalid"
        with self.lock:
            self.counters[key] += 1
            self.gate_ms.append(max(0.0, float(gate_ms)))
            if quality_score > 0.0:
                self.quality_scores.append(float(quality_score))

    def replace(self, candidates: tuple[GateCandidate, ...]) -> None:
        with self.lock:
            self.latest_replaced += 1
        for candidate in candidates:
            self.terminal("reject_invalid", candidate.metadata_gate_ms)

    def set_shape(self, camera_id: str, width: int, height: int) -> None:
        with self.lock:
            self.source_shapes[camera_id] = (int(width), int(height))

    def snapshot(self) -> dict:
        with self.lock:
            result = dict(self.counters)
            result.update(
                {
                    "gate_p50_ms": _pct(self.gate_ms, 0.50),
                    "gate_p95_ms": _pct(self.gate_ms, 0.95),
                    "quality_p50": _pct(self.quality_scores, 0.50),
                    "latest_replaced": self.latest_replaced,
                    "source_shapes": dict(self.source_shapes),
                }
            )
            return result


class ReIDQualityWorkerV1:
    """Asynchronous per-camera latest-slot gate; no FIFO frame queue."""

    def __init__(
        self,
        metrics: ReIDQualityMetricsV1,
        on_accepted: Callable[[GateCandidate, ReIDCropQualityDecision], None],
        *,
        min_width: int,
        min_height: int,
        min_aspect: float,
        max_aspect: float,
        max_clipped_fraction: float,
        severe_blur_variance: float,
        min_quality_score: float,
        map_read_flag: object,
    ) -> None:
        self.metrics = metrics
        self.on_accepted = on_accepted
        self.thresholds = {
            "min_width": int(min_width),
            "min_height": int(min_height),
            "min_aspect": float(min_aspect),
            "max_aspect": float(max_aspect),
            "max_clipped_fraction": float(max_clipped_fraction),
            "severe_blur_variance": float(severe_blur_variance),
            "min_quality_score": float(min_quality_score),
        }
        self.map_read_flag = map_read_flag
        self.cv = threading.Condition()
        self.pending: dict[str, GateFrameJob] = {}
        self.stop_requested = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        with self.cv:
            if self.thread is not None:
                return
            self.thread = threading.Thread(
                target=self._run,
                name="camera-v11-step4-reid-quality",
                daemon=True,
            )
            self.thread.start()

    def submit(self, job: GateFrameJob) -> None:
        replaced = None
        with self.cv:
            if self.stop_requested:
                replaced = job
            else:
                replaced = self.pending.get(job.camera_id)
                self.pending[job.camera_id] = job
                self.cv.notify()
        if replaced is not None:
            self.metrics.replace(replaced.candidates)

    def _take(self) -> GateFrameJob | None:
        with self.cv:
            while not self.stop_requested and not self.pending:
                self.cv.wait(timeout=0.25)
            if self.stop_requested:
                return None
            camera_id = next(iter(self.pending))
            return self.pending.pop(camera_id)

    def _reject_job(self, job: GateFrameJob) -> None:
        for candidate in job.candidates:
            self.metrics.terminal("reject_invalid", candidate.metadata_gate_ms)

    def _process(self, job: GateFrameJob) -> None:
        sample = job.sink.emit("try-pull-sample", 50_000_000)
        if sample is None:
            self._reject_job(job)
            return
        completed = 0
        try:
            sample_buffer = sample.get_buffer()
            actual_pts = int(sample_buffer.pts)
            if job.expected_pts >= 0 and actual_pts != job.expected_pts:
                raise RuntimeError(
                    f"native/detector PTS mismatch {actual_pts}!={job.expected_pts}"
                )
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            if width <= 0 or height <= 0:
                raise ValueError("invalid native frame dimensions")
            buffer = sample.get_buffer()
            map_started = time.perf_counter()
            ok, mapped = buffer.map(self.map_read_flag)
            if not ok:
                raise RuntimeError("native frame map failed")
            try:
                tight = width * 4
                size = int(getattr(mapped, "size", len(mapped.data)))
                stride = size // height if size % height == 0 else tight
                if stride < tight or size < stride * height:
                    raise RuntimeError(f"invalid native BGRx stride={stride} size={size}")
                raw = np.frombuffer(mapped.data, dtype=np.uint8, count=stride * height)
                bgrx = raw.reshape(height, stride)[:, :tight].reshape(height, width, 4)
                frame = bgrx[:, :, :3]
                self.metrics.set_shape(job.camera_id, width, height)
                map_share_ms = (
                    (time.perf_counter() - map_started)
                    * 1000.0
                    / max(1, len(job.candidates))
                )
                for candidate in job.candidates:
                    started = time.perf_counter()
                    decision = evaluate_reid_crop_quality(
                        frame,
                        candidate.bbox_xyxy,
                        candidate.detector_score,
                        **self.thresholds,
                    )
                    gate_ms = (
                        candidate.metadata_gate_ms
                        + map_share_ms
                        + (time.perf_counter() - started) * 1000.0
                    )
                    self.metrics.terminal(decision.reason, gate_ms, decision.quality_score)
                    completed += 1
                    if decision.accepted:
                        try:
                            self.on_accepted(candidate, decision)
                        except Exception:
                            # Step 1 has no downstream ReID. A future callback must not
                            # be allowed to kill the quality worker or camera runtime.
                            pass
            finally:
                buffer.unmap(mapped)
        except Exception:
            for candidate in job.candidates[completed:]:
                self.metrics.terminal("reject_invalid", candidate.metadata_gate_ms)

    def _run(self) -> None:
        while True:
            job = self._take()
            if job is None:
                return
            self._process(job)

    def close(self) -> None:
        with self.cv:
            if self.stop_requested:
                return
            self.stop_requested = True
            remaining = list(self.pending.values())
            self.pending.clear()
            self.cv.notify_all()
        for job in remaining:
            self._reject_job(job)
        thread = self.thread
        if thread is not None:
            thread.join(timeout=2.0)
            self.thread = None


class V11Step4ReIDQualityV1(V11Step3TrackingV2):
    """Frozen Step3 plus an asynchronous native-frame ReID crop quality side path."""

    def __init__(self) -> None:
        self.quality_sinks: dict[str, object] = {}
        self.quality_metrics = ReIDQualityMetricsV1()
        self.quality_closed = False
        self.quality_run_sec = max(
            0.0, float(os.environ.get("V11_STEP4_QUALITY_RUN_SEC", "0"))
        )
        self.quality_run_started = 0.0
        super().__init__()
        self.quality_min_score = max(
            0.18, min(0.90, float(os.environ.get("V11_STEP4_QUALITY_MIN_SCORE", "0.25")))
        )
        self.quality_worker = ReIDQualityWorkerV1(
            self.quality_metrics,
            self._accepted_crop,
            min_width=max(12, int(os.environ.get("V11_STEP4_QUALITY_MIN_WIDTH", "24"))),
            min_height=max(32, int(os.environ.get("V11_STEP4_QUALITY_MIN_HEIGHT", "64"))),
            min_aspect=max(0.40, float(os.environ.get("V11_STEP4_QUALITY_MIN_ASPECT", "0.85"))),
            max_aspect=min(10.0, float(os.environ.get("V11_STEP4_QUALITY_MAX_ASPECT", "6.50"))),
            max_clipped_fraction=min(
                0.50, float(os.environ.get("V11_STEP4_QUALITY_MAX_CLIPPED", "0.20"))
            ),
            severe_blur_variance=max(
                1.0, float(os.environ.get("V11_STEP4_QUALITY_BLUR_MIN", "7.0"))
            ),
            min_quality_score=max(
                0.10, float(os.environ.get("V11_STEP4_QUALITY_SCORE_MIN", "0.25"))
            ),
            map_read_flag=self.Gst.MapFlags.READ,
        )
        print(
            "CAMERA_V11_STEP4_REID_QUALITY_V1_ARCH "
            f"frozen_production_sha={FROZEN_PRODUCTION_SHA} "
            "source=native-decoded-before-detector-resize crop_coordinates=scaled-from-672x384 "
            "camera_thread=metadata-only gate_worker=async-cpu "
            "camera_queue=0 python_frame_queue=0 gate_pending=per-camera-latest-overwrite "
            "display_topology_changed=0 detector_schedule_changed=0 tracker_changed=0 "
            "reid_inference=0 gallery=0 pair_scoring=0 room_id=0 global_id=0 face=0 handoff=0",
            flush=True,
        )
        print(
            "CAMERA_V11_STEP4_REID_QUALITY_V1_POLICY "
            f"confirmed=required predicted=reject min_score={self.quality_min_score:.2f} "
            f"min_crop={self.quality_worker.thresholds['min_width']}x{self.quality_worker.thresholds['min_height']} "
            f"aspect={self.quality_worker.thresholds['min_aspect']:.2f}..{self.quality_worker.thresholds['max_aspect']:.2f} "
            f"max_clipped={self.quality_worker.thresholds['max_clipped_fraction']:.2f} "
            f"severe_blur_min={self.quality_worker.thresholds['severe_blur_variance']:.1f} "
            f"quality_min={self.quality_worker.thresholds['min_quality_score']:.2f}",
            flush=True,
        )

    def _build_camera(self, index, camera) -> None:
        # Build frozen Step2 exactly, then insert an additive tee before its resize.
        # The native branch has no queue element; its drop=true one-sample appsink
        # cannot accumulate frames or back-pressure on a full sample.
        super()._build_camera(index, camera)
        cid = camera.camera_id
        input_q = self.input_queues[cid]
        detector_convert = self.pipeline.get_by_name(f"step2_convert_{index}")
        if detector_convert is None:
            raise RuntimeError(f"{cid}: could not insert native quality tee")
        input_q.unlink(detector_convert)

        tee = self._make("tee", f"step4_quality_tee_{index}")
        native_convert = self._make("nvvideoconvert", f"step4_quality_convert_{index}")
        native_caps = self._make("capsfilter", f"step4_quality_caps_{index}")
        native_sink = self._make("appsink", f"step4_quality_sink_{index}")
        self._set_if(native_convert, "gpu-id", self.gpu_id)
        self._set_if(native_convert, "compute-hw", 1)
        native_caps.set_property(
            "caps",
            self.Gst.Caps.from_string("video/x-raw,format=BGRx,pixel-aspect-ratio=1/1"),
        )
        for name, value in (
            ("emit-signals", False),
            ("sync", False),
            ("async", False),
            ("drop", True),
            ("max-buffers", 1),
            ("enable-last-sample", False),
            ("wait-on-eos", False),
            ("qos", False),
            ("processing-deadline", 0),
        ):
            self._set_if(native_sink, name, value)
        for element in (tee, native_convert, native_caps, native_sink):
            self.pipeline.add(element)
        self._link(input_q, tee, f"{cid}:input->quality-tee")
        # Link detector first so the additive branch cannot get priority.
        self._link(tee, detector_convert, f"{cid}:quality-tee->frozen-detector")
        self._link(tee, native_convert, f"{cid}:quality-tee->native-convert")
        self._link(native_convert, native_caps, f"{cid}:native-convert->caps")
        self._link(native_caps, native_sink, f"{cid}:native-caps->sink")
        self.quality_sinks[cid] = native_sink

    def _accepted_crop(
        self, _candidate: GateCandidate, _decision: ReIDCropQualityDecision
    ) -> None:
        # Deliberate Step-1 stop boundary: the accepted native-resolution crop is
        # ready for a future asynchronous TRT submission, but no ReID runs here.
        return

    def _quality_track_update(
        self, _camera_id: str, _track_ids: tuple[str, ...], _captured_ns: int
    ) -> None:
        # Additive Step-2 extension seam. Step 1 intentionally has no state here.
        return

    def _consume_tracking(self, cid: str, boxes: list[list[float]], captured_ns: int) -> None:
        # Byte-for-byte behavior equivalent to frozen Step3's method until the
        # final nonblocking quality handoff.
        now = time.monotonic()
        if self.quality_run_started == 0.0:
            self.quality_run_started = now
        elif (
            self.quality_run_sec > 0.0
            and now - self.quality_run_started >= self.quality_run_sec
        ):
            self.stop_requested = True
        update = self.tracker.update(cid, boxes, captured_ns)
        self.stage_values["tracker"].append(float(update.step_ms))
        ids = tuple(snapshot.track_id for snapshot in update.snapshots)
        if len(ids) != len(set(ids)):
            self.track_duplicate_errors += 1
        prefix = f"{cid}-T"
        self.track_prefix_errors += sum(1 for track_id in ids if not track_id.startswith(prefix))
        self.track_updates[cid] += 1
        self.track_created[cid] += int(update.created)
        self.track_recovered[cid] += int(update.recovered)
        self.track_removed[cid] += int(update.removed)
        self.latest_track_ids[cid] = ids
        self._quality_track_update(cid, ids, int(captured_ns))

        eligible = []
        for snapshot in update.snapshots:
            started = time.perf_counter()
            self.quality_metrics.submitted()
            if (
                not snapshot.confirmed
                or snapshot.predicted
                or snapshot.since_detection_sec > 0.25
            ):
                self.quality_metrics.terminal(
                    "reject_predicted", (time.perf_counter() - started) * 1000.0
                )
                continue
            if float(snapshot.score) < self.quality_min_score:
                self.quality_metrics.terminal(
                    "reject_score", (time.perf_counter() - started) * 1000.0
                )
                continue
            eligible.append(
                GateCandidate(
                    camera_id=cid,
                    track_id=snapshot.track_id,
                    captured_ns=int(captured_ns),
                    bbox_xyxy=tuple(float(value) for value in snapshot.bbox_xyxy),
                    detector_score=float(snapshot.score),
                    metadata_gate_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
        if not eligible:
            return

        self.quality_worker.submit(
            GateFrameJob(
                camera_id=cid,
                captured_ns=int(captured_ns),
                sink=self.quality_sinks[cid],
                expected_pts=int(self.accepted_pts_ns.get(cid, -1)),
                candidates=tuple(eligible),
            )
        )

    def _print_quality_stats(self) -> None:
        row = self.quality_metrics.snapshot()
        terminal = sum(int(row[name]) for name in TERMINAL_REASONS)
        shapes = ",".join(
            f"{cid}:{width}x{height}"
            for cid, (width, height) in sorted(row["source_shapes"].items())
        )
        print(
            "CAMERA_V11_STEP4_REID_QUALITY_V1 "
            f"submitted={row['submitted']} accepted={row['accepted']} "
            f"reject_predicted={row['reject_predicted']} reject_score={row['reject_score']} "
            f"reject_size={row['reject_size']} reject_edge={row['reject_edge']} "
            f"reject_aspect={row['reject_aspect']} reject_blur={row['reject_blur']} "
            f"reject_invalid={row['reject_invalid']} "
            f"gate_p50={row['gate_p50_ms']:.3f}ms gate_p95={row['gate_p95_ms']:.3f}ms "
            f"quality_p50={row['quality_p50']:.3f} terminal={terminal} "
            f"latest_replaced={row['latest_replaced']} native={shapes or '-'}",
            flush=True,
        )

    def _print_stats(self) -> None:
        super()._print_stats()
        self._print_quality_stats()

    def run(self) -> int:
        self.quality_worker.start()
        try:
            return super().run()
        finally:
            self._close_quality()

    def _close_quality(self) -> None:
        if self.quality_closed:
            return
        self.quality_closed = True
        self.quality_worker.close()
        self._print_quality_stats()

    def close(self) -> None:
        self._close_quality()
        super().close()


def main() -> int:
    service = V11Step4ReIDQualityV1()

    def stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
