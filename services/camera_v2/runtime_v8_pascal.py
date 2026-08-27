from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue as pyqueue
import sys
import time
from collections import deque

from .bbox_policy_v7 import DisplaySizeState, expand_box, stable_size
from .runtime import CleanCameraRuntime, DETECT_W
from .runtime_quality import QualityCameraRuntime
from .visibility_policy_v72 import should_hold_last_good
from .yolo_trt86_batch6_bridge import yolo_trt86_batch6_worker


class PascalBatchLowLatencyRuntime(QualityCameraRuntime):
    """V8: Pascal-safe batch-6 TRT8.6 detector + independent NvDCF/display.

    DeepStream 7.1 ships TensorRT 10.x, while GTX 1050 Ti / SM 6.1 is a Pascal GPU
    that is no longer supported by TensorRT 10. Therefore V8 deliberately does not
    route the detector through gst-nvinfer. Detection stays in a TRT8.6 sidecar, but
    six per-camera batch-1 calls are replaced by one true batch-6 enqueue.

    More importantly, V8 removes SerializedGpuLaneRuntime entirely. NvDCF frames are
    never dropped because detection is running. Display remains on the independent
    source tee path so analytics cannot build a visible-frame backlog.
    """

    def __init__(self) -> None:
        self.min_display_track_conf = float(
            os.environ.get("CAMERA_V2_MIN_DISPLAY_TRACK_CONF", "0.28")
        )
        self.display_side_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_SIDE_MARGIN", "0.06")
        )
        self.display_top_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_TOP_MARGIN", "0.04")
        )
        self.display_bottom_margin = float(
            os.environ.get("CAMERA_V2_DISPLAY_BOX_BOTTOM_MARGIN", "0.07")
        )
        self.display_size_hold_sec = float(
            os.environ.get("CAMERA_V2_DISPLAY_SIZE_HOLD_SEC", "0.22")
        )
        self.display_shrink_alpha = float(
            os.environ.get("CAMERA_V2_DISPLAY_SHRINK_ALPHA", "0.42")
        )
        self.empty_hold_ms = max(
            180.0,
            min(500.0, float(os.environ.get("CAMERA_V2_DISPLAY_EMPTY_HOLD_MS", "350"))),
        )
        self.jump_diag_limit = float(
            os.environ.get("CAMERA_V2_TRACK_JUMP_DIAG_LIMIT", "1.00")
        )

        self.detector_budget = max(
            0.10,
            min(0.50, float(os.environ.get("CAMERA_V8_DETECT_GPU_BUDGET", "0.28"))),
        )
        self.detector_min_hz = max(
            0.30,
            min(1.50, float(os.environ.get("CAMERA_V8_DETECT_MIN_HZ", "0.70"))),
        )
        self.detector_max_hz = max(
            self.detector_min_hz,
            min(3.0, float(os.environ.get("CAMERA_V8_DETECT_MAX_HZ", "2.00"))),
        )
        self.detector_batch_hz = max(
            self.detector_min_hz,
            min(
                self.detector_max_hz,
                float(os.environ.get("CAMERA_V8_DETECT_INITIAL_HZ", "1.00")),
            ),
        )
        self.capture_batch_timeout = max(
            0.12,
            min(0.60, float(os.environ.get("CAMERA_V8_CAPTURE_BATCH_TIMEOUT", "0.30"))),
        )
        self.detector_ema_alpha = max(
            0.05,
            min(0.50, float(os.environ.get("CAMERA_V8_DETECT_EMA_ALPHA", "0.20"))),
        )

        self._display_sizes: dict[tuple[int, int], DisplaySizeState] = {}
        self._last_raw_boxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        self.v8_low_conf_filtered = 0
        self.v8_duplicates_suppressed = 0
        self.v8_teleport_events = 0
        self.v8_empty_holds = 0
        self.v8_empty_expires = 0
        self.v8_real_updates = 0
        self.v8_batch_calls = 0
        self.v8_capture_partial = 0
        self.v8_gpu_ms_ema = 0.0
        self.v8_roundtrip_ms_ema = 0.0
        self.v8_batch_age_ms = 0.0
        self.v8_batch_intervals = deque(maxlen=120)
        self.v8_last_batch_completed = 0.0

        super().__init__()
        self.display_track_max_age_ms = max(
            self.display_track_max_age_ms,
            self.empty_hold_ms + 50.0,
        )
        print(
            "CAMERA_V8_ARCH "
            f"cameras={len(self.cameras)} detector=TRT8.6/batch6 "
            f"tracker=NvDCF/{self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"detector_initial={self.detector_batch_hz:.2f}Hz/all-cameras "
            f"detector_budget={self.detector_budget:.2f} "
            "gpu_lane=removed tracker_drop_for_detector=0 display=independent predictor=0",
            flush=True,
        )

    def _preflight(self) -> None:
        if len(self.cameras) != 6:
            raise RuntimeError(
                f"V8 batch engine is fixed for exactly 6 cameras, got {len(self.cameras)}"
            )
        super()._preflight()

    def _prepare_tracker_files(self):
        self.track_width = max(
            320,
            min(DETECT_W, int(os.environ.get("CAMERA_V2_TRACK_WIDTH", "512"))),
        )
        self.track_height = max(
            192,
            min(384, int(os.environ.get("CAMERA_V2_TRACK_HEIGHT", "288"))),
        )
        lib, generated = CleanCameraRuntime._prepare_tracker_files(self)
        lines = generated.read_text(encoding="utf-8").splitlines()

        detector_floor = os.environ.get("CAMERA_V2_DETECT_CONF", "0.18")
        shadow_frames = max(12, int(round(self.track_fps * 1.20)))
        self._replace_yaml_key(lines, "minDetectorConfidence", detector_floor)
        self._replace_yaml_key(lines, "enableBboxUnClipping", "0")
        # Lower new-target IOU difference suppresses duplicate target creation.
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.22")
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.28")
        self._replace_yaml_key(lines, "probationAge", "2")
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._replace_yaml_key(lines, "earlyTerminationAge", "1")
        self._replace_yaml_key(lines, "minIou4TargetDuplicate", "0.94", required=False)
        self._replace_yaml_key(lines, "targetDuplicateRunInterval", "5", required=False)
        self._replace_yaml_key(lines, "useColorNames", "1", required=False)
        self._replace_yaml_key(lines, "useHog", "0", required=False)
        self._replace_yaml_key(lines, "featureImgSizeLevel", "2", required=False)
        self._replace_yaml_key(lines, "useHighPrecisionFeature", "0", required=False)
        self._replace_yaml_key(lines, "searchRegionPaddingScale", "1", required=False)
        self._replace_yaml_key(lines, "usePrediction4Assoc", "1", required=False)
        self._replace_yaml_key(
            lines, "minTrackingConfidenceDuringInactive", "0.40", required=False
        )
        self._insert_target_management_key(lines, "outputShadowTracks", "0")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_V8_NVDCF "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            f"min_detector_conf={detector_floor} min_tracker_conf=0.28 probation=2 "
            f"shadow_frames={shadow_frames} output_shadow=0 "
            "features=ColorNames hog=0 feature_level=2 high_precision=0",
            flush=True,
        )
        return lib, generated

    @staticmethod
    def _iou(a, b) -> float:
        inter = CleanCameraRuntime._intersection(a, b)
        if inter <= 0.0:
            return 0.0
        union = CleanCameraRuntime._area(a) + CleanCameraRuntime._area(b) - inter
        return inter / union if union > 0.0 else 0.0

    def _dedup_v8(self, tracks):
        ordered = sorted(tracks, key=lambda row: float(row[5]), reverse=True)
        kept = []
        suppressed = 0
        for row in ordered:
            box = row[1:5]
            area = max(1.0, self._area(box))
            duplicate = False
            for other in kept:
                other_box = other[1:5]
                other_area = max(1.0, self._area(other_box))
                inter = self._intersection(box, other_box)
                containment = inter / max(1.0, min(area, other_area))
                if self._iou(box, other_box) >= 0.72 or containment >= 0.94:
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                kept.append(row)
        return kept, suppressed

    def _stable_display_box(
        self,
        source_id: int,
        object_id: int,
        raw_box: tuple[float, float, float, float],
        now: float,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = raw_box
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        key = (int(source_id), int(object_id))
        previous = self._display_sizes.get(key)
        if previous is None:
            stable_w, stable_h = width, height
            width_expanded_at = height_expanded_at = now
        else:
            stable_w, width_expanded_at = stable_size(
                previous.width,
                width,
                previous.width_expanded_at,
                now,
                hold_sec=self.display_size_hold_sec,
                shrink_alpha=self.display_shrink_alpha,
            )
            stable_h, height_expanded_at = stable_size(
                previous.height,
                height,
                previous.height_expanded_at,
                now,
                hold_sec=self.display_size_hold_sec,
                shrink_alpha=self.display_shrink_alpha,
            )
        self._display_sizes[key] = DisplaySizeState(
            stable_w,
            stable_h,
            width_expanded_at,
            height_expanded_at,
            now,
        )
        base = (
            cx - 0.5 * stable_w,
            cy - 0.5 * stable_h,
            cx + 0.5 * stable_w,
            cy + 0.5 * stable_h,
        )
        return expand_box(
            base,
            self.display_width,
            self.display_height,
            side_margin=self.display_side_margin,
            top_margin=self.display_top_margin,
            bottom_margin=self.display_bottom_margin,
        )

    def _record_jump(
        self,
        source_id: int,
        object_id: int,
        box: tuple[float, float, float, float],
    ) -> None:
        key = (int(source_id), int(object_id))
        previous = self._last_raw_boxes.get(key)
        self._last_raw_boxes[key] = box
        if previous is None:
            return
        px1, py1, px2, py2 = previous
        x1, y1, x2, y2 = box
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        diag = math.hypot(max(2.0, px2 - px1), max(2.0, py2 - py1))
        if math.hypot(cx - pcx, cy - pcy) > self.jump_diag_limit * max(1.0, diag):
            self.v8_teleport_events += 1

    def _tracker_probe(self, _pad, info):
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rows = self.bridge.copy_tracks(buffer, max_rows=256)
            now = time.monotonic()
            grouped = {int(source_id): [] for source_id in self.index_camera}
            sx = self.display_width / float(self.track_width)
            sy = self.display_height / float(self.track_height)
            filtered = 0
            for row in rows:
                source_id = int(row["source_id"])
                if source_id not in grouped:
                    continue
                conf = float(row["tracker_confidence"])
                if conf < 0.0:
                    conf = float(row["confidence"])
                if conf < self.min_display_track_conf:
                    filtered += 1
                    continue
                left = float(row["left"]) * sx
                top = float(row["top"]) * sy
                right = (float(row["left"]) + float(row["width"])) * sx
                bottom = (float(row["top"]) + float(row["height"])) * sy
                object_id = int(row["object_id"])
                raw_box = (left, top, right, bottom)
                self._record_jump(source_id, object_id, raw_box)
                display_box = self._stable_display_box(source_id, object_id, raw_box, now)
                grouped[source_id].append(
                    (
                        object_id,
                        display_box[0],
                        display_box[1],
                        display_box[2],
                        display_box[3],
                        conf,
                    )
                )

            published = {}
            suppressed = 0
            for source_id, tracks in grouped.items():
                kept, count = self._dedup_v8(tracks)
                published[source_id] = kept
                suppressed += count

            held = 0
            expired = 0
            real_updates = 0
            with self.track_cache_lock:
                for source_id, tracks in published.items():
                    if tracks:
                        self.track_cache[source_id] = (now, tracks)
                        real_updates += 1
                        continue
                    previous = self.track_cache.get(source_id)
                    if (
                        previous is not None
                        and previous[1]
                        and should_hold_last_good(previous[0], now, self.empty_hold_ms)
                    ):
                        held += 1
                        continue
                    if previous is not None:
                        self.track_cache.pop(source_id, None)
                        expired += 1

                self.tracked_now = sum(
                    len(tracks) for _updated, tracks in self.track_cache.values()
                )
                self.tracker_batches += 1
                self.v8_low_conf_filtered += filtered
                self.v8_duplicates_suppressed += suppressed
                self.v8_empty_holds += held
                self.v8_empty_expires += expired
                self.v8_real_updates += real_updates
                active_keys = {
                    (source_id, int(track[0]))
                    for source_id, (_updated, tracks) in self.track_cache.items()
                    for track in tracks
                }

            for key, state in list(self._display_sizes.items()):
                if key not in active_keys and now - state.seen_at > 1.0:
                    self._display_sizes.pop(key, None)
                    self._last_raw_boxes.pop(key, None)
        except Exception as exc:
            print(
                f"CAMERA_V8_TRACK warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.PadProbeReturn.OK

    def _start_detector(self) -> None:
        if not self.detect_enabled:
            print("CAMERA_V8_DETECT enabled=0", flush=True)
            return
        ctx = mp.get_context("spawn")
        self.job_q = ctx.Queue(maxsize=1)
        self.result_q = ctx.Queue(maxsize=2)
        self.det_process = ctx.Process(
            target=yolo_trt86_batch6_worker,
            args=(self.job_q, self.result_q),
            name="camera-v8-trt86-batch6-bridge",
        )
        self.det_process.start()
        import threading

        self.det_thread = threading.Thread(
            target=self._detector_scheduler,
            name="camera-v8-detector-scheduler",
            daemon=True,
        )
        self.det_thread.start()

    def _ema(self, previous: float, current: float) -> float:
        if previous <= 0.0:
            return float(current)
        a = self.detector_ema_alpha
        return previous * (1.0 - a) + float(current) * a

    def _adapt_detector_hz(self) -> None:
        gpu_s = max(0.001, self.v8_gpu_ms_ema / 1000.0)
        wall_s = max(0.001, self.v8_roundtrip_ms_ema / 1000.0)
        by_gpu_budget = self.detector_budget / gpu_s
        # Leave at least 15% wall-time slack so the detector cannot self-backlog.
        by_wall = 0.85 / wall_s
        target = min(self.detector_max_hz, by_gpu_budget, by_wall)
        self.detector_batch_hz = max(self.detector_min_hz, target)
        # Base stats call this detect_hz. In V8 one batch covers all six cameras, so
        # batch frequency and per-camera detector frequency are the same number.
        self.detect_hz = self.detector_batch_hz

    def _detector_scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=60.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "V8 TRT86 batch startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "V8 TRT86 batch startup failed")
            return
        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_V8_DETECT_READY "
            f"model={ready.get('model')} backend={ready.get('backend')} batch=6 "
            f"initial={self.detector_batch_hz:.2f}Hz/all-cameras "
            "capture=coalesced-latest-six gpu_lane=none",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        versions = {cid: 0 for cid in ids}

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            if any(self.stats[cid].frames <= 0 for cid in ids):
                self.det_stop.wait(0.03)
                continue

            # Request all cameras together. This is the latency-critical change versus
            # six sequential JIT capture -> batch1 inference cycles.
            for cid in ids:
                self._request_capture(cid)

            deadline = time.monotonic() + self.capture_batch_timeout
            frames = []
            captured = []
            success = True
            for cid in ids:
                remaining = max(0.0, deadline - time.monotonic())
                row = self.mailbox.wait_new(cid, versions[cid], timeout=remaining)
                if row is None:
                    success = False
                    break
                version, captured_at, frame = row
                versions[cid] = version
                frames.append(frame)
                captured.append(captured_at)

            for cid in ids:
                self._clear_capture(cid)

            if not success or len(frames) != len(ids):
                with self.det_lock:
                    self.capture_timeouts += 1
                self.v8_capture_partial += 1
                self.det_stop.wait(0.01)
                continue

            try:
                self.job_q.put(
                    {"cameras": ids, "frames": frames, "captured": captured},
                    timeout=0.15,
                )
                result = self.result_q.get(timeout=10.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "V8 TRT86 batch result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "V8 TRT86 fatal")
                return
            if result.get("type") != "result":
                continue

            completed = time.monotonic()
            gpu_ms = float(result.get("batch_ms") or 0.0)
            roundtrip_ms = float(result.get("total_ms") or gpu_ms)
            self.v8_gpu_ms_ema = self._ema(self.v8_gpu_ms_ema, gpu_ms)
            self.v8_roundtrip_ms_ema = self._ema(self.v8_roundtrip_ms_ema, roundtrip_ms)
            self._adapt_detector_hz()

            if self.v8_last_batch_completed > 0.0:
                self.v8_batch_intervals.append(completed - self.v8_last_batch_completed)
            self.v8_last_batch_completed = completed
            self.v8_batch_calls += 1
            max_age_ms = 0.0
            total_boxes = 0
            for cid, captured_at in zip(ids, captured):
                detector_rows = result.get("boxes", {}).get(cid, [])
                boxes = self._map_detector_rows(detector_rows)
                self._publish_detector(cid, captured_at, boxes)
                age_ms = max(0.0, (completed - captured_at) * 1000.0)
                max_age_ms = max(max_age_ms, age_ms)
                total_boxes += len(boxes)
                with self.det_lock:
                    self.det_counts[cid] = len(boxes)
                    self.detector_times[cid].append(completed)

            self.v8_batch_age_ms = max_age_ms
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(ids)
                self.det_batch_ms = gpu_ms
                self.det_result_age_ms = max_age_ms
                self.det_error = ""

            if self.v8_batch_calls <= 5 or self.v8_batch_calls % 10 == 0:
                duty = self.detector_batch_hz * max(0.0, self.v8_gpu_ms_ema) / 1000.0
                print(
                    "CAMERA_V8_ADAPT "
                    f"call={self.v8_batch_calls} gpu_ema={self.v8_gpu_ms_ema:.1f}ms "
                    f"roundtrip_ema={self.v8_roundtrip_ms_ema:.1f}ms "
                    f"batch_hz={self.detector_batch_hz:.2f} per_camera_hz={self.detect_hz:.2f} "
                    f"duty_est={duty:.2f} age={max_age_ms:.0f}ms boxes={total_boxes}",
                    flush=True,
                )

            desired_interval = 1.0 / max(0.1, self.detector_batch_hz)
            elapsed = time.monotonic() - cycle_started
            self.det_stop.wait(max(0.002, desired_interval - elapsed))

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        duty = self.detector_batch_hz * max(0.0, self.v8_gpu_ms_ema) / 1000.0
        intervals = list(self.v8_batch_intervals)
        actual_batch_hz = 0.0
        if intervals:
            avg = sum(intervals) / len(intervals)
            actual_batch_hz = 1.0 / avg if avg > 0.0 else 0.0
        print(
            "CAMERA_V8_STATS "
            f"batch_calls={self.v8_batch_calls} "
            f"batch_target={self.detector_batch_hz:.2f}Hz batch_actual={actual_batch_hz:.2f}Hz "
            f"gpu_ema={self.v8_gpu_ms_ema:.1f}ms roundtrip_ema={self.v8_roundtrip_ms_ema:.1f}ms "
            f"gpu_duty_est={duty:.2f} result_age={self.v8_batch_age_ms:.0f}ms "
            f"capture_partial={self.v8_capture_partial} "
            f"real_updates={self.v8_real_updates} empty_holds={self.v8_empty_holds} "
            f"empty_expires={self.v8_empty_expires} low_conf_filtered={self.v8_low_conf_filtered} "
            f"duplicates_suppressed={self.v8_duplicates_suppressed} "
            f"teleport_events={self.v8_teleport_events} "
            "gpu_lane=0 tracker_skips_for_detector=0 predictor=0",
            flush=True,
        )
        return keep


def main() -> int:
    return PascalBatchLowLatencyRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
