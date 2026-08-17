from __future__ import annotations

import math
import os
import queue as pyqueue
import threading
import time
from collections import deque

# Production person tracking for GTX 1050 Ti 4 GB + six fixed CCTV streams.
# Keep the detector branch at the same 16:9 geometry as the 1280x720 camera
# frames. Ultralytics may stride-pad internally, but the pixels reaching YOLO are
# no longer geometrically stretched before inference.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "704")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "396")
os.environ.setdefault("CAMERA_V2_MICRO_BATCH", "2")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.08")
# High enough to preserve two strongly-overlapping people, but not so high that
# ordinary duplicate predictions survive into NvDCF.
os.environ.setdefault("CAMERA_V2_DETECT_IOU", "0.78")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")
os.environ.setdefault("CAMERA_V2_TRACKER_WIDTH", "480")
os.environ.setdefault("CAMERA_V2_TRACKER_HEIGHT", "288")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_SIDE_MARGIN", "0.00")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_TOP_MARGIN", "0.00")
os.environ.setdefault("CAMERA_V2_TRACK_BOX_BOTTOM_MARGIN", "0.00")
# Ultralytics already performs NMS. This second pass removes only clear
# duplicate/nested boxes, not legitimate adjacent people.
os.environ.setdefault("CAMERA_V2_DEDUP_IOU", "0.92")
os.environ.setdefault("CAMERA_V2_DEDUP_CONTAINMENT", "0.99")

from .detection import INFER_HEIGHT, INFER_WIDTH, MICRO_BATCH
from .detector_latency import DetectorLatencyCompensator, PreparedDetection
from .external_reid import ExternalReIDWorker
from .global_reid import GlobalReIDManager
from .person_tracking import CameraPersonTrackingV2 as _BaseTracking
from .tracker_profile import prepare_sparse_tracker_config, reid_backend


class CameraPersonTrackingFinal(_BaseTracking):
    """YOLO26m + NvDCF local tracking + conservative cross-camera identity.

    Geometry always belongs to YOLO/NvDCF. On Pascal, cross-camera appearance
    embeddings are extracted asynchronously on CPU from the already-captured YOLO
    frames. ReID never feeds boxes back into NvDCF, so an identity failure cannot
    destabilize the live camera wall or local tracking.
    """

    def __init__(self) -> None:
        self.detector_frames_applied = 0
        self.detector_target_hz = float(os.environ.get("CAMERA_V2_DETECT_TARGET_HZ", "3.0"))
        self.detector_min_hz = float(os.environ.get("CAMERA_V2_DETECT_MIN_HZ", "2.0"))
        self.detector_max_hz = float(os.environ.get("CAMERA_V2_DETECT_MAX_HZ", "3.6"))
        self.detector_min_idle = float(os.environ.get("CAMERA_V2_DETECT_MIN_IDLE_MS", "8")) / 1000.0
        self.detector_result_age_ms = 0.0
        self.detector_times: dict[str, deque[float]] = {}

        enabled = os.environ.get("CAMERA_V2_REID", "1").strip().lower() not in {"0", "false", "no", "off"}
        self.reid_mode = reid_backend() if enabled else "off"
        self.global_reid = GlobalReIDManager()
        self.reid_lock = threading.RLock()

        # nvstreammux is intentionally async and may push partial batches. A tracker
        # probe therefore often sees only one/few sources. Merge each fresh snapshot
        # into a short-lived cross-camera cache instead of replacing the other cams.
        self.track_snapshot_lock = threading.RLock()
        self.latest_tracks: dict[tuple[int, int], dict] = {}
        self.reid_track_cache_ttl = max(
            0.35, float(os.environ.get("CAMERA_V2_REID_TRACK_CACHE_TTL", "0.90"))
        )

        self.reid_last_submit: dict[tuple[int, int], float] = {}
        self.reid_vectors_seen = 0
        self.reid_last_batch = 0
        self.reid_error = ""
        self.reid_submitted = 0
        self.reid_match_misses = 0
        self.reid_track_cache_misses = 0
        self.reid_worker_not_ready = 0
        self.reid_track_rows_seen = 0

        self.reid_track_interval = max(
            0.6, float(os.environ.get("CAMERA_V2_REID_TRACK_INTERVAL", "1.5"))
        )
        self.reid_cycle_budget = max(
            1, min(3, int(os.environ.get("CAMERA_V2_REID_CROPS_PER_CYCLE", "1")))
        )
        self.reid_min_crop_h = max(
            24, int(os.environ.get("CAMERA_V2_REID_MIN_CROP_H", "42"))
        )
        self.reid_assoc_min_iou = max(
            0.0, min(1.0, float(os.environ.get("CAMERA_V2_REID_ASSOC_MIN_IOU", "0.08")))
        )
        self.reid_assoc_max_center = max(
            0.15, float(os.environ.get("CAMERA_V2_REID_ASSOC_MAX_CENTER", "0.48"))
        )
        self.external_reid: ExternalReIDWorker | None = None

        super().__init__()
        self._set_if(self.osd, "display-text", True)
        self.detector_target_hz = max(
            self.detector_min_hz, min(self.detector_max_hz, self.detector_target_hz)
        )
        self.detector_times = {cid: deque(maxlen=100) for cid in self.camera_index}
        self.latency_compensator = DetectorLatencyCompensator(
            self.frame_width, self.frame_height
        )

        if self.reid_mode == "external":
            self.external_reid = ExternalReIDWorker()
            # Model parse/warmup is isolated from GStreamer. Waiting briefly here
            # avoids throwing away the first useful person crops at startup.
            self.external_reid.ready_event.wait(
                timeout=max(1.0, float(os.environ.get("CAMERA_V2_REID_READY_TIMEOUT", "12")))
            )
            print(
                "CAMERA_REID backend=external-opencv-cpu "
                "reason=Pascal_SM6.1_not_supported_by_TensorRT10 "
                "nvtracker_reid=0 global_reid=1",
                flush=True,
            )
        elif self.reid_mode == "deepstream":
            print("CAMERA_REID backend=deepstream-tensorrt global_reid=1", flush=True)
        else:
            print("CAMERA_REID backend=off", flush=True)

    def _resolve_tracker_files(self):
        lib, stock_max_perf = super()._resolve_tracker_files()
        return lib, prepare_sparse_tracker_config(stock_max_perf)

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.external_reid is not None:
                self.external_reid.close()

    def _publish_prepared(
        self,
        cid: str,
        captured_t: float,
        prepared: list[PreparedDetection],
    ) -> None:
        with self.pending_lock:
            self.pending_seq += 1
            self.pending[cid] = (
                self.pending_seq,
                float(captured_t),
                list(prepared),
            )

    def _inject_detector_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        now = time.monotonic()
        boxes_added = 0
        frames_applied = 0
        max_age_ms = 0.0
        with self.pending_lock:
            pending = dict(self.pending)

        for cid, source_id in self.camera_index.items():
            row = pending.get(cid)
            if row is None:
                continue
            seq, captured_t, prepared = row
            if seq <= self.injected_seq.get(cid, 0):
                continue

            boxes, age_ms = self.latency_compensator.project(
                prepared, captured_t, now
            )
            result = self.bridge.apply_detector_result(buffer, source_id, boxes)
            if result == -2:
                # Current partial mux batch does not contain this source yet.
                continue
            if result < 0:
                continue

            self.injected_seq[cid] = seq
            frames_applied += 1
            boxes_added += result
            max_age_ms = max(max_age_ms, age_ms)

        if frames_applied or boxes_added:
            with self.det_lock:
                self.detector_frames_applied += frames_applied
                self.meta_boxes += boxes_added
                self.detector_result_age_ms = max_age_ms
        return self.Gst.PadProbeReturn.OK

    @staticmethod
    def _box_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = aa + bb - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def _track_box(row: dict) -> tuple[float, float, float, float]:
        left = float(row["left"])
        top = float(row["top"])
        return (
            left,
            top,
            left + float(row["width"]),
            top + float(row["height"]),
        )

    @staticmethod
    def _box_area(box) -> float:
        return max(1.0, float(box[2]) - float(box[0])) * max(
            1.0, float(box[3]) - float(box[1])
        )

    @staticmethod
    def _box_center(box) -> tuple[float, float]:
        return (
            (float(box[0]) + float(box[2])) * 0.5,
            (float(box[1]) + float(box[3])) * 0.5,
        )

    def _association_score(self, det_box, track_box) -> float | None:
        """Score a stale detector observation against a current NvDCF track.

        YOLO results arrive ~100-250ms after capture, so IoU-only matching can miss
        a walking person. We use projected detector geometry plus normalized center
        and size consistency, while keeping a strict enough gate to avoid attaching
        an embedding to the neighbouring person in crowded scenes.
        """
        iou = self._box_iou(det_box, track_box)
        dcx, dcy = self._box_center(det_box)
        tcx, tcy = self._box_center(track_box)
        tw = max(1.0, track_box[2] - track_box[0])
        th = max(1.0, track_box[3] - track_box[1])
        diag = max(24.0, math.hypot(tw, th))
        center_dist = math.hypot(dcx - tcx, dcy - tcy) / diag

        da = self._box_area(det_box)
        ta = self._box_area(track_box)
        size_similarity = min(da, ta) / max(da, ta)

        if size_similarity < 0.28:
            return None
        if iou < self.reid_assoc_min_iou and center_dist > self.reid_assoc_max_center:
            return None

        center_similarity = max(0.0, 1.0 - center_dist)
        return 0.68 * iou + 0.22 * center_similarity + 0.10 * size_similarity

    def _consume_external_reid(self) -> None:
        worker = self.external_reid
        if worker is None:
            return
        rows = worker.drain(16)
        if not rows:
            if worker.error:
                self.reid_error = worker.error
            return
        with self.reid_lock:
            self.global_reid.observe(rows, time.monotonic())
            self.reid_vectors_seen += len(rows)
            self.reid_last_batch = len(rows)
        self.reid_error = worker.error

    def _tracker_probe(self, _pad, info):
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK

        count = self.bridge.style_and_count_tracked(buffer)
        if count >= 0:
            with self.det_lock:
                self.tracked_now = count
                self.tracker_frames += 1

        try:
            if self.reid_mode == "external":
                now = time.monotonic()
                tracks = self.bridge.snapshot_tracks(buffer)
                with self.track_snapshot_lock:
                    # IMPORTANT: mux buffers may be partial. Merge fresh rows into
                    # the cache instead of replacing tracks from the other cameras.
                    for row in tracks:
                        item = dict(row)
                        item["_seen_at"] = now
                        key = (int(item["source_id"]), int(item["object_id"]))
                        self.latest_tracks[key] = item
                    self.reid_track_rows_seen += len(tracks)

                    stale = [
                        key
                        for key, row in self.latest_tracks.items()
                        if now - float(row.get("_seen_at", 0.0))
                        > self.reid_track_cache_ttl
                    ]
                    for key in stale:
                        self.latest_tracks.pop(key, None)
                        # If NvDCF later reuses a numeric local ID, it should be
                        # eligible for a fresh appearance sample immediately.
                        self.reid_last_submit.pop(key, None)
                self._consume_external_reid()

            elif self.reid_mode == "deepstream":
                rows = self.bridge.snapshot_reid(buffer)
                self.reid_last_batch = len(rows)
                if rows:
                    with self.reid_lock:
                        self.global_reid.observe(rows, time.monotonic())
                    self.reid_vectors_seen += len(rows)

            if self.reid_mode != "off":
                with self.reid_lock:
                    assignments = self.global_reid.label_assignments()
                self.bridge.apply_global_identity(buffer, assignments)
            else:
                self.bridge.apply_global_identity(buffer, [])

            if self.reid_mode != "external":
                self.reid_error = ""
        except Exception as exc:
            self.reid_error = f"{type(exc).__name__}: {exc}"
            try:
                self.bridge.apply_global_identity(buffer, [])
            except Exception:
                pass

        return self.Gst.PadProbeReturn.OK

    def _submit_external_reid(
        self,
        cid: str,
        frame,
        detections: list[tuple[tuple[float, float, float, float], float]],
        match_boxes: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        worker = self.external_reid
        if worker is None or worker.error or not detections:
            return
        if not worker.ready_event.is_set():
            self.reid_worker_not_ready += 1
            return

        source_id = int(self.camera_index[cid])
        now = time.monotonic()
        with self.track_snapshot_lock:
            tracks = [
                dict(row)
                for (sid, _oid), row in self.latest_tracks.items()
                if sid == source_id
                and now - float(row.get("_seen_at", 0.0)) <= self.reid_track_cache_ttl
            ]
        if not tracks:
            self.reid_track_cache_misses += 1
            return

        if match_boxes is None or len(match_boxes) != len(detections):
            match_boxes = [tuple(row[0]) for row in detections]

        pairs: list[tuple[float, int, int]] = []
        for di, det_box in enumerate(match_boxes):
            for ti, track in enumerate(tracks):
                score = self._association_score(det_box, self._track_box(track))
                if score is not None:
                    pairs.append((score, di, ti))

        pairs.sort(reverse=True)
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        matches: list[tuple[int, int]] = []
        for _score, di, ti in pairs:
            if di in used_dets or ti in used_tracks:
                continue
            used_dets.add(di)
            used_tracks.add(ti)
            matches.append((di, ti))

        if not matches:
            self.reid_match_misses += 1
            return

        # Prioritize local tracks that have not produced an embedding recently.
        ranked = []
        for di, ti in matches:
            track = tracks[ti]
            key = (source_id, int(track["object_id"]))
            ranked.append((self.reid_last_submit.get(key, 0.0), di, ti, key))
        ranked.sort(key=lambda item: item[0])

        submitted = 0
        sx = INFER_WIDTH / float(self.frame_width)
        sy = INFER_HEIGHT / float(self.frame_height)
        fh, fw = frame.shape[:2]

        for last_submit, di, ti, key in ranked:
            if submitted >= self.reid_cycle_budget:
                break
            if last_submit and now - last_submit < self.reid_track_interval:
                continue

            det_box, det_conf = detections[di]
            # Small context margin gives ReIdentificationNet a less brittle crop
            # without filling the embedding with desk/background pixels.
            bw = max(1.0, det_box[2] - det_box[0])
            bh = max(1.0, det_box[3] - det_box[1])
            crop_box = (
                det_box[0] - 0.03 * bw,
                det_box[1] - 0.02 * bh,
                det_box[2] + 0.03 * bw,
                det_box[3] + 0.03 * bh,
            )
            x1 = max(0, min(fw - 1, int(round(crop_box[0] * sx))))
            y1 = max(0, min(fh - 1, int(round(crop_box[1] * sy))))
            x2 = max(x1 + 1, min(fw, int(round(crop_box[2] * sx))))
            y2 = max(y1 + 1, min(fh, int(round(crop_box[3] * sy))))

            if y2 - y1 < self.reid_min_crop_h or x2 - x1 < 16:
                continue
            # Head/feet cut exactly by the image boundary makes appearance
            # embeddings unstable. Wait for a later, cleaner sample.
            if y1 <= 1 or y2 >= fh - 1:
                continue

            track = tracks[ti]
            crop = frame[y1:y2, x1:x2]
            if worker.submit(
                source_id=source_id,
                object_id=int(track["object_id"]),
                crop_bgr=crop,
                confidence=float(det_conf),
                tracker_confidence=float(track.get("tracker_confidence", 0.0) or 0.0),
            ):
                self.reid_last_submit[key] = now
                self.reid_submitted += 1
                submitted += 1

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO worker failed")
            return

        with self.det_lock:
            self.det_ready = True
        print(
            "CAMERA_TRACK_FINAL ready: "
            f"YOLO26m micro_batch={MICRO_BATCH} input={INFER_WIDTH}x{INFER_HEIGHT} "
            f"conf={os.environ.get('CAMERA_V2_DETECT_CONF')} "
            f"iou={os.environ.get('CAMERA_V2_DETECT_IOU')} "
            f"target={self.detector_target_hz:.1f}Hz/cam "
            f"range={self.detector_min_hz:.1f}-{self.detector_max_hz:.1f}Hz/cam "
            f"NvDCF={self.tracker_width}x{self.tracker_height} "
            f"device={ready.get('device')} cuda={ready.get('cuda')} "
            f"reid_backend={self.reid_mode} close_person=1 "
            "aspect_preserved=1 display_gap_bridge=1",
            flush=True,
        )

        ids = [camera.camera_id for camera in self.cameras]
        groups = [ids[i : i + MICRO_BATCH] for i in range(0, len(ids), MICRO_BATCH)]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        prefetched_group: tuple[str, ...] | None = None

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1
            group_key = tuple(group)

            if prefetched_group != group_key:
                self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.8)
            prefetched_group = None
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                self.det_stop.wait(0.025)
                continue

            frames = []
            captured = []
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
            self._clear_requests()

            next_group = groups[group_index % len(groups)]
            self._request_group(next_group)
            prefetched_group = tuple(next_group)

            try:
                self.job_q.put(
                    {"cameras": group, "frames": frames, "captured": captured},
                    timeout=0.3,
                )
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO batch error")
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            completed_t = time.monotonic()
            frame_by_cid = {cid: frame for cid, frame in zip(group, frames)}
            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(result["cameras"], result["captured"]):
                detections = self._dedup_and_expand(result["boxes"].get(cid, []))
                prepared = self.latency_compensator.prepare(cid, captured_t, detections)
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)
                ages_ms.append(max(0.0, (completed_t - captured_t) * 1000.0))
                self.detector_times[cid].append(completed_t)

                if self.reid_mode == "external":
                    frame = frame_by_cid.get(cid)
                    if frame is not None:
                        # Match against geometry projected to the tracker timestamp,
                        # but crop from the original detector frame. This fixes the
                        # previous detector-vs-live-track timing mismatch.
                        projected, _ = self.latency_compensator.project(
                            prepared, captured_t, completed_t
                        )
                        match_boxes = [
                            (row[0], row[1], row[2], row[3]) for row in projected
                        ]
                        self._submit_external_reid(
                            cid, frame, detections, match_boxes
                        )

            batch_ms = float(result.get("batch_ms") or 0.0)
            with self.det_lock:
                self.det_calls += 1
                self.det_inputs += len(group)
                self.det_batch_ms = batch_ms
                self.det_counts.update(counts)
                if ages_ms:
                    self.detector_result_age_ms = max(ages_ms)
                self.det_error = ""
                target_hz = self.detector_target_hz

            desired_call_interval = 1.0 / max(0.1, target_hz * len(groups))
            elapsed = time.monotonic() - cycle_started
            idle = max(self.detector_min_idle, desired_call_interval - elapsed)
            self.det_stop.wait(min(0.25, idle))

    @staticmethod
    def _recent_rate(times: deque[float], now: float, horizon: float = 5.0) -> float:
        while times and now - times[0] > horizon:
            times.popleft()
        if len(times) < 2:
            return 0.0
        span = max(0.2, times[-1] - times[0])
        return (len(times) - 1) / span

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        p95 = self._p95(self.wall_intervals_ms)
        now = time.monotonic()

        with self.det_lock:
            if p95 is not None:
                if p95 > 92.0:
                    self.detector_target_hz -= 0.45
                elif p95 > 80.0:
                    self.detector_target_hz -= 0.20
                elif p95 < 64.0 and self.det_ready:
                    self.detector_target_hz += 0.10
            self.detector_target_hz = max(
                self.detector_min_hz,
                min(self.detector_max_hz, self.detector_target_hz),
            )
            applied = self.detector_frames_applied
            tracked = self.tracked_now
            age_ms = self.detector_result_age_ms
            target_hz = self.detector_target_hz

        rates = {
            cid: self._recent_rate(rows, now) for cid, rows in self.detector_times.items()
        }
        rate_text = " ".join(
            f"{cid}:{rates.get(cid, 0.0):.1f}" for cid in self.camera_index
        )
        expected_skip = max(0.0, 20.0 / max(0.1, target_hz) - 1.0)
        print(
            "CAMERA_TRACK_FINAL "
            f"detector_frames={applied} tracked_now={tracked} "
            f"detector={INFER_WIDTH}x{INFER_HEIGHT}/micro{MICRO_BATCH} "
            f"target_hz={target_hz:.1f}/cam approx_skip={expected_skip:.1f}frames "
            f"actual_hz=[{rate_text}] result_age={age_ms:.0f}ms "
            f"tracker={self.tracker_width}x{self.tracker_height} "
            f"config={self.tracker_config} profile=max_perf max_targets=24 "
            "cascaded_assoc=1 display_gap_bridge=1 synthetic_tracker_boxes=0",
            flush=True,
        )

        with self.reid_lock:
            reid = self.global_reid.snapshot()
        stats = reid["stats"]
        worker_stats = self.external_reid.snapshot() if self.external_reid is not None else {}
        worker_error = worker_stats.get("error", "") if worker_stats else ""
        with self.track_snapshot_lock:
            track_cache = len(self.latest_tracks)

        print(
            "CAMERA_REID "
            f"backend={self.reid_mode} "
            f"ready={int(bool(worker_stats.get('ready', self.reid_mode != 'external')))} "
            f"vectors={self.reid_vectors_seen} "
            f"last_batch={self.reid_last_batch} submitted={self.reid_submitted} "
            f"track_rows={self.reid_track_rows_seen} track_cache={track_cache} "
            f"cache_miss={self.reid_track_cache_misses} assoc_miss={self.reid_match_misses} "
            f"worker_wait={self.reid_worker_not_ready} "
            f"globals={reid['global_count']} bindings={reid['local_bindings']} "
            f"strong={stats['strong_match']} merged={stats['merged']} "
            f"conflicts={stats['rejected_conflict']} "
            f"infer_ms={float(worker_stats.get('infer_ms', 0.0)):.1f} "
            f"queue={int(worker_stats.get('queued', 0))} dropped={int(worker_stats.get('dropped', 0))} "
            f"error={self.reid_error or worker_error or 'none'}",
            flush=True,
        )
        return keep


def main() -> int:
    return CameraPersonTrackingFinal().run()


if __name__ == "__main__":
    raise SystemExit(main())
