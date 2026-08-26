from __future__ import annotations

import os
import queue as pyqueue
import threading
import time
from pathlib import Path

from .detection import INFER_HEIGHT, INFER_WIDTH
from .person_tracking_pascal_trt86 import CameraPersonTrackingPascalTRT86
from .pose_gate_v2 import PoseGateClient


def _box_area(box) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_intersection(a, b) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a, b) -> float:
    inter = _box_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _box_containment(a, b) -> float:
    return _box_intersection(a, b) / max(1.0, min(_box_area(a), _box_area(b)))


def _center_ratio(a, b) -> float:
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


def _replace_tracker_key(lines: list[str], key: str, value: str, *, required: bool) -> bool:
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith(key + ":"):
            continue
        indent = line[: len(line) - len(stripped)]
        comment = ""
        if "#" in stripped:
            comment = "  #" + stripped.split("#", 1)[1]
        lines[index] = f"{indent}{key}: {value}{comment}"
        return True
    if required:
        raise RuntimeError(f"ML NvDCF config missing required key: {key}")
    return False


class CameraPersonTrackingTRT86PoseGate(CameraPersonTrackingPascalTRT86):
    """TRT8.6 detector + track-aware YOLO26s-pose gate + NvDCF.

    Pose is not a per-frame detector. Strong YOLO boxes and ambiguous boxes that
    already overlap a live NvDCF track bypass pose. Only genuinely new ambiguous
    candidates reach the crop-only S-pose worker; recent pose decisions are cached.
    """

    def __init__(self) -> None:
        self.pose_gate: PoseGateClient | None = None
        self._gate_logs = 0
        self._track_snapshot_lock = threading.RLock()
        self._track_boxes_raw: dict[str, tuple[float, list[tuple[float, float, float, float]]]] = {}
        self._raw_dedup_iou = float(os.environ.get("CAMERA_V2_RAW_DEDUP_IOU", "0.72"))
        self._raw_dedup_containment = float(os.environ.get("CAMERA_V2_RAW_DEDUP_CONTAINMENT", "0.88"))
        self._track_reuse_max_age = max(0.10, float(os.environ.get("CAMERA_V2_POSE_TRACK_REUSE_MAX_AGE_SEC", "0.50")))
        super().__init__()
        self.pose_gate = PoseGateClient()
        print(
            "CAMERA_ML_ARCH "
            "primary=YOLO26s/TRT8.6 pose_gate=YOLO26s-pose/crop-only/CPU/cached "
            "tracker=NvDCF pose_per_frame=0 tracker_reuse=1 global_id=off reid=off face=off",
            flush=True,
        )

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        # First retain the Pascal-safe sparse settings, then tighten only the
        # duplicate-target policy for this ML branch. The golden wall is untouched.
        path = CameraPersonTrackingPascalTRT86._stabilize_tracker_config(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        _replace_tracker_key(lines, "minIouDiff4NewTarget", "0.55", required=True)
        _replace_tracker_key(lines, "probationAge", "1", required=True)
        has_dup = _replace_tracker_key(lines, "minIou4TargetDuplicate", "0.85", required=False)
        has_interval = _replace_tracker_key(lines, "targetDuplicateRunInterval", "1", required=False)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_ML_NVDCF_DUP_GUARD "
            f"minIouDiff4NewTarget=0.55 probationAge=1 "
            f"minIou4TargetDuplicate={'0.85' if has_dup else 'unsupported'} "
            f"targetDuplicateRunInterval={'1' if has_interval else 'unsupported'}",
            flush=True,
        )
        return path

    def _dedup_raw_rows(self, rows):
        """Remove near-identical raw detector boxes before pose or NvDCF.

        This is intentionally stricter than generic NMS only for same-center,
        near-identical/nested boxes. It avoids running pose twice on the same person
        and prevents two detector metas from seeding two NvDCF IDs.
        """
        ordered = sorted(rows, key=lambda row: float(row[1]), reverse=True)
        kept = []
        for coords, score in ordered:
            box = tuple(float(v) for v in coords)
            duplicate = False
            for old_coords, _old_score in kept:
                old = tuple(float(v) for v in old_coords)
                iou = _box_iou(box, old)
                containment = _box_containment(box, old)
                center = _center_ratio(box, old)
                if iou >= self._raw_dedup_iou or (
                    containment >= self._raw_dedup_containment and center <= 0.20
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append((coords, float(score)))
        return kept

    def _tracker_probe(self, pad, info):
        """Snapshot live NvDCF geometry before display-only smoothing.

        Snapshots are converted back to detector coordinates and used only to decide
        whether pose is necessary. NvDCF remains the sole temporal tracking truth.
        """
        buffer = info.get_buffer()
        if buffer is not None:
            try:
                rows = self.bridge.copy_tracks(buffer, max_rows=256)
                source_to_cid = {int(source_id): cid for cid, source_id in self.camera_index.items()}
                by_camera: dict[str, list[tuple[float, float, float, float]]] = {
                    cid: [] for cid in self.camera_index
                }
                sx = float(INFER_WIDTH) / max(1.0, float(self.frame_width))
                sy = float(INFER_HEIGHT) / max(1.0, float(self.frame_height))
                for row in rows:
                    cid = source_to_cid.get(int(row.get("source_id", -1)))
                    if cid is None:
                        continue
                    tracker_conf = float(row.get("tracker_confidence", 0.0))
                    width = float(row.get("width", 0.0))
                    height = float(row.get("height", 0.0))
                    if tracker_conf < 0.05 or width < 8.0 or height < 16.0:
                        continue
                    left = float(row.get("left", 0.0))
                    top = float(row.get("top", 0.0))
                    by_camera[cid].append(
                        (
                            left * sx,
                            top * sy,
                            (left + width) * sx,
                            (top + height) * sy,
                        )
                    )
                now = time.monotonic()
                with self._track_snapshot_lock:
                    for cid, boxes in by_camera.items():
                        self._track_boxes_raw[cid] = (now, boxes)
            except Exception as exc:
                last = getattr(self, "_track_snapshot_warning", 0.0)
                now = time.monotonic()
                if now - last >= 5.0:
                    self._track_snapshot_warning = now
                    print(f"CAMERA_ML_TRACK_SNAPSHOT warning={type(exc).__name__}:{exc}", flush=True)
        return super()._tracker_probe(pad, info)

    def _trusted_track_boxes(self, cid: str) -> list[tuple[float, float, float, float]]:
        now = time.monotonic()
        with self._track_snapshot_lock:
            row = self._track_boxes_raw.get(cid)
            if row is None:
                return []
            captured, boxes = row
            if now - captured > self._track_reuse_max_age:
                return []
            return list(boxes)

    def _scheduler(self) -> None:
        assert self.result_q is not None and self.job_q is not None
        assert self.pose_gate is not None

        try:
            ready = self.result_q.get(timeout=40.0)
        except pyqueue.Empty:
            with self.det_lock:
                self.det_error = "YOLO TRT86 worker startup timeout"
            return
        if ready.get("type") != "ready":
            with self.det_lock:
                self.det_error = ready.get("error", "YOLO TRT86 worker failed")
            return

        with self.det_lock:
            self.det_ready = True

        all_ids = [camera.camera_id for camera in self.cameras]
        allowed = self._active_camera_set()
        ids = [cid for cid in all_ids if cid in allowed]
        if not ids:
            raise RuntimeError("CAMERA_V2_DETECT_ACTIVE_CAMERAS selected no cameras")

        print(
            "CAMERA_ML_READY "
            f"model={ready.get('model')} input={INFER_WIDTH}x{INFER_HEIGHT} micro_batch=1 "
            f"raw_conf={os.environ.get('CAMERA_V2_DETECT_CONF')} target={self.detector_target_hz:.2f}Hz/cam "
            f"tracker={self.tracker_width}x{self.tracker_height} active={','.join(ids)} "
            f"backend={ready.get('backend')} flow=TRT86->raw-dedup->track-aware-pose-gate->NvDCF "
            "capture=jit-latest-no-prefetch",
            flush=True,
        )

        groups = [[cid] for cid in ids]
        versions = {cid: 0 for cid in ids}
        group_index = 0
        age_log_n = 0

        while not self.det_stop.is_set():
            cycle_started = time.monotonic()
            group = groups[group_index % len(groups)]
            group_index += 1

            self._request_group(group)
            rows = self.mailbox.wait_group(group, versions, timeout=0.8)
            if rows is None:
                self._clear_requests()
                with self.det_lock:
                    self.capture_timeouts += 1
                    timeout_count = self.capture_timeouts
                if timeout_count <= 3 or timeout_count % 20 == 0:
                    print(f"CAMERA_ML_CAPTURE_TIMEOUT count={timeout_count} waiting={','.join(group)}", flush=True)
                self.det_stop.wait(0.025)
                continue

            frames = []
            captured = []
            frame_by_cid = {}
            for cid, row in zip(group, rows):
                version, captured_t, frame = row
                versions[cid] = version
                captured.append(captured_t)
                frames.append(frame)
                frame_by_cid[cid] = frame
            self._clear_requests()

            try:
                self.job_q.put({"cameras": group, "frames": frames, "captured": captured}, timeout=0.3)
                result = self.result_q.get(timeout=5.0)
            except pyqueue.Empty:
                with self.det_lock:
                    self.det_error = "YOLO TRT86 result timeout"
                self.det_stop.wait(0.05)
                continue

            if result.get("type") == "fatal":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO TRT86 fatal error")
                return
            if result.get("type") == "batch_error":
                with self.det_lock:
                    self.det_error = result.get("error", "YOLO TRT86 batch error")
                self.det_stop.wait(0.10)
                continue
            if result.get("type") != "result":
                continue

            counts: dict[str, int] = {}
            ages_ms: list[float] = []

            for cid, captured_t in zip(result["cameras"], result["captured"]):
                raw_all = list(result["boxes"].get(cid, []))
                raw_rows = self._dedup_raw_rows(raw_all)
                trusted = self._trusted_track_boxes(cid)
                gated_rows, gate = self.pose_gate.filter(
                    cid,
                    frame_by_cid[cid],
                    raw_rows,
                    trusted_boxes=trusted,
                )

                # A second conservative pass after pose protects the metadata
                # boundary even if model geometry changes slightly after validation.
                detections = self._dedup_and_expand(gated_rows)
                prepared = self.latency_compensator.prepare(cid, captured_t, detections)
                self._publish_prepared(cid, captured_t, prepared)
                counts[cid] = len(detections)

                completed_t = time.monotonic()
                age_ms = max(0.0, (completed_t - captured_t) * 1000.0)
                ages_ms.append(age_ms)
                self.detector_times[cid].append(completed_t)

                self._gate_logs += 1
                predup_removed = max(0, len(raw_all) - len(raw_rows))
                if (
                    self._gate_logs <= 18
                    or self._gate_logs % 30 == 0
                    or gate.pose_reject > 0
                    or gate.fallback > 0
                    or predup_removed > 0
                ):
                    print(
                        "CAMERA_ML_GATE "
                        f"cid={cid} raw={len(raw_all)} predup_removed={predup_removed} gate_in={gate.raw} "
                        f"direct={gate.direct} tracker_reuse={gate.tracker_reuse} "
                        f"cache_accept={gate.cache_accept} cache_reject={gate.cache_reject} "
                        f"pose_checked={gate.pose_checked} pose_accept={gate.pose_accept} pose_reject={gate.pose_reject} "
                        f"low_reject={gate.low_reject} overflow={gate.overflow} final={len(detections)} "
                        f"pose_ms={gate.pose_ms:.1f} fallback={gate.fallback}",
                        flush=True,
                    )

            self._update_freshness_budget(ages_ms)
            batch_ms = float(result.get("batch_ms") or 0.0)

            age_log_n += 1
            if ages_ms and (age_log_n <= 3 or age_log_n % 20 == 0):
                print(
                    "CAMERA_ML_FRESHNESS "
                    f"n={age_log_n} result_age={max(ages_ms):.1f}ms budget={self.max_detector_result_age_ms:.1f}ms "
                    f"trt_batch={batch_ms:.1f}ms",
                    flush=True,
                )

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
            self.det_stop.wait(idle)

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.pose_gate is not None:
                self.pose_gate.close()


def main() -> int:
    return CameraPersonTrackingTRT86PoseGate().run()


if __name__ == "__main__":
    raise SystemExit(main())
