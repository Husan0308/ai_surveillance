from __future__ import annotations

from collections import deque
import math
import time

from .camera_overlay import CameraAnkleHeatmapCoordinator as _BaseCameraHeatmap


class CameraAnkleHeatmapCoordinator(_BaseCameraHeatmap):
    """Camera heatmap v2 with per-person pose/bbox fusion.

    A pose sample only suppresses the detector fallback for the nearby person it
    explains. Other people in the same camera still contribute detector foot
    points, so a sparse single-person pose side path cannot make the heatmap look
    empty in a busy room.
    """

    def __init__(self, pose, frame_stores, config=None, detections=None):
        super().__init__(pose, frame_stores, config, detections=detections)
        # Keep the accumulator cheap, but make fallback frequent enough to build a
        # visible trail when CPU pose is sparse.
        self.fallback_every_n = min(self.fallback_every_n, 2)
        self.overlay_alpha = max(self.overlay_alpha, 0.34)
        self.overlay_threshold = min(self.overlay_threshold, 0.018)
        self.sigma_cells = max(self.sigma_cells, 3.0)
        self.dedupe_sec = min(self.dedupe_sec, 0.30)
        self.dedupe_distance = min(self.dedupe_distance, 0.018)
        self.pose_cover_sec = max(0.5, float(self.config.get("pose_cover_sec", 1.6)))
        self.pose_cover_distance = max(
            0.01, float(self.config.get("pose_cover_distance_norm", 0.085))
        )
        self._pose_contacts = {cid: deque(maxlen=32) for cid in self.frame_stores}
        self._covered_bbox_skips = {cid: 0 for cid in self.frame_stores}

    def _remember_pose_contact(self, camera_id, x, y, source_w, source_h, now):
        nx = float(x) / max(1.0, float(source_w))
        ny = float(y) / max(1.0, float(source_h))
        contacts = self._pose_contacts[camera_id]
        contacts.append((nx, ny, float(now)))

    def _is_pose_covered(self, camera_id, x, y, source_w, source_h, now):
        nx = float(x) / max(1.0, float(source_w))
        ny = float(y) / max(1.0, float(source_h))
        contacts = self._pose_contacts[camera_id]
        while contacts and now - contacts[0][2] > self.pose_cover_sec:
            contacts.popleft()
        return any(
            math.hypot(nx - px, ny - py) <= self.pose_cover_distance
            for px, py, _ts in contacts
        )

    def _run(self):
        while not self._stop.is_set():
            did_work = False
            try:
                pose_snapshot = self.pose.snapshot() if self.pose is not None else {}
                det_snapshot = self.detections.snapshot() if self.detections is not None else {}

                for camera_id in tuple(self._grids):
                    source_size = self._source_size(camera_id)
                    if not source_size:
                        continue
                    source_w, source_h = source_size
                    now = time.monotonic()
                    with self._lock:
                        self._decay_locked(camera_id, now)

                    # Add every fresh pose contact first and remember which part of
                    # the floor was explained by a real foot/pose result.
                    pose_result = pose_snapshot.get(camera_id)
                    if (
                        pose_result is not None
                        and int(pose_result.frame_id) > self._last_pose_frame[camera_id]
                    ):
                        self._last_pose_frame[camera_id] = int(pose_result.frame_id)
                        with self._lock:
                            for person in tuple(pose_result.people or ()):
                                contact = self._contact_from_pose(person)
                                if contact is None:
                                    self._ankle_skips[camera_id] += 1
                                    continue
                                x, y, weight, source = contact
                                added = self._add_sample_locked(
                                    camera_id, x, y, source_w, source_h, weight, source
                                )
                                self._remember_pose_contact(
                                    camera_id, x, y, source_w, source_h, now
                                )
                                did_work |= bool(added)

                    # Detector fallback is per person, not per camera. A pose for
                    # person A must not suppress person B/C/D from the heatmap.
                    detection = det_snapshot.get(camera_id)
                    if (
                        detection is not None
                        and int(detection.frame_id) > self._last_detection_frame[camera_id]
                    ):
                        self._last_detection_frame[camera_id] = int(detection.frame_id)
                        self._fallback_seen[camera_id] += 1
                        if self._fallback_seen[camera_id] % self.fallback_every_n == 0:
                            boxes = sorted(
                                tuple(detection.boxes or ()),
                                key=lambda item: float(item.confidence),
                                reverse=True,
                            )[: self.max_fallback_people]
                            with self._lock:
                                for box in boxes:
                                    point = self._bbox_contact(box)
                                    if point is None:
                                        continue
                                    x, y = point
                                    if self._is_pose_covered(
                                        camera_id,
                                        x,
                                        y,
                                        source_w,
                                        source_h,
                                        now,
                                    ):
                                        self._covered_bbox_skips[camera_id] += 1
                                        continue
                                    did_work |= self._add_sample_locked(
                                        camera_id,
                                        x,
                                        y,
                                        source_w,
                                        source_h,
                                        self.bbox_weight,
                                        "detector_bbox",
                                    )
                    with self._lock:
                        self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._errors += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
            if not did_work:
                self._stop.wait(self.poll_sec)

    def reset(self, camera_id=None):
        super().reset(camera_id)
        with self._lock:
            targets = [str(camera_id)] if camera_id is not None else list(self._grids)
            for cid in targets:
                self._pose_contacts[cid].clear()
                self._covered_bbox_skips[cid] = 0

    def snapshot(self):
        payload = super().snapshot()
        with self._lock:
            for cid, camera in (payload.get("cameras") or {}).items():
                camera["bbox_covered_by_pose_skips"] = int(
                    self._covered_bbox_skips.get(cid, 0)
                )
                camera["recent_pose_contacts"] = len(self._pose_contacts.get(cid, ()))
        payload["fusion"] = "ankle-first + per-unmatched-person bbox-bottom fallback"
        return payload
