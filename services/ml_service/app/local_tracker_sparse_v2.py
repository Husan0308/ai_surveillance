from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from services.ml_service.app.local_tracker import (
    Detection,
    LocalPersonTracker,
    TrackerUpdate,
    appearance_descriptor,
)


class SparseRecoveryPersonTracker(LocalPersonTracker):
    """Step 4 v2 tracker tuned for a sparse (~2 Hz) person detector.

    The v1 tracker already used ByteTrack-style high/low association, but its low-score
    second stage only considered tracks whose state was still ``tracked``. With a 2 Hz
    detector, one missed measurement immediately changes a confirmed track to ``lost``;
    a following low-confidence person detection could therefore no longer recover that
    ID. The result is exactly the fragmentation pattern seen on the live cameras.

    V2 keeps the same CPU-only geometry/appearance model and adds two bounded recovery
    stages before any new ID is created:

    * relaxed high-confidence reacquisition for recently lost confirmed tracks;
    * low-confidence recovery for recently lost confirmed tracks.

    Low-confidence boxes still never create a new ID. New high-confidence boxes are also
    duplicate-suppressed when they strongly overlap a track already matched in the same
    update. No ReID network, GPU tracker, NvDCF or cross-camera identity is introduced.
    """

    def __init__(
        self,
        *args,
        reacquire_thresh: float = 0.12,
        low_recovery_thresh: float = 0.10,
        low_recovery_sec: float = 1.6,
        duplicate_iou: float = 0.60,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reacquire_thresh = min(self.match_thresh, max(0.05, float(reacquire_thresh)))
        self.low_recovery_thresh = min(
            self.low_match_thresh, max(0.05, float(low_recovery_thresh))
        )
        self.low_recovery_sec = min(
            self.max_lost_sec, max(0.5, float(low_recovery_sec))
        )
        self.duplicate_iou = min(0.90, max(0.40, float(duplicate_iou)))

    def _is_duplicate_of_matched(self, det: Detection, matched_tracks: list[object]) -> bool:
        for track in matched_tracks:
            tbox = self._state_to_xyxy(track.state_vec)
            if self._iou(tbox, det.bbox) >= self.duplicate_iou:
                return True
        return False

    def update(self, detections: Iterable[Detection], timestamp: float) -> TrackerUpdate:
        started = time.perf_counter()
        timestamp = float(timestamp)
        rows = [d for d in detections if d.score >= self.low_thresh]
        high = [d for d in rows if d.score >= self.high_thresh]
        low = [d for d in rows if self.low_thresh <= d.score < self.high_thresh]

        live_tracks = [t for t in self._tracks if t.status != "removed"]
        for track in live_tracks:
            self._predict_to(track, timestamp)

        # Stage 1: normal high-confidence association, unchanged from v1.
        stage1_tracks = [
            t
            for t in live_tracks
            if t.status in ("tracked", "tentative", "lost")
            and timestamp - t.last_detection <= self.max_lost_sec
        ]
        matches1, unmatched_t1, unmatched_high = self._assign(
            stage1_tracks,
            high,
            timestamp,
            self.match_thresh,
            low_stage=False,
        )

        matched_track_ids: set[int] = set()
        matched_track_objects: list[object] = []
        recovered = 0
        for ti, di in matches1:
            track = stage1_tracks[ti]
            if self._update_track(track, high[di], timestamp):
                recovered += 1
            matched_track_ids.add(id(track))
            matched_track_objects.append(track)

        # Stage 1b: sparse-detector rescue. A confirmed recently-lost track gets one
        # relaxed geometry+appearance chance before an unmatched high box can mint a
        # new identity. This is bounded by max_lost_sec and the base pair-score's hard
        # distance/size gates, so appearance can never teleport a track across frame.
        rescue_tracks = [
            stage1_tracks[i]
            for i in unmatched_t1
            if stage1_tracks[i].hits >= self.confirm_hits
            and stage1_tracks[i].status in ("tracked", "lost")
            and timestamp - stage1_tracks[i].last_detection <= self.max_lost_sec
        ]
        rescue_high = [high[i] for i in unmatched_high]
        matches1b, _unmatched_rescue_tracks, unmatched_rescue_high = self._assign(
            rescue_tracks,
            rescue_high,
            timestamp,
            self.reacquire_thresh,
            low_stage=False,
        )
        for ti, di in matches1b:
            track = rescue_tracks[ti]
            if self._update_track(track, rescue_high[di], timestamp):
                recovered += 1
            matched_track_ids.add(id(track))
            matched_track_objects.append(track)
        remaining_high_indices = [unmatched_high[i] for i in unmatched_rescue_high]

        # Stage 2: ByteTrack-style low-confidence association, extended to recently
        # lost *confirmed* tracks. A low box can preserve/recover an identity but can
        # never create one. This is the key sparse-2Hz fragmentation fix.
        stage2_tracks = [
            t
            for t in live_tracks
            if id(t) not in matched_track_ids
            and t.hits >= self.confirm_hits
            and t.status in ("tracked", "lost")
            and timestamp - t.last_detection <= self.low_recovery_sec
        ]
        matches2, _unmatched_t2, _unmatched_low = self._assign(
            stage2_tracks,
            low,
            timestamp,
            self.low_recovery_thresh,
            low_stage=True,
        )
        for ti, di in matches2:
            track = stage2_tracks[ti]
            if self._update_track(track, low[di], timestamp):
                recovered += 1
            matched_track_ids.add(id(track))
            matched_track_objects.append(track)

        # Only genuinely unmatched high-confidence detections can create a new ID.
        # Strongly-overlapping boxes around a track already matched this update are
        # treated as detector duplicates instead of fragmenting the identity.
        created = 0
        for di in remaining_high_indices:
            det = high[di]
            if det.score < self.new_track_thresh:
                continue
            if self._is_duplicate_of_matched(det, matched_track_objects):
                continue
            self._new_track(det, timestamp)
            created += 1

        newly_lost = 0
        removed = 0
        for track in self._tracks:
            if track.status == "removed" or id(track) in matched_track_ids:
                continue
            since = timestamp - track.last_detection
            if track.status == "tentative":
                if since > self.tentative_ttl_sec:
                    track.status = "removed"
                    removed += 1
                continue
            if track.status == "tracked" and since > 1e-6:
                track.status = "lost"
                track.lost_since = timestamp
                newly_lost += 1
            if track.status == "lost" and since > self.max_lost_sec:
                track.status = "removed"
                removed += 1

        snapshots = []
        for track in self._tracks:
            if track.status == "removed":
                continue
            since = timestamp - track.last_detection
            if track.status == "tracked":
                snapshots.append(self._snapshot(track, timestamp, predicted=False))
            elif track.status == "tentative" and since <= 1e-6:
                snapshots.append(self._snapshot(track, timestamp, predicted=False))
            elif track.status == "lost" and track.hits >= self.confirm_hits and since <= self.shadow_sec:
                snapshots.append(self._snapshot(track, timestamp, predicted=True))

        active = sum(1 for t in self._tracks if t.status != "removed")
        step_ms = (time.perf_counter() - started) * 1000.0
        return TrackerUpdate(
            camera_id=self.camera_id,
            detections=len(rows),
            high_detections=len(high),
            low_detections=len(low),
            active=active,
            renderable=len(snapshots),
            matched_high=len(matches1) + len(matches1b),
            matched_low=len(matches2),
            created=created,
            recovered=recovered,
            newly_lost=newly_lost,
            removed=removed,
            snapshots=snapshots,
            step_ms=step_ms,
        )


class MultiCameraSparseRecoveryTracker:
    def __init__(self, camera_ids: Iterable[str], width: int, height: int, **kwargs) -> None:
        self.trackers = {
            cid: SparseRecoveryPersonTracker(cid, width, height, **kwargs) for cid in camera_ids
        }

    def update(
        self,
        camera_id: str,
        boxes: Iterable[Iterable[float]],
        frame_bgr: np.ndarray,
        captured_ns: int,
    ) -> TrackerUpdate:
        detections: list[Detection] = []
        for row in boxes:
            values = list(row)
            if len(values) != 5:
                continue
            x1, y1, x2, y2, score = (float(v) for v in values)
            bbox = np.array((x1, y1, x2, y2), dtype=np.float64)
            detections.append(
                Detection(
                    bbox=bbox,
                    score=score,
                    appearance=appearance_descriptor(frame_bgr, bbox),
                )
            )
        timestamp = captured_ns / 1_000_000_000.0
        return self.trackers[camera_id].update(detections, timestamp)
