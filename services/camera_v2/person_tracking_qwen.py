from __future__ import annotations

import os
import time

from .person_tracking_final import CameraPersonTrackingFinal
from .qwen_reid_verifier import QwenRoomReIDVerifier


class CameraPersonTrackingQwen(CameraPersonTrackingFinal):
    """Camera V2 runtime with asynchronous Qwen peer-camera verification.

    Fast path remains YOLO + NvDCF + TAO ReID. Qwen only audits same-room peer
    camera identities and can confirm two local tracks as one person or split an
    accidental shared Global ID after repeated independent visual votes.
    """

    def __init__(self) -> None:
        self.qwen_reid: QwenRoomReIDVerifier | None = None
        super().__init__()
        if self.reid_mode != "off":
            self.qwen_reid = QwenRoomReIDVerifier()
            q = self.qwen_reid
            print(
                "CAMERA_QWEN_REID "
                f"enabled={int(q.enabled)} url={q.url} model={q.model} "
                "scope=same-room-peer-cameras async=1 reversible=1 votes=2 "
                f"room_map={self.global_reid.room_map}",
                flush=True,
            )

    def run(self) -> int:
        try:
            return super().run()
        finally:
            if self.qwen_reid is not None:
                self.qwen_reid.close()

    def _remember_qwen_visuals(self, cid, frame, detections, match_boxes=None) -> None:
        verifier = self.qwen_reid
        if verifier is None or not verifier.enabled or frame is None or not detections:
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
            return

        if match_boxes is None or len(match_boxes) != len(detections):
            match_boxes = [
                (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                for row in detections
            ]

        pairs = []
        for di, det_box in enumerate(match_boxes):
            for ti, track in enumerate(tracks):
                score = self._association_score(det_box, self._track_box(track))
                if score is not None:
                    pairs.append((score, di, ti))
        pairs.sort(reverse=True)

        used_dets = set()
        used_tracks = set()
        sx = frame.shape[1] / float(self.frame_width)
        sy = frame.shape[0] / float(self.frame_height)
        fh, fw = frame.shape[:2]

        for assoc_score, di, ti in pairs:
            if di in used_dets or ti in used_tracks:
                continue
            used_dets.add(di)
            used_tracks.add(ti)
            dx1, dy1, dx2, dy2, det_conf = [float(v) for v in detections[di]]

            # Keep Qwen memory cleaner than the detector display: partial border
            # crops are allowed for tracking but are poor identity evidence.
            if (
                dx1 <= 2.0
                or dy1 <= 2.0
                or dx2 >= float(self.frame_width - 2)
                or dy2 >= float(self.frame_height - 2)
            ):
                continue
            bw = max(1.0, dx2 - dx1)
            bh = max(1.0, dy2 - dy1)
            x1 = max(0, min(fw - 1, int(round((dx1 - 0.025 * bw) * sx))))
            y1 = max(0, min(fh - 1, int(round((dy1 - 0.015 * bh) * sy))))
            x2 = max(x1 + 1, min(fw, int(round((dx2 + 0.025 * bw) * sx))))
            y2 = max(y1 + 1, min(fh, int(round((dy2 + 0.020 * bh) * sy))))
            if y2 - y1 < self.reid_min_crop_h or x2 - x1 < 14:
                continue

            track = tracks[ti]
            tracker_conf = float(track.get("tracker_confidence", 0.0) or 0.0)
            quality = max(float(det_conf), tracker_conf) * max(0.25, float(assoc_score))
            verifier.remember(
                (source_id, int(track["object_id"])),
                frame[y1:y2, x1:x2],
                quality,
            )

    def _submit_external_reid(self, cid, frame, detections, match_boxes=None) -> None:
        # Fast TAO ReID remains unchanged. Qwen gets a separate visual memory and
        # never blocks or alters detector/tracker scheduling.
        super()._submit_external_reid(cid, frame, detections, match_boxes)
        self._remember_qwen_visuals(cid, frame, detections, match_boxes)

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        verifier = self.qwen_reid
        if verifier is not None and verifier.enabled and self.reid_mode != "off":
            try:
                with self.reid_lock:
                    verifier.service(self.global_reid, time.monotonic())
            except Exception as exc:
                verifier.last_error = f"service:{type(exc).__name__}: {exc}"
        return result

    def _print_stats(self) -> bool:
        keep = super()._print_stats()
        verifier = self.qwen_reid
        if verifier is not None:
            q = verifier.snapshot()
            print(
                "CAMERA_QWEN_REID "
                f"enabled={int(bool(q['enabled']))} visual_tracks={q['visual_tracks']} "
                f"requests={q['requests']} responses={q['responses']} pending={q['pending']} "
                f"same={q['same']} different={q['different']} uncertain={q['uncertain']} "
                f"merges={q['merges']} splits={q['splits']} cannot_links={q['cannot_links']} "
                f"latency_ms={float(q['latency_ms']):.0f} failed={q['failed']} dropped={q['dropped']} "
                f"last={q['last_verdict']} error={q['error'] or 'none'}",
                flush=True,
            )
        return keep


def main() -> int:
    # Qwen verifier is intentionally optional. If server is unavailable the fast
    # TAO/room-memory ReID continues running and the verifier reports errors only.
    os.environ.setdefault("CAMERA_V2_QWEN_VERIFY", "1")
    return CameraPersonTrackingQwen().run()


if __name__ == "__main__":
    raise SystemExit(main())
