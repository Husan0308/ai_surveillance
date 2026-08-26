from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .pose_gate_v2 import PoseGateClient as _BasePoseGateClient
from .pose_gate_v2 import PoseGateMetrics, _same_region


@dataclass
class _RejectState:
    box: tuple[float, float, float, float]
    hits: int
    expires_at: float


class PoseGateClient(_BasePoseGateClient):
    """Recall-first wrapper around the cached S-pose gate.

    A single failed pose crop is not enough evidence to delete a plausible YOLO
    person candidate. CCTV people are often seated, clipped by furniture, small,
    or partially outside the crop, which can legitimately hide keypoints. Strong
    YOLO boxes and live-track reuse still bypass pose in the base gate. For a new
    ambiguous candidate, this wrapper requires repeated negative pose evidence in
    the same spatial region before the candidate is actually rejected.
    """

    def __init__(self) -> None:
        super().__init__()
        # Disable the base one-shot negative cache. Positive cache stays active.
        # We replace negative caching with an explicit multi-hit hysteresis below.
        self.negative_ttl = 0.0
        self.soft_keep_conf = max(
            self.min_conf,
            float(os.environ.get("CAMERA_V2_POSE_GATE_SOFT_KEEP_CONF", "0.14")),
        )
        self.reject_hits_required = max(
            2,
            min(4, int(os.environ.get("CAMERA_V2_POSE_GATE_REJECT_HITS", "2"))),
        )
        self.reject_window = max(
            4.0,
            float(os.environ.get("CAMERA_V2_POSE_GATE_REJECT_WINDOW_SEC", "10.0")),
        )
        self._rejects: dict[str, list[_RejectState]] = {}
        print(
            "CAMERA_POSE_GATE_HYSTERESIS "
            f"soft_keep_conf={self.soft_keep_conf:.2f} "
            f"reject_hits={self.reject_hits_required} "
            f"window={self.reject_window:.1f}s one_shot_negative_cache=0",
            flush=True,
        )

    @staticmethod
    def _row_key(row) -> tuple[float, ...]:
        coords, score = row
        return tuple(round(float(v), 3) for v in coords) + (round(float(score), 5),)

    def _purge_rejects(self, cid: str, now: float) -> list[_RejectState]:
        live = [row for row in self._rejects.get(cid, []) if row.expires_at > now]
        self._rejects[cid] = live[-24:]
        return live

    @staticmethod
    def _same_reject_region(a, b) -> bool:
        return _same_region(
            a,
            b,
            iou_min=0.52,
            containment_min=0.80,
            center_max=0.22,
        )

    def _clear_reject_for_box(self, cid: str, box, now: float) -> None:
        live = self._purge_rejects(cid, now)
        self._rejects[cid] = [
            row for row in live if not self._same_reject_region(box, row.box)
        ]

    def _register_reject(self, cid: str, box, now: float) -> int:
        live = self._purge_rejects(cid, now)
        for row in reversed(live):
            if self._same_reject_region(box, row.box):
                row.hits += 1
                row.box = tuple(float(v) for v in box)
                row.expires_at = now + self.reject_window
                return row.hits
        live.append(
            _RejectState(
                tuple(float(v) for v in box),
                1,
                now + self.reject_window,
            )
        )
        self._rejects[cid] = live[-24:]
        return 1

    def filter(self, cid: str, frame, rows, trusted_boxes=None) -> tuple[list, PoseGateMetrics]:
        now = time.monotonic()
        accepted, metrics = super().filter(
            cid,
            frame,
            rows,
            trusted_boxes=trusted_boxes,
        )

        accepted_keys = {self._row_key(row) for row in accepted}
        soft_holds = 0
        confirmed_rejects = 0
        output = list(accepted)

        # A successful detector/pose/track decision immediately clears any older
        # negative streak for the same physical region.
        for coords, _score in accepted:
            self._clear_reject_for_box(cid, coords, now)

        for row in rows:
            coords, score_value = row
            score = float(score_value)
            if self._row_key(row) in accepted_keys:
                continue
            # Very weak candidates remain fail-closed. Hysteresis is only for
            # plausible persons where a pose miss may be caused by occlusion.
            if score < self.soft_keep_conf:
                continue

            hits = self._register_reject(cid, coords, now)
            if hits < self.reject_hits_required:
                output.append((coords, score))
                soft_holds += 1
            else:
                confirmed_rejects += 1

        # The caller performs a second geometry de-dup before NvDCF, so adding a
        # provisional row here cannot create a persistent duplicate track.
        metrics.final = len(output)
        setattr(metrics, "soft_hold", soft_holds)
        setattr(metrics, "confirmed_reject", confirmed_rejects)

        if soft_holds or confirmed_rejects:
            print(
                "CAMERA_POSE_HYSTERESIS "
                f"cid={cid} soft_hold={soft_holds} "
                f"confirmed_reject={confirmed_rejects} "
                f"required={self.reject_hits_required}",
                flush=True,
            )
        return output, metrics
