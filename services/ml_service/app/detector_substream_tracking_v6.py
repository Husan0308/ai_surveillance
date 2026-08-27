from __future__ import annotations

import os
import time

from services.ml_service.app.detector_substream_tracking_v5 import (
    DetectorSubstreamTrackingV5Service,
)
from services.ml_service.app.local_tracker_sparse_v6 import MultiCameraBodyEnvelopeTracker
from services.ml_service.app.trt86_detector import CONTENT_H, INPUT_W


class DetectorSubstreamTrackingV6Service(DetectorSubstreamTrackingV5Service):
    """Step 4 V6: sticky body envelope + strict lost-track geometry."""

    def __init__(self) -> None:
        os.environ.setdefault("ML_TRACK_LOW_RECOVERY_SEC", "1.8")
        super().__init__()

        self.track_lost_low_jump_diag = float(
            os.environ.get("ML_TRACK_LOST_LOW_JUMP_DIAG", "1.05")
        )
        self.track_lost_high_jump_diag = float(
            os.environ.get("ML_TRACK_LOST_HIGH_JUMP_DIAG", "1.35")
        )
        self.track_render_expand_alpha = float(
            os.environ.get("ML_TRACK_RENDER_EXPAND_ALPHA", "0.82")
        )
        self.track_render_contract_alpha = float(
            os.environ.get("ML_TRACK_RENDER_CONTRACT_ALPHA", "0.14")
        )
        self.track_render_expand_max_step = float(
            os.environ.get("ML_TRACK_RENDER_EXPAND_MAX_STEP", "0.72")
        )
        self.track_render_contract_max_step = float(
            os.environ.get("ML_TRACK_RENDER_CONTRACT_MAX_STEP", "0.22")
        )
        self.track_envelope_pad_x = float(
            os.environ.get("ML_TRACK_ENVELOPE_PAD_X", "0.07")
        )
        self.track_envelope_pad_top = float(
            os.environ.get("ML_TRACK_ENVELOPE_PAD_TOP", "0.04")
        )
        self.track_envelope_pad_bottom = float(
            os.environ.get("ML_TRACK_ENVELOPE_PAD_BOTTOM", "0.03")
        )
        self.track_envelope_compact_extra_x = float(
            os.environ.get("ML_TRACK_ENVELOPE_COMPACT_EXTRA_X", "0.06")
        )

        appearance_weight = min(
            0.30, max(0.0, float(os.environ.get("ML_TRACK_APPEARANCE_WEIGHT", "0.22")))
        )
        confirm_hits = max(1, int(os.environ.get("ML_TRACK_CONFIRM_HITS", "2")))
        tentative_ttl_sec = max(
            0.3, float(os.environ.get("ML_TRACK_TENTATIVE_TTL_SEC", "0.9"))
        )

        self.trackers = MultiCameraBodyEnvelopeTracker(
            (camera.camera_id for camera in self.cameras),
            INPUT_W,
            CONTENT_H,
            low_thresh=self.track_low_thresh,
            high_thresh=self.track_high_thresh,
            new_track_thresh=self.track_new_thresh,
            confirm_hits=confirm_hits,
            tentative_ttl_sec=tentative_ttl_sec,
            shadow_sec=self.track_shadow_sec,
            max_lost_sec=self.track_max_lost_sec,
            appearance_weight=appearance_weight,
            reacquire_thresh=self.track_reacquire_thresh,
            low_recovery_thresh=self.track_low_recovery_thresh,
            low_recovery_sec=self.track_low_recovery_sec,
            duplicate_iou=self.track_duplicate_iou,
            low_appearance_weight=self.track_low_appearance_weight,
            low_appearance_floor=self.track_low_appearance_floor,
            live_duplicate_iou=self.track_live_duplicate_iou,
            lost_velocity_half_life_sec=self.track_lost_velocity_half_life,
            nested_duplicate_ios=self.track_nested_duplicate_ios,
            nested_duplicate_app_floor=self.track_nested_duplicate_app_floor,
            nested_duplicate_center_frac=self.track_nested_duplicate_center_frac,
            render_anchor_alpha=self.track_render_anchor_alpha,
            render_size_alpha=self.track_render_size_alpha,
            render_recovery_size_alpha=self.track_render_recovery_size_alpha,
            render_max_size_step=self.track_render_max_size_step,
            render_velocity_gain=self.track_render_velocity_gain,
            lost_low_jump_diag=self.track_lost_low_jump_diag,
            lost_high_jump_diag=self.track_lost_high_jump_diag,
            render_expand_alpha=self.track_render_expand_alpha,
            render_contract_alpha=self.track_render_contract_alpha,
            render_expand_max_step=self.track_render_expand_max_step,
            render_contract_max_step=self.track_render_contract_max_step,
            envelope_pad_x=self.track_envelope_pad_x,
            envelope_pad_top=self.track_envelope_pad_top,
            envelope_pad_bottom=self.track_envelope_pad_bottom,
            envelope_compact_extra_x=self.track_envelope_compact_extra_x,
        )

    def _record_tracks(self, update, cid: str, seq: int, total_n: int) -> None:
        object_logs = bool(self.track_log_objects)
        self.track_log_objects = False
        try:
            super()._record_tracks(update, cid, seq, total_n)
        finally:
            self.track_log_objects = object_logs

        if not object_logs:
            return

        captured_ns = int(getattr(update, "captured_ns", 0) or 0)
        lag_ms = 0.0
        if captured_ns > 0:
            lag_ms = max(0.0, (time.monotonic_ns() - captured_ns) / 1_000_000.0)

        for snap in update.snapshots:
            b = snap.bbox_norm
            v = snap.velocity_norm_s
            print(
                "ML_TRACK_OBJECT_V6 "
                f"camera={cid} id={snap.track_id} state={snap.state} "
                f"confirmed={int(snap.confirmed)} predicted={int(snap.predicted)} "
                f"score={snap.score:.3f} hits={snap.hits} "
                f"box_norm={b[0]:.4f},{b[1]:.4f},{b[2]:.4f},{b[3]:.4f} "
                f"vel_norm_s={v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{v[3]:.4f} "
                f"since_det={snap.since_detection_sec:.3f}s "
                f"metadata_lag_ms={lag_ms:.1f}",
                flush=True,
            )

    def run(self) -> int:
        print(
            "ML_STEP4_V6_PROFILE "
            "algorithm=v5-low-score+v6-body-envelope "
            f"low_recovery={self.track_low_recovery_sec:.1f}s "
            f"lost_jump_low={self.track_lost_low_jump_diag:.2f}diag "
            f"lost_jump_high={self.track_lost_high_jump_diag:.2f}diag "
            f"expand_alpha={self.track_render_expand_alpha:.2f} "
            f"contract_alpha={self.track_render_contract_alpha:.2f} "
            f"pad_x={self.track_envelope_pad_x:.2f}+compact "
            f"pad_top={self.track_envelope_pad_top:.2f} "
            f"pad_bottom={self.track_envelope_pad_bottom:.2f} "
            "association_box=raw render_box=body-envelope latency_stamp=1 "
            "pose=0 gpu_tracker=0 nvdcf=0",
            flush=True,
        )
        return super().run()


def main() -> int:
    service = DetectorSubstreamTrackingV6Service()
    try:
        return service.run()
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
