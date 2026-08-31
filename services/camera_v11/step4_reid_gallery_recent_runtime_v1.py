from __future__ import annotations

from .step4_reid_gallery_runtime_v1 import V11Step4ReIDGalleryRuntimeV1


class V11Step4ReIDGalleryRecentRuntimeV1(V11Step4ReIDGalleryRuntimeV1):
    """Keep bounded recent ReID evidence when a local track leaves active snapshots.

    Step2's gallery already has a 12-second expiry window, but the original runtime
    called ``touch_active`` on every tracker update, which deleted a gallery
    immediately when that local track was absent from the current snapshot set.
    Later same-room matching needs recently-seen tracklet evidence, especially when
    one CCTV view temporarily loses the person.  This additive runtime preserves the
    existing gallery until its configured expiry while keeping scheduler submission
    strictly limited to currently active track IDs.
    """

    def __init__(self) -> None:
        super().__init__()
        print(
            "CAMERA_V11_STEP4_REID_GALLERY_RECENT_V1 "
            "retention=bounded-expiry active_set_immediate_delete=0 "
            "expiry_sec=12.0 scheduler_active_only=1 thresholds_changed=0",
            flush=True,
        )

    def _quality_track_update(
        self, camera_id: str, track_ids: tuple[str, ...], captured_ns: int
    ) -> None:
        active = frozenset(str(track_id) for track_id in track_ids)
        self.active_track_ids[camera_id] = active

        # Do not erase a gallery just because the tracker temporarily stops
        # publishing that local track.  Keep the already-bounded recent evidence
        # and expire it by age instead.  This does not create new samples for an
        # inactive track and does not change any ReID score or identity threshold.
        self.reid_gallery.cleanup_expired(int(captured_ns))

        # Sampling cadence state is only useful for currently active local tracks.
        # Dropping this timestamp lets a genuinely recovered/re-created track sample
        # immediately without retaining any frame/crop queue.
        with self.reid_submit_lock:
            for key in [
                key
                for key in self.reid_submit_at
                if key[0] == camera_id and key[1] not in active
            ]:
                del self.reid_submit_at[key]
