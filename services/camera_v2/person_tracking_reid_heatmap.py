from __future__ import annotations

import time

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .person_tracking_reid import CameraPersonTrackingReID, DISPLAY_HOLD_BINDING_SEC


class CameraPersonTrackingReIDHeatmap(
    CameraPersonTrackingReID,
    CameraPersonTrackingHeatmap,
):
    """ReID runtime with native heatmap and unique live people accounting.

    Python MRO is intentional:
    ReID tracker probe -> Heatmap tracker probe -> stable NvDCF tracker probe.
    This preserves the known live camera/tracking path while adding heatmap
    accumulation and per-camera render visibility as a side concern.
    """

    def active_people_count(self) -> int:
        """Count currently visible people once per Global ID.

        Tracks without a Global ID are temporarily counted by local camera/track
        key. Once ReID binds multiple camera-local tracks to the same Global ID,
        they collapse to one live person.
        """
        now = time.monotonic()
        bindings = self.identity.bindings() if self.identity is not None else {}
        active: set[tuple] = set()
        max_age = max(0.75, float(DISPLAY_HOLD_BINDING_SEC))

        with self._reid_lock:
            last_seen = dict(self._last_real_track_seen)

        for key, seen_at in last_seen.items():
            if now - float(seen_at) > max_age:
                continue
            binding = bindings.get(key)
            if binding and int(binding.get("global_id") or 0) > 0:
                active.add(("global", int(binding["global_id"])))
            else:
                active.add(("local", str(key[0]), int(key[1])))
        return len(active)

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        # The parent stable tracker temporarily writes the raw local-track count.
        # Replace it with the live Global-ID-aware count before UI metrics read it.
        try:
            unique_now = self.active_people_count()
            with self.det_lock:
                self.tracked_now = unique_now
        except Exception:
            pass
        return result


def main() -> int:
    return CameraPersonTrackingReIDHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
