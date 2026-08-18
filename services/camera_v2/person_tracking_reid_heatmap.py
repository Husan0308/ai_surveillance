from __future__ import annotations

import os
from pathlib import Path
import time

# Keep the proven 736x416 detector canvas so the GTX-class GPU remains smooth,
# but admit lower-confidence person observations. This specifically helps hard
# poses (reclining, foreshortening, partial occlusion) without increasing pixel
# cost. NvDCF still owns temporal stability after a detection is accepted.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "736")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "416")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.04")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from . import person_tracking_reid as _reid_module

# The native display smoother bridges up to ~1.8 s of sparse-detector misses.
# Keep Global-ID styling/counters alive for the same bounded visual interval.
_reid_module.DISPLAY_HOLD_BINDING_SEC = 1.90
CameraPersonTrackingReID = _reid_module.CameraPersonTrackingReID
DISPLAY_HOLD_BINDING_SEC = _reid_module.DISPLAY_HOLD_BINDING_SEC


class CameraPersonTrackingReIDHeatmap(
    CameraPersonTrackingReID,
    CameraPersonTrackingHeatmap,
):
    """ReID runtime with native heatmap and live unique occupancy counters."""

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        """Tune NvDCF for difficult poses without exposing unreliable shadow data.

        Shadow tracking keeps an internal target alive between detector matches,
        but NVIDIA does not report shadow targets as normal downstream objects.
        Visual gap bridging therefore stays in native_display_smoother.c; here we
        only keep NvDCF's internal target alive long enough to reacquire it.
        """
        stabilized = CameraPersonTrackingReID._stabilize_tracker_config(path)
        replacements = {
            "minDetectorConfidence": "0.04",
            "maxShadowTrackingAge": "50",
            "earlyTerminationAge": "3",
            "minTrackingConfidenceDuringInactive": "0.15",
            "outputShadowTracks": "0",
        }
        lines = stabilized.read_text(encoding="utf-8").splitlines()
        found: set[str] = set()
        output: list[str] = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            replaced = False
            for key, value in replacements.items():
                if stripped.startswith(key + ":"):
                    comment = ""
                    if "#" in stripped:
                        comment = "  #" + stripped.split("#", 1)[1]
                    output.append(f"{indent}{key}: {value}{comment}")
                    found.add(key)
                    replaced = True
                    break
            if not replaced:
                output.append(line)

        missing = sorted(set(replacements) - found)
        if missing:
            raise RuntimeError(
                "Generated NvDCF config missing persistence keys: "
                + ", ".join(missing)
            )
        stabilized.write_text("\n".join(output) + "\n", encoding="utf-8")
        return stabilized

    def live_people_counts(self) -> dict[str, int]:
        """Return live unique Total/Known/Unknown counts used by the Qt cards.

        A positive Global ID is shown in green by the live overlay and counts as
        Known. A currently visible track without a Global ID is shown as Unknown.
        Multiple camera-local tracks bound to the same Global ID count once.
        """
        now = time.monotonic()
        bindings = self.identity.bindings() if self.identity is not None else {}
        known_globals: set[int] = set()
        unknown_tracks: set[tuple[str, int]] = set()
        max_age = max(1.90, float(DISPLAY_HOLD_BINDING_SEC))

        with self._reid_lock:
            last_seen = dict(self._last_real_track_seen)

        for key, seen_at in last_seen.items():
            if now - float(seen_at) > max_age:
                continue
            binding = bindings.get(key)
            global_id = int((binding or {}).get("global_id") or 0)
            if global_id > 0:
                known_globals.add(global_id)
            else:
                unknown_tracks.add((str(key[0]), int(key[1])))

        known = len(known_globals)
        unknown = len(unknown_tracks)
        return {
            "total": known + unknown,
            "known": known,
            "unknown": unknown,
        }

    def active_people_count(self) -> int:
        return int(self.live_people_counts()["total"])

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        try:
            counts = self.live_people_counts()
            with self.det_lock:
                self.tracked_now = int(counts["total"])
        except Exception:
            pass
        return result


def main() -> int:
    return CameraPersonTrackingReIDHeatmap().run()


if __name__ == "__main__":
    raise SystemExit(main())
