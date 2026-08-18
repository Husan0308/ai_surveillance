from __future__ import annotations

import os
import time
from pathlib import Path

# Slightly more detector detail helps reclined/foreshortened people without
# changing the camera decode/display path. These values are applied before the
# detector modules are imported, so their own setdefault() calls keep them.
os.environ.setdefault("CAMERA_V2_DETECT_WIDTH", "736")
os.environ.setdefault("CAMERA_V2_DETECT_HEIGHT", "416")
os.environ.setdefault("CAMERA_V2_DETECT_CONF", "0.06")
os.environ.setdefault("CAMERA_V2_MAX_DET", "40")

from .person_tracking_heatmap import CameraPersonTrackingHeatmap
from .person_tracking_reid import CameraPersonTrackingReID, DISPLAY_HOLD_BINDING_SEC


class CameraPersonTrackingReIDHeatmap(
    CameraPersonTrackingReID,
    CameraPersonTrackingHeatmap,
):
    """ReID runtime with native heatmap, stable focus geometry and track hold."""

    def __init__(self) -> None:
        self._focus_geometry_state: bool | None = None
        super().__init__()
        # show-source can be changed by the Qt controller while PLAYING. Keep the
        # tiler output geometry synchronized with the mode: 2x3 grid uses the
        # historical 1280x1080 wall (640x360 tiles), single-camera focus uses a
        # true 16:9 output so the camera is never stretched vertically.
        self.GLib.timeout_add(100, self._sync_focus_geometry)

    @staticmethod
    def _stabilize_tracker_config(path: Path) -> Path:
        """Keep a visible NvDCF box through short detector misses.

        YOLO already sees the reclining person intermittently; the distracting
        failure is the box disappearing between those detections. Let NvDCF emit
        a bounded shadow prediction for up to ~3.5 s at 20 FPS, with a conservative
        inactive confidence floor. A fresh detector observation immediately takes
        ownership again.
        """
        stabilized = CameraPersonTrackingReID._stabilize_tracker_config(path)
        replacements = {
            "maxShadowTrackingAge": "70",
            "earlyTerminationAge": "3",
            "minTrackingConfidenceDuringInactive": "0.15",
            "outputShadowTracks": "1",
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

    def _sync_focus_geometry(self) -> bool:
        try:
            prop = self.tiler.find_property("show-source")
            if prop is None:
                return True
            focused = int(self.tiler.get_property("show-source")) >= 0
            if focused == self._focus_geometry_state:
                return True

            if focused:
                width = max(1280, int(os.environ.get("CAMERA_V2_FOCUS_WIDTH", "1920")))
                height = max(720, int(os.environ.get("CAMERA_V2_FOCUS_HEIGHT", "1080")))
            else:
                width = max(640, int(os.environ.get("CAMERA_V2_WALL_WIDTH", "1280")))
                height = max(540, int(os.environ.get("CAMERA_V2_WALL_HEIGHT", "1080")))

            self.tiler.set_property("width", width)
            self.tiler.set_property("height", height)
            self.wall_width = width
            self.wall_height = height
            self._focus_geometry_state = focused
            print(
                f"CAMERA_FOCUS_GEOMETRY focused={int(focused)} output={width}x{height}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"CAMERA_FOCUS_GEOMETRY warning={type(exc).__name__}:{exc}",
                flush=True,
            )
        return True

    def active_people_count(self) -> int:
        """Count currently visible people once per Global ID."""
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
