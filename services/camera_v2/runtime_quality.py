from __future__ import annotations

import time

from .runtime import CleanCameraRuntime


class QualityCameraRuntime(CleanCameraRuntime):
    """Tighten sparse-detector/NvDCF duplicate handling without touching display."""

    @staticmethod
    def _iou(a, b) -> float:
        inter = CleanCameraRuntime._intersection(a, b)
        if inter <= 0.0:
            return 0.0
        union = CleanCameraRuntime._area(a) + CleanCameraRuntime._area(b) - inter
        return inter / union if union > 0.0 else 0.0

    def _map_detector_rows(self, rows):
        mapped = super()._map_detector_rows(rows)
        ordered = sorted(mapped, key=lambda row: float(row[4]), reverse=True)
        kept = []
        for row in ordered:
            box = row[:4]
            area = max(1.0, self._area(box))
            duplicate = False
            for other in kept:
                other_box = other[:4]
                inter = self._intersection(box, other_box)
                containment = inter / max(1.0, min(area, self._area(other_box)))
                if self._iou(box, other_box) >= 0.68 or containment >= 0.90:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(row)
        return kept

    def _prepare_tracker_files(self):
        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.22")
        self._replace_yaml_key(lines, "minIou4TargetDuplicate", "0.72", required=False)
        self._replace_yaml_key(lines, "targetDuplicateRunInterval", "1", required=False)
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.12")
        self._replace_yaml_key(lines, "probationAge", "0")
        shadow_frames = max(20, int(round(self.track_fps * 5.0)))
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._insert_target_management_key(lines, "outputShadowTracks", "1")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_QUALITY_NVDCF "
            f"new_target_iou=0.22 duplicate_iou=0.72 duplicate_interval=1 "
            f"min_tracker_conf=0.12 shadow_frames={shadow_frames} probation=0",
            flush=True,
        )
        return lib, generated

    def _tracker_probe(self, pad, info):
        result = super()._tracker_probe(pad, info)
        now = time.monotonic()
        suppressed = 0
        with self.track_cache_lock:
            for source_id, (updated, tracks) in list(self.track_cache.items()):
                if now - updated > 0.5 or len(tracks) < 2:
                    continue
                ordered = sorted(tracks, key=lambda row: float(row[5]), reverse=True)
                kept = []
                for row in ordered:
                    box = row[1:5]
                    area = max(1.0, self._area(box))
                    duplicate = False
                    for other in kept:
                        other_box = other[1:5]
                        inter = self._intersection(box, other_box)
                        containment = inter / max(1.0, min(area, self._area(other_box)))
                        if self._iou(box, other_box) >= 0.80 or containment >= 0.94:
                            duplicate = True
                            break
                    if duplicate:
                        suppressed += 1
                    else:
                        kept.append(row)
                self.track_cache[source_id] = (updated, kept)
            if suppressed:
                self.tracked_now = sum(len(row[1]) for row in self.track_cache.values())
        if suppressed:
            print(f"CAMERA_QUALITY_DEDUP suppressed={suppressed}", flush=True)
        return result


def main() -> int:
    return QualityCameraRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
