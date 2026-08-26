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
        # Detector-side de-dup before NvDCF sees the metadata. This is deliberately
        # stricter than the base runtime because the exported TensorRT output can
        # contain highly-overlapping person rows at the low operating threshold.
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

        # DeepStream 7.1 supported TargetManagement knobs only. NVIDIA documents
        # minIouDiff4NewTarget as the duplicate-new-target guard: lowering it makes
        # an overlapping fresh detector box less likely to spawn a second track.
        # Do NOT add newer/foreign keys such as minIou4TargetDuplicate or
        # targetDuplicateRunInterval; DS 7.1 reports them as unknown parameters.
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.30")
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.12")
        self._replace_yaml_key(lines, "probationAge", "0")
        self._replace_yaml_key(lines, "earlyTerminationAge", "2")
        shadow_frames = max(20, int(round(self.track_fps * 5.0)))
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        self._insert_target_management_key(lines, "outputShadowTracks", "1")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_QUALITY_NVDCF "
            f"new_target_iou=0.30 min_tracker_conf=0.12 "
            f"shadow_frames={shadow_frames} early_termination=2 probation=0 "
            "ds71_supported_only=1",
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
