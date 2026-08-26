from __future__ import annotations

import os
import sys
import time

from .runtime import DETECT_CONTENT_H, DETECT_W, CleanCameraRuntime


class QualityCameraRuntime(CleanCameraRuntime):
    """Pascal quality/runtime tuning without putting analytics on the display path.

    The tracker intentionally runs at a smaller spatial resolution than the YOLO
    detector. Detector coordinates are rescaled explicitly before metadata is
    injected into NvDCF. Track de-duplication is performed before the cache is
    published so the display thread can never observe the pre-dedup batch.
    """

    @staticmethod
    def _iou(a, b) -> float:
        inter = CleanCameraRuntime._intersection(a, b)
        if inter <= 0.0:
            return 0.0
        union = CleanCameraRuntime._area(a) + CleanCameraRuntime._area(b) - inter
        return inter / union if union > 0.0 else 0.0

    def _map_detector_rows(self, rows):
        # Detector output is always 672x384 with a 3/378/3 letterbox. NvDCF may
        # use a smaller surface, so scale both axes instead of merely clamping X.
        x_scale = self.track_width / float(DETECT_W)
        y_scale = self.track_height / float(DETECT_CONTENT_H)
        mapped = []
        for coords, conf in rows:
            x1, y1, x2, y2 = [float(v) for v in coords]
            x1 *= x_scale
            x2 *= x_scale
            y1 = (y1 - 3.0) * y_scale
            y2 = (y2 - 3.0) * y_scale
            x1 = max(0.0, min(float(self.track_width - 1), x1))
            x2 = max(0.0, min(float(self.track_width - 1), x2))
            y1 = max(0.0, min(float(self.track_height - 1), y1))
            y2 = max(0.0, min(float(self.track_height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mapped.append((x1, y1, x2, y2, float(conf)))

        # Detector-side overlap suppression before NvDCF sees candidates.
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
        # GTX 1050 Ti budget: 512x288@10 Hz is ~29% less tracker input pixel-work
        # than 672x384@8 Hz, while the temporal update interval improves from
        # 125 ms to 100 ms. Detector resolution remains 672x384 independently.
        self.track_width = max(
            320,
            min(DETECT_W, int(os.environ.get("CAMERA_V2_TRACK_WIDTH", "512"))),
        )
        self.track_height = max(
            192,
            min(384, int(os.environ.get("CAMERA_V2_TRACK_HEIGHT", "288"))),
        )

        lib, generated = super()._prepare_tracker_files()
        lines = generated.read_text(encoding="utf-8").splitlines()

        # DeepStream 7.1 supported knobs only. NVIDIA documents a lower
        # minIouDiff4NewTarget as the mechanism for suppressing duplicate targets.
        self._replace_yaml_key(lines, "minDetectorConfidence", "0.18")
        self._replace_yaml_key(lines, "minIouDiff4NewTarget", "0.22")
        self._replace_yaml_key(lines, "minTrackerConfidence", "0.12")
        self._replace_yaml_key(lines, "probationAge", "1")
        self._replace_yaml_key(lines, "earlyTerminationAge", "1")
        shadow_frames = max(20, int(round(self.track_fps * 5.0)))
        self._replace_yaml_key(lines, "maxShadowTrackingAge", str(shadow_frames))
        # Shadow targets remain available internally for re-acquisition, but we do
        # not export shadow history/meta downstream. The visible wall uses only
        # active object metadata copied into the cache.
        self._insert_target_management_key(lines, "outputShadowTracks", "0")
        generated.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(
            "CAMERA_QUALITY_NVDCF "
            f"tracker={self.track_width}x{self.track_height}@{self.track_fps:.1f}Hz "
            "min_detector_conf=0.18 new_target_iou=0.22 min_tracker_conf=0.12 "
            f"shadow_frames={shadow_frames} early_termination=1 probation=1 "
            "output_shadow=0 ds71_supported_only=1",
            flush=True,
        )
        return lib, generated

    def _dedup_tracks(self, tracks):
        # Keep the strongest track if NvDCF briefly has two IDs around the same
        # person. Combining IoU and containment catches both nearly-equal boxes and
        # the common nested old/new-box case while preserving side-by-side people.
        ordered = sorted(tracks, key=lambda row: float(row[5]), reverse=True)
        kept = []
        suppressed = 0
        for row in ordered:
            box = row[1:5]
            area = max(1.0, self._area(box))
            duplicate = False
            for other in kept:
                other_box = other[1:5]
                other_area = max(1.0, self._area(other_box))
                inter = self._intersection(box, other_box)
                containment = inter / max(1.0, min(area, other_area))
                if self._iou(box, other_box) >= 0.58 or containment >= 0.88:
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                kept.append(row)
        return kept, suppressed

    def _tracker_probe(self, _pad, info):
        """Copy -> scale -> dedup -> publish as one atomic cache transaction."""
        if not self.analytics_enabled:
            return self.Gst.PadProbeReturn.OK
        buffer = info.get_buffer()
        if buffer is None:
            return self.Gst.PadProbeReturn.OK
        try:
            rows = self.bridge.copy_tracks(buffer, max_rows=256)
            now = time.monotonic()
            grouped = {}
            sx = self.display_width / float(self.track_width)
            sy = self.display_height / float(self.track_height)
            for row in rows:
                source_id = int(row["source_id"])
                left = float(row["left"]) * sx
                top = float(row["top"]) * sy
                right = (float(row["left"]) + float(row["width"])) * sx
                bottom = (float(row["top"]) + float(row["height"])) * sy
                conf = float(row["tracker_confidence"])
                if conf < 0.0:
                    conf = float(row["confidence"])
                grouped.setdefault(source_id, []).append(
                    (
                        int(row["object_id"]),
                        left,
                        top,
                        right,
                        bottom,
                        conf,
                    )
                )

            published = {}
            suppressed = 0
            for source_id, tracks in grouped.items():
                kept, count = self._dedup_tracks(tracks)
                published[source_id] = kept
                suppressed += count

            with self.track_cache_lock:
                for source_id, tracks in published.items():
                    self.track_cache[source_id] = (now, tracks)
                self.tracked_now = sum(len(tracks) for tracks in published.values())
                self.tracker_batches += 1

            if suppressed:
                print(
                    f"CAMERA_QUALITY_DEDUP suppressed={suppressed} publish=atomic",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"CAMERA_CLEAN_TRACK warning={type(exc).__name__}:{exc}",
                file=sys.stderr,
                flush=True,
            )
        return self.Gst.PadProbeReturn.OK


def main() -> int:
    return QualityCameraRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
