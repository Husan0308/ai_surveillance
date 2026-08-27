from __future__ import annotations

from .runtime import DETECT_CONTENT_H, DETECT_W
from .runtime_v93_motion_display import PascalMotionDisplayRuntime


class PascalXMapRuntime(PascalMotionDisplayRuntime):
    """V9.4: fix only detector -> tracker X-coordinate scaling.

    The detector branch captures a 672x378 image and pads it to 672x384 before
    TensorRT.  The tracker mux runs at 512x288.  V9.3 correctly removed the 3-pixel
    vertical letterbox and scaled Y from 378 -> tracker_height, but X was only
    clamped to tracker_width instead of being scaled from 672 -> tracker_width.

    That fed horizontally distorted detector corrections into NvDCF.  V9.4 changes
    only this coordinate transform.  V9.3 tracker cadence, detector cadence/budget,
    confidence thresholds, stale-result policy, bbox hold/smoothing, and display
    compensation remain unchanged for a clean one-variable A/B test.
    """

    def __init__(self) -> None:
        super().__init__()
        x_scale = self.track_width / float(DETECT_W)
        y_scale = self.track_height / float(DETECT_CONTENT_H)
        print(
            "CAMERA_V94_XMAP "
            f"detector_content={DETECT_W}x{DETECT_CONTENT_H} "
            f"tracker={self.track_width}x{self.track_height} "
            f"x_scale={x_scale:.6f} y_scale={y_scale:.6f} "
            "x_fix=scale-not-clamp y_policy=unchanged one_behavior_change=1",
            flush=True,
        )

    def _map_detector_rows(self, rows):
        mapped: list[tuple[tuple[float, float, float, float], float]] = []
        x_scale = self.track_width / float(DETECT_W)
        y_scale = self.track_height / float(DETECT_CONTENT_H)

        for coords, conf in rows:
            x1, y1, x2, y2 = [float(v) for v in coords]

            # Detector X is in the 672-wide TensorRT image.  Tracker metadata on
            # tracker_mux is in tracker_width coordinates, so X must be scaled just
            # like Y.  V9.3 omitted these two multiplications and merely clipped X,
            # which shifted/widened corrections toward the right side of the frame.
            x1 = x1 * x_scale
            x2 = x2 * x_scale

            # Existing V9.3 Y contract: remove the 3px top letterbox, then map the
            # 378px real detector content to the tracker-mux height.
            y1 = (y1 - 3.0) * y_scale
            y2 = (y2 - 3.0) * y_scale

            x1 = max(0.0, min(float(self.track_width - 1), x1))
            x2 = max(0.0, min(float(self.track_width - 1), x2))
            y1 = max(0.0, min(float(self.track_height - 1), y1))
            y2 = max(0.0, min(float(self.track_height - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            mapped.append(((x1, y1, x2, y2), float(conf)))

        # Keep the exact existing duplicate suppression policy.  This step is only
        # about putting detector geometry into the correct tracker coordinate space.
        mapped.sort(key=lambda item: item[1], reverse=True)
        kept: list[tuple[tuple[float, float, float, float], float]] = []
        for box, conf in mapped:
            duplicate = False
            area = max(1.0, self._area(box))
            for other, _ in kept:
                inter = self._intersection(box, other)
                union = area + self._area(other) - inter
                iou = inter / union if union > 0 else 0.0
                containment = inter / max(1.0, min(area, self._area(other)))
                if iou >= 0.82 or containment >= 0.94:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((box, conf))

        return [(x1, y1, x2, y2, conf) for (x1, y1, x2, y2), conf in kept]


def main() -> int:
    return PascalXMapRuntime().run()


if __name__ == "__main__":
    raise SystemExit(main())
