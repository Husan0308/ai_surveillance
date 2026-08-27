#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication

from scripts.step4_visual_debug_viewer import (
    MainStreamSource as LegacyMainStreamSource,
    TrackLogTailer,
    Viewer as LegacyViewer,
    VisualTrack,
)
from services.ml_service.app.visual_box_policy import (
    bounded_center_prediction,
    visual_track_is_fresh,
)
from services.shared.camera_config import CameraConfig, load_settings


class QualityMainStreamSource(LegacyMainStreamSource):
    """Main-stream visual path optimized for image integrity, not detector freshness."""

    def __init__(self, camera: CameraConfig, width: int, height: int, latency_ms: int) -> None:
        super().__init__(camera, width, height, latency_ms)
        # The old viewer converted the main stream to 960x540 with bilinear filtering,
        # then Qt commonly enlarged it again. V5 uses 720p and GPU cubic scaling.
        self._set_if(self.convert, "interpolation-method", 2)
        # Visual main stream should prefer complete frames. The detector substream keeps
        # drop-on-latency enabled separately because detector freshness has different goals.
        self._set_if(self.source, "drop-on-latency", False)

    def _configure_rtsp_child(self, _bin, _sub_bin, element) -> None:
        super()._configure_rtsp_child(_bin, _sub_bin, element)
        factory = element.get_factory()
        if factory is None or factory.get_name() != "rtspsrc":
            return
        self._set_if(element, "latency", self.latency_ms)
        self._set_if(element, "drop-on-latency", False)


class FreshTrackLogTailer(TrackLogTailer):
    """Never leave a visually stale box frozen on screen if metadata stalls."""

    def snapshot(self) -> list[VisualTrack]:
        now = time.monotonic()
        return [
            row
            for row in super().snapshot()
            if visual_track_is_fresh(now - row.seen_at, max_age_sec=1.20)
        ]


class ViewerV5(LegacyViewer):
    """720p viewer with short, center-only and displacement-bounded interpolation."""

    def __init__(self, source: QualityMainStreamSource, tracks, camera_id: str) -> None:
        super().__init__(source, tracks, camera_id)
        self.resize(1280, 720)
        self.setWindowTitle(f"Step 4 V5 Visual Acceptance - {camera_id}")

    @staticmethod
    def _predict(row: VisualTrack) -> tuple[float, float, float, float]:
        return bounded_center_prediction(
            row.box,
            row.velocity,
            time.monotonic() - row.seen_at,
            max_predict_sec=0.20,
            max_dx_width_frac=0.20,
            max_dy_height_frac=0.12,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.image is None or self.image.isNull():
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Connecting high-quality main stream..."
            )
            return

        iw, ih = self.image.width(), self.image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox = 0.5 * (self.width() - dw)
        oy = 0.5 * (self.height() - dh)
        target = self.rect()
        target.setRect(int(ox), int(oy), int(dw), int(dh))
        painter.drawImage(target, self.image)

        tracks = [row for row in self.tracks.snapshot() if row.confirmed]
        font = QFont("Sans Serif", 10)
        font.setBold(True)
        painter.setFont(font)

        for row in tracks:
            x1, y1, x2, y2 = self._predict(row)
            left = ox + x1 * dw
            top = oy + y1 * dh
            right = ox + x2 * dw
            bottom = oy + y2 * dh
            predicted = row.predicted or row.state == "lost"
            color = QColor(255, 196, 0) if predicted else QColor(0, 235, 120)
            pen = QPen(color, 2.0)
            if predicted:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                int(left),
                int(top),
                int(max(1.0, right - left)),
                int(max(1.0, bottom - top)),
            )

            short_id = row.track_id.split("-")[-1]
            label = f"{short_id} {row.score:.2f}" + (" PRED" if predicted else "")
            metrics = painter.fontMetrics()
            tw = metrics.horizontalAdvance(label) + 10
            th = metrics.height() + 6
            ly = max(oy, top - th)
            painter.fillRect(int(left), int(ly), int(tw), int(th), QColor(0, 0, 0, 180))
            painter.setPen(color)
            painter.drawText(int(left + 5), int(ly + th - 5), label)

        header = (
            f"{self.camera_id}  MAIN 1280x720  {self.video_fps:.1f} FPS  "
            f"tracks={len(tracks)}  Step4 V5 bounded-box"
        )
        painter.fillRect(12, 12, 620, 30, QColor(0, 0, 0, 170))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(22, 33, header)

        error = self.source.last_error or self.tracks.last_error
        if error:
            painter.fillRect(12, 48, min(self.width() - 24, 900), 28, QColor(120, 0, 0, 190))
            painter.setPen(QColor(255, 230, 230))
            painter.drawText(22, 68, error[:150])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show a sharp main stream with Step 4 V5 bounded tracker overlay."
    )
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--track-log", default="/tmp/ML_STEP4_V5_VISUAL.log")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--latency-ms", type=int, default=120)
    args = parser.parse_args()

    settings = load_settings()
    by_id = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera not in by_id:
        raise SystemExit(f"unknown camera {args.camera}; available={','.join(by_id)}")

    log_path = Path(args.track_log)
    tracks = FreshTrackLogTailer(log_path, args.camera)
    source = QualityMainStreamSource(
        by_id[args.camera],
        max(640, args.width),
        max(360, args.height),
        max(80, args.latency_ms),
    )

    print(
        "STEP4_V5_VISUAL_DEBUG "
        f"camera={args.camera} source=main-stream resolution={args.width}x{args.height} "
        f"latency={args.latency_ms}ms drop_on_latency=0 scale=gpu-cubic "
        f"tracker_log={log_path} predict_max=0.20s size_prediction=0 stale_box_max=1.20s "
        "production_camera_service_modified=0",
        flush=True,
    )
    print("STEP4_V5_VISUAL_KEYS q/esc=quit f=fullscreen", flush=True)

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = ViewerV5(source, tracks, args.camera)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
