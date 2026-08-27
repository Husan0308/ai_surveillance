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
    FRAME_PREFIX,
    TrackLogTailer,
    VisualTrack,
    _kv,
)
from scripts.step4_visual_debug_viewer_v5 import QualityMainStreamSource, ViewerV5
from services.ml_service.app.visual_box_policy import bounded_center_prediction
from services.shared.camera_config import load_settings

OBJECT_V6_PREFIX = "ML_TRACK_OBJECT_V6 "


class LagAwareTrackLogTailer(TrackLogTailer):
    """Read V6 body-envelope rows and retain capture->metadata latency."""

    def _parse_object_v6(self, line: str) -> None:
        fields = _kv(line)
        if fields.get("camera") != self.camera_id:
            return
        tid = fields.get("id", "")
        if not tid:
            return
        try:
            box = tuple(float(v) for v in fields["box_norm"].split(","))
            vel = tuple(float(v) for v in fields["vel_norm_s"].split(","))
            if len(box) != 4 or len(vel) != 4:
                return
            lag_sec = max(
                0.0,
                min(0.40, float(fields.get("metadata_lag_ms", "0")) / 1000.0),
            )
            row = VisualTrack(
                track_id=tid,
                state=fields.get("state", "tracked"),
                confirmed=fields.get("confirmed", "0") == "1",
                predicted=fields.get("predicted", "0") == "1",
                score=float(fields.get("score", "0")),
                box=(box[0], box[1], box[2], box[3]),
                velocity=(vel[0], vel[1], vel[2], vel[3]),
                seen_at=time.monotonic(),
            )
            setattr(row, "metadata_lag_sec", lag_sec)
        except (KeyError, ValueError):
            return
        with self._lock:
            self._tracks[tid] = row

    def _run(self) -> None:
        while not self._stop.is_set() and not self.path.exists():
            self.last_error = f"waiting for {self.path}"
            self._stop.wait(0.20)
        if self._stop.is_set():
            return

        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(0, 2)
                self.last_error = ""
                while not self._stop.is_set():
                    line = fh.readline()
                    if not line:
                        self._stop.wait(0.01)
                        continue
                    if line.startswith(FRAME_PREFIX):
                        self._parse_frame(line[len(FRAME_PREFIX) :])
                    elif line.startswith(OBJECT_V6_PREFIX):
                        self._parse_object_v6(line[len(OBJECT_V6_PREFIX) :])
        except Exception as exc:
            self.last_error = str(exc)


class FreshLagAwareTrackLogTailer(LagAwareTrackLogTailer):
    def snapshot(self) -> list[VisualTrack]:
        now = time.monotonic()
        return [
            row
            for row in super().snapshot()
            if 0.0 <= now - row.seen_at <= 1.10
        ]


class ViewerV6(ViewerV5):
    """Sharp 720p main stream + latency-compensated V6 body envelope."""

    def __init__(self, source: QualityMainStreamSource, tracks, camera_id: str) -> None:
        super().__init__(source, tracks, camera_id)
        self.resize(1280, 720)
        self.setWindowTitle(f"Step 4 V6 Body Envelope - {camera_id}")

    @staticmethod
    def _predict(row: VisualTrack) -> tuple[float, float, float, float]:
        metadata_lag = float(getattr(row, "metadata_lag_sec", 0.0))
        visual_age = max(0.0, time.monotonic() - row.seen_at)
        return bounded_center_prediction(
            row.box,
            row.velocity,
            visual_age + metadata_lag,
            max_predict_sec=0.34,
            max_dx_width_frac=0.45,
            max_dy_height_frac=0.30,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.image is None or self.image.isNull():
            painter.setPen(QColor(230, 230, 230))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Connecting high-quality main stream...",
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

        lag_values = []
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
            lag_values.append(float(getattr(row, "metadata_lag_sec", 0.0)) * 1000.0)

        lag_ms = sum(lag_values) / len(lag_values) if lag_values else 0.0
        header = (
            f"{self.camera_id}  MAIN 1280x720  {self.video_fps:.1f} FPS  "
            f"tracks={len(tracks)}  V6 BODY-ENVELOPE  lag={lag_ms:.0f}ms"
        )
        painter.fillRect(12, 12, 720, 30, QColor(0, 0, 0, 170))
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(22, 33, header)

        error = self.source.last_error or self.tracks.last_error
        if error:
            painter.fillRect(12, 48, min(self.width() - 24, 900), 28, QColor(120, 0, 0, 190))
            painter.setPen(QColor(255, 230, 230))
            painter.drawText(22, 68, error[:150])


def build_source_and_tracks(args):
    settings = load_settings()
    by_id = {camera.camera_id: camera for camera in settings.cameras}
    if args.camera not in by_id:
        raise SystemExit(
            f"unknown camera {args.camera}; available={','.join(by_id)}"
        )

    tracks = FreshLagAwareTrackLogTailer(Path(args.track_log), args.camera)
    source = QualityMainStreamSource(
        by_id[args.camera],
        max(640, args.width),
        max(360, args.height),
        max(80, args.latency_ms),
    )
    return source, tracks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show sharp main stream with V6 sticky body-envelope tracker overlay."
    )
    parser.add_argument("--camera", default="CAM-01")
    parser.add_argument("--track-log", default="/tmp/ML_STEP4_V6_VISUAL.log")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--latency-ms", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, tracks = build_source_and_tracks(args)

    print(
        "STEP4_V6_VISUAL_DEBUG "
        f"camera={args.camera} source=main-stream resolution={args.width}x{args.height} "
        f"latency={args.latency_ms}ms drop_on_latency=0 scale=gpu-cubic "
        f"tracker_log={args.track_log} metadata_lag_comp=1 predict_max=0.34s "
        "size_prediction=0 body_envelope=tracker stale_box_max=1.10s",
        flush=True,
    )
    print("STEP4_V6_VISUAL_KEYS q/esc=quit f=fullscreen", flush=True)

    app = QApplication(sys.argv)
    tracks.start()
    source.start()
    viewer = ViewerV6(source, tracks, args.camera)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
