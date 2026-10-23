#!/usr/bin/env python3
"""Render deterministic, synthetic UI-only provisional-ID overlay evidence.

This utility deliberately does not load video, detector, tracker, or ReID data.
It exercises the same compact label and badge helpers used by VideoPane.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.mv3dt_staging.live_tracking.live_bridge import (
    clamp_box_to_frame,
    paint_camera_provisional_badge,
    paint_camera_track_labels,
)


def _panel(painter: QPainter, *, x: int, camera: str, target_id: str,
           local_ids: tuple[str, ...], identity_map: dict[str, dict[str, str]]) -> list[dict]:
    frame_rect = (x, 78, 650, 450)
    frame_x, frame_y, frame_width, frame_height = frame_rect
    painter.fillRect(*frame_rect, QColor("#273b49" if camera == "CAM-02" else "#3b3547"))

    # Abstract room details and person silhouettes: intentionally not camera footage.
    painter.fillRect(frame_x + 18, frame_y + 325, frame_width - 36, 3, QColor("#8d9ba3"))
    painter.fillRect(frame_x + 515, frame_y + 42, 4, 290, QColor("#73818a"))
    boxes = [
        (frame_x + 194, frame_y + 82, 122, 250),
        (frame_x + 258, frame_y + 89, 122, 244),
        (frame_x + 342, frame_y + 76, 104, 258),
        (frame_x - 12, frame_y + 111, 86, 212),  # exercise boundary clipping
    ]
    ids = (target_id, *local_ids)
    objects = []
    for index, (box, local_id) in enumerate(zip(boxes, ids)):
        clipped = clamp_box_to_frame(frame_rect, box)
        assert clipped is not None
        left, top, width, height, was_clamped = clipped
        # A simple silhouette keeps the artifact visibly synthetic.
        center = left + width // 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#d5dce0"))
        painter.drawEllipse(center - 19, top + 18, 38, 42)
        painter.setBrush(QColor("#8398a5" if index % 2 == 0 else "#9d8eaa"))
        painter.drawRoundedRect(center - 28, top + 60, 56, max(50, height - 66), 9, 9)

        painter.setPen(QPen(QColor("#38e58f"), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(left, top, width, height)
        painter.setPen(QPen(QColor("#ffd15b" if camera == "CAM-05" else "#63ceff"), 3))
        painter.setBrush(QColor("#ffd15b" if camera == "CAM-05" else "#63ceff"))
        painter.drawEllipse(center - 5, top + height - 5, 10, 10)
        objects.append({
            # Pass the original box to the same renderer used in VideoPane so
            # the shared clamp path, not just the synthetic painter, is tested.
            "left": box[0], "top": box[1], "width": box[2], "height": box[3],
            "tracker_id": local_id, "confidence": 0.88 - index * 0.025,
        })

    painter.setFont(QFont("Sans Serif", 10, QFont.Weight.Bold))
    metrics = QFontMetrics(painter.font())
    camera_identity_map = identity_map.get(camera, {})
    rendered = paint_camera_track_labels(
        painter, frame_rect, objects, camera_identity_map, metrics,
        Qt.TextElideMode.ElideRight, QColor(0, 0, 0, 220), QColor("#ffffff"),
    )
    badge = paint_camera_provisional_badge(
        painter, frame_rect, identity_map, camera, set(ids), metrics,
        Qt.TextElideMode.ElideRight, QColor("#542578"), QColor("#ffffff"),
    )

    assert len(rendered) == len(ids)
    target = next(row for row in rendered if row["tracker_id"] == target_id)
    assert f"local {target_id}" in target["text"] and "G001?" in target["text"]
    assert badge is not None and "G001?" in badge["text"]
    assert rendered[-1]["box_was_clamped"] is True
    for row in rendered:
        rx, ry, rw, rh = row["rect"]
        assert frame_x <= rx <= frame_x + frame_width - rw
        assert frame_y <= ry <= frame_y + frame_height - rh
    for index, first in enumerate(rendered):
        for second in rendered[index + 1:]:
            a, b = first["rect"], second["rect"]
            assert (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]
                    or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])
    # In this deliberately crowded-but-bounded fixture, the primary GID labels
    # must find a clear placement that does not cover another detection box.
    assert target["conflict"] is False
    return rendered


def _withdrawal_check(painter: QPainter, x: int) -> None:
    frame_rect = (x, 566, 300, 78)
    painter.fillRect(*frame_rect, QColor("#2b3037"))
    obj = [{"left": x + 18, "top": 589, "width": 66, "height": 48,
            "tracker_id": "0", "confidence": 0.91}]
    font = QFont("Sans Serif", 9, QFont.Weight.Bold)
    metrics = QFontMetrics(font)
    painter.setFont(font)
    local_only = paint_camera_track_labels(
        painter, frame_rect, obj, {}, metrics, Qt.TextElideMode.ElideRight,
        QColor(0, 0, 0, 220), QColor("#ffffff"),
    )
    badge = paint_camera_provisional_badge(
        painter, frame_rect, {}, "CAM-05", {"0"}, metrics,
        Qt.TextElideMode.ElideRight, QColor("#542578"), QColor("#ffffff"),
    )
    assert len(local_only) == 1
    assert local_only[0]["text"] == "local 0 · det 0.91"
    assert "G001" not in local_only[0]["text"]
    assert badge is None
    painter.setPen(QColor("#ffffff"))
    painter.drawText(x + 312, 614, "withdrawal fixture: local-only label; obsolete G001? removed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QGuiApplication.instance() or QGuiApplication([])
    assert app is not None

    image = QImage(1380, 660, QImage.Format.Format_ARGB32)
    image.fill(QColor("#101820"))
    painter = QPainter(image)
    painter.setFont(QFont("Sans Serif", 16, QFont.Weight.Bold))
    painter.setPen(QColor("#ffffff"))
    painter.drawText(20, 34, "SYNTHETIC UI-ONLY TEST — NOT CAMERA FOOTAGE")
    painter.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
    mapping = {
        "CAM-02": {"4": {"provisional_global_id": "G001"}},
        "CAM-05": {"0": {"provisional_global_id": "G001"}},
    }
    _panel(painter, x=20, camera="CAM-02", target_id="4",
           local_ids=("5", "9", "22"), identity_map=mapping)
    _panel(painter, x=700, camera="CAM-05", target_id="0",
           local_ids=("1", "2", "21"), identity_map=mapping)
    painter.setPen(QColor("#f2f5f7"))
    painter.drawText(20, 67, "CAM-02 · local NvDCF IDs + provisional candidate")
    painter.drawText(700, 67, "CAM-05 · local NvDCF IDs + provisional candidate")
    _withdrawal_check(painter, 20)
    painter.end()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(args.output), "PNG"):
        raise RuntimeError(f"could not save fixture image to {args.output}")
    if args.output.stat().st_size < 10_000:
        raise RuntimeError("fixture image unexpectedly small")
    print("UI_FIXTURE_ASSERTIONS=PASS")
    print("CAM02_COMPACT_PROVISIONAL_LABEL=PASS")
    print("CAM05_COMPACT_PROVISIONAL_LABEL=PASS")
    print("BOUNDARY_CLAMP_AND_LABEL_PLACEMENT=PASS")
    print("CANDIDATE_WITHDRAWAL_LABEL_REMOVAL=PASS")
    print(f"SYNTHETIC_SCREENSHOT={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
