#!/usr/bin/env python3
"""CPU-only image of the production-staging marker painter with injected IDs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.mv3dt_staging.map_fusion import (
    ACCEPTED_STATUS, ARRIVAL_DOMAIN, IDENTITY_AUTHORITY,
    ExperimentalMapFusion, paint_map_fusion_markers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    app = QGuiApplication.instance() or QGuiApplication([])
    assert app is not None
    now = 10_000_000_000
    states = {}
    for camera, sid, local, xy in (("CAM-02", 0, "4", [500, 400]),
                                  ("CAM-05", 1, "0", [540, 420])):
        states[camera] = {"camera_id": camera, "source_id": sid, "frame_index": 10 + sid,
            "source_pts_ns": 1000 + sid, "updated_mono_ns": now,
            "observation_time_domain": ARRIVAL_DOMAIN, "status": "LIVE",
            "markers": [{"camera_id": camera, "source_id": sid, "frame_index": 10 + sid,
                "source_pts_ns": 1000 + sid, "local_tracker_id": local,
                "layout_xy": xy, "jump_candidate": False}]}
    bindings = {camera: {state["markers"][0]["local_tracker_id"]: {
        "provisional_global_id": "G001", "status": ACCEPTED_STATUS,
        "identity_memory_state": "ACTIVE", "persistent": False,
        "identity_expires_monotonic_ns": now + 600_000_000_000}}
        for camera, state in states.items()}
    identity = {"authority": IDENTITY_AUTHORITY, "revision": 1,
                "sampled_monotonic_ns": now, "assignments": bindings}
    fusion = ExperimentalMapFusion(layout_width=1448, layout_height=1086)
    snapshots = [fusion.update(states, identity, now_mono_ns=now, mode="local"),
                 fusion.update(states, identity, now_mono_ns=now)]
    withdrawn = {**identity, "revision": 2, "assignments": {"CAM-02": {}, "CAM-05": {}}}
    snapshots.append(fusion.update(states, withdrawn, now_mono_ns=now))
    assert [len(s["markers"]) for s in snapshots] == [2, 1, 2]
    assert snapshots[1]["markers"][0]["label"] == "G001?"
    assert all("G001" not in m["label"] for m in snapshots[2]["markers"])
    assert "global:G001" not in fusion._trails

    image = QImage(1410, 590, QImage.Format.Format_ARGB32)
    image.fill(QColor("#131b24"))
    painter = QPainter(image)
    painter.setPen(QColor("white")); painter.setFont(QFont("Sans Serif", 16, QFont.Weight.Bold))
    painter.drawText(18, 30, "SYNTHETIC UI FIXTURE — injected identity bindings, no camera inference")
    painter.setFont(QFont("Sans Serif", 11))
    painter.drawText(18, 55, "EXPERIMENTAL / UNCALIBRATED · map pixels · identity and exposure synchronization unverified")
    evidence = []
    titles = ("A: Local / Diagnostic", "B: Experimental Global Fusion", "C: Association withdrawn")
    for i, (title, snapshot) in enumerate(zip(titles, snapshots)):
        x, y, w, h = 18 + i * 464, 104, 442, 390
        painter.setPen(QColor("white")); painter.setFont(QFont("Sans Serif", 12, QFont.Weight.Bold))
        painter.drawText(x, 87, title)
        painter.fillRect(x, y, w, h, QColor("#e3e7e4"))
        painter.setPen(QPen(QColor("#bdc6c4"), 1))
        for gx in range(x, x + w, 30): painter.drawLine(gx, y, gx, y + h)
        for gy in range(y, y + h, 30): painter.drawLine(x, gy, x + w, gy)
        rendered = paint_map_fusion_markers(painter, snapshot,
                                          (x, y, w, h, w / 1448, h / 1086))
        assert not rendered["labels_hidden"]
        for row in rendered["markers"]:
            rx, ry, rw, rh = row["label_rect"]
            assert x <= rx <= x + w - rw and y <= ry <= y + h - rh
        evidence.append({"title": title, "rendered": rendered})
        painter.setPen(QColor("white")); painter.setFont(QFont("Sans Serif", 11))
        painter.drawText(x, 519, f"Primary markers: {len(rendered['markers'])}")
    painter.drawText(18, 558, "Original camera/local IDs and exact frame/PTS contributors remain in marker metadata.")
    painter.end()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output.parent, 0o700)
    if not image.save(str(args.output), "PNG"):
        raise RuntimeError("synthetic fixture image save failed")
    os.chmod(args.output, 0o600)
    metadata = args.output.with_suffix(".json")
    metadata.write_text(json.dumps({"synthetic_only": True, "policy": fusion.policy(),
                                     "evidence": evidence}, indent=2) + "\n")
    os.chmod(metadata, 0o600)
    print("FUSION_AND_WITHDRAWAL_UI=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
