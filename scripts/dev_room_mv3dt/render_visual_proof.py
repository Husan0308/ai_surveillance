#!/usr/bin/env python3
"""Render 1x2 Dev Room views plus BEV with persistent global_person_id."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import yaml

CAMS = ("CAM-01", "CAM-04")
from services.mv3dt_room.presence import (
    CurrentPresencePositionFilter,
    active_observations,
    application_id,
    assert_presence_contract,
)
VIDEOS = ("cam_00.mp4", "cam_01.mp4")
COLORS = {"Person_01": (80, 220, 80), "Person_02": (40, 165, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity-jsonl", required=True, type=Path)
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    rows = defaultdict(lambda: defaultdict(list))
    for line in args.identity_jsonl.read_text().splitlines():
        row = json.loads(line)
        rows[row["frame"]][row["camera_id"]].append(row)

    transforms = yaml.safe_load((args.dataset_dir / "transforms.yml").read_text())
    world_to_px = np.asarray(transforms["T_ov2px"], np.float64).reshape(3, 3)
    map_image = cv2.imread(str(args.dataset_dir / "map.png"))
    if map_image is None:
        raise RuntimeError("map image missing")

    captures = [cv2.VideoCapture(str(args.video_dir / name)) for name in VIDEOS]
    if not all(cap.isOpened() for cap in captures):
        raise RuntimeError("video open failed")
    fps = 20.0
    output_size = (1280, 720)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
    if not writer.isOpened():
        raise RuntimeError("video writer open failed")

    map_scale = min(1260 / map_image.shape[1], 330 / map_image.shape[0])
    map_view = cv2.resize(map_image, None, fx=map_scale, fy=map_scale)
    map_x = (1280 - map_view.shape[1]) // 2
    map_y = 380 + (330 - map_view.shape[0]) // 2
    trajectories = defaultdict(lambda: deque(maxlen=240))
    position_filter = CurrentPresencePositionFilter()
    presence_disagreements = []
    duplicate_source_rows = 0
    presence_assertion_frames = 0

    frame_num = 0
    while True:
        decoded = [cap.read() for cap in captures]
        if not all(ok for ok, _ in decoded):
            break
        canvas = np.zeros((720, 1280, 3), np.uint8)
        source_frame_rows = [
            row
            for camera in CAMS
            for row in rows[frame_num][camera]
        ]
        _, frame_rows = active_observations(source_frame_rows, frame_num)
        duplicate_source_rows += max(0, len(source_frame_rows) - len(frame_rows))
        for index, ((_, image), camera) in enumerate(zip(decoded, CAMS)):
            panel = cv2.resize(image, (640, 360))
            sx, sy = 640 / image.shape[1], 360 / image.shape[0]
            for row in [item for item in frame_rows if item["camera_id"] == camera]:
                l, t, r, b = row["bbox"]
                identity = application_id(row)
                color = COLORS.get(identity, (255, 180, 60))
                p1, p2 = (int(l * sx), int(t * sy)), (int(r * sx), int(b * sy))
                cv2.rectangle(panel, p1, p2, color, 2)
                label = identity
                cv2.putText(panel, label, (p1[0], max(24, p1[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
                cv2.putText(panel, f"native:{row['native_track_id']}", (p1[0], min(352, p2[1] + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            cv2.putText(panel, camera, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            canvas[0:360, index * 640:(index + 1) * 640] = panel

        canvas[map_y:map_y + map_view.shape[0], map_x:map_x + map_view.shape[1]] = map_view
        by_identity = position_filter.update(frame_rows)
        active_ids = {application_id(row) for row in frame_rows}
        stale_ids = set(trajectories) - active_ids
        for gid in list(trajectories):
            if gid not in active_ids:
                del trajectories[gid]
        for gid, point in by_identity.items():
            hp = world_to_px @ np.asarray([point[0], point[1], 1.0])
            if abs(hp[2]) < 1e-9:
                continue
            px, py = hp[:2] / hp[2]
            pixel = (int(map_x + px * map_scale), int(map_y + py * map_scale))
            trajectories[gid].append(pixel)
        rendered_ids = set()
        rendered_marker_ids = []
        for gid in sorted(active_ids):
            trail = trajectories.get(gid)
            if not trail:
                continue
            color = COLORS.get(gid, (255, 180, 60))
            if len(trail) > 1:
                cv2.polylines(canvas, [np.asarray(trail, np.int32)], False, color, 2)
            cv2.circle(canvas, trail[-1], 8, color, -1)
            cv2.putText(canvas, gid, (trail[-1][0] + 10, trail[-1][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            rendered_ids.add(gid)
            rendered_marker_ids.append(gid)
        duplicate_ids = {
            identity
            for identity in rendered_marker_ids
            if rendered_marker_ids.count(identity) > 1
        }
        presence_result = assert_presence_contract(
            frame_rows,
            rendered_marker_ids,
            duplicate_ids=duplicate_ids,
            native_rendered_ids=(),
            stale_rendered_ids=stale_ids & set(rendered_marker_ids),
        )
        presence_assertion_frames += 1
        expected = set(presence_result["active_camera_ids"])
        disagreement = sorted(expected ^ rendered_ids)
        if disagreement:
            presence_disagreements.append({
                "frame": frame_num,
                "active": sorted(expected),
                "rendered": sorted(rendered_ids),
            })
        cv2.putText(canvas, "Persistent application identity (OSNet + world/time); native ID is debug only",
                    (20, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {frame_num:04d}  t={frame_num / fps:06.2f}s",
                    (20, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        frame_num += 1

    for cap in captures:
        cap.release()
    writer.release()
    report = {
        "frames": frame_num,
        "fps": fps,
        "duration_seconds": frame_num / fps,
        "resolution": list(output_size),
        "output": str(args.output),
        "layout": "CAM-01 | CAM-04 over common BEV",
        "labels": {"white-shirt": "Person_01", "black-shirt": "Person_02"},
        "native_id_display": "small debug text only",
        "presence_disagreements": presence_disagreements,
        "frames_with_presence_disagreement": len(presence_disagreements),
        "presence_assertion_frames": presence_assertion_frames,
        "duplicate_source_rows_suppressed": duplicate_source_rows,
        "bev_presence_policy": "current active camera observation only; no stale last position",
        "presence_assertions": "no marker without active observation; one marker per application ID; no native/orphan/stale marker",
    }
    (args.output.parent / "visual_proof.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
