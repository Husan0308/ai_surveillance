#!/usr/bin/env python3
"""Render short dual-camera clips with unmodified raw MV3DT IDs/timestamps."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from pathlib import Path

import cv2
import numpy as np

DB = Path("/tmp/entrance-180s-review.9WdL9e/B-events.sqlite3")
VIDEOS = {
    "CAM-02": Path("/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4"),
    "CAM-05": Path("/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4"),
}
FPS = 30
PANEL_W, PANEL_H = 960, 540
FOOTER_H = 42
CLIPS = (
    ("gid7-to-gid8", 2050, 2740, "Raw MV3DT IDs 7→8 candidate; identity remains UNCERTAIN"),
    ("gid12-to-gid13", 3600, 3950, "Raw MV3DT IDs 12→13 candidate; identity remains UNCERTAIN"),
    ("gid8-vs-gid10", 2600, 3110, "Raw MV3DT GID 8 / GID 10 sequence; user label UNCERTAIN"),
    ("gid8-vs-gid11", 2900, 3110, "GID 8 / 11 overlap near frame 3087; likely duplicate box, not identity merge"),
)


def load_events(cam: str, start: int, end: int):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    frames = {
        int(r["frame_id"]): r["event_timestamp"]
        for r in con.execute(
            "SELECT frame_id,event_timestamp FROM frame_messages WHERE camera_id=? AND frame_id>=? AND frame_id<?",
            (cam, start, end),
        )
    }
    boxes: dict[int, list[tuple[str, dict]]] = {}
    for r in con.execute(
        "SELECT frame_id,mv3dt_global_id,bbox_json FROM observations WHERE camera_id=? AND frame_id>=? AND frame_id<? ORDER BY frame_id,mv3dt_global_id",
        (cam, start, end),
    ):
        if r["bbox_json"]:
            boxes.setdefault(int(r["frame_id"]), []).append((r["mv3dt_global_id"], json.loads(r["bbox_json"])))
    con.close()
    return frames, boxes


def color_for(raw_id: str):
    # Deterministic visible color; this is a visualization palette only.
    palette = [(70, 230, 100), (60, 210, 245), (255, 170, 55), (230, 90, 230), (60, 170, 255), (255, 100, 90), (125, 225, 245)]
    try:
        return palette[int(raw_id) % len(palette)]
    except ValueError:
        return palette[sum(map(ord, raw_id)) % len(palette)]


def put_label(img, text, x, y, color, font_scale=0.5, thick=2):
    f = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, f, font_scale, thick)
    x = max(0, min(img.shape[1] - tw - 6, int(x)))
    y = max(th + base + 3, min(img.shape[0] - base - 2, int(y)))
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - base - 1), f, font_scale, (0, 0, 0), thick, cv2.LINE_AA)


def render_panel(cam, raw_frame, frame_no, event_ts, boxes):
    if raw_frame.shape[:2] != (1080, 1920):
        raise RuntimeError(f"Unexpected source image shape for {cam}: {raw_frame.shape}")
    img = cv2.resize(raw_frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    scale_x, scale_y = PANEL_W / 1920.0, PANEL_H / 1080.0
    # Source frames and event boxes are both 1920x1080 coordinate-space records.
    for gid, b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in (b["leftX"] * scale_x, b["topY"] * scale_y, b["rightX"] * scale_x, b["bottomY"] * scale_y)]
        c = color_for(gid)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
        put_label(img, f"RAW GID {gid}", x1, y1 - 3, c, 0.48, 1)
    local_t = frame_no / FPS
    stamp = event_ts or "event timestamp unavailable"
    put_label(img, f"{cam} | frame {frame_no} | t={local_t:.3f}s (nominal) | {stamp}", 10, 30, (24, 28, 36), 0.42, 1)
    return img


def render_clip(name: str, start: int, end: int, title: str, outdir: Path):
    streams = {}
    metadata = {}
    for cam, video in VIDEOS.items():
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        streams[cam] = cap
        metadata[cam] = load_events(cam, start, end)

    out = outdir / f"{name}.mp4"
    errlog = outdir / f"{name}.ffmpeg.log"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{PANEL_W*2}x{PANEL_H+FOOTER_H}",
        "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
    ]
    with errlog.open("wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            count = 0
            for frame_no in range(start, end):
                panels = []
                for cam in ("CAM-02", "CAM-05"):
                    ok, raw = streams[cam].read()
                    if not ok:
                        raise RuntimeError(f"Decode stopped: {cam} frame {frame_no}")
                    timestamps, boxes = metadata[cam]
                    panels.append(render_panel(cam, raw, frame_no, timestamps.get(frame_no), boxes.get(frame_no, [])))
                joined = np.hstack(panels)
                footer = np.zeros((FOOTER_H, PANEL_W * 2, 3), dtype=np.uint8)
                cv2.putText(footer, "Raw event IDs and per-camera event timestamps; paired panels do not assert frame-level synchronization (approx. +/-150 ms uncertainty).", (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 220, 165), 1, cv2.LINE_AA)
                proc.stdin.write(np.vstack((joined, footer)).tobytes())
                count += 1
            proc.stdin.close()
            rc = proc.wait(timeout=120)
            if rc:
                raise RuntimeError(f"ffmpeg returned {rc} for {name}")
        finally:
            for cap in streams.values():
                cap.release()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    out.chmod(0o600)
    errlog.chmod(0o600)
    return out, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=True)
    args.output.chmod(0o700)
    for name, start, end, title in CLIPS:
        path, count = render_clip(name, start, end, title, args.output)
        print(f"{path}\tframes={count}\tduration={count/FPS:.3f}s\tlabel={title}")


if __name__ == "__main__":
    main()
