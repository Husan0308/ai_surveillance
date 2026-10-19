#!/usr/bin/env python3
"""Create read-only visual-review sheets from preserved Entrance Room events."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DB = Path("/tmp/entrance-180s-review.9WdL9e/B-events.sqlite3")
VIDEOS = {
    "CAM-02": Path("/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_00.mp4"),
    "CAM-05": Path("/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/entrance-20261018-mCcX3SDI/dataset/videos/cam_01.mp4"),
}
OUT = Path("/tmp/entrance-reid-mux-fix-2bJ7d9c9/contact-sheets")
PANEL_W = 960
PANEL_H = 540
TITLE_H = 40
HEADER_H = 74
BG = (25, 29, 36)
WHITE = (245, 245, 245)
COLORS = {"7": (255, 190, 60), "8": (70, 230, 100), "12": (60, 210, 245), "13": (210, 90, 255), "10": (50, 160, 255), "11": (255, 80, 80)}


def obs(camera: str, gid: str, frame: int) -> dict:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT frame_id,mv3dt_global_id,bbox_json,media_time_seconds_nominal FROM observations "
        "WHERE camera_id=? AND mv3dt_global_id=? AND frame_id=?",
        (camera, gid, frame),
    ).fetchone()
    con.close()
    if row is None:
        raise ValueError(f"No preserved observation: {camera} GID {gid} frame {frame}")
    return dict(row)


def frame_at(camera: str, frame: int) -> Image.Image:
    cap = cv2.VideoCapture(str(VIDEOS[camera]))
    if not cap.isOpened():
        raise RuntimeError(f"Video open failed: {VIDEOS[camera]}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not decode {camera} frame {frame}")
    h, w = bgr.shape[:2]
    if (w, h) != (1920, 1080):
        raise RuntimeError(f"Unexpected {camera} image size {(w, h)}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate_panel(camera: str, gid: str, frame: int, label: str, extra_gids=()) -> Image.Image:
    image = frame_at(camera, frame)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    wanted = [gid, *extra_gids]
    marks = ",".join("?" for _ in wanted)
    rows = con.execute(
        f"SELECT frame_id,mv3dt_global_id,bbox_json FROM observations WHERE camera_id=? AND frame_id=? AND mv3dt_global_id IN ({marks})",
        (camera, frame, *wanted),
    ).fetchall()
    con.close()
    draw = ImageDraw.Draw(image)
    for row in rows:
        ident = row["mv3dt_global_id"]
        b = json.loads(row["bbox_json"])
        color = COLORS.get(ident, (255, 255, 0))
        box = (float(b["leftX"]), float(b["topY"]), float(b["rightX"]), float(b["bottomY"]))
        draw.rectangle(box, outline=color, width=6)
        draw.rectangle((box[0], max(0, box[1] - 37), box[0] + 175, box[1]), fill=color)
        draw.text((box[0] + 6, max(0, box[1] - 35)), f"raw GID {ident}", fill=(0, 0, 0), font=font(24, True))
    image.thumbnail((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (PANEL_W, PANEL_H + TITLE_H), BG)
    panel.paste(image, ((PANEL_W - image.width) // 2, TITLE_H))
    ImageDraw.Draw(panel).text((12, 8), label, fill=WHITE, font=font(20, True))
    return panel


def save_transition(filename: str, title: str, old_gid: str, old_obs: list[tuple[str, int]], new_gid: str, new_obs: list[tuple[str, int]], note: str):
    # 2 columns by 2 rows: each column is the raw-ID segment, each row a camera.
    canvas = Image.new("RGB", (2 * PANEL_W, HEADER_H + 2 * (PANEL_H + TITLE_H)), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), title, fill=WHITE, font=font(30, True))
    draw.text((18, 43), note, fill=(235, 210, 135), font=font(17))
    old_map = dict(old_obs)
    new_map = dict(new_obs)
    for col, (gid, mapping, tag) in enumerate(((old_gid, old_map, "previous raw ID"), (new_gid, new_map, "new raw ID"))):
        for row, cam in enumerate(("CAM-02", "CAM-05")):
            fr = mapping[cam]
            entry = obs(cam, gid, fr)
            t = float(entry["media_time_seconds_nominal"])
            panel = annotate_panel(cam, gid, fr, f"{tag}: {cam}, raw GID {gid}, frame {fr}, nominal {t:.2f}s")
            canvas.paste(panel, (col * PANEL_W, HEADER_H + row * (PANEL_H + TITLE_H)))
    path = OUT / filename
    canvas.save(path)
    return path


def save_pair(filename: str, title: str, items: list[tuple[str, int, str]], note: str, extras: dict[int, tuple[str, ...]] | None = None):
    # A 2-panel comparison preserves full context; boxes are event-log boxes.
    canvas = Image.new("RGB", (2 * PANEL_W, HEADER_H + PANEL_H + TITLE_H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), title, fill=WHITE, font=font(30, True))
    draw.text((18, 43), note, fill=(235, 210, 135), font=font(17))
    for i, (cam, fr, gid) in enumerate(items):
        entry = obs(cam, gid, fr)
        t = float(entry["media_time_seconds_nominal"])
        extra = (extras or {}).get(i, ())
        panel = annotate_panel(cam, gid, fr, f"{cam}, raw GID {gid}, frame {fr}, nominal {t:.2f}s", extra)
        canvas.paste(panel, (i * PANEL_W, HEADER_H))
    path = OUT / filename
    canvas.save(path)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    for p in VIDEOS.values():
        if not p.is_file():
            raise FileNotFoundError(p)
    paths = []
    paths.append(save_transition(
        "candidate-gid7-to-gid8.png", "Candidate: GID 7 → GID 8",
        "7", [("CAM-02", 2416), ("CAM-05", 2521)],
        "8", [("CAM-02", 2672), ("CAM-05", 2633)],
        "Candidate only — please label SAME_PERSON / DIFFERENT_PEOPLE / UNCERTAIN. Cross-camera time sync is unverified (about ±150 ms); panels are separate event snapshots.",
    ))
    paths.append(save_transition(
        "candidate-gid12-to-gid13.png", "Candidate: GID 12 → GID 13",
        "12", [("CAM-02", 3671), ("CAM-05", 3704)],
        "13", [("CAM-02", 3907), ("CAM-05", 3787)],
        "Candidate only — please label SAME_PERSON / DIFFERENT_PEOPLE / UNCERTAIN. Cross-camera time sync is unverified (about ±150 ms); panels are separate event snapshots.",
    ))
    # GID 8 and GID 10 are sequential in the preserved event rows, not a
    # simultaneous co-visible pair; this is an appearance candidate only.
    paths.append(save_pair(
        "candidate-gid8-versus-gid10.png", "Candidate comparison: GID 8 vs GID 10",
        [("CAM-05", 2637, "8"), ("CAM-05", 2816, "10")],
        "Sequential observations 179 frames apart; not simultaneous hard-negative proof. Label SAME_PERSON / DIFFERENT_PEOPLE / UNCERTAIN.",
    ))
    # At frame 3087 the raw log contains both IDs with overlapping boxes.
    paths.append(save_pair(
        "candidate-gid8-versus-gid11.png", "Candidate comparison: GID 8 vs GID 11",
        [("CAM-05", 3087, "8"), ("CAM-05", 3087, "11")],
        "Same camera/frame 3087; boxes overlap heavily. This may be duplicate observations, not a hard negative. Please label SAME_PERSON / DIFFERENT_PEOPLE / UNCERTAIN.",
        extras={0: ("11",), 1: ("8",)},
    ))
    for p in paths:
        p.chmod(0o600)
        print(p)


if __name__ == "__main__":
    main()
