#!/usr/bin/env python3
"""Create private, same-run annotated review clips from preserved MV3DT output.

No inference is run. IDs are displayed exactly as captured in run06; no IDs are
aliased or merged. The source videos are read-only inputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
RUN = Path("/tmp/entrance-nvdcf-reid-globalid-20261019-run06")
VIDEO_DIR = Path(
    "/home/apsidal/devroom-new-calibration/entrance-mv3dt-visual-test/"
    "entrance-20261018-mCcX3SDI/dataset/videos"
)
OUT = HERE / "identity-review-run06"
OUT.mkdir(mode=0o700, parents=True, exist_ok=True)
os.chmod(OUT, 0o700)
os.umask(0o077)

sys.path.insert(0, str(HERE))
from analyze_globalid_reid_run import feature_rows, frame_file_map  # noqa: E402


CAMERAS = {
    0: {"name": "CAM-02", "file": "cam_00.mp4"},
    1: {"name": "CAM-05", "file": "cam_01.mp4"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pts(source_id: int) -> list[float]:
    path = VIDEO_DIR / CAMERAS[source_id]["file"]
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [float(frame["best_effort_timestamp_time"]) for frame in json.loads(result.stdout)["frames"]]


def load_tracker_rows():
    root = RUN / "run" / "tracker-kitti-dump"
    rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for path in sorted(root.glob("00_???_??????.txt")):
        _, stream_s, frame_s = path.stem.split("_")
        source_id, frame = int(stream_s), int(frame_s)
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 9:
                continue
            try:
                rows[(source_id, frame)].append(
                    {
                        "raw_id": int(fields[1]),
                        "bbox": [float(value) for value in fields[5:9]],
                        "row": line,
                    }
                )
            except ValueError:
                continue
    return rows


def load_adoptions():
    decoder = json.JSONDecoder()
    result: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for line in (RUN / "mqtt-raw-hex.log").open():
        topic, encoded = line.rstrip("\n").split("|", 1)
        payload = bytes.fromhex(encoded).decode("utf-8")
        cursor = 0
        while cursor < len(payload):
            while cursor < len(payload) and payload[cursor].isspace():
                cursor += 1
            if cursor >= len(payload):
                break
            message, cursor = decoder.raw_decode(payload, cursor)
            source_id = {"/trck/cam1": 0, "/trck/cam2": 1}.get(topic)
            if source_id is not None and str(message.get("type")) == "3":
                result[(source_id, int(message["frameNum"]))].append(
                    (int(message["targetId"]), int(message["adoptedId"]))
                )
            if cursor < len(payload) and payload[cursor] == "|":
                cursor += 1
    return result


def draw_panel(source_id: int, frame_index: int, frame, pts: float, tracker_rows,
               features, adoptions, width=960, height=540):
    scale_x = width / frame.shape[1]
    scale_y = height / frame.shape[0]
    panel = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    cam_name = CAMERAS[source_id]["name"]
    cv2.rectangle(panel, (0, 0), (width, 42), (18, 18, 18), -1)
    title = f"{cam_name} | source={source_id} frame={frame_index} PTS={pts:.6f}s"
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (255, 255, 255), 1, cv2.LINE_AA)
    adoption_events = adoptions.get((source_id, frame_index), [])
    for event_index, (left_id, adopted_id) in enumerate(adoption_events):
        text = f"ADOPTED_ID raw {left_id} -> {adopted_id} (tracker lineage only)"
        cv2.putText(panel, text, (12, 58 + event_index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 235, 255), 2, cv2.LINE_AA)
    rows = tracker_rows.get((source_id, frame_index), [])
    # Keep labels readable when detections overlap: put a color-keyed raw-ID
    # legend in a compact, dark strip instead of stacking text at box corners.
    # Box geometry and per-camera ID colors remain unchanged.
    if rows:
        legend_y = 52 + len(adoption_events) * 20
        row_height = 20
        legend_bottom = min(height - 1, legend_y + len(rows) * row_height + 5)
        overlay = panel.copy()
        cv2.rectangle(overlay, (6, legend_y - 16), (min(width - 1, 380), legend_bottom),
                      (12, 12, 12), -1)
        cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
    for index, row in enumerate(rows):
        x1, y1, x2, y2 = row["bbox"]
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        # Color is camera-scoped so equal numeric IDs across cameras are not
        # visually presented as confirmed cross-camera identity matches.
        color_seed = (row["raw_id"] * 73 + source_id * 131) % 255
        color = (40 + color_seed // 2, 220 - color_seed // 3, 70 + color_seed // 4)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, 2)
        has_feature = row["raw_id"] in features.get((source_id, frame_index), set())
        label = f"raw GID {row['raw_id']} | {'ReID 256D' if has_feature else 'no feature'}"
        y_text = legend_y + index * row_height
        cv2.rectangle(panel, (12, y_text - 12), (23, y_text + 2), color, -1)
        cv2.putText(panel, label, (30, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, color, 1, cv2.LINE_AA)
    return panel


def render_clip(name, start_s, end_s, source_ids, pts_by_source, tracker_rows,
                features, adoptions):
    caps = {}
    start_frames = {}
    end_frames = {}
    for source_id in source_ids:
        pts = pts_by_source[source_id]
        start_frames[source_id] = max(0, bisect_left(pts, start_s))
        end_frames[source_id] = min(len(pts), bisect_left(pts, end_s))
        if start_frames[source_id] >= end_frames[source_id]:
            raise RuntimeError(f"Empty frame window for source {source_id} / {name}")
        cap = cv2.VideoCapture(str(VIDEO_DIR / CAMERAS[source_id]["file"]))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not read {VIDEO_DIR / CAMERAS[source_id]['file']}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames[source_id])
        caps[source_id] = cap

    output_path = OUT / f"{name}.mp4"
    pair = len(source_ids) == 2
    out_size = (1920, 540) if pair else (960, 540)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed for {output_path}")
    decoded_counts = Counter()
    frame_cursor = dict(start_frames)
    frame_limit = max(end_frames.values())
    try:
        # The two input clips are rendered against the nominal recording
        # timeline for review only. Each source displays its own exact PTS.
        for master_frame in range(min(start_frames.values()), frame_limit):
            panels = []
            for source_id in source_ids:
                frame_index = frame_cursor[source_id]
                cap = caps[source_id]
                if frame_index >= end_frames[source_id]:
                    blank = 0 * __import__("numpy").zeros((540, 960, 3), dtype="uint8")
                    cv2.putText(blank, f"{CAMERAS[source_id]['name']} outside clip window",
                                (25, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    panels.append(blank)
                    continue
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Decode stopped at source={source_id} frame={frame_index}")
                pts = pts_by_source[source_id][frame_index]
                panels.append(draw_panel(source_id, frame_index, frame, pts,
                                         tracker_rows, features, adoptions))
                frame_cursor[source_id] += 1
                decoded_counts[source_id] += 1
            canvas = __import__("numpy").hstack(panels) if pair else panels[0]
            writer.write(canvas)
    finally:
        writer.release()
        for cap in caps.values():
            cap.release()
    os.chmod(output_path, 0o600)
    return {
        "path": str(output_path),
        "sha256": sha256(output_path),
        "timeline_start_s": start_s,
        "timeline_end_s": end_s,
        "output_frames": int(decoded_counts[source_ids[0]]),
        "per_source_frames": {str(s): int(decoded_counts[s]) for s in source_ids},
        "source_frame_ranges": {
            str(s): [start_frames[s], end_frames[s] - 1] for s in source_ids
        },
        "note": "Review timeline only; the two cameras are not asserted exposure-synchronized.",
    }


def main():
    kitti, _ = frame_file_map(RUN / "run" / "tracker-kitti-dump")
    features, _, _ = feature_rows(RUN / "reid-feature-output")
    tracker_rows = load_tracker_rows()
    adoptions = load_adoptions()
    pts_by_source = {source_id: load_pts(source_id) for source_id in CAMERAS}
    clips = [
        render_clip("return_candidate_7_to_8_timewindow_run06", 72.0, 93.0,
                    [0, 1], pts_by_source, tracker_rows, features, adoptions),
        render_clip("return_candidate_12_to_13_timewindow_run06", 119.0, 135.0,
                    [0, 1], pts_by_source, tracker_rows, features, adoptions),
        render_clip("overlap_candidate_8_vs_10_cam05_run06", 91.5, 96.0,
                    [1], pts_by_source, tracker_rows, features, adoptions),
        render_clip("overlap_candidate_8_vs_11_cam05_run06", 96.5, 104.0,
                    [1], pts_by_source, tracker_rows, features, adoptions),
    ]
    annotations = [
        {
            "candidate": "prior_B_GID7_to_GID8_reappearance_window",
            "current_run06_observation": "see paired clip; numeric IDs are run06-only and are not equated with B IDs",
            "label": "UNCERTAIN",
            "valid_labels": ["SAME_PERSON", "DIFFERENT_PEOPLE", "UNCERTAIN"],
        },
        {
            "candidate": "prior_B_GID12_to_GID13_reappearance_window",
            "current_run06_observation": "see paired clip; numeric IDs are run06-only and are not equated with B IDs",
            "label": "UNCERTAIN",
            "valid_labels": ["SAME_PERSON", "DIFFERENT_PEOPLE", "UNCERTAIN"],
        },
        {
            "candidate": "prior_B_GID8_vs_GID10_overlap_window_CAM05",
            "current_run06_observation": "actual raw IDs shown in clip; visual correspondence unreviewed",
            "label": "UNCERTAIN",
            "valid_labels": ["SAME_PERSON", "DIFFERENT_PEOPLE", "UNCERTAIN"],
        },
        {
            "candidate": "prior_B_GID8_vs_GID11_overlap_window_CAM05",
            "current_run06_observation": "prior user note at frame 3087: likely duplicate box; carried as provisional only",
            "label": "UNCERTAIN",
            "valid_labels": ["SAME_PERSON", "DIFFERENT_PEOPLE", "UNCERTAIN"],
        },
    ]
    manifest = {
        "run": str(RUN),
        "camera_mapping": {str(k): v["name"] for k, v in CAMERAS.items()},
        "source_video_sha256": {
            str(s): sha256(VIDEO_DIR / CAMERAS[s]["file"]) for s in CAMERAS
        },
        "feature_rows": sum(len(ids) for ids in features.values()),
        "feature_observation_rows": sum(len(ids) for ids in features.values()),
        "tracker_frame_groups": len(kitti),
        "clips": clips,
        "annotation_template": annotations,
        "limitations": [
            "Camera timing is displayed from each source's encoded PTS; the paired panels do not imply exposure synchronization.",
            "Raw IDs are run-scoped MV3DT outputs; no cross-run numeric-ID equivalence is asserted.",
            "No persistent person IDs are assigned.",
        ],
    }
    manifest_path = OUT / "clip-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)
    csv_path = OUT / "identity-annotation-template.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate", "label", "notes"])
        writer.writeheader()
        for item in annotations:
            writer.writerow({"candidate": item["candidate"], "label": item["label"], "notes": item["current_run06_observation"]})
    os.chmod(csv_path, 0o600)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
