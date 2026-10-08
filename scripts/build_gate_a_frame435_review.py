#!/usr/bin/env python3
"""Render source-indexed CAM-01 frame 430..437 Gate A adjudication panels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".runtime/gate-a-direct-file-20261005-r3/frame435-review"
VIDEO = ROOT / ".runtime/amc-transformer-personpresent-MtbZPbuh/amc_inputs/videos/cam_00.mp4"
ORACLE_CSV = ROOT / ".runtime/sixcam-local6-20261004-b/shadow-assisted-detector-ab-v4/CAM01-per-frame.csv"
R0_AUDIT = ROOT / ".runtime/sixcam-local6-20261004-b/replay-local6-type2-shadow-age/run/logs/probe/frame_path_audit.jsonl"
R1 = ROOT / ".runtime/gate-a-direct-file-20261005-r3/R1"
R2 = ROOT / ".runtime/gate-a-direct-file-20261005-r3/R2"
OLD_T2 = ROOT / ".runtime/gate-a-tracker-t2-20261005"
EXPECTED_VIDEO_SHA = "1e2818e022b58f2f6608b9af24682f9dbc0d628ee80c7754f0c95a7c943f70f1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def box(value):
    if not value:
        return None
    return [float(x) for x in json.loads(value)]


def draw_panel(image, title: str, boxes: list[tuple[list[float], str, tuple[int, int, int]]]):
    output = image.copy()
    for (x, y, w, h), label, color in boxes:
        p1 = (round(x), round(y))
        p2 = (round(x + w), round(y + h))
        cv2.rectangle(output, p1, p2, color, 3, cv2.LINE_AA)
        cv2.putText(output, label, (p1[0], max(28, p1[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, color, 2, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1], 45), (24, 24, 24), -1)
    cv2.putText(output, title, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (245, 245, 245), 2, cv2.LINE_AA)
    return output


def selected_rows(frame_rows: list[dict], source: str | None = None):
    selected = []
    for row in frame_rows:
        b = box(row.get("chosen_bbox"))
        if not b:
            continue
        kind = row.get("chosen_source", "?")
        color = (60, 230, 60) if kind == "LOCAL_ACTIVE" else (0, 220, 255)
        selected.append((b, f"{kind} {row.get('chosen_native_track_id', '')}", color))
    return selected


def read_rows_at(path: Path, predicate):
    result = []
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if predicate(item):
                result.append(item)
    return result


def main():
    if sha(VIDEO) != EXPECTED_VIDEO_SHA:
        raise SystemExit("pinned CAM-01 video hash mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    per_frame = {frame: [] for frame in range(430, 438)}
    for row in load_rows(ORACLE_CSV):
        frame = int(row["frame"])
        if frame in per_frame:
            per_frame[frame].append(row)

    r0 = {}
    for row in read_rows_at(R0_AUDIT, lambda x: x.get("record") == "frame" and x.get("stage") == "tracker"
                            and x.get("mapped_camera_id") == "CAM-01" and 430 <= int(x["frame_num"]) <= 437):
        r0[int(row["frame_num"])] = row
    r1 = {frame: read_rows_at(R1 / "normalized-tracks.jsonl",
                              lambda x, f=frame: x.get("camera_id") == "CAM-01" and
                              int(x["source_local_frame_num"]) == f) for frame in range(430, 438)}
    r2 = {frame: read_rows_at(R2 / "normalized-tracks.jsonl",
                              lambda x, f=frame: int(x["source_id"]) == 1 and
                              int(x["source_local_frame_num"]) == f) for frame in range(430, 438)}
    old_t2 = {frame: read_rows_at(OLD_T2 / "tracks.jsonl",
                                 lambda x, f=frame: int(x["source_id"]) == 0 and int(x["frame"]) == f)
              for frame in range(430, 438)}

    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise SystemExit("could not open pinned CAM-01 video")
    frames = {}
    current = 0
    while current <= 437:
        ok, image = capture.read()
        if not ok:
            raise SystemExit(f"video ended before source-local frame {current}")
        if current >= 430:
            frames[current] = image
        current += 1
    capture.release()

    contact_panels = []
    contact_metadata = []
    for frame, image in frames.items():
        det = read_rows_at(R1 / "normalized-detections.jsonl",
                           lambda x, f=frame: x.get("camera_id") == "CAM-01" and
                           int(x["source_local_frame_num"]) == f)
        det_boxes = [(x["box"], f"DET {float(x['confidence']):.3f}", (0, 220, 255)) for x in det]
        oracle = selected_rows(per_frame[frame])
        r1_boxes = [(x["box"], f"ACTIVE id={x['object_id']}", (255, 160, 40)) for x in r1[frame]]
        r2_boxes = [(x["box"], f"ACTIVE id={x['object_id']}", (255, 0, 220)) for x in r2[frame]]
        old_boxes = [(x["box"], f"LEGACY T2 id={x['object_id']}", (0, 100, 255)) for x in old_t2[frame]]
        panels = [
            draw_panel(image, f"SOURCE — frame {frame}", []),
            draw_panel(image, f"PGIE DET — frame {frame}", det_boxes),
            draw_panel(image, f"APPROVED v4 — frame {frame}", oracle),
            draw_panel(image, f"R1 integrated 2-source — frame {frame}", r1_boxes),
            draw_panel(image, f"R2 6-input duplicate source_id=1 — frame {frame}", r2_boxes),
            draw_panel(image, f"Prior T2 (index-only, PTS not validated) — frame {frame}", old_boxes),
        ]
        if frame == 435:
            for name, panel in zip(("source", "pgie", "approved-v4", "R1-two-source", "R2-six-input-source1", "prior-T2-unvalidated"), panels):
                cv2.imwrite(str(OUT / f"CAM01-frame435-{name}.png"), panel)
            thumb = [cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA) for panel in panels]
            top = cv2.hconcat(thumb[:3])
            bottom = cv2.hconcat(thumb[3:])
            cv2.imwrite(str(OUT / "CAM01-frame435-six-panel.jpg"), cv2.vconcat([top, bottom]),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            roi = image[620:1080, 100:600]
            cv2.imwrite(str(OUT / "CAM01-frame435-person-proposal-crop.png"), roi)
        contact_panels.append(cv2.resize(panels[4], (480, 270), interpolation=cv2.INTER_AREA))
        contact_metadata.append({
            "source_local_frame": frame,
            "source_pts_ns": 100_000_000 + frame * 50_000_000,
            "pgie_proposals": len(det), "approved_output_boxes": len(oracle),
            "R1_active_boxes": len(r1[frame]), "R2_source1_active_boxes": len(r2[frame]),
            "prior_T2_boxes_index_only": len(old_t2[frame]),
        })
    rows = [cv2.hconcat(contact_panels[i:i + 4]) for i in range(0, len(contact_panels), 4)]
    cv2.imwrite(str(OUT / "CAM01-frames430-437-R2-six-input-contact.jpg"), cv2.vconcat(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 94])
    manifest = {
        "source_video": str(VIDEO), "source_sha256": EXPECTED_VIDEO_SHA,
        "source_frames": "decoded sequentially from index 0; indices 430..437 extracted",
        "source_pts_ns": "100000000 + frame_index*50000000 as verified by direct-file source-local log",
        "approved_reference_csv": str(ORACLE_CSV),
        "R0_tracker_source": str(R0_AUDIT),
        "R1_candidate": str(R1), "R2_candidate": str(R2),
        "prior_T2_caveat": "Prior T2 output is shown as historical index-only context; source-local PTS/frame integrity failed and it is not treated as ground truth.",
        "R2_caveat": "source_id=1 is one of three diagnostic duplicate CAM-01 file sources in a six-input batch; not a unique live-camera result.",
        "frames": contact_metadata,
        "human_adjudication": "Pending visual inspection",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(OUT), "files": sorted(p.name for p in OUT.iterdir()),
                      "frame435": contact_metadata[5]}, indent=2))


if __name__ == "__main__":
    main()
