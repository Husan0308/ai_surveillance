"""Extract exact indexed evidence frames around validated native-track lifetimes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "frame":
                rows.append(value)
    return rows


def read_markers(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def objects_by_id(row: dict) -> dict[int, dict]:
    return {int(item["native_track_id"]): item for item in row.get("painted_boxes", [])}


def movement_window(rows: list[dict], target_id: int, start: int, stop: int) -> dict | None:
    best: dict | None = None
    for left in range(max(0, start), min(stop, len(rows))):
        first = objects_by_id(rows[left]).get(target_id)
        if first is None:
            continue
        # A 100-source-frame separation is approximately the requested five seconds.
        target_source = int(rows[left]["source_frame_num"]) + 100
        right = left + 1
        while right < min(stop, len(rows)) and int(rows[right]["source_frame_num"]) < target_source:
            right += 1
        if right >= min(stop, len(rows)):
            continue
        second = objects_by_id(rows[right]).get(target_id)
        if second is None:
            continue
        a, b = first["bbox"], second["bbox"]
        c0 = ((float(a[0]) + float(a[2])) / 2, (float(a[1]) + float(a[3])) / 2)
        c1 = ((float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2)
        distance = math.hypot(c1[0] - c0[0], c1[1] - c0[1])
        candidate = {"start_index": left, "end_index": right,
                     "source_frame_delta": int(rows[right]["source_frame_num"])
                     - int(rows[left]["source_frame_num"]),
                     "center_delta_pixels": distance,
                     "bbox_start": a, "bbox_end": b}
        if best is None or distance > best["center_delta_pixels"]:
            best = candidate
    return best


def extract_frame(video: Path, index: int, destination: Path) -> None:
    # Recorder writes one CFR 20 fps output frame per machine-row entry.
    at_seconds = index / 20.0
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss",
                    f"{at_seconds:.6f}", "-i", str(video), "-frames:v", "1",
                    "-q:v", "2", "-y", str(destination)], check=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg did not extract row-aligned frame {index}")


def build(capture_root: Path, targets: dict[str, int]) -> dict:
    result = {"capture_root": str(capture_root), "index_mapping":
              "machine-row order maps one-to-one to the CFR 20 fps annotated MP4 frame order",
              "cameras": {}}
    for camera, target_id in targets.items():
        directory = capture_root / camera
        rows = read_rows(directory / "machine-rows.jsonl")
        markers = read_markers(directory / "markers.jsonl")
        by_event = {marker["event"]: marker for marker in markers}
        baseline = by_event.get("empty-baseline-end")
        positive = by_event.get("person-positive-start")
        if positive is None or target_id not in {
                int(item["native_track_id"]) for item in positive.get("objects", [])}:
            result["cameras"][camera] = {"status": "PENDING", "reason":
                                         "no positive marker for selected native track"}
            continue
        baseline_source = int(baseline["source_frame_num"]) if baseline else -1
        positive_source = int(positive["source_frame_num"])
        target_indices = [i for i, row in enumerate(rows)
                          if target_id in objects_by_id(row)]
        entry_candidates = [i for i in target_indices
                            if int(rows[i]["source_frame_num"]) > baseline_source]
        positive_index = next((i for i, row in enumerate(rows)
                               if int(row["shared_sequence"]) == int(positive["shared_sequence"])), None)
        after_positive = [i for i in target_indices
                          if int(rows[i]["source_frame_num"]) >= positive_source]
        if not entry_candidates or not after_positive or positive_index is None:
            result["cameras"][camera] = {"status": "PENDING", "reason":
                                         "entry, positive frame, or post-baseline track rows missing"}
            continue
        entry_index = entry_candidates[0]
        exit_index = after_positive[-1]
        exit_source = int(rows[exit_index]["source_frame_num"])
        post_index = next((i for i in range(exit_index + 1, len(rows))
                           if int(rows[i]["source_frame_num"]) >= exit_source + 100
                           and target_id not in objects_by_id(rows[i])
                           and all(target_id not in objects_by_id(rows[j])
                                   for j in range(exit_index + 1, i + 1))), None)
        movement = movement_window(rows, target_id, positive_index, exit_index + 1)
        events = {"entry": entry_index, "person_positive": positive_index,
                  "exit_last_box": exit_index}
        if movement:
            events["movement_start"] = movement["start_index"]
            events["movement_end"] = movement["end_index"]
        if post_index is not None:
            events["post_exit"] = post_index
        event_dir = directory / "event-review"
        event_dir.mkdir(exist_ok=True)
        for name, index in events.items():
            extract_frame(directory / "entry-exit-annotated.mp4", index,
                          event_dir / f"{name}-frame-{index:06d}.jpg")
        event_rows = {name: {"row_index": index, "shared_sequence": rows[index]["shared_sequence"],
                             "source_frame_num": rows[index]["source_frame_num"],
                             "tracker_frame_num": rows[index]["tracker_frame_num"],
                             "pts_ns": rows[index]["pts_ns"],
                             "image_bbox_sync": rows[index]["image_bbox_sync"],
                             "object_count": rows[index]["object_count"],
                             "target_object": objects_by_id(rows[index]).get(target_id)}
                      for name, index in events.items()}
        record = {"camera_id": camera, "target_native_track_id": target_id,
                  "empty_baseline_available": baseline is not None,
                  "entry_row_index": entry_index, "entry_row": rows[entry_index],
                  "positive_marker": positive, "movement_window": movement,
                  "last_track_row_index": exit_index, "last_track_row": rows[exit_index],
                  "post_exit_row_index": post_index,
                  "post_exit_absence_frames": (None if post_index is None else
                      int(rows[post_index]["source_frame_num"]) - exit_source),
                  "post_exit_pass": post_index is not None,
                  "event_frames": event_rows,
                  "screenshots_directory": str(event_dir)}
        record["status"] = "EVIDENCE_COMPLETE_REVIEW_REQUIRED" if post_index is not None else "PENDING_EXIT"
        (event_dir / "event-evidence.json").write_text(json.dumps(record, indent=2) + "\n")
        result["cameras"][camera] = {"status": record["status"],
            "selected_track": target_id, "entry_index": entry_index,
            "last_track_index": exit_index, "post_exit_index": post_index,
            "movement_center_delta_pixels": None if movement is None else movement["center_delta_pixels"],
            "post_exit_absence_frames": record["post_exit_absence_frames"],
            "event_evidence": str(event_dir / "event-evidence.json")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[], help="CAM-XX=native_track_id")
    args = parser.parse_args()
    root = args.capture_root.resolve()
    if not root.is_relative_to(ROOT / ".runtime"):
        raise ValueError("capture root must remain under .runtime")
    targets = {}
    for value in args.target:
        camera, raw_id = value.split("=", 1)
        if camera in targets or camera not in {f"CAM-{i:02d}" for i in range(1, 7)}:
            raise ValueError(f"invalid/duplicate camera target: {camera}")
        targets[camera] = int(raw_id)
    summary = build(root, targets)
    out = root / "event-review-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
