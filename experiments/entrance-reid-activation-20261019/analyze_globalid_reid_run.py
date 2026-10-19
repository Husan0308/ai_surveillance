#!/usr/bin/env python3
"""Read-only consistency audit for one isolated DeepStream 9.1 MV3DT run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def parse_payloads(path: Path):
    decoder = json.JSONDecoder()
    errors = []
    for line_number, line in enumerate(path.open(), 1):
        try:
            topic, payload_hex = line.rstrip("\n").split("|", 1)
            payload = bytes.fromhex(payload_hex).decode("utf-8")
            cursor = 0
            while cursor < len(payload):
                while cursor < len(payload) and payload[cursor].isspace():
                    cursor += 1
                if cursor == len(payload):
                    break
                message, cursor = decoder.raw_decode(payload, cursor)
                yield topic, message
                if cursor < len(payload) and payload[cursor] == "|":
                    cursor += 1
                elif cursor < len(payload) and not payload[cursor].isspace():
                    raise ValueError(f"unexpected payload separator at {cursor}")
        except Exception as exc:  # evidence audit must count, not hide, malformed records
            errors.append({"line": line_number, "error": str(exc)})
    if errors:
        raise ValueError(f"{len(errors)} malformed MQTT publication(s); first={errors[0]}")


def frame_file_map(directory: Path):
    rows = defaultdict(set)
    files = defaultdict(list)
    for path in sorted(directory.glob("00_???_??????.txt")):
        _, stream, frame = path.stem.split("_")
        stream_id, frame_id = int(stream), int(frame)
        files[stream_id].append(frame_id)
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                try:
                    rows[(stream_id, frame_id)].add(int(fields[1]))
                except ValueError:
                    continue
    return rows, files


def kitti_boxes(directory: Path):
    boxes = defaultdict(list)
    for path in sorted(directory.glob("00_???_??????.txt")):
        _, stream, frame = path.stem.split("_")
        key = (int(stream), int(frame))
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 9:
                continue
            try:
                boxes[key].append((int(fields[1]), *(float(value) for value in fields[5:9])))
            except ValueError:
                continue
    return boxes


def box_iou(left, right) -> float:
    _, lx1, ly1, lx2, ly2 = left
    _, rx1, ry1, rx2, ry2 = right
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def continuity_proxies(boxes):
    by_stream = defaultdict(list)
    for (stream, frame), rows in boxes.items():
        by_stream[stream].append((frame, rows))
    changes_04 = []
    changes_08 = []
    for stream, frames in by_stream.items():
        rows_by_frame = dict(frames)
        for frame in sorted(rows_by_frame):
            current = rows_by_frame.get(frame + 1)
            if current is None:
                continue
            for previous_box in rows_by_frame[frame]:
                if not current:
                    continue
                match = max(current, key=lambda candidate: box_iou(previous_box, candidate))
                overlap = box_iou(previous_box, match)
                if overlap >= 0.4 and previous_box[0] != match[0]:
                    item = {"source": stream, "frame_from": frame, "frame_to": frame + 1,
                            "id_from": previous_box[0], "id_to": match[0], "iou": overlap}
                    changes_04.append(item)
                    if overlap >= 0.8:
                        changes_08.append(item)
    duplicates = []
    for (stream, frame), rows in boxes.items():
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                overlap = box_iou(left, right)
                if left[0] != right[0] and overlap >= 0.5:
                    duplicates.append({"source": stream, "frame": frame,
                                       "id_a": left[0], "id_b": right[0], "iou": overlap})
    return {
        "adjacent_frame_id_change_iou_ge_0_4": len(changes_04),
        "adjacent_frame_id_change_iou_ge_0_8": len(changes_08),
        "same_camera_frame_distinct_id_overlap_iou_ge_0_5": len(duplicates),
        "id_change_examples": changes_04,
        "overlap_examples": duplicates,
        "warning": "Continuity surrogates only; not ground-truth ID switches, fragmentation, or false-merge metrics.",
    }


def feature_rows(directory: Path):
    rows = defaultdict(set)
    files = defaultdict(list)
    vector_stats = defaultdict(lambda: {"rows": 0, "bad_rows": 0, "norm_min": math.inf,
                                        "norm_max": 0.0, "norm_sum": 0.0})
    for path in sorted(directory.glob("00_???_??????.txt")):
        _, stream, frame = path.stem.split("_")
        stream_id, frame_id = int(stream), int(frame)
        files[stream_id].append(frame_id)
        stats = vector_stats[stream_id]
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                tracker_id = int(fields[0])
                vector = [float(value) for value in fields[1:]]
                if len(vector) != 256 or not all(math.isfinite(value) for value in vector):
                    raise ValueError("feature shape/non-finite value")
                norm = math.sqrt(sum(value * value for value in vector))
                if not math.isfinite(norm) or norm == 0:
                    raise ValueError("zero/non-finite feature norm")
                stats["rows"] += 1
                stats["norm_min"] = min(stats["norm_min"], norm)
                stats["norm_max"] = max(stats["norm_max"], norm)
                stats["norm_sum"] += norm
                rows[(stream_id, frame_id)].add(tracker_id)
            except Exception:
                stats["bad_rows"] += 1
    for stats in vector_stats.values():
        stats["norm_mean"] = stats["norm_sum"] / stats["rows"] if stats["rows"] else None
        del stats["norm_sum"]
        if stats["norm_min"] == math.inf:
            stats["norm_min"] = None
    return rows, files, vector_stats


def summarize_frames(files):
    summary = {}
    for stream, frames in sorted(files.items()):
        unique = sorted(set(frames))
        missing = sorted(set(range(unique[0], unique[-1] + 1)) - set(unique)) if unique else []
        summary[str(stream)] = {
            "files": len(frames),
            "unique_frames": len(unique),
            "first_frame": unique[0] if unique else None,
            "last_frame": unique[-1] if unique else None,
            "missing_frames_within_range": missing[:100],
            "missing_frame_count_within_range": len(missing),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    kitti, kitti_files = frame_file_map(run / "run" / "tracker-kitti-dump")
    boxes = kitti_boxes(run / "run" / "tracker-kitti-dump")
    features, feature_files, vector_stats = feature_rows(run / "reid-feature-output")

    mqtt_counts = Counter()
    mqtt_ids = defaultdict(set)
    adoptions = []
    message_frames = defaultdict(set)
    topic_stream = {"/trck/cam1": 0, "/trck/cam2": 1}
    for topic, message in parse_payloads(run / "mqtt-raw-hex.log"):
        event_type = str(message.get("type", "missing"))
        mqtt_counts[f"{topic}:type_{event_type}"] += 1
        stream = topic_stream.get(topic)
        if stream is None:
            continue
        frame = message.get("frameNum")
        target = message.get("targetId")
        if isinstance(frame, int):
            message_frames[(stream, frame)].add(event_type)
        if isinstance(target, int) and event_type in {"1", "2"}:
            mqtt_ids[(stream, frame)].add(target)
        if event_type == "3":
            adoptions.append({
                "camera_source": stream,
                "camera_topic": topic,
                "frame": frame,
                "old_target_id": message.get("targetId"),
                "adopted_id": message.get("adoptedId"),
                "message_timestamp": message.get("messageTimeStamp"),
                "adopted_id_creation_timestamp": message.get("targetIDTimeStamp"),
            })

    common_mqtt_frames = sorted(set(mqtt_ids) & set(kitti))
    exact_mqtt_kitti = sum(mqtt_ids[key] == kitti[key] for key in common_mqtt_frames)
    common_feature_frames = sorted(set(features) & set(kitti))
    exact_feature_kitti = sum(features[key] == kitti[key] for key in common_feature_frames)
    feature_id_matches = sum(len(features[key] & kitti[key]) for key in common_feature_frames)
    feature_id_total = sum(len(features[key]) for key in common_feature_frames)
    kitti_id_total = sum(len(kitti[key]) for key in common_feature_frames)

    shared_global_ids = sorted(
        set().union(*(ids for (stream, _frame), ids in mqtt_ids.items() if stream == 0))
        & set().union(*(ids for (stream, _frame), ids in mqtt_ids.items() if stream == 1))
    )
    output = {
        "run_directory": str(run),
        "frames": {"kitti_tracker": summarize_frames(kitti_files),
                   "reid_feature": summarize_frames(feature_files)},
        "feature_vectors": {str(stream): stats for stream, stats in sorted(vector_stats.items())},
        "mqtt_event_counts": dict(sorted(mqtt_counts.items())),
        "mqtt_payloads": sum(mqtt_counts.values()),
        "adopted_id_events": adoptions,
        "camera_scoped_target_ids": {
            str(stream): sorted(set().union(*(ids for (s, _f), ids in mqtt_ids.items() if s == stream)))
            for stream in (0, 1)
        },
        "target_ids_appearing_on_both_topics": shared_global_ids,
        "same_frame_crosswalk": {
            "mqtt_vs_kitti_common_stream_frames": len(common_mqtt_frames),
            "mqtt_vs_kitti_exact_id_set_frames": exact_mqtt_kitti,
            "feature_vs_kitti_common_stream_frames": len(common_feature_frames),
            "feature_vs_kitti_exact_id_set_frames": exact_feature_kitti,
            "feature_id_rows": feature_id_total,
            "feature_ids_found_in_kitti": feature_id_matches,
            "kitti_object_rows": kitti_id_total,
        },
        "continuity_proxies": continuity_proxies(boxes),
        "interpretation": (
            "Same-frame source-scoped identity/feature crosswalk only. Numeric IDs are not joined across cameras. "
            "MQTT shared target IDs are algorithm outputs, not visual identity ground truth."
        ),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
