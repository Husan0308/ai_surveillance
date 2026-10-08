"""Frame-aligned public NvDCF metadata for the two CAM-04 continuity windows."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

WINDOWS = ((1045, 1179), (1250, 1344))
STATE_NAMES = {0: "EMPTY", 1: "ACTIVE", 2: "INACTIVE", 3: "TENTATIVE",
               4: "PROJECTED", 5: "QUASIACTIVE"}


def audit(path: Path, camera: str) -> dict[int, dict]:
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("record") != "frame" or row.get("stage") not in {"pgie", "tracker"}:
            continue
        if row.get("mapped_camera_id") != camera:
            continue
        frame = int(row["frame_num"])
        if frame not in rows:
            rows[frame] = {"pts": row.get("pts"), "pgie": [], "tracker": []}
        key = "pgie" if row["stage"] == "pgie" else "tracker"
        rows[frame][key] = [
            {"native_track_id": obj.get("object_id"),
             "confidence": obj.get("confidence"),
             "tracker_confidence": obj.get("tracker_confidence"),
             "bbox": obj.get("bbox")}
            for obj in row.get("objects", [])
            if obj.get("class_id") == 0
            and (key == "pgie" or int(obj.get("object_id", -1)) >= 0)
        ]
    return rows


def past_states(path: Path, camera: str) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["camera_id"] != camera:
                continue
            frame = int(row["frame"])
            rows[frame].append({
                "native_track_id": row["native_track_id"],
                "state": row["nvdcf_state"], "age": int(row["track_age"]),
                "tracker_confidence": float(row["tracker_confidence"]),
                "visibility": float(row["visibility"]),
                "bbox": [float(row[k]) for k in ("left", "top", "width", "height")],
                "source": "NVDS_TRACKER_PAST_FRAME_META",
            })
    return rows


def shadow_states(directory: Path, camera: str) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = defaultdict(list)
    source_index = {"CAM-01": 0, "CAM-04": 1}[camera]
    public_csv = directory / "tracker_shadow_meta.csv"
    if public_csv.is_file():
        with public_csv.open() as stream:
            for row in csv.DictReader(stream):
                if row["camera_id"] != camera:
                    continue
                frame = int(row["frame"])
                rows[frame].append({
                    "native_track_id": int(row["native_track_id"]),
                    "age": int(row["age"]), "state": row["state"],
                    "tracker_confidence": float(row["tracker_confidence"]),
                    "visibility": float(row["visibility"]),
                    "bbox": [float(row[k]) for k in ("left", "top", "width", "height")],
                    "source": "NVDS_TRACKER_SHADOW_LIST_META",
                })
        return rows
    shadow_directory = directory / "shadow" if (directory / "shadow").is_dir() else directory
    if not shadow_directory.is_dir():
        return rows
    for path in sorted(shadow_directory.glob("*.txt")):
        match = re.fullmatch(r"\d+_(\d+)_(\d+)\.txt", path.name)
        if not match or int(match[1]) != source_index:
            continue
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 19:
                raise ValueError(f"unexpected DeepStream shadow sample in {path}: {len(fields)} fields")
            left, top, right, bottom = map(float, fields[5:9])
            rows[int(fields[0])].append({
                "native_track_id": int(fields[1]),
                "age": None,
                "state": STATE_NAMES.get(int(fields[17]), f"UNKNOWN_{fields[17]}"),
                "tracker_confidence": float(fields[16]),
                "visibility": float(fields[18]),
                "bbox": [left, top, right-left, bottom-top],
                "source": "DEEPSTREAM_SHADOW_TRACK_DUMP",
            })
    return rows


def build_report(v13_dir: Path, type1_dir: Path, type2_dir: Path, output: Path) -> dict:
    files = {
        "v13": v13_dir / "frame_path_audit.jsonl",
        "type1": type1_dir / "frame_path_audit.jsonl",
        "type2": type2_dir / "frame_path_audit.jsonl",
    }
    if v13_dir.name == "probe":
        files["v13"] = v13_dir / "frame_path_audit.jsonl"
    reports = {}
    for camera in ("CAM-01", "CAM-04"):
        datasets = {key: audit(path, camera) for key, path in files.items()}
        states = {
            "v13": past_states(v13_dir / "tracker_state.csv" if v13_dir.name == "probe" else v13_dir / "tracker_state.csv", camera),
            "type1": past_states(type1_dir / "tracker_state.csv", camera),
            "type2": past_states(type2_dir / "tracker_state.csv", camera),
        }
        shadows = {
            "v13": shadow_states(v13_dir, camera),
            "type1": shadow_states(type1_dir, camera),
            "type2": shadow_states(type2_dir, camera),
        }
        frames = []
        for start, end in WINDOWS if camera == "CAM-04" else ((1480, 1500), (1308, 1330)):
            for frame in range(start, end + 1):
                record = {"camera_id": camera, "frame": frame, "window": [start, end]}
                for key in ("v13", "type1", "type2"):
                    sample = datasets[key].get(frame, {})
                    record[f"{key}_pts"] = sample.get("pts")
                    record[f"{key}_pgie"] = sample.get("pgie", [])
                    record[f"{key}_tracker_output"] = sample.get("tracker", [])
                    record[f"{key}_past_frame_state"] = states[key].get(frame, [])
                    record[f"{key}_shadow_records"] = shadows[key].get(frame, [])
                    record[f"{key}_state_status"] = (
                        "OBSERVED" if states[key].get(frame) else "NOT_EXPOSED_FOR_THIS_FRAME"
                    )
                frames.append(record)
        reports[camera] = frames
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                               for camera_rows in reports.values() for row in camera_rows))
    summary = {
        "output_jsonl": str(output),
        "windows": {camera: [[r[0], r[1]] for r in (WINDOWS if camera == "CAM-04" else ((1480, 1500), (1308, 1330)))]
                    for camera in reports},
        "frames": {camera: len(rows) for camera, rows in reports.items()},
        "state_semantics": "only exact-frame public past-frame/shadow samples are named; missing samples are not inferred",
        "v13_shadow_dump_present": (v13_dir / "shadow").is_dir(),
        "type1_shadow_dump_present": (type1_dir / "shadow").is_dir(),
        "type2_shadow_dump_present": (type2_dir / "shadow").is_dir(),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v13", type=Path, required=True)
    parser.add_argument("--type1", type=Path, required=True)
    parser.add_argument("--type2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.v13, args.type1, args.type2, args.output), indent=2))


if __name__ == "__main__":
    main()
