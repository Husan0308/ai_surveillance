#!/usr/bin/env python3
"""Summarize one CAM-01/CAM-04 source startup run."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STAGES = ("nvstreammux", "pgie", "tracker")

def load_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    probe = run / "run/logs/probe"
    audit = list(load_jsonl(probe / "frame_path_audit.jsonl"))
    events = list(load_jsonl(probe / "source_health.jsonl"))
    readiness = {}
    try:
        readiness = json.loads((probe / "readiness.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    first = {}
    counts = defaultdict(lambda: Counter())
    pairs = defaultdict(Counter)
    for row in audit:
        if row.get("record") != "frame":
            continue
        stage = str(row.get("stage"))
        camera = str(row.get("mapped_camera_id"))
        source = int(row.get("source_id", -1))
        pad = int(row.get("pad_index", -1))
        key = (camera, stage)
        counts[key]["frames"] += 1
        pairs[(source, pad, camera)]["frames"] += 1
        if row.get("bInferDone") is True:
            counts[key]["bInferDone"] += 1
        objects = row.get("objects") or []
        persons = [obj for obj in objects if str(obj.get("label", "")).lower() == "person" or int(obj.get("class_id", -1)) == 0]
        counts[key]["person_frames"] += bool(persons)
        counts[key]["person_detections"] += len(persons)
        first.setdefault(key, {"frame_num": row.get("frame_num"), "pts": row.get("pts"), "ntp_timestamp": row.get("ntp_timestamp")})
        if persons:
            first.setdefault((camera, "first_person_detection"), {"frame_num": row.get("frame_num"), "pts": row.get("pts"), "ntp_timestamp": row.get("ntp_timestamp")})
    log_text = ""
    try:
        log_text = (run / "logs/deepstream.log").read_text(errors="replace")
    except OSError:
        pass
    warning_lines = [line.strip() for line in log_text.splitlines() if re.search(r"(WARN|warning|ERROR|error|EOS|reconnect|state change)", line)]
    result = {
        "run": str(run),
        "readiness": readiness,
        "first": {f"{camera}:{stage}": value for (camera, stage), value in first.items()},
        "stage_counts": {f"{camera}:{stage}": dict(value) for (camera, stage), value in sorted(counts.items())},
        "source_pad_counts": {f"source={source}/pad={pad}/{camera}": dict(value) for (source, pad, camera), value in sorted(pairs.items())},
        "source_health_events": events,
        "warning_error_eos_lines": warning_lines[-200:],
        "pass_within_startup_gate": all(
            counts[(camera, stage)]["frames"] > 0
            for camera in ("CAM-01", "CAM-04") for stage in STAGES
        ) and bool(readiness.get("ready")),
    }
    output = run / "startup_report.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass_within_startup_gate"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
