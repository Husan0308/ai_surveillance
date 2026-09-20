#!/usr/bin/env python3
"""Validate one intentional camera isolation/recovery run with YOLO26m + NvDCF."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.yolo26m_person.check_isolation import assess as assess_detector_isolation
from scripts.yolo26m_person.check_gate import load_jsonl_evidence
from scripts.yolo26m_tracker.check_gate import analyze_tracks, parse_tracker_rows

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    out = args.directory
    text = (out / "pipeline.log").read_text()

    base = assess_detector_isolation(out)
    failures = list(base["failures"])

    if "tracker=1" not in text or "->nvtracker(" not in text:
        failures.append("Missing NvDCF tracker graph evidence")
    if "TRACKER_GRAPH nvtracker(NvDCF_stable_person,width=960,height=544,batch=6,gpu=0,reid=0)" not in text:
        failures.append("Missing frozen NvDCF stable-person/no-ReID profile evidence")

    tracker_log = {}
    for i in range(1, 7):
        cid = f"CAM-{i:02d}"
        rows = parse_tracker_rows(text, cid)
        if not rows:
            failures.append(f"{cid}: missing NvDCF telemetry")
            continue
        last = rows[-1]
        tracker_log[cid] = {
            "frames": int(last.get("frames", 0)),
            "objects": int(last.get("objects", 0)),
            "untracked": int(last.get("untracked", 0)),
            "duplicate_ids": int(last.get("duplicate_ids", 0)),
            "unique_ids": int(last.get("unique_ids", 0)),
        }
        if last.get("frames", 0) <= 0:
            failures.append(f"{cid}: tracker frame counter did not advance")
        if last.get("untracked", 0) != 0:
            failures.append(f"{cid}: untracked person metadata survived NvDCF")
        if last.get("duplicate_ids", 0) != 0:
            failures.append(f"{cid}: duplicate tracker ID appeared in one frame")

    tracks = load_jsonl_evidence(out / "tracks.jsonl")
    if tracks["malformed"]:
        failures.append("Malformed tracker JSONL evidence")
    records = tracks["records"]
    analysis = analyze_tracks(records) if records else {
        "duplicate_frame_ids": 0,
        "per_source": {},
    }
    if analysis["duplicate_frame_ids"]:
        failures.append("Duplicate tracker ID record exists within one source/frame")

    target = (base.get("isolation") or {}).get("target")
    target_sid = int(target[-2:]) - 1 if target else None

    # tracking-id-reset-mode=1 intentionally allows a new local ID for the
    # interrupted stream after GST_NVEVENT_STREAM_RESET. Do not require ID
    # continuity across the target outage. Instead, ensure any post-recovery
    # tracker evidence is structurally valid and peers keep producing tracks
    # when people are present.
    per_source_records = {sid: 0 for sid in range(6)}
    for row in records:
        sid = int(row["source_id"])
        if sid in per_source_records:
            per_source_records[sid] += 1

    for sid in range(6):
        cid = f"CAM-{sid+1:02d}"
        objects = tracker_log.get(cid, {}).get("objects", 0)
        if objects > 0 and per_source_records[sid] == 0:
            failures.append(f"{cid}: tracker objects reported but tracks.jsonl has no records")

    report = {
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "isolation": base.get("isolation"),
        "tracker_log": tracker_log,
        "track_analysis": analysis,
        "target_source_id": target_sid,
        "target_id_continuity_required": False,
        "target_reset_semantics": "tracking-id-reset-mode=1 allows new IDs after stream reset",
        "runtime_error_diagnostics": base.get("runtime_error_diagnostics", []),
        "benign_teardown_diagnostics": base.get("benign_teardown_diagnostics", []),
        "detector_overlap_validation": base.get("overlap_validation"),
    }
    (out / "tracker_isolation_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
