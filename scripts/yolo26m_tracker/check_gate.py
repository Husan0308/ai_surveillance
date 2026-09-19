#!/usr/bin/env python3
"""Assess the six-camera YOLO26m + NvDCF 60-second tracker gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.yolo26m_person.check_gate import (
    analyze_overlaps,
    assess as assess_detector_runtime,
    load_jsonl_evidence,
)

NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_tracker_rows(text: str, cid: str):
    rows = []
    prefix = f"{cid} TRACK "
    for line in text.splitlines():
        if line.startswith(prefix):
            rows.append({k: float(v) for k, v in re.findall(rf"(\w+)=({NUMBER_RE})", line)})
    return rows


def analyze_tracks(records):
    per_source = {}
    duplicate_frame_ids = 0
    seen_frame_ids = set()
    tracks = {}

    for r in records:
        sid = int(r["source_id"])
        frame = int(r["frame"])
        oid = int(r["object_id"])
        key = (sid, frame, oid)
        if key in seen_frame_ids:
            duplicate_frame_ids += 1
        seen_frame_ids.add(key)

        t = tracks.setdefault((sid, oid), {"frames": set(), "first": frame, "last": frame})
        t["frames"].add(frame)
        t["first"] = min(t["first"], frame)
        t["last"] = max(t["last"], frame)

    for (sid, oid), t in tracks.items():
        observations = len(t["frames"])
        span = t["last"] - t["first"] + 1
        density = observations / span if span > 0 else 0.0
        src = per_source.setdefault(sid, {
            "records": 0,
            "unique_ids": 0,
            "tracks_ge_20_frames": 0,
            "tracks_ge_100_frames": 0,
            "longest_observations": 0,
            "longest_span_frames": 0,
            "best_density": 0.0,
        })
        src["unique_ids"] += 1
        src["records"] += observations
        src["longest_observations"] = max(src["longest_observations"], observations)
        src["longest_span_frames"] = max(src["longest_span_frames"], span)
        src["best_density"] = max(src["best_density"], density)
        if observations >= 20:
            src["tracks_ge_20_frames"] += 1
        if observations >= 100:
            src["tracks_ge_100_frames"] += 1

    for sid in range(6):
        per_source.setdefault(sid, {
            "records": 0,
            "unique_ids": 0,
            "tracks_ge_20_frames": 0,
            "tracks_ge_100_frames": 0,
            "longest_observations": 0,
            "longest_span_frames": 0,
            "best_density": 0.0,
        })

    return {
        "duplicate_frame_ids": duplicate_frame_ids,
        "per_source": {f"CAM-{sid+1:02d}": per_source[sid] for sid in range(6)},
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    out = args.directory
    text = (out / "pipeline.log").read_text()

    report = assess_detector_runtime(text)
    failures = report["failures"]

    if "tracker=1" not in text or "->nvtracker(" not in text:
        failures.append("Missing NvDCF tracker graph evidence")
    if "TRACKER_GRAPH nvtracker(NvDCF_stable_person,width=960,height=544,batch=6,gpu=0,reid=0)" not in text:
        failures.append("Missing frozen NvDCF stable-person/no-ReID profile evidence")

    tracker_log = {}
    for i in range(1, 7):
        cid = f"CAM-{i:02d}"
        rows = parse_tracker_rows(text, cid)
        if not rows:
            failures.append(f"{cid}: missing tracker telemetry")
            continue
        last = rows[-1]
        tracker_log[cid] = {
            "frames": int(last.get("frames", 0)),
            "objects": int(last.get("objects", 0)),
            "untracked": int(last.get("untracked", 0)),
            "duplicate_ids": int(last.get("duplicate_ids", 0)),
            "unique_ids": int(last.get("unique_ids", 0)),
        }
        if last.get("frames", 0) < 900:
            failures.append(f"{cid}: insufficient tracked frames")
        if last.get("untracked", 0) != 0:
            failures.append(f"{cid}: untracked person metadata survived NvDCF")
        if last.get("duplicate_ids", 0) != 0:
            failures.append(f"{cid}: duplicate object ID appeared in one frame")
    report["tracker_log"] = tracker_log

    preview = json.loads((out / "preview.json").read_text())
    if not preview.get("enabled") or not preview.get("alive_at_end"):
        failures.append("Preview unavailable or exited")

    with (out / "gpu.csv").open() as f:
        gpu_rows = list(csv.DictReader(f))
    if not gpu_rows:
        failures.append("Missing GPU telemetry")
        gpu = []
    else:
        start = float(gpu_rows[0]["time"])
        gpu = []
        proc = []
        for r in gpu_rows:
            base = {
                k: float(r[k])
                for k in ("time", "memory_used_mib", "gpu_pct", "decoder_pct", "encoder_pct")
            }
            gpu.append(base)
            raw = (r.get("process_memory_used_mib") or "").strip()
            if raw:
                proc.append((base["time"] - start, float(raw)))
        steady_proc = [v for elapsed, v in proc if elapsed >= 30]
        if len(steady_proc) < 5:
            failures.append("Insufficient process-specific tracker VRAM evidence")
        elif max(steady_proc) - min(steady_proc) > 128:
            failures.append(
                f"NvDCF steady process VRAM spread exceeds 128 MiB: "
                f"{max(steady_proc)-min(steady_proc):.3f} MiB"
            )
        report["tracker_gpu"] = {
            "device_memory_used_mib": {
                "min": min(r["memory_used_mib"] for r in gpu),
                "max": max(r["memory_used_mib"] for r in gpu),
                "mean": statistics.mean(r["memory_used_mib"] for r in gpu),
            },
            "process_memory_samples": len(proc),
            "steady_after_sec": 30,
            "process_steady_min_mib": min(steady_proc) if steady_proc else None,
            "process_steady_max_mib": max(steady_proc) if steady_proc else None,
            "process_steady_spread_mib": (
                max(steady_proc) - min(steady_proc) if steady_proc else None
            ),
        }

    video = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,nb_read_packets",
            "-show_entries", "format=duration", "-of", "json",
            str(out / "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"),
        ],
        capture_output=True, text=True, timeout=60,
    )
    try:
        v = json.loads(video.stdout)
        st = v["streams"][0]
        assert video.returncode == 0 and float(v["format"]["duration"]) >= 58
        assert (st["width"], st["height"], st["codec_name"]) == (1920, 1620, "h264")
        assert abs(int(st["nb_read_packets"]) - report["output_frames"]) <= 16
        report["recording"] = v
    except (ValueError, KeyError, IndexError, AssertionError, json.JSONDecodeError):
        failures.append("Finalized tracker recording validation failed")

    detections = load_jsonl_evidence(out / "detections.jsonl")
    if detections["malformed"]:
        failures.append("Malformed detector JSONL during tracker run")
    overlap = analyze_overlaps(detections["records"], nms_threshold=0.45)
    report["detector_overlap_validation"] = {
        k: v for k, v in overlap.items() if k != "evidence"
    }
    if overlap["pairs_over_nms_threshold"]:
        failures.append("Detector duplicate boxes survived NMS during tracker run")

    tracks = load_jsonl_evidence(out / "tracks.jsonl")
    if tracks["malformed"]:
        failures.append("Malformed tracker JSONL evidence")
    records = tracks["records"]
    if not records:
        failures.append("Missing tracked-object evidence")

    for r in records:
        vals = [
            float(r["detector_confidence"]),
            float(r["tracker_confidence"]),
            *[float(x) for x in r["box"]],
        ]
        if int(r["source_id"]) not in range(6) or int(r["object_id"]) < 0:
            failures.append("Invalid tracker source/object ID metadata")
            break
        if not all(math.isfinite(x) for x in vals):
            failures.append("Non-finite tracker metadata")
            break
        if float(r["box"][2]) <= 0 or float(r["box"][3]) <= 0:
            failures.append("Invalid tracker bbox geometry")
            break

    analysis = analyze_tracks(records)
    report["track_analysis"] = analysis
    if analysis["duplicate_frame_ids"]:
        failures.append("Duplicate tracker ID record exists within one source/frame")

    for cid, stats in analysis["per_source"].items():
        objects = tracker_log.get(cid, {}).get("objects", 0)
        if objects >= 20 and stats["tracks_ge_20_frames"] == 0:
            failures.append(f"{cid}: no track persisted for at least 20 observed frames")

    visual_path = out / "tracker_visual_review.json"
    if not visual_path.exists():
        failures.append("Missing separate NvDCF visual ID-stability review")
    else:
        visual = json.loads(visual_path.read_text())
        report["tracker_visual_review"] = visual
        recording = out / "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"
        if visual.get("status") != "PASS" or visual.get("recording_sha256") != sha256(recording):
            failures.append("NvDCF visual review did not pass for this recording")

    report["status"] = "BLOCKED" if failures else "PASS"
    (out / "tracker_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
