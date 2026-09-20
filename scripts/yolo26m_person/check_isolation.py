#!/usr/bin/env python3
"""Validate one intentional camera isolation/recovery run with YOLO26m enabled."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.yolo26m_person.check_gate import (
    analyze_overlaps,
    load_jsonl_evidence,
)

SUMMARY_RE = re.compile(
    r"^GROUP ISOLATION_SUMMARY "
    r"target=(CAM-0[1-6]) healthy_peers=(\d+) outage_sec=(\d+) "
    r"peer_frames_during=(\d+) target_frames_during=(\d+) "
    r"target_recovered_frames=(\d+) status=(PASS|BLOCKED)$",
    re.MULTILINE,
)


def classify_runtime_diagnostics(text: str, summary: dict | None) -> tuple[list[str], list[str]]:
    diagnostic_re = re.compile(r"GROUP FATAL|CRITICAL|PARSER_ERROR|\\bERROR\\s")
    rows = text.splitlines()
    diagnostics = [(i, line) for i, line in enumerate(rows) if diagnostic_re.search(line)]
    if not diagnostics:
        return [], []

    target = summary.get("target") if summary else None
    if not target or summary.get("status") != "PASS":
        return [line for _, line in diagnostics], []

    start_marker = f"{target} ISOLATION setting only this source to NULL"
    resume_marker = f"{target} ISOLATION recreating only this nvurisrcbin"
    try:
        start_idx = next(i for i, line in enumerate(rows) if start_marker in line)
        resume_idx = next(i for i, line in enumerate(rows) if i > start_idx and resume_marker in line)
    except StopIteration:
        return [line for _, line in diagnostics], []

    benign_patterns = (
        re.compile(
            r"ERROR\\s+v4l2allocator\\b.*"
            r"<nvv4l2decoder\\d+:pool:src:allocator> "
            r"failed queueing buffer \\d+: Bad file descriptor$"
        ),
        re.compile(
            r"ERROR\\s+v4l2bufferpool\\b.*"
            r"<nvv4l2decoder\\d+:pool:src> could not queue a buffer \\d+$"
        ),
    )

    blocking: list[str] = []
    benign: list[str] = []
    for idx, line in diagnostics:
        in_intentional_teardown = start_idx < idx < resume_idx
        exact_benign = any(pattern.search(line) for pattern in benign_patterns)
        if in_intentional_teardown and exact_benign:
            benign.append(line)
        else:
            blocking.append(line)
    return blocking, benign


def assess(directory: Path) -> dict:
    text = (directory / "pipeline.log").read_text()
    failures: list[str] = []

    matches = list(SUMMARY_RE.finditer(text))
    summary = None
    if len(matches) != 1:
        failures.append(f"Expected exactly one isolation summary, found {len(matches)}")
    else:
        m = matches[0]
        summary = {
            "target": m.group(1),
            "healthy_peers": int(m.group(2)),
            "outage_sec": int(m.group(3)),
            "peer_frames_during": int(m.group(4)),
            "target_frames_during": int(m.group(5)),
            "target_recovered_frames": int(m.group(6)),
            "status": m.group(7),
        }
        if summary["status"] != "PASS":
            failures.append("Native isolation summary is BLOCKED")
        if summary["healthy_peers"] != 5:
            failures.append("Isolation did not retain five healthy peer sources")
        if summary["target_frames_during"] > 5:
            failures.append("Target source advanced unexpectedly while intentionally isolated")
        if summary["peer_frames_during"] < summary["outage_sec"] * 75:
            failures.append("Healthy peers did not advance enough during target outage")
        if summary["target_recovered_frames"] < 100:
            failures.append("Target did not recover at least 100 frames")

    if "inference=1" not in text or "nvinfer(YOLO26m,FP16,batch=6,interval=0)" not in text:
        failures.append("Missing YOLO26m DeepStream inference graph evidence")
    if "DeepStream-NMS(iou=0.45,conf=0.25)" not in text:
        failures.append("Missing validated DeepStream NMS=0.45 graph evidence")

    if not re.search(
        r"GROUP END .*fatal=0 shared_errors=0 shared_warnings=0 isolation=completed",
        text,
    ):
        failures.append("Missing clean completed-isolation group shutdown")

    runtime_error_diagnostics, benign_teardown_diagnostics = (
        classify_runtime_diagnostics(text, summary)
    )
    if runtime_error_diagnostics:
        failures.append("Fatal/runtime/parser error diagnostic present")

    for i in range(1, 7):
        cid = f"CAM-{i:02d}"
        end = re.search(
            rf"^{cid} END .*hardware_decoder=(\d+) errors=(\d+) warnings=(\d+) "
            rf"pts_backwards=(\d+) pts_duplicates=(\d+)$",
            text,
            re.MULTILINE,
        )
        if not end:
            failures.append(f"{cid}: missing final source completion")
            continue
        decoder, errors, warnings, backwards, duplicates = map(int, end.groups())
        if decoder != 1 or errors or warnings or backwards or duplicates:
            failures.append(f"{cid}: dirty final decoder/error/timestamp counters")

    yolo_rows = [
        {k: float(v) for k, v in re.findall(r"(\w+)=([-+]?\d+(?:\.\d+)?)", line)}
        for line in text.splitlines()
        if line.startswith("YOLO STATS ")
    ]
    if not yolo_rows:
        failures.append("Missing YOLO telemetry")
    else:
        last = yolo_rows[-1]
        for key in ("errors", "parser_errors", "parser_rejected"):
            if last.get(key, 1) != 0:
                failures.append(f"YOLO {key} is non-zero")
        if last.get("inferred_frames", 0) <= 0:
            failures.append("YOLO inference did not advance")

    preview_path = directory / "preview.json"
    if not preview_path.exists():
        failures.append("Missing preview evidence")
    else:
        preview = json.loads(preview_path.read_text())
        if not preview.get("enabled") or not preview.get("alive_at_end"):
            failures.append("Realtime preview did not stay alive")

    jsonl = load_jsonl_evidence(directory / "detections.jsonl")
    records = jsonl["records"]
    if jsonl["malformed"]:
        failures.append("Malformed detection JSONL evidence")
    if not records:
        failures.append("Missing person metadata evidence")

    overlap = analyze_overlaps(records, nms_threshold=0.45) if records else {
        "nms_iou_threshold": 0.45,
        "pairs_over_nms_threshold": 0,
        "pairs_iou_gte_090": 0,
        "pairs_iou_gte_095": 0,
        "max_person_iou": 0.0,
        "worst_pair": None,
        "evidence": [],
    }
    if overlap["pairs_over_nms_threshold"]:
        failures.append("Duplicate person boxes survived NMS IoU=0.45")

    report = {
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "isolation": summary,
        "jsonl_rows": len(records),
        "runtime_error_diagnostics": runtime_error_diagnostics,
        "benign_teardown_diagnostics": benign_teardown_diagnostics,
        "overlap_validation": {k: v for k, v in overlap.items() if k != "evidence"},
    }
    (directory / "isolation_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    report = assess(args.directory)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
