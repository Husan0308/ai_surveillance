#!/usr/bin/env python3
"""Diagnose detector->NvDCF retention losses from an archived A/B replay.

This is read-only. It consumes the per-frame audit JSONL plus tracker_state.csv
already emitted by the native diagnostic run. It does not edit production or
runtime configs.

The confidence-multiset matcher is intentionally a diagnostic proxy: for
matched detector-backed tracker objects DeepStream preserves the detector
confidence in object metadata, so same-frame detector/tracker confidences can be
paired without needing private NvDCF association scores. Shadow-only tracker
objects do not consume detector proposals in this matcher.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
AB_ROOT = ROOT / ".runtime/final-detector-ab-20261002-uSpPqjLY"


def _q(value: object) -> float:
    return round(float(value), 6)


def match_by_confidence(pgie_objects: list[dict], tracker_objects: list[dict]) -> tuple[int, list[float]]:
    """Return (matched_count, unmatched_detector_confidences)."""
    available = Counter()
    for obj in tracker_objects:
        try:
            conf = _q(obj.get("confidence", -1.0))
        except (TypeError, ValueError):
            continue
        if conf >= 0.0:
            available[conf] += 1

    matched = 0
    unmatched: list[float] = []
    for obj in pgie_objects:
        try:
            conf = _q(obj.get("confidence", -1.0))
        except (TypeError, ValueError):
            continue
        if available[conf] > 0:
            available[conf] -= 1
            matched += 1
        else:
            unmatched.append(conf)
    return matched, unmatched


def contiguous_windows(frames: Iterable[int]) -> list[dict]:
    seq = sorted(set(int(x) for x in frames))
    if not seq:
        return []
    out: list[dict] = []
    start = prev = seq[0]
    for frame in seq[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        out.append({"start": start, "end": prev, "frames": prev - start + 1})
        start = prev = frame
    out.append({"start": start, "end": prev, "frames": prev - start + 1})
    return out


def resolve_run(label: str, explicit: Path | None) -> Path:
    if explicit is not None:
        run = explicit.resolve()
        if not run.is_dir():
            raise SystemExit(f"run directory not found: {run}")
        return run

    label_dir = AB_ROOT / "runs" / label
    candidates = sorted(p for p in label_dir.glob("replay-*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no replay directories found under: {label_dir}")
    return candidates[-1]


def load_audit(path: Path, camera_filter: set[str] | None) -> dict[str, dict[int, dict[str, list[dict]]]]:
    frames: dict[str, dict[int, dict[str, list[dict]]]] = defaultdict(lambda: defaultdict(dict))
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if rec.get("record") != "frame":
                continue
            stage = rec.get("stage")
            if stage not in {"pgie", "tracker"}:
                continue
            cam = str(rec.get("mapped_camera_id") or "")
            if not cam:
                continue
            if camera_filter and cam not in camera_filter:
                continue
            try:
                frame = int(rec["frame_num"])
            except (KeyError, TypeError, ValueError):
                continue
            frames[cam][frame][stage] = list(rec.get("objects") or [])
    return frames


def load_tracker_state(path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    out: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    if not path.is_file():
        return out
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cam = row.get("camera_id") or ""
            try:
                frame = int(float(row.get("frame") or "nan"))
            except ValueError:
                continue
            out[(cam, frame)].append(row)
    return out


def band(conf: float) -> str:
    if conf < 0.50:
        return "below_0.50"
    if conf < 0.70:
        return "0.50_to_0.70"
    return "at_least_0.70"


def analyze_camera(
    camera: str,
    frame_map: dict[int, dict[str, list[dict]]],
    tracker_state: dict[tuple[str, int], list[dict[str, str]]],
) -> dict:
    detector_total = matched_total = 0
    unmatched_frames: list[int] = []
    unmatched_by_band = Counter()
    detector_by_band = Counter()
    top_frames: list[dict] = []
    inactive_overlap_frames = 0

    for frame in sorted(frame_map):
        stages = frame_map[frame]
        pgie = stages.get("pgie", [])
        tracker = stages.get("tracker", [])
        if not pgie:
            continue

        detector_total += len(pgie)
        for obj in pgie:
            try:
                detector_by_band[band(float(obj.get("confidence", -1.0)))] += 1
            except (TypeError, ValueError):
                pass

        matched, unmatched = match_by_confidence(pgie, tracker)
        matched_total += matched
        if not unmatched:
            continue

        unmatched_frames.append(frame)
        for conf in unmatched:
            unmatched_by_band[band(conf)] += 1

        states = tracker_state.get((camera, frame), [])
        inactive = [
            {
                "native_track_id": r.get("native_track_id"),
                "tracker_confidence": r.get("tracker_confidence"),
                "track_age": r.get("track_age"),
            }
            for r in states
            if (r.get("nvdcf_state") or "").upper() == "INACTIVE"
        ]
        if inactive:
            inactive_overlap_frames += 1

        top_frames.append(
            {
                "frame": frame,
                "pgie_count": len(pgie),
                "tracker_count": len(tracker),
                "unmatched_count": len(unmatched),
                "unmatched_confidences": unmatched,
                "inactive_state_rows": inactive,
            }
        )

    windows = contiguous_windows(unmatched_frames)
    windows.sort(key=lambda x: (-x["frames"], x["start"]))
    top_frames.sort(key=lambda x: (-x["unmatched_count"], x["frame"]))

    band_summary = {}
    for name in ("below_0.50", "0.50_to_0.70", "at_least_0.70"):
        det = detector_by_band[name]
        miss = unmatched_by_band[name]
        band_summary[name] = {
            "detector_proposals": det,
            "unmatched_proxy": miss,
            "matched_proxy": det - miss,
            "proxy_retention_percent": (100.0 * (det - miss) / det) if det else None,
        }

    return {
        "camera": camera,
        "detector_proposals": detector_total,
        "matched_by_confidence_proxy": matched_total,
        "unmatched_by_confidence_proxy": detector_total - matched_total,
        "proxy_retention_percent": (100.0 * matched_total / detector_total) if detector_total else None,
        "frames_with_unmatched": len(unmatched_frames),
        "frames_with_unmatched_and_inactive_state_row": inactive_overlap_frames,
        "unmatched_by_confidence_band": band_summary,
        "longest_unmatched_windows": windows[:20],
        "top_unmatched_frames": top_frames[:40],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="nvdcf-retention-v6-newtarget035")
    ap.add_argument("--run", type=Path)
    ap.add_argument("--camera", action="append", dest="cameras")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    run = resolve_run(args.label, args.run)
    probe = run / "run/logs/probe"
    audit = probe / "frame_path_audit.jsonl"
    state_csv = probe / "tracker_state.csv"
    if not audit.is_file():
        raise SystemExit(f"frame audit not found: {audit}")

    camera_filter = set(args.cameras) if args.cameras else None
    frames = load_audit(audit, camera_filter)
    states = load_tracker_state(state_csv)

    report = {
        "run": str(run),
        "method": "same-frame detector/tracker confidence multiset proxy",
        "note": (
            "Diagnostic only: this does not replace the canonical acceptance analyzer. "
            "Use it to localize which detector confidence bands and frames remain unmatched."
        ),
        "tracker_state_csv_present": state_csv.is_file(),
        "cameras": {
            cam: analyze_camera(cam, frame_map, states)
            for cam, frame_map in sorted(frames.items())
        },
    }

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
