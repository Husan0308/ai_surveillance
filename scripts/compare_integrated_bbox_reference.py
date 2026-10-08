#!/usr/bin/env python3
"""Compare integrated local6 outputs with the human-approved v4 box reference."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runtime/integrated-bbox-reference-replay-20261005"
APPROVED = ROOT / ".runtime/sixcam-local6-20261004-b/shadow-assisted-detector-ab-v4"


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def box(value):
    if not value:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if len(value) != 4:
        return None
    x, y, w, h = map(float, value)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pctl(values, pct):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * pct / 100
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] * (hi - index) + values[hi] * (index - lo)


def contiguous_miss(missed_frames):
    best = cur = prev = 0
    for frame in sorted(set(missed_frames)):
        cur = cur + 1 if frame == prev + 1 else 1
        best = max(best, cur)
        prev = frame
    return best


def analyze_camera(camera, run_dir, mux_width=2560, mux_height=1440):
    expected_by_frame = defaultdict(list)
    reference_pts_by_frame = {}
    ref_path = APPROVED / f"{camera.replace('-', '')}-per-frame.csv"
    with ref_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            b = box(row.get("chosen_bbox"))
            if b:
                reference_pts_by_frame[int(row["frame"])] = int(row["pts"])
                expected_by_frame[int(row["frame"])].append({
                    "box": b,
                    "pgie_box": box(row.get("pgie_bbox")),
                    "source": row.get("chosen_source", ""),
                    "native_id": row.get("chosen_native_track_id", ""),
                    "v13_iou": float(row["v13_iou"]) if row.get("v13_iou") else None,
                })
    candidate_by_frame = defaultdict(list)
    raw_tracker_by_frame = defaultdict(list)
    # source slots 0 and 3 are the original CAM-01/CAM-04 files; slots 1/2
    # and 4/5 are duplicated batch-fill diagnostics only.
    source_id = {"CAM-01": 0, "CAM-04": 3}[camera]
    sx, sy = 1920.0 / mux_width, 1080.0 / mux_height
    for row in read_jsonl(run_dir / "tracks.jsonl"):
        if int(row.get("source_id", -1)) == source_id:
            b = box(row.get("box"))
            if b and 0 <= int(row["frame"]) < 2400:
                raw_tracker_by_frame[int(row["frame"])].append(b)
                candidate_by_frame[int(row["frame"])].append({
                    "box": (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), "native_id": row.get("object_id"),
                    "tracker_confidence": row.get("tracker_confidence"),
                    "detector_confidence": row.get("detector_confidence"),
                    "source": "TRACKER",
                })
    det_frames = set()
    detector_boxes_by_frame = Counter()
    detector_conf_by_frame = defaultdict(list)
    detector_by_frame = defaultdict(list)
    for row in read_jsonl(run_dir / "detections.jsonl"):
        if int(row.get("source_id", -1)) == source_id and 0 <= int(row["frame"]) < 2400:
            det_frames.add(int(row["frame"]))
            detector_boxes_by_frame[int(row["frame"])] += 1
            detected_box = box(row.get("box"))
            if detected_box:
                detector_by_frame[int(row["frame"])].append(detected_box)
            if isinstance(row.get("confidence"), (float, int)):
                detector_conf_by_frame[int(row["frame"])].append(float(row["confidence"]))
    fallback_frames = set()
    fallback_rows = 0
    fallback_rows_reference = 0
    fallback_ids = set()
    fallback_by_frame = Counter()
    for row in read_jsonl(run_dir / "shadow_fallbacks.jsonl"):
        if row.get("camera_id") == camera and 0 <= int(row["frame_num"]) < 2400:
            fallback_rows += 1
            fallback_by_frame[int(row["frame_num"])] += 1
            fallback_frames.add(int(row["frame_num"]))
            fallback_ids.add(row.get("matched_shadow_native_id"))
            b = box(row.get("detector_bbox"))
            if b:
                if int(row["frame_num"]) in expected_by_frame:
                    fallback_rows_reference += 1
                candidate_by_frame[int(row["frame_num"])].append({
                    "box": (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy),
                    "native_id": row.get("matched_shadow_native_id"),
                    "tracker_confidence": row.get("shadow_confidence"),
                    "detector_confidence": row.get("detector_confidence"),
                    "source": "PGIE_SHADOW_FALLBACK",
                })

    decision_by_frame = defaultdict(list)
    decision_path = run_dir / "tracker_decisions.jsonl"
    if decision_path.exists():
        for row in read_jsonl(decision_path):
            if int(row.get("source_id", -1)) == source_id and 0 <= int(row.get("frame", -1)) < 2400:
                decision_by_frame[int(row["frame"])].append(row)
    invalid_by_frame = set()
    invalid_path = run_dir / "invalid_shadow_geometry.jsonl"
    if invalid_path.exists():
        for row in read_jsonl(invalid_path):
            if int(row.get("source_id", -1)) == source_id:
                invalid_by_frame.add(int(row.get("frame", -1)))
    approved_fallback_disposition = Counter()
    approved_fallback_ids = []
    matched_reference_id_by_frame = defaultdict(list)

    all_frames = sorted(expected_by_frame)
    expected_boxes = sum(map(len, expected_by_frame.values()))
    rendered_boxes = sum(map(len, (candidate_by_frame[f] for f in all_frames)))
    matched = []
    misses = []
    false_candidates = []
    matched_frames = set()
    for frame in all_frames:
        refs = expected_by_frame[frame]
        cands = candidate_by_frame.get(frame, [])
        pairs = sorted(((iou(r["box"], c["box"]), ri, ci)
                        for ri, r in enumerate(refs) for ci, c in enumerate(cands)
                        if iou(r["box"], c["box"]) > 0.0), reverse=True)
        used_r, used_c = set(), set()
        matched_candidate_for_reference = {}
        for score, ri, ci in pairs:
            if ri in used_r or ci in used_c:
                continue
            used_r.add(ri); used_c.add(ci)
            matched_candidate_for_reference[ri] = cands[ci]
            matched.append(score)
            if score >= 0.80:
                matched_frames.add(frame)
        for ri, ref in enumerate(refs):
            if ri not in used_r:
                misses.append(frame)
            candidate = matched_candidate_for_reference.get(ri)
            if candidate and candidate.get("native_id") is not None:
                matched_reference_id_by_frame[frame].append(str(candidate["native_id"]))
            if ref.get("source") == "SHADOW_ID_PLUS_CURRENT_PGIE":
                if candidate and candidate["source"] == "TRACKER":
                    approved_fallback_disposition["candidate_ACTIVE"] += 1
                    if candidate.get("native_id") is not None:
                        approved_fallback_ids.append((frame, str(candidate["native_id"])))
                elif candidate and candidate["source"] == "PGIE_SHADOW_FALLBACK":
                    approved_fallback_disposition["candidate_FALLBACK"] += 1
                    if candidate.get("native_id") is not None:
                        approved_fallback_ids.append((frame, str(candidate["native_id"])))
                else:
                    pg = ref.get("pgie_box") or ref["box"]
                    decision = None
                    for possible in decision_by_frame.get(frame, []):
                        db = box(possible.get("detection", {}).get("bbox"))
                        if db and iou(pg, db) >= 0.30:
                            decision = possible
                            break
                    if frame in invalid_by_frame:
                        reason = "invalid_shadow"
                    elif decision and decision.get("outcome") == "AMBIGUOUS":
                        reason = "ambiguous"
                    elif decision and decision.get("outcome") in ("NO_CURRENT_SHADOW", "NO_INACTIVE_SHADOW"):
                        reason = "no_current_shadow"
                    elif decision and decision.get("outcome") == "SPATIAL_GATE_REJECT":
                        reason = "other_spatial_gate"
                    else:
                        reason = "candidate_NONE_other"
                    approved_fallback_disposition[reason] += 1
        for ci, cand in enumerate(cands):
            if ci not in used_c:
                false_candidates.append({"frame": frame, "bbox": cand["box"], "native_id": cand["native_id"]})
    expected_frames = set(all_frames)
    no_candidate_frames = [f for f in all_frames if not candidate_by_frame.get(f)]
    source_count = Counter(row["source"] for frame in all_frames for row in candidate_by_frame.get(frame, []))
    association_proposals = 0
    unmatched_detection_by_frame = Counter()
    for frame in sorted(set(detector_by_frame) | set(candidate_by_frame)):
        detections = detector_by_frame.get(frame, [])
        active = raw_tracker_by_frame.get(frame, [])
        pairs = sorted(((iou(d, t), di, ti) for di, d in enumerate(detections)
                        for ti, t in enumerate(active) if iou(d, t) >= 0.30), reverse=True)
        used_d, used_t = set(), set()
        for _, di, ti in pairs:
            if di not in used_d and ti not in used_t:
                used_d.add(di); used_t.add(ti)
        associated = len(used_d) + fallback_by_frame[frame]
        association_proposals += associated
        unmatched_detection_by_frame[frame] = max(0, len(detections) - associated)
    unmatched_frames = [f for f, count in unmatched_detection_by_frame.items() if count]
    candidate_pts = {}
    for row in read_jsonl(run_dir / "frames.jsonl"):
        if int(row.get("source_id", -1)) == source_id and int(row.get("frame", -1)) in expected_by_frame:
            candidate_pts[int(row["frame"])] = int(row["pts_ns"])
    pts_offsets_ms = [(candidate_pts[f] - reference_pts_by_frame[f]) / 1e6
                      for f in expected_by_frame.keys() & candidate_pts.keys()]
    warning_matches = re.findall(
        rf"CAM-{int(camera[-2:]):02d} TRACK .*?ambiguous=(\d+).*?invalid_shadow_geometry=(\d+)",
        (run_dir / "native.log").read_text(errors="replace"),
    )
    latest_stats = tuple(map(int, warning_matches[-1])) if warning_matches else None
    unmatched_approved = len(misses)
    report = {
        "camera_id": camera,
        "approved_reference_frames": len(expected_frames),
        "approved_reference_boxes": expected_boxes,
        "candidate_rendered_boxes_on_reference_frames": rendered_boxes,
        "candidate_person_positive_reference_frames": sum(bool(candidate_by_frame.get(f)) for f in all_frames),
        "candidate_detector_frames_on_reference_frames": sum(f in det_frames for f in all_frames),
        "candidate_detector_boxes_on_reference_frames": sum(detector_boxes_by_frame[f] for f in all_frames),
        "candidate_detector_confidence_p05_p50_p95": [
            pctl([conf for f in all_frames for conf in detector_conf_by_frame[f]], x) for x in (5, 50, 95)
        ],
        "matched_box_pairs": len(matched),
        "bbox_iou_p05_p50_p95_min": [pctl(matched, 5), median(matched) if matched else None,
                                      pctl(matched, 95), min(matched) if matched else None],
        "reference_boxes_iou_ge_0_80": sum(v >= 0.80 for v in matched),
        "reference_boxes_iou_ge_0_50": sum(v >= 0.50 for v in matched),
        "missed_approved_boxes": unmatched_approved,
        "missed_approved_frames": sorted(set(misses)),
        "longest_consecutive_missed_approved_frames": contiguous_miss(misses),
        "candidate_boxes_not_matching_approved_box": len(false_candidates),
        "candidate_unmatched_examples": false_candidates[:20],
        "active_target_box_count": sum(1 for frame in all_frames for row in candidate_by_frame.get(frame, [])
                                        if row["source"] == "TRACKER"),
        "fallback_box_count_on_reference_frames": fallback_rows_reference,
        "fallback_box_count_full_replay": fallback_rows,
        "fallback_frame_count_full_replay": len(fallback_frames),
        "fallback_native_ids": sorted(x for x in fallback_ids if x is not None),
        "candidate_box_sources_on_reference_frames": dict(source_count),
        "approved_v4_fallback_disposition": dict(approved_fallback_disposition),
        "approved_v4_fallback_disposition_total": sum(approved_fallback_disposition.values()),
        "candidate_native_ids_matched_to_approved_fallbacks": sorted(set(i for _, i in approved_fallback_ids)),
        "candidate_native_id_changes_across_adjacent_matched_reference_frames": sum(
            1 for a, b in zip(
                sorted((f, ids[0]) for f, ids in matched_reference_id_by_frame.items() if ids),
                sorted((f, ids[0]) for f, ids in matched_reference_id_by_frame.items() if ids)[1:]
            ) if b[0] == a[0] + 1 and b[1] != a[1]
        ),
        "ambiguous_match_counter_full_replay": latest_stats[0] if latest_stats else None,
        "invalid_shadow_geometry_counter_full_replay": latest_stats[1] if latest_stats else None,
        "candidate_minus_approved_pts_offset_ms_p05_p50_p95": [pctl(pts_offsets_ms, x) for x in (5, 50, 95)],
        "tracker_retention_vs_approved_boxes": (len(matched) / expected_boxes if expected_boxes else None),
        "detector_proposals_first_2400_frames": sum(detector_boxes_by_frame.values()),
        "detector_proposals_associated_to_active_or_approved_shadow_fallback_iou_ge_0_30": association_proposals,
        "proposal_to_tracker_or_fallback_retention_iou_ge_0_30": (association_proposals / sum(detector_boxes_by_frame.values())
                                                                     if sum(detector_boxes_by_frame.values()) else None),
        "unassociated_detector_proposals_first_2400_frames": sum(unmatched_detection_by_frame.values()),
        "longest_consecutive_frame_run_with_any_unassociated_detector_proposal": contiguous_miss(unmatched_frames),
        "strict_geometric_match_rate_iou_ge_0_80": (sum(v >= 0.80 for v in matched) / expected_boxes if expected_boxes else None),
        "frames_without_candidate_box": no_candidate_frames,
        "candidate_coordinate_normalization": f"candidate mux coordinates {mux_width}x{mux_height} scaled to pinned 1920x1080 source coordinates",
        "definition_note": "Box comparison is exact camera/frame greedy IoU matching after coordinate normalization; IoU >=0.80 is a strict geometry-equivalence diagnostic, not substituted for the approved policy. Unmatched candidate boxes mean deviation from approved output, not a ground-truth false positive. Missed approved box frames are box-level misses and do not imply every person in that frame was missing.",
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mux-width", type=int, default=2560)
    parser.add_argument("--mux-height", type=int, default=1440)
    args = parser.parse_args()
    result = {camera: analyze_camera(camera, args.run, args.mux_width, args.mux_height)
              for camera in ("CAM-01", "CAM-04")}
    output = args.output or args.run / "approved-reference-comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
