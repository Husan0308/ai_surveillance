#!/usr/bin/env python3
"""CPU-only replay of cross-ID canonical duplicate decisions from saved rows.

This does not execute or emulate YOLO/NvDCF. It consumes their recorded exact
frame outputs and verifier context proposals, then runs the documented
conservative canonical policy to evaluate positive/negative metadata fixtures.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / ".runtime/entrance-r1-history-duplicate-replay-20261009"
DEFAULT_OUT = ROOT / ".runtime/entrance-crossid-duplicate-candidate-20261010"


def rows(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_num}: invalid JSON: {exc}") from exc


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def valid(box: list[float], width: float, height: float) -> bool:
    return (len(box) == 4 and all(math.isfinite(v) for v in box) and
            box[0] >= 0 and box[1] >= 0 and box[2] > 0 and box[3] > 0 and
            box[0] + box[2] <= width + 0.1 and box[1] + box[3] <= height + 0.1)


def center_inside(inner: list[float], outer: list[float]) -> bool:
    cx, cy = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2
    return outer[0] <= cx <= outer[0] + outer[2] and outer[1] <= cy <= outer[1] + outer[3]


def map_context_to_frame(local: list[float], detector_bbox: list[float],
                         frame_width: float, frame_height: float,
                         surface_width: float, surface_height: float,
                         candidate_centered_crop: bool = False) -> list[float]:
    """Mirror shadow_context_compute_crop + shadow_context_to_frame."""
    dx, dy, dw, dh = detector_bbox
    if candidate_centered_crop:
        side = int(math.ceil(max(dw, dh) * 1.8))
        side = max(256, min(640, side, int(surface_width), int(surface_height))) & ~1
    else:
        side = min(int(round(surface_width / 3.0)), int(surface_height)) & ~1
    if side < 2:
        raise ValueError("invalid verifier crop")
    scale_x, scale_y = surface_width / frame_width, surface_height / frame_height
    center_x, center_y = (dx + dw / 2) * scale_x, (dy + dh / 2) * scale_y
    crop_x = int(round(center_x - side / 2))
    crop_y = int(round(center_y - side / 2))
    crop_x = max(0, min(int(surface_width) - side, crop_x)) & ~1
    crop_y = max(0, min(int(surface_height) - side, crop_y)) & ~1
    local_scale = 640.0 / side
    x, y, w, h = local
    return [(x / local_scale + crop_x) / scale_x,
            (y / local_scale + crop_y) / scale_y,
            w / local_scale / scale_x,
            h / local_scale / scale_y]


def admit(candidate_id: int, current: dict[int, list[float]], previous_ids: set[int],
          proposals: list[dict[str, Any]], width: float, height: float,
          frame_current: bool) -> dict[str, Any]:
    if not frame_current:
        return {"admit": False, "reason": "STALE_OR_MISSING_FRAME"}
    if len(current) == 0 or any(not valid(b, width, height) for b in current.values()):
        return {"admit": False, "reason": "INVALID_TRACK_GEOMETRY"}
    if candidate_id not in current or candidate_id in previous_ids:
        return {"admit": False, "reason": "NOT_A_NEW_ACTIVE_BIRTH"}
    if len(proposals) < 2 or any(not valid(p["bbox"], width, height) for p in proposals):
        return {"admit": False, "reason": "INSUFFICIENT_OR_INVALID_CONTEXT"}
    cand = current[candidate_id]
    eligible = []
    for owner_id in sorted(previous_ids & current.keys()):
        owner = current[owner_id]
        if owner_id == candidate_id or not center_inside(cand, owner):
            continue
        area_ratio = cand[2] * cand[3] / (owner[2] * owner[3])
        if not 0 < area_ratio <= 0.60:
            continue
        owner_scores = [iou(owner, p["bbox"]) for p in proposals]
        cand_scores = [iou(cand, p["bbox"]) for p in proposals]
        owner_best = max(range(len(owner_scores)), key=owner_scores.__getitem__)
        cand_best = max(range(len(cand_scores)), key=cand_scores.__getitem__)
        owner_runner = max((score for i, score in enumerate(owner_scores) if i != owner_best), default=0.0)
        cand_runner = max((score for i, score in enumerate(cand_scores) if i != cand_best), default=0.0)
        owner_iou = owner_scores[owner_best]
        cand_iou = cand_scores[owner_best]
        if (owner_best != cand_best or owner_iou < 0.65 or owner_iou - owner_runner < 0.10 or
                cand_scores[cand_best] - cand_runner < 0.10 or not 0.25 <= cand_iou <= 0.40 or
                owner_iou - cand_iou < 0.28):
            continue
        independent = False
        for proposal_index, proposal in enumerate(proposals):
            if proposal_index == owner_best:
                continue
            for other_id in previous_ids & current.keys():
                if other_id in (candidate_id, owner_id):
                    continue
                other_scores = [iou(current[other_id], p["bbox"]) for p in proposals]
                other_score = other_scores[proposal_index]
                other_runner = max((s for i, s in enumerate(other_scores) if i != proposal_index), default=0.0)
                if other_score >= 0.65 and other_score - other_runner >= 0.10:
                    independent = True
                    break
            if independent:
                break
        if not independent:
            continue
        if any(other_id not in (candidate_id, owner_id) and
               iou(current[other_id], proposals[owner_best]["bbox"]) >= owner_iou - 0.10
               for other_id in current):
            continue
        eligible.append({"candidate_id": candidate_id, "owner_id": owner_id,
                         "proposal_index": owner_best, "owner_iou": owner_iou,
                         "candidate_iou": cand_iou, "margin": owner_iou - cand_iou,
                         "area_ratio": area_ratio})
    if len(eligible) == 1:
        return {"admit": True, "reason": "UNIQUE_ESTABLISHED_TRACK_OWNS_CONTEXT_INSTANCE", **eligible[0]}
    return {"admit": False, "reason": "MULTIPLE_PLAUSIBLE_OWNERS" if eligible else "NO_STRONG_CONTEXT_EXCLUSIVITY"}


def admit_birth_batch(events: list[dict[str, Any]], current: dict[int, list[float]],
                      previous_ids: set[int], width: float, height: float,
                      frame_current: bool) -> list[dict[str, Any]]:
    decisions = [admit(int(event["candidate_id"]), current, previous_ids,
                       event["proposals"], width, height, frame_current)
                 for event in events]
    owner_claims: dict[int, int] = defaultdict(int)
    for decision in decisions:
        if decision["admit"]:
            owner_claims[int(decision["owner_id"])] += 1
    for decision in decisions:
        if decision["admit"] and owner_claims[int(decision["owner_id"])] > 1:
            decision["admit"] = False
            decision["reason"] = "AMBIGUOUS_MULTIPLE_BIRTHS_CLAIM_OWNER"
    return decisions


def read_dataset(directory: Path, camera: str) -> dict[str, Any]:
    canonical = {}
    for row in rows(directory / "canonical-objects.jsonl"):
        if row.get("camera_id") == camera:
            canonical[int(row["frame_num"])] = row

    track_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows(directory / "tracks.jsonl"):
        track_groups[(int(row["source_id"]), int(row["frame"]))].append(row)

    states: dict[tuple[int, int], dict[str, Any]] = {}
    state_path = directory / "tracker-frame-states.jsonl"
    if state_path.exists():
        for row in rows(state_path):
            if row.get("camera_id") == camera:
                states[(int(row["source_id"]), int(row["frame"]))] = row

    # Select the tracker source slot by matching canonical native IDs in each
    # frame. A repeated-file diagnostic has multiple slots, but IDs are source
    # local; the matching is validated per frame rather than inferred from order.
    frame_rows: dict[int, dict[str, Any]] = {}
    source_votes: dict[int, int] = defaultdict(int)
    frame_source: dict[int, int] = {}
    for frame, row in canonical.items():
        object_ids = {int(o["native_track_id"]) for o in row.get("objects", [])}
        choices = []
        for (source_id, group_frame), group in track_groups.items():
            if group_frame != frame:
                continue
            ids = {int(t["object_id"]) for t in group}
            intersection = len(object_ids & ids)
            exact_box_matches = 0
            for obj in row.get("objects", []):
                track = next((t for t in group if int(t["object_id"]) == int(obj["native_track_id"])), None)
                if track and len(track.get("box", [])) == 4:
                    xyxy = obj.get("bbox", [])
                    xywh = track["box"]
                    if len(xyxy) == 4 and max(abs(xyxy[i] - v) for i, v in enumerate(
                            [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]])) < 2.0:
                        exact_box_matches += 1
            choices.append((intersection, exact_box_matches, source_id, group))
        if choices:
            choices.sort(reverse=True, key=lambda v: (v[0], v[1]))
            best = choices[0]
            if best[0] >= min(1, len(object_ids)):
                frame_source[frame] = best[2]
                frame_rows[frame] = {int(t["object_id"]): [float(x) for x in t["box"]] for t in best[3]}
                source_votes[best[2]] += 1
    source_id = max(source_votes, key=source_votes.get) if source_votes else None
    if source_id is not None:
        # Normalize to the modal slot, preserving only frames whose per-frame
        # mapping agrees with it. Any mismatch is a hard coverage gap.
        for frame in list(frame_rows):
            if frame_source.get(frame) != source_id:
                frame_rows.pop(frame)

    shadow_boxes: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for (state_source, frame), state in states.items():
        if source_id is not None and state_source != source_id:
            continue
        for shadow in state.get("shadow_targets", []):
            box = shadow.get("bbox")
            if isinstance(box, list) and len(box) == 4 and all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                shadow_boxes[frame][int(shadow["id"])] = [float(v) for v in box]
    if not state_path.exists():
        for row in rows(directory / "tracker_decisions.jsonl"):
            if row.get("camera_id") != camera:
                continue
            if source_id is not None and int(row.get("source_id", -1)) != source_id:
                continue
            frame = int(row["frame"])
            for shadow in row.get("shadow_candidates", []):
                if shadow.get("state") == "INACTIVE":
                    box = shadow.get("bbox")
                    if isinstance(box, list) and len(box) == 4 and all(
                            isinstance(v, (int, float)) and math.isfinite(v) for v in box):
                        shadow_boxes[frame][int(shadow["native_id"])] = [float(v) for v in box]

    events: dict[int, list[dict[str, Any]]] = defaultdict(list)
    command_path = directory / "command-redacted.json"
    candidate_centered_crop = False
    if command_path.exists():
        command = json.loads(command_path.read_text(encoding="utf-8"))
        candidate_centered_crop = bool(command.get("candidate_centered_crop", False))
    for event in rows(directory / "lowconf-verifier.jsonl"):
        if (event.get("camera_id") != camera or event.get("verification_kind") != "ACTIVE_TRACK_BIRTH" or
                event.get("status") != "CONFIRMED"):
            continue
        frame = int(event["source_frame_num"])
        canonical_row = canonical.get(frame)
        if not canonical_row or int(event["source_pts_ns"]) != int(canonical_row["pts_ns"]):
            continue
        frame_width = float(canonical_row.get("bbox_coordinate_width", canonical_row.get("pipeline_width", 1920)))
        frame_height = float(canonical_row.get("bbox_coordinate_height", canonical_row.get("pipeline_height", 1080)))
        surface_width = float(canonical_row.get("pipeline_width", frame_width))
        surface_height = float(canonical_row.get("pipeline_height", frame_height))
        detector_bbox = [float(v) for v in event.get("detector_bbox", [])]
        if len(detector_bbox) != 4:
            continue
        proposals = [{"confidence": float(p["confidence"]),
                      "bbox": map_context_to_frame([float(v) for v in p["bbox"]],
                          detector_bbox, frame_width, frame_height,
                          surface_width, surface_height, candidate_centered_crop)}
                     for p in event.get("proposals", [])]
        events[frame].append({"candidate_id": int(event["native_track_id"]),
                              "pts_ns": int(event["source_pts_ns"]),
                              "proposals": proposals,
                              "verifier_confidence": float(event.get("verifier_confidence", 0.0))})

    published_ids = {
        frame: {int(obj["native_track_id"]) for obj in row.get("objects", [])}
        for frame, row in canonical.items()
    }
    return {"camera": camera, "directory": directory, "canonical": canonical,
            "tracks": frame_rows, "published_ids": published_ids, "source_id": source_id,
            "frame_source": frame_source, "shadow_boxes": shadow_boxes, "events": events}


def evaluate(data: dict[str, Any]) -> dict[str, Any]:
    frames = sorted(data["canonical"])
    links: dict[int, dict[str, Any]] = {}
    suppressed_by_frame: dict[int, list[int]] = defaultdict(list)
    event_results = []
    prior_ids: set[int] = set()
    previous_frame = None
    released = 0
    continuity_gaps = 0
    for frame in frames:
        row = data["canonical"][frame]
        # tracker_apply_cross_id_context_duplicate_filter() receives the
        # current `published` vector after active-birth verification. Raw
        # NvDCF rows that were withheld upstream must not be considered owners
        # here. Use tracker rows for geometry, intersected with IDs that the
        # saved canonical output actually published on this exact frame.
        raw_current_tracks = data["tracks"].get(frame, {})
        published_ids = data.get("published_ids", {}).get(frame,
            set(raw_current_tracks))
        current_tracks = {track_id: box for track_id, box in raw_current_tracks.items()
                          if track_id in published_ids}
        if previous_frame is None or frame != previous_frame + 1:
            prior_ids = set()
            links.clear()
        exact_previous = previous_frame is not None and frame == previous_frame + 1
        width, height = float(row.get("bbox_coordinate_width", row.get("pipeline_width", 1920))), float(
            row.get("bbox_coordinate_height", row.get("pipeline_height", 1080)))
        # New links require same-frame confirmed context evidence, active birth,
        # and a complete preceding tracker frame.
        frame_events = data["events"].get(frame, [])
        frame_decisions = admit_birth_batch(
            frame_events, current_tracks, prior_ids, width, height,
            exact_previous and all(event["pts_ns"] == int(row["pts_ns"])
                                   for event in frame_events))
        for event, decision in zip(frame_events, frame_decisions):
            candidate_id = event["candidate_id"]
            if decision["admit"]:
                owner_id = decision["owner_id"]
                if candidate_id not in links and all(v["owner_id"] != owner_id for v in links.values()):
                    candidate, owner = current_tracks[candidate_id], current_tracks[owner_id]
                    links[candidate_id] = {
                        "owner_id": owner_id,
                        "offset_x": ((candidate[0] + candidate[2] / 2) - (owner[0] + owner[2] / 2)) / owner[2],
                        "offset_y": ((candidate[1] + candidate[3] / 2) - (owner[1] + owner[3] / 2)) / owner[3],
                        "area_ratio": candidate[2] * candidate[3] / (owner[2] * owner[3]),
                        "last_frame": frame - 1,
                    }
                else:
                    decision = {"admit": False, "reason": "LEDGER_CONFLICT"}
            event_results.append({"frame": frame, "candidate_id": candidate_id,
                                  "verifier_confidence": event["verifier_confidence"], **decision})

        for candidate_id, relation in list(links.items()):
            if relation["last_frame"] + 1 != frame:
                links.pop(candidate_id)
                released += 1
                continue
            owner_active = relation["owner_id"] in current_tracks
            candidate_active = candidate_id in current_tracks
            owner = current_tracks.get(relation["owner_id"], data["shadow_boxes"].get(frame, {}).get(relation["owner_id"]))
            candidate = current_tracks.get(candidate_id, data["shadow_boxes"].get(frame, {}).get(candidate_id))
            if owner is None or candidate is None:
                links.pop(candidate_id)
                released += 1
                continue
            ratio = candidate[2] * candidate[3] / (owner[2] * owner[3])
            off_x = ((candidate[0] + candidate[2] / 2) - (owner[0] + owner[2] / 2)) / owner[2]
            off_y = ((candidate[1] + candidate[3] / 2) - (owner[1] + owner[3] / 2)) / owner[3]
            still_same = (center_inside(candidate, owner) and 0.05 < ratio <= 0.75 and
                          abs(off_x - relation["offset_x"]) <= 0.25 and
                          abs(off_y - relation["offset_y"]) <= 0.24 and
                          abs(ratio - relation["area_ratio"]) <= 0.25)
            if not still_same:
                links.pop(candidate_id)
                released += 1
                continue
            if owner_active and candidate_active:
                suppressed_by_frame[frame].append(candidate_id)
            elif not candidate_active:
                continuity_gaps += 1
            relation["last_frame"] = frame
        # active_previous_frame comes from raw NvDCF target history, even if a
        # given row was withheld from canonical publication on that frame.
        prior_ids = set(raw_current_tracks)
        previous_frame = frame

    base_total = sum(len(r.get("objects", [])) for r in data["canonical"].values())
    removed_total = sum(len(v) for v in suppressed_by_frame.values())
    return {
        "camera": data["camera"], "directory": str(data["directory"]),
        "source_slot": data["source_id"], "canonical_frames": len(frames),
        "frame_first": frames[0] if frames else None, "frame_last": frames[-1] if frames else None,
        "unique_consecutive_frames": bool(frames and frames == list(range(frames[0], frames[-1] + 1))),
        "track_frames_mapped": len(data["tracks"]), "context_birth_events": len(event_results),
        "context_birth_decisions": event_results,
        "admitted_births": sum(bool(e.get("admit")) for e in event_results),
        "rejected_births": sum(not bool(e.get("admit")) for e in event_results),
        "raw_canonical_rows": base_total, "candidate_suppressed_rows": removed_total,
        "candidate_canonical_rows": base_total - removed_total,
        "suppressed_by_frame": {str(k): v for k, v in suppressed_by_frame.items()},
        "shadow_continuity_frames": continuity_gaps, "released_links": released,
        "decision_limitations": "Metadata-only policy replay, not inference, human annotation, or exact-r1 full-video replay.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam02-dir", type=Path, default=DEFAULT_BASE / "cam02")
    parser.add_argument("--cam05-dir", type=Path, default=DEFAULT_BASE / "cam05-history-candidate-throttled-250ms")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for directory, camera in ((args.cam02_dir, "CAM-02"), (args.cam05_dir, "CAM-05")):
        data = read_dataset(directory, camera)
        result = evaluate(data)
        results.append(result)
        output = args.output_dir / f"{camera.lower()}-metadata-replay.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{camera}: frames={result['canonical_frames']} slot={result['source_slot']} "
              f"births={result['admitted_births']}/{result['context_birth_events']} "
              f"suppressed_rows={result['candidate_suppressed_rows']} "
              f"range={result['frame_first']}..{result['frame_last']}")
        for event in result["context_birth_decisions"]:
            print(f"  frame={event['frame']} candidate={event['candidate_id']} "
                  f"owner={event.get('owner_id')} admit={event.get('admit')} reason={event.get('reason')}")
    summary = {"mode": "CPU metadata-only", "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
