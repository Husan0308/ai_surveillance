#!/usr/bin/env python3
"""Read-only lifecycle/identity evidence; never changes acceptance or decisions.

Consumes the exact proposal association audit, rather than a confidence-count
proxy. Public SDK shadow samples describe their recorded frame only. The SDK
does not expose association scores, shadow age, or private rejection reasons;
these remain null, and geometric hypotheses are not reported as proven causes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

STATE_NAMES = {0: "EMPTY", 1: "ACTIVE", 2: "INACTIVE", 3: "TENTATIVE",
               4: "PROJECTED", 5: "QUASIACTIVE"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_values(text):
    # OpenCV's YAML directive is not standard YAML. Comments are never values.
    text = re.sub(r"^%YAML:1\.0\s*$", "", text, flags=re.MULTILINE)
    return yaml.safe_load(text)


def iou(a, b):
    x = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    intersection = x * y
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentiles(values):
    return {f"p{p}": float(np.percentile(values, p)) for p in (1, 5, 50, 95, 99)} if values else {}


def shadow_sample(line, camera):
    fields = line.split()
    if len(fields) != 19:
        raise ValueError("expected 19 DeepStream 9.1 shadow fields")
    left, top, right, bottom = map(float, fields[5:9])
    return dict(camera=camera, frame=int(fields[0]), native=str(int(fields[1])),
                state=STATE_NAMES[int(fields[17])], confidence=float(fields[16]),
                visibility=float(fields[18]), bbox=[left, top, right-left, bottom-top],
                shadow_age=None, termination_reason=None, origin="CURRENT_SDK_SHADOW")


def spans(frames):
    result = []
    for frame in sorted(set(frames)):
        if result and frame == result[-1]["end"] + 1:
            result[-1]["end"] = frame
        else:
            result.append(dict(start=frame, end=frame))
    for row in result:
        row["frames"] = row["end"] - row["start"] + 1
        row["seconds"] = row["frames"] / 20.0
    return result


def public_lifecycle(probe):
    samples = defaultdict(list)
    for row in csv_rows(probe / "tracker_state.csv"):
        samples[row["camera_id"], int(row["frame"])].append(dict(
            camera=row["camera_id"], frame=int(row["frame"]),
            native=str(int(row["native_track_id"])), state=row["nvdcf_state"],
            confidence=float(row["tracker_confidence"]), visibility=float(row["visibility"]),
            bbox=[float(row[k]) for k in ("left", "top", "width", "height")],
            track_age=int(row["track_age"]), shadow_age=None, termination_reason=None,
            origin="PAST_FRAME_METADATA"))
    for path in sorted((probe / "shadow").glob("*.txt")):
        match = re.fullmatch(r"\d+_(\d+)_(\d+)\.txt", path.name)
        if not match or int(match[1]) not in (0, 1):
            raise ValueError(f"unmapped shadow filename: {path.name}")
        camera = ("CAM-01", "CAM-04")[int(match[1])]
        for line in path.read_text().splitlines():
            if line.strip():
                sample = shadow_sample(line, camera)
                samples[camera, sample["frame"]].append(sample)
    return samples


def diagnose(run, associations, output):
    run, output = Path(run), Path(output)
    probe = run / "run/logs/probe"
    output.mkdir(parents=True, exist_ok=False)
    proposals = jsonl(associations)
    grouped = defaultdict(list)
    for row in proposals:
        grouped[row["camera"], row["frame"]].append(row)
    posts = defaultdict(list)
    native_posts = defaultdict(list)
    for row in csv_rows(probe / "posttracker_geometry.csv"):
        sample = dict(camera=row["camera_id"], frame=int(row["frame"]), pts=row["pts"],
                      native=str(int(row["native_track_id"])),
                      confidence=float(row["tracker_confidence"]),
                      detector_associated=row["detector_associated"] == "1",
                      bbox=[float(row[k]) for k in ("rect_left", "rect_top", "rect_width", "rect_height")])
        posts[sample["camera"], sample["frame"]].append(sample)
        native_posts[sample["camera"], sample["native"]].append(sample)
    lifecycle = public_lifecycle(probe)
    config = config_values((run / "run/config_tracker.yml").read_text())
    details, summaries = [], {}
    for camera in ("CAM-01", "CAM-04"):
        all_rows = [p for p in proposals if p["camera"] == camera]
        missing = [p for p in all_rows if not p["tracker_association_emitted"]]
        for proposal in missing:
            frame, box = proposal["frame"], proposal["nvdcf_input_bbox_xywh"]
            before, after = [], []
            # These are nearest spatial outputs, NOT asserted physical IDs.
            for delta in range(1, 41):
                if not before:
                    before = [dict(s, comparison_iou=iou(box, s["bbox"])) for s in posts[camera, frame-delta]
                              if iou(box, s["bbox"]) >= .2]
                if not after:
                    after = [dict(s, comparison_iou=iou(box, s["bbox"])) for s in posts[camera, frame+delta]
                             if iou(box, s["bbox"]) >= .2]
                if before and after:
                    break
            tracked_ids = {s["native"] for s in before + after}
            targets = [dict(s, comparison_iou=iou(box, s["bbox"]))
                       for s in lifecycle[camera, frame]
                       if s["native"] in tracked_ids or iou(box, s["bbox"]) > .01]
            other_proposal_iou = max((iou(box, p["nvdcf_input_bbox_xywh"])
                                     for p in grouped[camera, frame] if p is not proposal), default=0)
            current_output_iou = max((iou(box, s["bbox"]) for s in posts[camera, frame]), default=0)
            state_counts = Counter(s["state"] for s in targets)
            classification = "PRIVATE_NvDCF_REJECTION_UNRESOLVED"
            if state_counts["TENTATIVE"]:
                classification = "TENTATIVE_LIFECYCLE_OBSERVED"
            elif state_counts["INACTIVE"]:
                classification = "INACTIVE_ASSOCIATION_LOSS_OBSERVED"
            elif not proposal["tracker_input_observed"]:
                classification = "TRACKER_INPUT_UNOBSERVED"
            details.append(dict(proposal=proposal, nearest_spatial_output_before=before,
                                nearest_spatial_output_after=after, current_frame_lifecycle=targets,
                                max_other_proposal_iou=other_proposal_iou,
                                max_current_output_iou=current_output_iou,
                                duplicate_like_geometry=other_proposal_iou >= .5 or current_output_iou >= .5,
                                new_target_suppression_proven=False,
                                matching_score=None, physical_label="REQUIRES_VISUAL_REVIEW",
                                classification=classification))
        camera_details = [d for d in details if d["proposal"]["camera"] == camera]
        summaries[camera] = dict(
            proposals=len(all_rows), associated=len(all_rows)-len(missing),
            exact_retention_percent=100*(len(all_rows)-len(missing))/len(all_rows),
            unmatched_confidence=percentiles([p["pgie_confidence"] for p in missing]),
            deficit_windows=spans([p["frame"] for p in missing]),
            native_count=len({p["native_track_id"] for p in all_rows if p["native_track_id"] is not None}),
            duplicate_like_unmatched=sum(d["duplicate_like_geometry"] for d in camera_details),
            classifications=dict(Counter(d["classification"] for d in camera_details)))
    decisions = jsonl(run / "identity-live/identity_decision_trace.jsonl")
    events = json.loads((run / "identity-live/identity_events.json").read_text())
    global_rows = jsonl(run / "identity-live/global_identity.jsonl")
    creations, trajectories = [], []
    for event in events:
        if event["event"] != "NEW_CONFIRMED":
            continue
        camera, native = event["camera_id"], str(event["native_track_id"])
        traces = [t for t in decisions if t["allocation_branch"] == "pending_new_identity"
                  and t["observation"]["camera_id"] == camera
                  and str(t["observation"]["native_track_id"]) == native]
        samples = native_posts[camera, native]
        creations.append(dict(event=event, decision_traces=traces,
                              first_native_output=min((s["frame"] for s in samples), default=None),
                              native_output_frames=len(samples)))
    for (camera, native), samples in sorted(native_posts.items()):
        published = [p for p in global_rows if p["camera_id"] == camera
                     and str(p["native_track_id"]) == native]
        trajectories.append(dict(camera=camera, native=native,
                                 first_frame=min(s["frame"] for s in samples),
                                 last_frame=max(s["frame"] for s in samples),
                                 canonical_ids=sorted({p["global_person_id"] for p in published}),
                                 physical_trajectory="UNREVIEWED", root_cause="NOT_INFERRED_FROM_REID"))
    # Only observed transitions. No stale sample is carried forward as current.
    by_native = defaultdict(lambda: defaultdict(set))
    for samples in lifecycle.values():
        for s in samples:
            by_native[s["camera"], s["native"]][s["frame"]].add(s["state"])
    transitions = []
    for (camera, native), frames in sorted(by_native.items()):
        previous = None
        for frame, states in sorted(frames.items()):
            states = sorted(states)
            if previous and previous[1] != states:
                transitions.append(dict(camera=camera, native=native, frame=frame,
                                        previous_sample_frame=previous[0], before=previous[1], after=states,
                                        interval_not_fully_observed=frame != previous[0]+1))
            previous = frame, states
    result = dict(run=str(run.resolve()), acceptance_modified=False,
                  hashes={"tracker": sha(run / "run/config_tracker.yml"),
                          "pgie": sha(run / "run/config_pgie.txt"), "associations": sha(associations)},
                  effective_tracker=config, cameras=summaries,
                  unavailable_private_fields=["association_scores", "termination_reason", "shadow_age"],
                  identity_creations=creations, local_to_canonical=trajectories)
    (output / "diagnostic-summary.json").write_text(json.dumps(result, indent=2))
    (output / "unmatched-proposals.jsonl").write_text("".join(json.dumps(d)+"\n" for d in details))
    (output / "lifecycle-transitions.json").write_text(json.dumps(transitions, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--associations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = diagnose(args.run, args.associations, args.output)
    print(json.dumps({"cameras": report["cameras"], "creations": [c["event"] for c in report["identity_creations"]]}, indent=2))


if __name__ == "__main__":
    main()
