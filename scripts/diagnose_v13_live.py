"""Read-only deficit evidence. Private NvDCF reasons remain explicitly unknown."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.dev_room_mv3dt.audit_replay_acceptance import _bbox_iou
from scripts.dev_room_mv3dt.verify_validated_assets import sha256
from scripts.run_room_candidate import save


def overlap(a, b):
    return _bbox_iou([a[0], a[1], a[0]+a[2], a[1]+a[3]],
                     [b[0], b[1], b[0]+b[2], b[1]+b[3]])


def window_evidence(camera, window, proposals, states):
    begin, end = window["start"], window["end"]
    selected = [p for p in proposals if p["camera"] == camera]
    missing = [p for p in selected if begin <= p["frame"] <= end and not p["tracker_association_emitted"]]
    evidence = []
    for proposal in missing:
        before = [p for p in selected if p["frame"] == begin - 1 and p["tracker_association_emitted"]]
        previous = max(before, key=lambda p: overlap(p["bbox_xywh"], proposal["bbox_xywh"]), default=None)
        if previous is not None and overlap(previous["bbox_xywh"], proposal["bbox_xywh"]) < .3:
            previous = None
        after = [p for p in selected if end < p["frame"] <= end + 10
                 and p["tracker_association_emitted"] and overlap(p["bbox_xywh"], proposal["bbox_xywh"]) >= .3]
        following = min(after, key=lambda p: p["frame"], default=None)
        native = previous["native_track_id"] if previous else None
        target_states = [s for s in states if s["camera_id"] == camera and s["native_track_id"] == native
                         and begin-10 <= int(s["frame"]) <= end+10]
        evidence.append({"proposal": proposal, "previous_associated_proposal": previous,
            "first_geometrically_matching_return": following, "target_metadata": target_states,
            "classification": "inactive_target_during_association_deficit" if any(s["nvdcf_state"] == "INACTIVE" for s in target_states)
                              else "association_deficit_private_reason_unavailable",
            "physical_ground_truth": "UNAVAILABLE; detector proposal is not proof of a person",
            "private_rejection_reason": "NOT_EXPORTED; no threshold or suppression cause inferred"})
    return {"window": window, "unassociated_proposals": evidence}


def diagnose(run: Path, output: Path):
    if output.exists():
        raise ValueError("new evidence output required")
    proposals_path = run / "proposal-association.jsonl"
    states_path = run / "run/logs/probe/tracker_state.csv"
    retention_path = run / "retention.json"
    proposals = [json.loads(line) for line in proposals_path.read_text().splitlines()]
    with states_path.open() as handle:
        states = list(csv.DictReader(handle))
    retention = json.loads(retention_path.read_text())
    report = {"run": str(run), "input_hashes": {str(p): sha256(p) for p in (proposals_path, states_path, retention_path)},
        "decisions_changed": False, "proposals_removed": 0, "cameras": {}}
    for camera, data in retention["cameras"].items():
        report["cameras"][camera] = [window_evidence(camera, w, proposals, states)
                                   for w in data["association_deficit_windows"]]
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.run, args.output)


if __name__ == "__main__":
    main()
