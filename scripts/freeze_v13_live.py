"""F5 independent acceptance summary: exact V13, complete replay, real UI.

This does not change scoring, tracker policy, identity decisions or production.
Physical keyframes are the same frozen review used by prior acceptance, not a
claim of dense person-level ground truth. Proposal retention is reported apart.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess

from scripts.dev_room_mv3dt.audit_replay_acceptance import _bbox_iou, presence_frame
from scripts.run_room_candidate import ROOT, save
from scripts.run_v13_room_candidate import protected_v13, V13_SHA
from services.mv3dt_room.presence import CurrentPresencePositionFilter, application_id

F5 = ROOT / ".runtime/freeze/F5-v13-live"
REFERENCE = ROOT / ".runtime/full-pipeline-detector-ab-20261002-anjxPxXh/ground-truth-review/person-reference.json"


def load(path):
    return json.loads(Path(path).read_text())


def evidence_save(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError(f"existing evidence differs; refusing overwrite: {path}")
        return
    save(path, value)


def same_candidate_chain(manifests):
    """Require the live run and every replay to validate identical assets."""
    def key(row):
        native = row["native"]
        source_sha = native.get("source_sha256", native.get("staged_source_sha256"))
        if not native.get("binary_sha256") or not source_sha:
            raise ValueError("candidate provenance requires exact binary and source SHA256 values")
        return (native["binary_sha256"], source_sha,
                row["tracker_sha256"], row["pose"]["engine_sha256"], row["configs"]["config_pgie.txt"])
    return bool(manifests) and len({key(row) for row in manifests}) == 1 and all(
        row["gallery_preexisting"] is False for row in manifests) and len(
            {row["gallery"] for row in manifests}) == len(manifests)


def match_references(refs, objects):
    # Identical one-to-one visible-body IoU >= .3 rule from frozen score_people.
    state = {0: (0., [])}
    for ri, ref in enumerate(refs):
        updated = dict(state)
        a = ref["bbox"]
        a = [a[0], a[1], a[0] + a[2], a[1] + a[3]]
        for mask, (score, pairs) in state.items():
            for oi, ob in enumerate(objects):
                b = ob["bbox"]
                value = _bbox_iou(a, [b[0], b[1], b[0] + b[2], b[1] + b[3]])
                if mask >> oi & 1 or value < .3:
                    continue
                newmask = mask | 1 << oi
                item = score + value, pairs + [(ri, oi, value)]
                if newmask not in updated or item[0] > updated[newmask][0]:
                    updated[newmask] = item
        state = updated
    return max(state.values(), key=lambda x: (len(x[1]), x[0]))[1]


def reviewed_person_present(run: Path, output: Path | None = None) -> dict:
    reference = load(REFERENCE)
    dets, tracks, canonical = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in csv.DictReader((run / "run/logs/probe/pretracker_geometry.csv").open()):
        dets[row["camera_id"], int(row["frame"])].append({"bbox": [float(row[k]) for k in ("left", "top", "width", "height")]})
    for row in csv.DictReader((run / "run/logs/probe/posttracker_geometry.csv").open()):
        tracks[row["camera_id"], int(row["frame"])].append({"bbox": [float(row["rect_" + k]) for k in ("left", "top", "width", "height")],
            "native": row["native_track_id"], "associated": row["detector_associated"] == "1"})
    for row in map(json.loads, (run / "identity-live/global_identity.jsonl").read_text().splitlines()):
        canonical[row["camera_id"], int(row["frame"])].append(row)
    results, merges, switches, last = {}, [], [], {}
    for camera, observations in reference["observations"].items():
        visible = detected = tracked = associated = 0
        detail = []
        for frame_s, refs in sorted(observations.items(), key=lambda p: int(p[0])):
            frame = int(frame_s)
            dm, tm = match_references(refs, dets[camera, frame]), match_references(refs, tracks[camera, frame])
            visible += len(refs)
            detected += len(dm)
            tracked += len(tm)
            apps = defaultdict(set)
            for ri, oi, _ in tm:
                ob = tracks[camera, frame][oi]
                associated += ob["associated"]
                person = refs[ri]["physical"]
                for row in canonical[camera, frame]:
                    app = application_id(row)
                    if str(row["native_track_id"]) == ob["native"] and app != "Unknown":
                        apps[app].add(person)
                        key = camera, person
                        if key in last and last[key] != app:
                            switches.append({"camera": camera, "frame": frame, "person": person, "before": last[key], "after": app})
                        last[key] = app
            for app, people in apps.items():
                if len(people) > 1:
                    merges.append({"camera": camera, "frame": frame, "canonical": app, "physical_people": sorted(people)})
            detail.append({"frame": frame, "visible": len(refs), "detected": len(dm), "tracked": len(tm), "canonical_mapping": {a: sorted(p) for a, p in apps.items()}})
        results[camera] = {"visible": visible, "detected": detected, "tracked": tracked,
            "detector_recall_percent": 100 * detected / visible, "tracker_coverage_percent": 100 * tracked / visible,
            "detector_associated_tracker_coverage_percent": 100 * associated / visible, "frames": detail}
    report = {"reference": str(REFERENCE), "scope": reference["scope"], "dense_recall": "NOT_MEASURED",
        "cameras": results, "sampled_false_merges": merges, "sampled_identity_changes": switches}
    evidence_save(output or run / "reviewed-physical-sample.json", report)
    return report


def publication(run: Path) -> dict:
    by_frame = defaultdict(list)
    for row in map(json.loads, (run / "identity-live/global_identity.jsonl").read_text().splitlines()):
        by_frame[int(row["frame"])].append(row)
    filt, previous = CurrentPresencePositionFilter(), {}
    duplicates = ghosts = disagreements = 0
    maxjump = 0.
    for frame in range(max(by_frame, default=-1) + 1):
        active, positions, markers = presence_frame(by_frame[frame], frame, filt)
        current = {application_id(r) for r in active}
        duplicates += len(markers) - len(set(markers))
        ghosts += len(set(markers) - current)
        renderable = {application_id(r) for r in active if r.get("world") and len(r["world"]) >= 2}
        disagreements += len(renderable ^ set(markers))
        for app, pos in positions.items():
            if app in previous and previous[app][0] == frame - 1:
                maxjump = max(maxjump, math.dist(pos, previous[app][1]))
        previous = {a: (frame, p) for a, p in positions.items()}
    counts = defaultdict(lambda: defaultdict(int))
    for row in map(json.loads, (run / "run/logs/probe/frame_path_audit.jsonl").read_text().splitlines()):
        if row.get("record") != "frame":
            continue
        camera, stage = row["mapped_camera_id"], row["stage"]
        counts[camera][stage + "_frames"] += 1
        counts[camera][stage + "_objects"] += len(row["objects"])
    log = (run / "logs/deepstream.log").read_text(errors="replace")
    perf = [(float(a), float(b)) for a, b in re.findall(r"\*\*PERF:\s+([0-9.]+)\s+\([^)]*\)\s+([0-9.]+)", log)]
    report = {"frames": dict(counts), "published_identity_observations": sum(map(len, by_frame.values())),
        "bev_duplicates": duplicates, "bev_ghosts": ghosts, "presence_disagreements": disagreements,
        "max_bev_jump_projection_units": maxjump, "metric_scale": "UNPROVEN; no calibration change",
        "steady_fps": {c: sum(p[i] for p in perf[-10:]) / len(perf[-10:]) for i, c in enumerate(("CAM-01", "CAM-04"))} if perf else {}}
    evidence_save(run / "publication-audit.json", report)
    return report


def replay_checks(canonical: Path, present: Path, empty: Path) -> dict:
    formal = load(canonical / "formal-acceptance.json")
    reviewed = reviewed_person_present(present)
    results = {"canonical": {"formal": formal["status"], "ids": formal["identities"], "checks": formal["checks"]},
        "person_present_sample": reviewed, "runs": {}}
    checks = {"canonical_formal": formal["status"] == "PASS",
        "person_present_sampled_false_merges_zero": not reviewed["sampled_false_merges"]}
    for dataset, run in (("canonical", canonical), ("person-present", present), ("empty-room", empty)):
        result = load(run / "result.json")
        pub = publication(run)
        checks[dataset + "_pipeline_provenance"] = result["status"] == "PASS" and result["loaded_tracker_sha256"] == V13_SHA
        expected = 12031 if dataset == "empty-room" else 2400
        cams = result["retention"]["cameras"]
        for camera, row in cams.items():
            key = dataset + "_" + camera
            checks[key + "_complete_frames"] = row["pgie_frames"] == row["tracker_frames"] == expected
            if dataset != "empty-room":
                checks[key + "_retention"] = row["detector_associated_retention_percent"] >= 98
                checks[key + "_gap"] = max((w["frames"] for w in row["association_deficit_windows"]), default=0) <= 10
        checks[dataset + "_bev_zero"] = pub["bev_duplicates"] == pub["bev_ghosts"] == pub["presence_disagreements"] == 0
        checks[dataset + "_fps"] = len(pub["steady_fps"]) == 2 and all(19 <= fps <= 21 for fps in pub["steady_fps"].values())
        if dataset == "empty-room":
            checks["empty_published_tracks_zero"] = all(r.get("tracker_objects", 0) == 0 for r in pub["frames"].values())
            checks["empty_canonical_zero"] = result["canonical_ids"] == [] and pub["published_identity_observations"] == 0
        results["runs"][dataset] = {"publication": pub, "tracker": cams, "path": str(run), "pipeline_result": result["status"]}
    results["checks"] = checks
    results["status"] = "PASS" if all(checks.values()) else "FAIL"
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--live-result", type=Path, help="explicit preserved failure/post-mortem result")
    parser.add_argument("--canonical", type=Path, default=F5 / "canonical")
    parser.add_argument("--present", type=Path, default=F5 / "person-present")
    parser.add_argument("--empty", type=Path, default=F5 / "empty-room")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    replay = replay_checks(args.canonical, args.present, args.empty)
    save(args.output / "replay-regression.json", replay)
    live = load(args.live_result or args.live / "result.json")
    chain = same_candidate_chain([load(p / "loaded-config-manifest.json") for p in
                                 (args.canonical, args.present, args.empty, args.live / "room-pair/live-current")])
    python = ROOT / ".runtime/full-stack-venv/bin/python"
    commands = [[str(python), "-B", "-m", "pytest", "-q", "tests", f"--junitxml={args.output / 'unit.xml'}"],
        ["bash", "tests/native/run_bbox_history_lifecycle_tests.sh"], ["bash", "scripts/start_full_live_stack.sh", "--preflight"],
        [str(python), "-I", "-B", "-m", "pip", "check"],
        [str(python), "-B", "scripts/freeze_repo_baseline.py", "--output", ".runtime/freeze/F0-repo-baseline", "--verify"],
        ["git", "diff", "--check"]]
    results = []
    with (args.output / "test-results.txt").open("x") as log:
        for command in commands:
            proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                env=dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", QT_QPA_PLATFORM="offscreen"))
            log.write(json.dumps(command) + "\n" + proc.stdout + proc.stderr + "\n")
            results.append({"command": command, "exit": proc.returncode})
    syntax_errors = []
    for name in set(subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard"], text=True).splitlines()):
        if name.endswith(".py"):
            try:
                ast.parse((ROOT / name).read_text(), filename=name)
            except (SyntaxError, UnicodeError) as exc:
                syntax_errors.append(str(exc))
        elif name.endswith(".sh"):
            if subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True).returncode:
                syntax_errors.append(name)
    before = load(args.live / "protected-before.json")
    after = protected_v13()
    save(args.output / "protected-before.json", before)
    save(args.output / "protected-after.json", after)
    passed = (replay["status"] == live["status"] == "PASS" and chain and before == after
              and not syntax_errors and all(r["exit"] == 0 for r in results))
    save(args.output / "final-acceptance.json", {"status": "PASS" if passed else "FAIL", "commands": results,
        "protected_unchanged": before == after, "syntax_errors": syntax_errors, "live": str(args.live),
        "replay_status": replay["status"], "live_status": live["status"], "identical_candidate_chain": chain,
        "production_promoted": False})
    print(json.dumps({"status": "PASS" if passed else "FAIL", "protected_unchanged": before == after}), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
