#!/usr/bin/env python3
"""Validate durable Dev Room identity memory against a captured replay."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from services.mv3dt_room.global_identity_manager import GlobalIdentityManager
from services.mv3dt_room.identity_gallery_store import IdentityGalleryStore
from services.mv3dt_room.presence import assert_presence_contract


def iso(value: str, delta: timedelta = timedelta(0)) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return (parsed + delta).astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def row_for_embedding(rows_by_key, record):
    candidates = rows_by_key.get((record["camera_id"], int(record["native_track_id"])), [])
    after = [row for row in candidates if int(row.get("frame", -1)) >= int(record["frame"])]
    pool = after or candidates
    return min(pool, key=lambda row: abs(int(row.get("frame", -1)) - int(record["frame"]))) if pool else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gallery-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identity_dir = args.run / "identity-live"
    rows = [json.loads(line) for line in (identity_dir / "global_identity.jsonl").read_text().splitlines()]
    rows_by_key = {}
    for row in rows:
        rows_by_key.setdefault((row.get("camera_id"), int(row.get("native_track_id", -1))), []).append(row)
    for values in rows_by_key.values():
        values.sort(key=lambda row: int(row.get("frame", -1)))
    records = [json.loads(line) for line in (identity_dir / "embedding_debug.jsonl").read_text().splitlines()]

    representative = {}
    for record in records:
        row = row_for_embedding(rows_by_key, record)
        if row is None:
            continue
        application = row.get("canonical_global_person_id") or row.get("application_id")
        if application and application != "Unknown" and application not in representative:
            representative[application] = (record, row)
    applications = sorted(representative)
    if len(applications) != 2:
        raise AssertionError(f"expected the replay’s two inferred identities, got {applications}")

    # The current state is intentionally empty during the simulated absence.
    assert_presence_contract([], set())
    results = []
    false_matches = []
    similarities = []
    reacquisition_ms = []
    for gap in (timedelta(minutes=5), timedelta(hours=1), timedelta(hours=6)):
        store = IdentityGalleryStore(args.gallery_db)
        manager = GlobalIdentityManager(np.empty((0, 512), np.float32), gallery_store=store)
        for expected in applications:
            record, source_row = representative[expected]
            obs = {
                "camera_id": record["camera_id"],
                "native_track_id": int(record["native_track_id"]) + 10000,
                "frame": int(record["frame"]) + int(gap.total_seconds() * 20.0) + 100000,
                "timestamp": iso(record["timestamp"], gap),
                "bbox": tuple(record["bbox"]),
                "world": tuple(record["world"]),
                "crop_quality_usable": True,
                "crop_quality": {"area_fraction": 0.05, "inside_fraction": 1.0, "height": 500.0},
            }
            started = time.perf_counter_ns()
            recovered, evidence = manager.resolve_existing(obs, np.asarray(record["vector"], dtype=np.float32), [])
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            observed = manager.application_id(recovered) if recovered is not None else "Unknown"
            if observed != expected:
                false_matches.append({"gap": gap.total_seconds(), "expected": expected, "observed": observed})
            if evidence.get("appearance_similarity") is not None:
                similarities.append(float(evidence["appearance_similarity"]))
            reacquisition_ms.append(elapsed)
            results.append({
                "gap_seconds": gap.total_seconds(),
                "expected": expected,
                "observed": observed,
                "success": observed == expected,
                "reason": evidence.get("reason"),
                "similarity": evidence.get("appearance_similarity"),
                "latency_ms": elapsed,
                "visible_presence_after_absence": False,
            })
        if manager.next_number != 3:
            raise AssertionError(f"unexpected new identity allocation at {gap}: {manager.next_number - 1}")
        store.close()

    restart_store = IdentityGalleryStore(args.gallery_db)
    restarted = GlobalIdentityManager(np.empty((0, 512), np.float32), gallery_store=restart_store)
    restart_record, _ = representative[applications[0]]
    restart_obs = {
        "camera_id": restart_record["camera_id"],
        "native_track_id": 20000,
        "frame": 200000,
        "timestamp": iso(restart_record["timestamp"], timedelta(hours=1)),
        "bbox": tuple(restart_record["bbox"]),
        "world": tuple(restart_record["world"]),
        "crop_quality_usable": True,
        "crop_quality": {"area_fraction": 0.05, "inside_fraction": 1.0, "height": 500.0},
    }
    recovered, evidence = restarted.resolve_existing(
        restart_obs, np.asarray(restart_record["vector"], dtype=np.float32), []
    )
    restart_ok = restarted.application_id(recovered) == applications[0]
    assert_presence_contract([], set())
    if not restart_ok:
        raise AssertionError(f"restart recovery failed: {evidence}")
    restart_report = {
        "success": restart_ok,
        "application": applications[0],
        "reason": evidence.get("reason"),
        "similarity": evidence.get("appearance_similarity"),
        "identities_loaded": restarted.persistence_stats["identities_loaded"],
        "gallery_entries_loaded": restarted.persistence_stats["gallery_entries_loaded"],
        "visible_presence_restored": False,
    }
    restart_store.close()

    payload = {
        "dataset_run": str(args.run),
        "applications_inferred_from_replay": applications,
        "reacquisition": results,
        "all_reacquisitions_successful": not false_matches,
        "false_matches": false_matches,
        "new_identity_allocations": 0,
        "osnet_similarity": {
            "count": len(similarities),
            "min": min(similarities) if similarities else None,
            "median": float(np.median(similarities)) if similarities else None,
            "max": max(similarities) if similarities else None,
        },
        "reacquisition_latency_ms": {
            "count": len(reacquisition_ms),
            "median": float(np.median(reacquisition_ms)) if reacquisition_ms else None,
            "p95": float(np.quantile(reacquisition_ms, 0.95)) if reacquisition_ms else None,
            "max": max(reacquisition_ms) if reacquisition_ms else None,
        },
        "sqlite_restart": restart_report,
        "visible_presence_contract": "no marker during all simulated absences",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
