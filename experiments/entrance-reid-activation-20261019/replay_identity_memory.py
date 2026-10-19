#!/usr/bin/env python3
"""Feed validated same-frame MV3DT/NvDCF embeddings into fail-closed memory.

This diagnostic intentionally has no calibrated threshold profile and no
validated world coordinates. It may retain verified feature vectors in the
temporary in-memory gallery, but must not accept an appearance identity match.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from services.mv3dt_staging.person_identity_memory import (  # noqa: E402
    IdentityObservation,
    MemoryConfig,
    PersonIdentityMemory,
)


def mqtt_ids(path: Path):
    decoder = json.JSONDecoder()
    topic_to_source = {"/trck/cam1": 0, "/trck/cam2": 1}
    ids = defaultdict(set)
    malformed = 0
    for line in path.open():
        try:
            topic, payload_hex = line.rstrip("\n").split("|", 1)
            source = topic_to_source[topic]
            payload = bytes.fromhex(payload_hex).decode("utf-8")
            cursor = 0
            while cursor < len(payload):
                while cursor < len(payload) and payload[cursor].isspace():
                    cursor += 1
                if cursor >= len(payload):
                    break
                message, cursor = decoder.raw_decode(payload, cursor)
                if str(message.get("type")) in {"1", "2"}:
                    frame = message.get("frameNum")
                    raw_id = message.get("targetId")
                    if isinstance(frame, int) and isinstance(raw_id, int):
                        ids[(source, frame)].add(raw_id)
                if cursor < len(payload) and payload[cursor] == "|":
                    cursor += 1
                elif cursor < len(payload) and not payload[cursor].isspace():
                    raise ValueError("invalid message delimiter")
        except Exception:
            malformed += 1
    if malformed:
        raise ValueError(f"{malformed} malformed MQTT payload publications")
    return ids


def kitti_ids(path: Path):
    ids = defaultdict(set)
    for file_path in path.glob("00_???_??????.txt"):
        _, stream, frame = file_path.stem.split("_")
        key = (int(stream), int(frame))
        for line in file_path.read_text(errors="replace").splitlines():
            columns = line.split()
            if len(columns) >= 2:
                try:
                    ids[key].add(int(columns[1]))
                except ValueError:
                    pass
    return ids


def read_vectors(path: Path):
    output = []
    duplicate_keys = Counter()
    invalid_rows = 0
    for file_path in sorted(path.glob("00_???_??????.txt")):
        _, stream, frame = file_path.stem.split("_")
        source, frame_id = int(stream), int(frame)
        for line in file_path.read_text(errors="replace").splitlines():
            columns = line.split()
            if not columns:
                continue
            try:
                raw_id = int(columns[0])
                vector = np.asarray([float(value) for value in columns[1:]], dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if (vector.size != 256 or not np.isfinite(vector).all()
                        or not math.isfinite(norm) or not 0.9999 <= norm <= 1.0001):
                    raise ValueError("not a finite normalized 256D vector")
                key = (source, frame_id, raw_id)
                duplicate_keys[key] += 1
                output.append((source, frame_id, raw_id, vector))
            except Exception:
                invalid_rows += 1
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    return output, invalid_rows, duplicate_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    target_ids = mqtt_ids(run / "mqtt-raw-hex.log")
    tracker_ids = kitti_ids(run / "run" / "tracker-kitti-dump")
    vectors, invalid_rows, duplicate_rows = read_vectors(run / "reid-feature-output")

    # These are verified vector records, not subjective crop-quality scores.
    # A zero quality plus zero minimum admits valid vectors to a bounded test
    # gallery. No threshold profile and no validated location are supplied.
    memory = PersonIdentityMemory(
        MemoryConfig(
            retention_seconds=30.0,
            active_timeout_seconds=1.5,
            max_embeddings_per_person=8,
            min_embedding_quality=0.0,
            min_similarity=None,
            min_best_second_margin=None,
            threshold_profile_id=None,
            require_validated_location=True,
        ),
        run_id="entrance-run06-reid-only-offline",
    )
    accepted_rows = 0
    skipped_no_mqtt_crosswalk = 0
    skipped_no_kitti_crosswalk = 0
    statuses = Counter()
    camera_names = {0: "CAM-02", 1: "CAM-05"}
    for source, frame, raw_id, vector in sorted(vectors, key=lambda item: (item[1], item[0], item[2])):
        if raw_id not in tracker_ids.get((source, frame), set()):
            skipped_no_kitti_crosswalk += 1
            continue
        if raw_id not in target_ids.get((source, frame), set()):
            skipped_no_mqtt_crosswalk += 1
            continue
        decision = memory.observe(IdentityObservation(
            run_id="entrance-run06-reid-only-offline",
            camera_id=camera_names[source],
            raw_id=str(raw_id),
            timestamp_seconds=frame / 30.0,
            embedding=vector,
            embedding_quality=0.0,
            world_xy=None,
            world_geometry_validated=False,
            raw_id_scope="mv3dt_global",
            frame_id=frame,
        ))
        accepted_rows += 1
        statuses[decision["match_status"]] += 1

    output = {
        "run_directory": str(run),
        "raw_tracker_ids_modified": False,
        "source_mapping": {"0": "CAM-02", "1": "CAM-05"},
        "records": {
            "verified_vectors_read": len(vectors),
            "invalid_vector_rows": invalid_rows,
            "duplicate_same_source_frame_raw_id_rows": duplicate_rows,
            "fed_to_identity_memory_after_exact_source_frame_crosswalk": accepted_rows,
            "skipped_without_kitti_identity_crosswalk": skipped_no_kitti_crosswalk,
            "skipped_without_mqtt_target_crosswalk": skipped_no_mqtt_crosswalk,
        },
        "memory": memory.snapshot(),
        "decision_status_counts": dict(sorted(statuses.items())),
        "automatic_reidentification_matches": sum(
            count for status, count in statuses.items() if status == "appearance_reidentified"
        ),
        "policy": (
            "No calibrated appearance threshold/profile and no validated world geometry are configured. "
            "This replay can retain validated vectors in a bounded in-memory gallery but cannot merge raw IDs. "
            "The frame/30 timeline is nominal replay time, not verified cross-camera exposure synchronization."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
