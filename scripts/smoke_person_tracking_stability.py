from __future__ import annotations

from collections import defaultdict
import json
import os
import time
from urllib.request import Request, urlopen

BASE_URL = os.getenv("ML_SMOKE_URL", "http://127.0.0.1:8001").rstrip("/")
DURATION_SEC = float(os.getenv("ML_TRACK_STABILITY_SEC", "45"))
INTERVAL_SEC = float(os.getenv("ML_TRACK_STABILITY_INTERVAL_SEC", "1.0"))
CAMERA_FILTER = os.getenv("ML_TRACK_STABILITY_CAMERA", "").strip()


def _json(path: str, timeout: float = 3.0) -> dict:
    request = Request(f"{BASE_URL}{path}", headers={"Connection": "close"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    print(
        f"=== Person tracking stability: {BASE_URL} duration={DURATION_SEC:.0f}s "
        f"interval={INTERVAL_SEC:.1f}s ===",
        flush=True,
    )
    health = _json("/health")
    tracker = dict(health.get("tracker") or {})
    if tracker.get("state") != "ready":
        print(f"PERSON_TRACK_STABILITY=FAIL tracker not ready: {tracker}", flush=True)
        return 1

    camera_rows = list(_json("/cameras").get("cameras") or [])
    camera_ids = [str(row["id"]) for row in camera_rows]
    if CAMERA_FILTER:
        if CAMERA_FILTER not in camera_ids:
            print(f"PERSON_TRACK_STABILITY=FAIL unknown camera {CAMERA_FILTER}", flush=True)
            return 1
        camera_ids = [CAMERA_FILTER]

    baseline_created = {
        str(row["id"]): int((row.get("tracking") or {}).get("created_tracks") or 0)
        for row in camera_rows
        if str(row["id"]) in camera_ids
    }
    seen_ids: dict[str, set[int]] = defaultdict(set)
    previous_ids: dict[str, tuple[int, ...]] = {camera_id: () for camera_id in camera_ids}
    previous_count: dict[str, int] = {camera_id: 0 for camera_id in camera_ids}
    id_set_changes = defaultdict(int)
    same_count_id_changes = defaultdict(int)
    occupancy_changes = defaultdict(int)
    zero_samples = defaultdict(int)
    samples = defaultdict(int)
    peak_active = defaultdict(int)

    deadline = time.monotonic() + max(1.0, DURATION_SEC)
    while time.monotonic() < deadline:
        health = _json("/health")
        tracker = dict(health.get("tracker") or {})
        if tracker.get("state") != "ready":
            print(f"PERSON_TRACK_STABILITY=FAIL tracker state={tracker}", flush=True)
            return 1

        for camera_id in camera_ids:
            payload = _json(f"/tracks/{camera_id}")
            result = dict(payload.get("result") or {})
            tracks = list(result.get("tracks") or [])
            ids = tuple(sorted(int(track["track_id"]) for track in tracks))
            if len(ids) != len(set(ids)):
                print(
                    f"PERSON_TRACK_STABILITY=FAIL {camera_id} duplicate IDs {list(ids)}",
                    flush=True,
                )
                return 1

            count = len(ids)
            samples[camera_id] += 1
            peak_active[camera_id] = max(peak_active[camera_id], count)
            seen_ids[camera_id].update(ids)
            if count == 0:
                zero_samples[camera_id] += 1

            if samples[camera_id] > 1:
                if ids != previous_ids[camera_id]:
                    id_set_changes[camera_id] += 1
                if count != previous_count[camera_id]:
                    occupancy_changes[camera_id] += 1
                elif count > 0 and ids != previous_ids[camera_id]:
                    same_count_id_changes[camera_id] += 1

            previous_ids[camera_id] = ids
            previous_count[camera_id] = count

        time.sleep(max(0.1, INTERVAL_SEC))

    final_rows = {
        str(row["id"]): row for row in list(_json("/cameras").get("cameras") or [])
    }
    quality_warnings = 0
    for camera_id in camera_ids:
        tracking = dict(final_rows.get(camera_id, {}).get("tracking") or {})
        created_now = int(tracking.get("created_tracks") or 0)
        created_delta = max(0, created_now - baseline_created.get(camera_id, 0))
        distinct = len(seen_ids[camera_id])
        same_count_switches = int(same_count_id_changes[camera_id])
        print(
            f"[STABLE] {camera_id} samples={samples[camera_id]} peak={peak_active[camera_id]} "
            f"distinct_seen={distinct} created_delta={created_delta} "
            f"id_set_changes={id_set_changes[camera_id]} "
            f"occupancy_changes={occupancy_changes[camera_id]} "
            f"same_count_id_changes={same_count_switches} zero_samples={zero_samples[camera_id]}",
            flush=True,
        )

        # This is a diagnostic warning, not a hard failure. Entry/exit can
        # legitimately create IDs. A same-count ID-set change is more suspicious
        # because occupancy did not change while identity did.
        if same_count_switches >= 3:
            quality_warnings += 1
            print(
                f"[QUALITY] {camera_id} WARN repeated same-count ID changes; ByteTrack tuning needed",
                flush=True,
            )

    print(
        f"PERSON_TRACK_STABILITY=PASS quality_warnings={quality_warnings}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
