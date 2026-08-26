from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# `python scripts/probe_camera_service_shm.py` puts scripts/ at sys.path[0], not
# the repository root. Make this diagnostic executable directly from the repo
# without requiring users to export PYTHONPATH or install the project package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.camera_service.app.shm_frame import LatestFrameMmapReader


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe latest-frame camera_service SHM without ML")
    parser.add_argument("--dir", default="/dev/shm/ai_surveillance")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--cameras", default="CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06")
    parser.add_argument("--min-hz", type=float, default=1.50)
    parser.add_argument("--max-age-ms", type=float, default=250.0)
    args = parser.parse_args()

    root = Path(args.dir)
    ids = [row.strip() for row in args.cameras.split(",") if row.strip()]
    readers = {}
    # The first snapshot visible when the probe attaches may have been published
    # before the probe started. Treat it as a baseline sequence, not as a fresh
    # update; otherwise a healthy 2 Hz producer can legitimately look ~500 ms
    # "stale" if the probe happens to attach just before the next publication.
    last_seq: dict[str, int | None] = {cid: None for cid in ids}
    active_since: dict[str, float | None] = {cid: None for cid in ids}
    counts = {cid: 0 for cid in ids}
    ages = {cid: [] for cid in ids}
    started = time.monotonic()
    deadline = started + max(2.0, args.seconds)

    try:
        while time.monotonic() < deadline:
            for cid in ids:
                if cid not in readers:
                    path = root / f"{cid.lower().replace('-', '_')}.frame"
                    if path.exists():
                        readers[cid] = LatestFrameMmapReader(path)
                reader = readers.get(cid)
                if reader is None:
                    continue

                snap = reader.latest()
                if snap is None:
                    continue

                if last_seq[cid] is None:
                    # Establish an attach-time baseline. Freshness/rate statistics
                    # begin with the next publication after this point.
                    last_seq[cid] = snap.seq
                    active_since[cid] = time.monotonic()
                    continue

                if snap.seq <= int(last_seq[cid]):
                    continue

                last_seq[cid] = snap.seq
                counts[cid] += 1
                age_ms = max(0.0, (time.monotonic_ns() - snap.captured_ns) / 1_000_000.0)
                ages[cid].append(age_ms)
                if snap.width != 672 or snap.height != 378 or snap.stride != 2016:
                    raise RuntimeError(
                        f"{cid}: bad geometry {snap.width}x{snap.height} stride={snap.stride}"
                    )
            time.sleep(0.01)
    finally:
        for reader in readers.values():
            reader.close()

    finished = time.monotonic()
    failures = []
    for cid in ids:
        camera_started = active_since[cid]
        elapsed = max(0.001, finished - camera_started) if camera_started is not None else 0.0
        hz = counts[cid] / elapsed if elapsed > 0.0 else 0.0
        age_avg = sum(ages[cid]) / len(ages[cid]) if ages[cid] else 0.0
        age_max = max(ages[cid]) if ages[cid] else 0.0
        seq = 0 if last_seq[cid] is None else int(last_seq[cid])
        print(
            f"CAMERA_SERVICE_SHM_PROBE {cid} hz={hz:.2f} "
            f"age_avg={age_avg:.1f}ms age_max={age_max:.1f}ms seq={seq}",
            flush=True,
        )
        if camera_started is None:
            failures.append(f"{cid}: SHM file not observed")
            continue
        if hz < args.min_hz:
            failures.append(f"{cid}: update rate {hz:.2f}Hz < {args.min_hz:.2f}Hz")
        if not ages[cid]:
            failures.append(f"{cid}: no post-attach updates")
        elif age_max > args.max_age_ms:
            failures.append(f"{cid}: post-attach stale frame max_age={age_max:.1f}ms")

    if failures:
        print("CAMERA_SERVICE_SHM_PROBE_FAIL " + " | ".join(failures), flush=True)
        return 2
    print("CAMERA_SERVICE_SHM_PROBE_OK cameras=" + str(len(ids)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
