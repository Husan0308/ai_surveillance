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
    parser.add_argument("--max-age-ms", type=float, default=500.0)
    args = parser.parse_args()

    root = Path(args.dir)
    ids = [row.strip() for row in args.cameras.split(",") if row.strip()]
    readers = {}
    last_seq = {cid: 0 for cid in ids}
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
                if snap is None or snap.seq <= last_seq[cid]:
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

    elapsed = max(0.001, time.monotonic() - started)
    failures = []
    for cid in ids:
        hz = counts[cid] / elapsed
        age_avg = sum(ages[cid]) / len(ages[cid]) if ages[cid] else 0.0
        age_max = max(ages[cid]) if ages[cid] else 0.0
        print(
            f"CAMERA_SERVICE_SHM_PROBE {cid} hz={hz:.2f} "
            f"age_avg={age_avg:.1f}ms age_max={age_max:.1f}ms seq={last_seq[cid]}",
            flush=True,
        )
        if hz < args.min_hz:
            failures.append(f"{cid}: update rate {hz:.2f}Hz < {args.min_hz:.2f}Hz")
        if age_max > args.max_age_ms:
            failures.append(f"{cid}: stale frame max_age={age_max:.1f}ms")

    if failures:
        print("CAMERA_SERVICE_SHM_PROBE_FAIL " + " | ".join(failures), flush=True)
        return 2
    print("CAMERA_SERVICE_SHM_PROBE_OK cameras=" + str(len(ids)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
