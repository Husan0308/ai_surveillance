from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from services.mv3dt_staging.event_store import CAMERA_MAPS, DEFAULT_FPS, EXPECTED_FRAMES, import_event_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a preserved MV3DT Kafka text log into a staging-only SQLite store.")
    parser.add_argument("--event-log", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help="nominal normalized video rate; does not assert source PTS")
    parser.add_argument("--expected-frames", type=int, default=EXPECTED_FRAMES,
                        help="nominal per-camera frame count recorded as dataset metadata")
    parser.add_argument("--camera-profile", choices=sorted(CAMERA_MAPS), default="dev-room",
                        help="explicit input sensor-to-physical-camera mapping")
    parser.add_argument("--cam00-sha256", default="")
    parser.add_argument("--cam01-sha256", default="")
    parser.add_argument("--calibration-sha256", default="")
    args = parser.parse_args()
    provenance = {
        "cam00_sha256": args.cam00_sha256,
        "cam01_sha256": args.cam01_sha256,
        "calibration_zip_sha256": args.calibration_sha256,
    }
    provenance = {key: value for key, value in provenance.items() if value}
    result = import_event_log(args.event_log, args.db, fps=args.fps,
                              expected_frames=args.expected_frames, provenance=provenance,
                              camera_mapping=CAMERA_MAPS[args.camera_profile])
    os.chmod(args.db, 0o600)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

