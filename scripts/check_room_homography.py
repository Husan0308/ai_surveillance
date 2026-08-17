from __future__ import annotations

from services.camera_v2.manual_geometry_reid import ManualRoomHomography


EXPECTED_PAIRS = ((0, 3), (1, 4), (2, 5))


def main() -> int:
    calibration = ManualRoomHomography()
    snap = calibration.snapshot()
    print(f"CALIBRATION_FILE={snap['path']}")
    for source in snap["camera_sources"]:
        row = calibration.cameras[source]
        print(
            f"source={source} room={row.room_id} "
            f"reprojection_rmse={row.reprojection_rmse_m:.4f}m"
        )
    for error in snap["errors"]:
        print(f"WARNING {error}")

    missing = []
    for a, b in EXPECTED_PAIRS:
        if a not in calibration.cameras or b not in calibration.cameras:
            missing.append((a, b))
            continue
        if calibration.cameras[a].room_id != calibration.cameras[b].room_id:
            print(f"ERROR pair {a}-{b} maps to different room ids")
            return 2

    if missing:
        print("CALIBRATION_READY=NO missing_pairs=" + ",".join(f"{a}-{b}" for a, b in missing))
        return 1
    print("CALIBRATION_READY=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
