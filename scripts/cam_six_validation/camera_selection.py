"""Camera selection helpers for the diagnostic-only six-camera validator."""

CAMERA_IDS = ("CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06")


def camera_ids_for_run(only_camera: str | None = None) -> tuple[str, ...]:
    if only_camera is None:
        return CAMERA_IDS
    if only_camera not in CAMERA_IDS:
        raise ValueError("camera selection must be CAM-01 through CAM-06")
    return (only_camera,)


def recording_filename(camera_ids: tuple[str, ...]) -> str:
    if camera_ids == CAMERA_IDS:
        return "CAM-01_CAM-02_CAM-03_CAM-04_CAM-05_CAM-06.mkv"
    if len(camera_ids) == 1 and camera_ids[0] in CAMERA_IDS:
        return f"{camera_ids[0]}.mkv"
    raise ValueError("recording filename requires the default six cameras or one selected camera")


def validate_camera_mode(only_camera: str | None, person_detection: bool) -> None:
    if only_camera is not None and person_detection:
        raise ValueError("--only-camera is restricted to the capture-only validator")
