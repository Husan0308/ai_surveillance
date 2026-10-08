from __future__ import annotations

from scripts.run_integrated_bbox_reference_replay import CAMERAS, INPUTS, reference_camera_config


def test_recorded_reference_mode_uses_six_sources_without_camera_credentials():
    config = reference_camera_config()
    assert [line for line in config.splitlines() if line.startswith("[CAM-")] == [
        f"[{camera}]" for camera in CAMERAS
    ]
    assert "127.0.0.1:8554/cam01" not in config  # RTSP URI is base64 in the source profile.
    assert "username=" in config and "password=" in config
    assert "preview_enabled=false" in config
    assert set(INPUTS) == {"CAM-01", "CAM-04"}


def test_reference_pair_is_pinned_to_the_approved_video_contract():
    assert all(len(digest) == 64 and path.name.endswith(".mp4") for path, digest in INPUTS.values())
