from __future__ import annotations

import json
import hashlib

import cv2
import numpy as np
import pytest

from services.mv3dt_staging.calibration_editor import LIVE_NATIVE_SPACE
from services.mv3dt_staging.native_map_preview import (
    diagnose_projected_motion,
    load_validated_candidate,
    project_anchor,
)


def _candidate(tmp_path, *, include_holdouts=True):
    layout_path = tmp_path / "layout.png"
    layout = np.zeros((300, 400, 3), np.uint8)
    assert cv2.imwrite(str(layout_path), layout)
    cam02 = tmp_path / "cam02.mp4"
    cam05 = tmp_path / "cam05.mp4"
    cam02.write_bytes(b"raw-cam02")
    cam05.write_bytes(b"raw-cam05")

    # Coordinates are generated from a deterministic projective transform.
    h = np.asarray([[0.22, 0.01, 30.0], [-0.02, 0.25, 20.0],
                    [0.00001, 0.00002, 1.0]], dtype=float)
    image_points = [(100, 100), (500, 100), (500, 500), (100, 500),
                    (280, 180), (350, 400), (180, 360)]
    points = []
    for index, xy in enumerate(image_points):
        mapped = h @ np.asarray([xy[0], xy[1], 1.0])
        point_id = f"P{index + 1:02d}"
        is_holdout = include_holdouts and index >= 5
        points.append({
            "point_id": point_id,
            "cam02": list(xy),
            "cam05": list(xy),
            "layout": (mapped[:2] / mapped[2]).tolist(),
            "role": "heldout" if is_holdout else "fit",
            "provenance": "operator_selected_new_floor_landmark",
            "prior_alignment_reference": False,
            "independent_of_prior_fit": bool(is_holdout),
            "heldout_attestation": "operator_attests_new_physical_target_not_in_prior_fit_eight" if is_holdout else None,
            "cam02_source_sha256": hashlib.sha256(cam02.read_bytes()).hexdigest(),
            "cam05_source_sha256": hashlib.sha256(cam05.read_bytes()).hexdigest(),
            "cam02_coordinate_space": LIVE_NATIVE_SPACE,
            "cam05_coordinate_space": LIVE_NATIVE_SPACE,
            "layout_verified": True,
            "layout_source_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
        })
    payload = {
        "schema_version": 2,
        "coordinate_space": LIVE_NATIVE_SPACE,
        "camera_mapping": {"cam02": "CAM-02", "cam05": "CAM-05"},
        "camera_coordinate_dimensions": {"cam02": [1920, 1080], "cam05": [1920, 1080]},
        "source_files": {
            "cam02": {"sha256": hashlib.sha256(cam02.read_bytes()).hexdigest(), "width": 3200, "height": 1800},
            "cam05": {"sha256": hashlib.sha256(cam05.read_bytes()).hexdigest(), "width": 2560, "height": 1440},
        },
        "layout": {"sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest()},
        "points": points,
    }
    alignment_path = tmp_path / "LIVE_NATIVE_ALIGNMENT-v0001.json"
    alignment_path.write_text(json.dumps(payload))
    return alignment_path, {"cam02": cam02, "cam05": cam05}, layout_path


def test_preview_gate_requires_two_new_independent_holdouts_per_camera(tmp_path) -> None:
    alignment, videos, layout = _candidate(tmp_path, include_holdouts=False)
    with pytest.raises(ValueError, match=">=2 NEW independent HELD-OUT"):
        load_validated_candidate(alignment, video_paths=videos, layout_path=layout)


def test_preview_gate_fits_live_native_alignment_in_1920_space(tmp_path) -> None:
    alignment, videos, layout = _candidate(tmp_path)
    matrices, diagnostics, _ = load_validated_candidate(
        alignment, video_paths=videos, layout_path=layout)
    assert set(matrices) == {"cam02", "cam05"}
    assert diagnostics["cam02"]["fit_point_count"] == 5
    assert diagnostics["cam02"]["heldout_point_count"] == 2
    assert diagnostics["cam02"]["heldout_rms_layout_px"] < 1e-4


def test_bottom_center_anchor_projection_and_bounds() -> None:
    identity = np.eye(3)
    assert project_anchor(identity, [10, 20, 30, 40], 100, 100) == (25.0, 60.0)
    assert project_anchor(identity, [90, 10, 20, 20], 100, 100) is None
    assert project_anchor(identity, [0, 0, -1, 2], 100, 100) is None


def test_projected_motion_flags_large_jumps_and_nearby_local_id_replacements() -> None:
    tracks = {
        "CAM-02": {
            0: [{"object_id": "7", "box": [0, 0, 20, 20]}],
            1: [{"object_id": "7", "box": [100, 0, 20, 20]}],
            2: [{"object_id": "8", "box": [105, 0, 20, 20]},
                {"object_id": "9", "box": [200, 0, 20, 20]}],
        },
        "CAM-05": {},
    }
    local_pts = {
        "CAM-02": {0: 0, 1: 50_000_000, 2: 100_000_000},
        "CAM-05": {},
    }
    result = diagnose_projected_motion(
        tracks, local_pts,
        {"cam02": np.eye(3), "cam05": np.eye(3)},
        layout_width=150,
        layout_height=100,
    )
    cam02 = result["cameras"]["CAM-02"]
    assert len(cam02["large_projected_jump_candidates"]) == 1
    assert cam02["large_projected_jump_candidates"][0]["local_tracker_id"] == "7"
    assert cam02["out_of_bounds_projected_anchors"] == 1
    assert cam02["invalid_projected_anchors"] == 0
    assert cam02["out_of_bounds_events"][0]["local_tracker_id"] == "9"
    assert any(event["from_local_tracker_id"] == "7" and
               event["to_local_tracker_id"] == "8"
               for event in cam02["possible_local_id_replacement_candidates"])
    assert "not physical distance" in result["units_warning"]
