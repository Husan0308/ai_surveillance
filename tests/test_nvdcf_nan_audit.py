import json

from scripts.audit_nvdcf_invalid_shadow_geometry import analyze


def test_invalid_shadow_is_inactive_and_not_emitted_as_canonical(tmp_path):
    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    write("invalid_shadow_geometry.jsonl", [{
        "source_id": 0, "camera_id": "CAM-01", "frame": 5, "pts_ns": 10,
        "native_track_id": 42, "age": 4, "state_value": 2,
        "confidence_finite": True, "bbox_finite": [True] * 4, "bbox_positive": False,
    }])
    write("detections.jsonl", [{"source_id": 0, "frame": 5, "class_id": 0,
                                "confidence": 0.8, "box": [1, 2, 3, 4]}])
    write("tracks.jsonl", [{"source_id": 0, "frame": 5, "object_id": 99,
                            "box": [1, 2, 3, 4]}])
    write("shadow_fallbacks.jsonl", [])
    write("frames.jsonl", [{"source_id": 0, "frame": 5}])
    (tmp_path / "native.log").write_text(
        "Width and height of input and output bounding boxes from "
        "vpiSubmitDCFTrackerLocalizeBatch() are NAN!!\n"
    )

    report = analyze(tmp_path)
    cam = report["per_camera"]["CAM-01"]
    assert cam["mapped_state_names"] == {"2": "INACTIVE/shadow"}
    assert cam["all_bbox_components_finite"] is True
    assert cam["all_bbox_rows_have_nonpositive_width_or_height"] is True
    assert cam["invalid_rows_with_current_detector_person_proposal"] == 1
    assert cam["invalid_rows_same_native_id_active_output"] == 0
    assert report["canonical_feed_safety"]["invalid_geometry_reaches_canonical_output"] is False
    assert report["native_vpi_nan_warning_lines"] == 1
