from services.ml_service.vision_v3.rfdetr_detection import ProtectiveBoxManager


def _cfg():
    return {
        "side_margin": 0.10,
        "top_margin": 0.08,
        "bottom_margin": 0.12,
        "sitting_extra_side": 0.05,
        "sitting_extra_bottom": 0.04,
        "sitting_aspect_threshold": 1.55,
        "duplicate_iou": 0.80,
        "hold_sec": 1.10,
        "predict_sec": 0.70,
        "expand_alpha": 0.88,
        "shrink_alpha": 0.20,
        "position_alpha": 0.82,
        "velocity_alpha": 0.38,
        "min_width": 12,
        "min_height": 24,
    }


def test_envelope_contains_raw_person_box():
    manager = ProtectiveBoxManager(1280, 720, _cfg())
    raw = (500.0, 100.0, 620.0, 620.0)
    manager.update("CAM-01", 1.0, [(raw, 0.9)])
    x1, y1, x2, y2, confidence = manager.render("CAM-01", 1.0)[0]
    assert x1 < raw[0]
    assert y1 < raw[1]
    assert x2 > raw[2]
    assert y2 > raw[3]
    assert confidence == 0.9


def test_sitting_person_gets_extra_horizontal_and_bottom_guard():
    manager = ProtectiveBoxManager(1280, 720, _cfg())
    standing = manager._guard((100.0, 100.0, 200.0, 500.0))
    sitting = manager._guard((400.0, 200.0, 600.0, 430.0))
    standing_raw_w = 100.0
    sitting_raw_w = 200.0
    assert standing[2] - standing[0] > standing_raw_w
    assert sitting[2] - sitting[0] > sitting_raw_w
    # Sitting/crouched shapes intentionally receive a stronger side envelope.
    standing_side_ratio = ((standing[2] - standing[0]) - standing_raw_w) / standing_raw_w
    sitting_side_ratio = ((sitting[2] - sitting[0]) - sitting_raw_w) / sitting_raw_w
    assert sitting_side_ratio > standing_side_ratio


def test_duplicate_boxes_are_suppressed():
    manager = ProtectiveBoxManager(1280, 720, _cfg())
    rows = [
        ((100.0, 100.0, 250.0, 600.0), 0.91),
        ((102.0, 102.0, 249.0, 598.0), 0.70),
    ]
    manager.update("CAM-01", 1.0, rows)
    assert len(manager.render("CAM-01", 1.0)) == 1


def test_box_expands_fast_but_shrinks_slowly():
    manager = ProtectiveBoxManager(1280, 720, _cfg())
    manager.update("CAM-01", 1.0, [((500.0, 100.0, 600.0, 600.0), 0.9)])
    first = manager.render("CAM-01", 1.0)[0]
    manager.update("CAM-01", 1.2, [((510.0, 140.0, 590.0, 560.0), 0.9)])
    second = manager.render("CAM-01", 1.2)[0]
    # A single tighter detector observation must not instantly clip the envelope.
    assert (second[2] - second[0]) > 0.80 * (first[2] - first[0])
    assert (second[3] - second[1]) > 0.80 * (first[3] - first[1])
