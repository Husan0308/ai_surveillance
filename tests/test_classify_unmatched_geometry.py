import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dev_room_mv3dt/classify_unmatched_geometry.py"

spec = importlib.util.spec_from_file_location("classify_unmatched_geometry", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_iou_identity_and_disjoint():
    a = (10.0, 10.0, 20.0, 20.0)
    assert mod.iou(a, a) == 1.0
    assert mod.iou(a, (100.0, 100.0, 5.0, 5.0)) == 0.0


def test_pick_common_aliases():
    fields = ["camera_id", "frame_num", "confidence", "left", "top", "width", "height"]
    assert mod.pick(fields, "camera") == "camera_id"
    assert mod.pick(fields, "frame") == "frame_num"
    assert mod.pick(fields, "confidence") == "confidence"


def test_unmatched_by_confidence_is_multiset_safe():
    frames = {
        10: {
            "pgie": [
                {"confidence": 0.25},
                {"confidence": 0.25},
                {"confidence": 0.9},
            ],
            "tracker": [
                {"confidence": 0.25},
                {"confidence": 0.9},
                {"confidence": -0.1},
            ],
        }
    }
    assert mod.unmatched_by_confidence(frames) == [{"frame": 10, "confidence": 0.25}]


def test_band_boundaries():
    assert mod.band(0.49) == "below_0.50"
    assert mod.band(0.50) == "0.50_to_0.70"
    assert mod.band(0.69) == "0.50_to_0.70"
    assert mod.band(0.70) == "at_least_0.70"
