import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dev_room_mv3dt/diagnose_retention_losses.py"

spec = importlib.util.spec_from_file_location("diagnose_retention_losses", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_match_by_confidence_is_multiset_safe():
    pgie = [
        {"confidence": 0.25},
        {"confidence": 0.25},
        {"confidence": 0.8},
    ]
    tracker = [
        {"confidence": 0.25},
        {"confidence": 0.8},
        {"confidence": -0.1},  # shadow-only output must not consume a detector box
    ]
    matched, unmatched = mod.match_by_confidence(pgie, tracker)
    assert matched == 2
    assert unmatched == [0.25]


def test_contiguous_windows():
    assert mod.contiguous_windows([1, 2, 3, 7, 8, 10]) == [
        {"start": 1, "end": 3, "frames": 3},
        {"start": 7, "end": 8, "frames": 2},
        {"start": 10, "end": 10, "frames": 1},
    ]


def test_confidence_bands():
    assert mod.band(0.49) == "below_0.50"
    assert mod.band(0.50) == "0.50_to_0.70"
    assert mod.band(0.69) == "0.50_to_0.70"
    assert mod.band(0.70) == "at_least_0.70"
