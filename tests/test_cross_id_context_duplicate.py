import subprocess
from pathlib import Path

from scripts.replay_cross_id_context_duplicate_candidate import admit_birth_batch, evaluate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/native/test_cross_id_context_duplicate.cpp"


def test_metadata_replay_fails_open_when_two_births_claim_one_owner():
    proposals = [
        {"confidence": .806152, "bbox": [376.768, 414.357, 183.036, 249.286]},
        {"confidence": .819336, "bbox": [549.714, 492.571, 263.571, 300.714]},
    ]
    births = [{"candidate_id": 20, "proposals": proposals},
              {"candidate_id": 21, "proposals": proposals}]
    current = {
        20: [431.812, 540.750, 126.376, 126.000],
        21: [431.812, 540.750, 126.376, 126.000],
        3: [376.875, 414.000, 178.500, 205.500],
        4: [550.500, 493.875, 262.500, 275.250],
        5: [803.250, 699.000, 288.000, 244.500],
    }
    decisions = admit_birth_batch(births, current, {3, 4, 5}, 1920, 1080, True)
    assert len(decisions) == 2
    assert all(not decision["admit"] for decision in decisions)
    assert all(decision["reason"] == "AMBIGUOUS_MULTIPLE_BIRTHS_CLAIM_OWNER"
               for decision in decisions)


def test_cross_id_context_duplicate_policy_positive_negative_and_epoch_safety(tmp_path):
    executable = tmp_path / "test-cross-id-context-duplicate"
    subprocess.run([
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(ROOT), str(SOURCE), "-o", str(executable),
    ], check=True, cwd=ROOT)
    subprocess.run([str(executable)], check=True, cwd=ROOT)


def test_metadata_replay_does_not_use_raw_withheld_track_as_duplicate_owner() -> None:
    candidate = [102.938, 161.812, 103.688, 250.875]
    other_a = [463.688, 332.812, 196.125, 285.375]
    other_b = [610.5, 483.75, 235.5, 300.0]
    withheld_owner = [103.875, 156.0, 216.0, 331.5]
    proposals = [
        {"confidence": .759766, "bbox": [462.857, 332.143, 177.143, 305.714]},
        {"confidence": .910645, "bbox": [103.929, 157.5, 214.286, 331.071]},
    ]
    data = {
        "camera": "CAM-02",
        "directory": Path("saved-fixture"),
        "source_id": 1,
        "canonical": {
            9: {"pts_ns": 90, "pipeline_width": 1920, "pipeline_height": 1080,
                "bbox_coordinate_width": 1920, "bbox_coordinate_height": 1080,
                "objects": [{"native_track_id": 44}]},
            10: {"pts_ns": 100, "pipeline_width": 1920, "pipeline_height": 1080,
                 "bbox_coordinate_width": 1920, "bbox_coordinate_height": 1080,
                 "objects": [{"native_track_id": 51}, {"native_track_id": 1},
                             {"native_track_id": 2}]},
        },
        "published_ids": {9: {44}, 10: {51, 1, 2}},
        "tracks": {
            9: {44: withheld_owner},
            # ID 44 still exists in raw NvDCF rows but active-birth
            # publication rejected it at frame 10. It is not an eligible
            # owner in the native canonical filter either.
            10: {44: withheld_owner, 51: candidate, 1: other_a, 2: other_b},
        },
        "shadow_boxes": {},
        "events": {10: [{"candidate_id": 51, "pts_ns": 100,
                         "proposals": proposals, "verifier_confidence": .910645}]},
    }
    result = evaluate(data)
    decision = result["context_birth_decisions"][0]
    assert decision["admit"] is False
    assert decision["reason"] == "NO_STRONG_CONTEXT_EXCLUSIVITY"
    assert result["suppressed_by_frame"] == {}
    assert result["candidate_canonical_rows"] == result["raw_canonical_rows"]
