import pytest

from scripts.evaluate_detector_annotations import evaluate_frame, evaluate_review, match_iou_matrix


def test_iou_matching_is_one_to_one_when_multiple_proposals_overlap_one_person():
    result = evaluate_frame(
        [[0, 0, 10, 10]],
        [[0, 0, 10, 10], [1, 1, 9, 9]],
        threshold=0.5,
    )
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_matching_maximizes_qualified_pair_count_before_iou_sum():
    # Greedy highest-IoU-first would take (0, 0) at .90 and leave only one
    # match; the maximum-cardinality assignment correctly takes both .51s.
    matches = match_iou_matrix([[0.90, 0.51], [0.51, 0.49]], threshold=0.50)
    assert {(i, j) for i, j, _ in matches} == {(0, 1), (1, 0)}


def test_matching_returns_no_pairs_for_empty_or_below_threshold_inputs():
    assert match_iou_matrix([], threshold=0.5) == []
    assert match_iou_matrix([[0.49]], threshold=0.5) == []


def test_review_evaluator_refuses_unannotated_frames():
    review = {
        "images": [{
            "camera": "CAM-02",
            "annotation_complete": False,
            "human_verified_annotations": [],
            "proposals": {"YOLO26m": []},
        }]
    }
    with pytest.raises(ValueError, match="no frames marked annotation_complete"):
        evaluate_review(review, "YOLO26m", threshold=0.5)


def test_review_evaluator_scores_only_completed_frames():
    review = {
        "images": [
            {
                "camera": "CAM-02",
                "annotation_complete": True,
                "human_verified_annotations": [{"bbox_xyxy": [0, 0, 10, 10]}],
                "proposals": {"YOLO26m": [{"xyxy": [0, 0, 10, 10]}]},
            },
            {
                "camera": "CAM-05",
                "annotation_complete": False,
                "human_verified_annotations": [],
                "proposals": {"YOLO26m": []},
            },
        ]
    }
    result = evaluate_review(review, "YOLO26m", threshold=0.5)
    assert result["totals"] == {
        "tp": 1, "fp": 0, "fn": 0, "frames": 1,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert result["by_camera"] == {"CAM-02": {"tp": 1, "fp": 0, "fn": 0, "frames": 1}}
