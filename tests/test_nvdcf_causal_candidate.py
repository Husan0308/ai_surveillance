from pathlib import Path

from scripts.dev_room_mv3dt.diagnose_canonical_retention import config_values

ROOT = Path(__file__).resolve().parents[1]


def test_v9_changes_only_pose_anchor_from_effective_v8():
    before = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v8_early2.yml').read_text())
    # The archived V8 runner, not its comment-matched template, defines baseline.
    before['TargetManagement']['earlyTerminationAge'] = 2
    after = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v9_bbox_anchor.yml').read_text())
    assert before['PoseEstimator']['poseEstimatorType'] == 1
    assert after['PoseEstimator']['poseEstimatorType'] == 0
    after['PoseEstimator']['poseEstimatorType'] = 1
    assert before == after


def test_v10_changes_only_local_iou_from_effective_v8():
    before = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v8_early2.yml').read_text())
    before['TargetManagement']['earlyTerminationAge'] = 2
    after = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v10_iou004.yml').read_text())
    assert after['DataAssociator']['minMatchingScore4Iou'] == .04
    after['DataAssociator']['minMatchingScore4Iou'] = .10
    assert before == after


def test_v11_changes_one_geometry_factor_from_v10():
    before = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v10_iou004.yml').read_text())
    after = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v11_size004.yml').read_text())
    assert after['DataAssociator']['minMatchingScore4SizeSimilarity'] == .04
    after['DataAssociator']['minMatchingScore4SizeSimilarity'] = .30
    assert before == after


def test_v12_discards_rejected_relaxations_and_changes_only_suppression():
    before = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v8_early2.yml').read_text())
    before['TargetManagement']['earlyTerminationAge'] = 2
    after = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v12_newtarget050.yml').read_text())
    assert after['TargetManagement']['minIouDiff4NewTarget'] == .50
    after['TargetManagement']['minIouDiff4NewTarget'] = .35
    assert before == after


def test_v13_changes_only_probation_from_v12():
    before = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v12_newtarget050.yml').read_text())
    after = config_values((ROOT / 'config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml').read_text())
    assert after['TargetManagement']['probationAge'] == 0
    after['TargetManagement']['probationAge'] = 2
    assert before == after
