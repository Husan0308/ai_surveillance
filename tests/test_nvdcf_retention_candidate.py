import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention.yml"
CANDIDATE_V2 = ROOT / "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v2.yml"
PRODUCTION = ROOT / "config/mv3dt_dev_room/config_tracker.yml"
APP = ROOT / "config/mv3dt_dev_room/config_deepstream.txt"
YOLO = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"


class NvDCFRetentionCandidateTests(unittest.TestCase):
    def test_candidate_aligns_lifecycle_with_fixed_yolo_floor(self):
        text = CANDIDATE.read_text()
        for item in (
            "minTrackerConfidence: 0.20",
            "probationAge: 2",
            "tentativeDetectorConfidence: 0.25",
            "outputShadowTracks: 0",
        ):
            self.assertIn(item, text)

    def test_safety_sensitive_tracker_settings_are_held_fixed(self):
        text = CANDIDATE.read_text()
        for item in (
            "minIouDiff4NewTarget: 0.22656630527418112",
            "maxShadowTrackingAge: 162",
            "earlyTerminationAge: 1",
            "minMatchingScore4Overall: 0.4",
            "minMatchingScore4SizeSimilarity: 0.4",
            "minMatchingScore4Iou: 0.1393522182207021",
            "minMatchingScore4VisualSimilarity: 0.0520394823204932",
            "minPeerTrackletMatchScore: 0.48",
            "maxPeerToPredDistance4Fusion: 1.35",
        ):
            self.assertIn(item, text)


    def test_v2_changes_only_overall_association_floor_beyond_v1_lifecycle(self):
        text = CANDIDATE_V2.read_text()
        for item in (
            "minTrackerConfidence: 0.20",
            "probationAge: 2",
            "tentativeDetectorConfidence: 0.25",
            "minMatchingScore4Overall: 0.35",
            "minMatchingScore4SizeSimilarity: 0.4",
            "minMatchingScore4Iou: 0.1393522182207021",
            "minMatchingScore4VisualSimilarity: 0.0520394823204932",
            "minIouDiff4NewTarget: 0.22656630527418112",
            "maxShadowTrackingAge: 162",
            "earlyTerminationAge: 1",
            "minPeerTrackletMatchScore: 0.48",
            "maxPeerToPredDistance4Fusion: 1.35",
            "outputShadowTracks: 0",
        ):
            self.assertIn(item, text)

    def test_v2_patch_helper_migrates_v1_fail_closed(self):
        text = (
            ROOT / "scripts/dev_room_mv3dt/patch_final_detector_ab_for_nvdcf_retention_v2.py"
        ).read_text()
        self.assertIn("refusing to edit non-runtime file", text)
        self.assertIn("production tracker changed during runtime patch", text)
        self.assertIn("'DataAssociator.minMatchingScore4Overall': .35", text)
        self.assertIn('"migrated_from": migrated_from', text)

    def test_production_tracker_is_not_rebased_to_candidate_values(self):
        text = PRODUCTION.read_text()
        self.assertIn("minTrackerConfidence: 0.6957540479571296", text)
        self.assertIn("probationAge: 5", text)
        self.assertIn("tentativeDetectorConfidence: 0.70167245554449", text)
        self.assertNotIn("outputShadowTracks: 1", text)

    def test_production_app_does_not_reference_experimental_candidate(self):
        text = APP.read_text()
        self.assertIn("ll-config-file=config_tracker.yml", text)
        self.assertNotIn("config_tracker_NvDCF_yolo26m_retention.yml", text)
        self.assertNotIn("shadow-track-output-dir=", text)

    def test_yolo_detector_contract_is_fixed(self):
        text = YOLO.read_text()
        for item in (
            "network-mode=2",
            "interval=0",
            "infer-dims=3;640;640",
            "pre-cluster-threshold=0.25",
            "nms-iou-threshold=0.45",
        ):
            self.assertIn(item, text)

    def test_staging_helper_is_runtime_only(self):
        text = (ROOT / "scripts/dev_room_mv3dt/stage_nvdcf_retention.py").read_text()
        self.assertIn('ROOT / ".runtime"', text)
        self.assertIn("refusing to stage outside", text)
        self.assertIn("production tracker changed while staging", text)


if __name__ == "__main__":
    unittest.main()
