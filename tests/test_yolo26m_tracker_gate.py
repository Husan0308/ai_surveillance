import unittest

from scripts.yolo26m_tracker.check_gate import analyze_tracks


class NvDCFTrackerGateTests(unittest.TestCase):
    def test_stable_track_is_counted(self):
        records = [
            {
                "source_id": 0,
                "frame": i,
                "object_id": 7,
                "detector_confidence": 0.9,
                "tracker_confidence": 0.8,
                "box": [10, 20, 100, 200],
            }
            for i in range(1, 121)
        ]
        report = analyze_tracks(records)
        cam = report["per_source"]["CAM-01"]
        self.assertEqual(report["duplicate_frame_ids"], 0)
        self.assertEqual(cam["unique_ids"], 1)
        self.assertEqual(cam["tracks_ge_20_frames"], 1)
        self.assertEqual(cam["tracks_ge_100_frames"], 1)
        self.assertEqual(cam["longest_observations"], 120)

    def test_duplicate_id_same_frame_is_rejected(self):
        records = [
            {"source_id": 2, "frame": 10, "object_id": 4},
            {"source_id": 2, "frame": 10, "object_id": 4},
        ]
        report = analyze_tracks(records)
        self.assertEqual(report["duplicate_frame_ids"], 1)

    def test_two_people_same_frame_are_valid_when_ids_differ(self):
        records = [
            {"source_id": 1, "frame": 10, "object_id": 11},
            {"source_id": 1, "frame": 10, "object_id": 12},
        ]
        report = analyze_tracks(records)
        self.assertEqual(report["duplicate_frame_ids"], 0)
        self.assertEqual(report["per_source"]["CAM-02"]["unique_ids"], 2)

    def test_tracker_source_avoids_removed_ds9_properties(self):
        from pathlib import Path
        text = Path("scripts/yolo26m_tracker/tracker.hpp").read_text()
        self.assertNotIn('"enable-batch-process"', text)
        self.assertNotIn('"enable-past-frame"', text)
        for name in (
            '"tracker-width"', '"tracker-height"', '"ll-lib-file"',
            '"ll-config-file"', '"gpu-id"', '"display-tracking-id"',
            '"compute-hw"', '"tracking-id-reset-mode"',
        ):
            self.assertIn(name, text)

    def test_stable_person_profile_is_no_reid_and_accuracy_tuned(self):
        from pathlib import Path
        text = Path("config/deepstream/config_tracker_NvDCF_stable_person.yml").read_text()
        for item in (
            "associationMatcherType: 1",
            "maxShadowTrackingAge: 90",
            "minTrackerConfidence: 0.15",
            "useColorNames: 1",
            "useHog: 1",
            "featureImgSizeLevel: 3",
            "reidType: 0",
        ):
            self.assertIn(item, text)

        tracker = Path("scripts/yolo26m_tracker/tracker.hpp").read_text()
        self.assertIn("/config/config_tracker_NvDCF_stable_person.yml", tracker)
        self.assertNotIn("config_tracker_NvDCF_perf.yml", tracker)

    def test_graph_marker_matches_stable_profile(self):
        from pathlib import Path
        detection = Path("scripts/yolo26m_person/detection.hpp").read_text()
        checker = Path("scripts/yolo26m_tracker/check_gate.py").read_text()
        marker = "nvtracker(NvDCF_stable_person,960x544)"
        profile_marker = (
            "TRACKER_GRAPH nvtracker("
            "NvDCF_stable_person,width=960,height=544,batch=6,gpu=0,reid=0)"
        )
        self.assertIn(marker, detection)
        self.assertIn('"->nvtracker("', checker)
        self.assertIn(profile_marker, checker)
        self.assertNotIn("nvtracker(NvDCF_perf,960x544)", detection)

    def test_tracker_preflight_requests_gpu_access(self):
        from pathlib import Path
        text = Path("scripts/validate_cam_six.py").read_text()
        self.assertIn('"--gpus", "device=0"', text)
        self.assertIn(
            '"NVIDIA_DRIVER_CAPABILITIES=compute,utility,video"',
            text,
        )
        self.assertIn(
            '"GST_REGISTRY=/tmp/nvtracker-preflight-registry.bin"',
            text,
        )

    def test_tracker_isolation_checker_documents_reset_semantics(self):
        from pathlib import Path
        text = Path("scripts/yolo26m_tracker/check_isolation.py").read_text()
        self.assertIn("tracking-id-reset-mode=1 allows new IDs after stream reset", text)
        self.assertIn('"target_id_continuity_required": False', text)

    def test_fragmented_track_does_not_fake_long_observation_count(self):
        records = [
            {"source_id": 0, "frame": i * 10, "object_id": 99}
            for i in range(10)
        ]
        report = analyze_tracks(records)
        cam = report["per_source"]["CAM-01"]
        self.assertEqual(cam["longest_observations"], 10)
        self.assertEqual(cam["tracks_ge_20_frames"], 0)


if __name__ == "__main__":
    unittest.main()
