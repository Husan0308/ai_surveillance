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
