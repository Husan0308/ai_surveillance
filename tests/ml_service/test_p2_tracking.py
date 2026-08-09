import unittest
from services.ml_service.tracking.evaluation import run_comparison
from tests.ml_service.test_tracking import camera,CONFIG
from services.ml_service.tracking.tracker_manager import TrackerManager

class P2TrackingEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=run_comparison()
    def test_short_occlusion_continuity(self):
        self.assertFalse(self.report["current_baseline"]["A_short_5"]["continuity"])
        self.assertTrue(self.report["current"]["A_short_5"]["continuity"])
    def test_long_occlusion_is_not_kept_indefinitely(self):
        self.assertEqual(self.report["current"]["B_long_15"]["fragments"],1)
    def test_crossing_and_similar_overlap_do_not_switch(self):
        for tracker in ("current","official_bytetrack"):
            self.assertEqual(self.report[tracker]["C_crossing"]["id_switches"],0)
            self.assertEqual(self.report[tracker]["D_similar_overlap"]["false_merges"],0)
    def test_nvdcf_capability_is_reported_honestly(self):
        self.assertTrue(self.report["nvdcf"]["runtime_available"])
        self.assertFalse(self.report["nvdcf"]["comparable_external_detection_adapter"])
    def test_stable_track_output_contract(self):
        manager=TrackerManager(CONFIG);stamp=1000.0
        first=manager.update(camera("CAM-01",1,[((0,0,20,40),.9)],stamp)).tracks[0]
        self.assertEqual(first.camera_id,"CAM-01");self.assertEqual(first.local_track_id,1)
        self.assertEqual(first.first_seen,stamp);self.assertEqual(first.last_seen,stamp)
        self.assertIsNotNone(first.predicted_bbox);self.assertFalse(first.confirmed)
        second=manager.update(camera("CAM-01",2,[((1,0,21,40),.9)],stamp+.1)).tracks[0]
        self.assertTrue(second.confirmed);self.assertAlmostEqual(second.age_seconds,.1)

if __name__=="__main__":unittest.main()
