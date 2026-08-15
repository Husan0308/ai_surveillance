from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SimpleDetectionArchitectureTests(unittest.TestCase):
    def test_full_config_is_preserved_for_later(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        self.assertTrue(bool(core["detector"]["enabled"]))
        self.assertTrue(bool(core["reid"]["enabled"]))
        self.assertTrue(bool(core["face"]["enabled"]))
        self.assertEqual(core["face"]["provider"], "CUDAExecutionProvider")
        self.assertEqual(core["reid"]["model_name"], "osnet_ain_x1_0")

    def test_default_ml_entrypoint_uses_simple_app(self):
        main = (ROOT / "services/ml_service/core_v1/main.py").read_text(encoding="utf-8")
        simple = (ROOT / "services/ml_service/core_v1/simple_app.py").read_text(encoding="utf-8")
        compile(simple, "simple_app.py", "exec")
        self.assertIn("from .simple_app import app, core_cfg", main)
        self.assertIn('"profile": "simple-clear-detection-v1"', simple)
        self.assertIn('"capture_output_width": 960', simple)
        self.assertIn('"capture_output_height": 540', simple)
        self.assertIn('"max_display_width": 960', simple)
        self.assertIn('"max_display_height": 540', simple)
        self.assertIn('"jpeg_quality": 88', simple)
        self.assertIn("EventDrivenJpegPublisher", simple)
        self.assertNotIn("RoomConsensusGlobalReIdCoordinator", simple)
        self.assertNotIn("CudaFaceRecognitionService", simple)
        self.assertIn('"reid": {"enabled": False', simple)
        self.assertIn('"face": {"enabled": False', simple)

    def test_detector_input_stays_small_even_when_display_is_clearer(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        height, width = [int(value) for value in core["detector"]["imgsz"]]
        self.assertEqual((height, width), (416, 736))
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        self.assertIn("cv2.resize(frame.image,(self.input_w,self.input_h)", detector)
        self.assertIn('"source_w":int(frame.width)', detector)
        self.assertIn('"source_h":int(frame.height)', detector)
        self.assertIn("_map_full_boxes", detector)

    def test_default_frontend_is_only_six_camera_wall(self):
        main = (ROOT / "services/frontend/core_v1/main.py").read_text(encoding="utf-8")
        wall = (ROOT / "services/frontend/core_v1/simple_detection_wall.py").read_text(encoding="utf-8")
        compile(wall, "simple_detection_wall.py", "exec")
        self.assertIn("simple_detection_wall", main)
        self.assertNotIn("operator_dashboard_people_roster", main)
        self.assertIn('CAMERA_IDS = [f"CAM-{index:02d}" for index in range(1, 7)]', wall)
        self.assertIn("QGridLayout", wall)
        self.assertIn("index // 3, index % 3", wall)
        self.assertIn("SmoothFrameReader", wall)
        self.assertNotIn("SideBar", wall)
        self.assertNotIn("RightPanel", wall)
        self.assertNotIn("PersonManagementPage", wall)
        self.assertNotIn("Enrollment", wall)

    def test_full_ui_and_face_stack_remain_available_but_not_default(self):
        for relative in (
            "services/ml_service/core_v1/app.py",
            "services/ml_service/core_v1/room_consensus_reid.py",
            "services/ml_service/core_v1/face_service_cuda.py",
            "services/frontend/core_v1/operator_dashboard_mukammal.py",
            "services/frontend/core_v1/operator_dashboard_face.py",
            "services/frontend/core_v1/operator_dashboard_people_roster.py",
        ):
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_capture_pipeline_remains_latest_only(self):
        deepstream = (ROOT / "services/ml_service/cameras/deepstream.py").read_text(encoding="utf-8")
        camera_worker = (ROOT / "services/ml_service/core_v1/camera_worker.py").read_text(encoding="utf-8")
        latest_frame = (ROOT / "services/ml_service/core_v1/latest_frame.py").read_text(encoding="utf-8")
        self.assertIn("leaky=downstream", deepstream)
        self.assertIn("drop=true max-buffers=1", deepstream)
        self.assertIn("single-slot frame store", camera_worker)
        self.assertIn("self._frame", latest_frame)
        self.assertNotIn("self._history", latest_frame)

    def test_detector_is_person_only_and_freshness_gated(self):
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        config = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["detector"]
        self.assertIn('"classes": [0]', detector)
        self.assertLessEqual(float(config["max_submit_age_ms"]), 300.0)
        self.assertLessEqual(float(config["max_result_age_ms"]), 700.0)
        self.assertFalse(bool(config["roi_second_pass"]["enabled"]))

    def test_full_tracker_ownership_code_is_untouched(self):
        tracking = (ROOT / "services/ml_service/core_v1/tracking_publisher.py").read_text(encoding="utf-8")
        ownership = (ROOT / "services/ml_service/core_v1/ownership_tracker.py").read_text(encoding="utf-8")
        self.assertIn("OwnershipLockedTracker", tracking)
        self.assertIn("ownership_quarantine", ownership)


if __name__ == "__main__":
    unittest.main()
