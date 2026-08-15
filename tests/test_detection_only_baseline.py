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

    def test_default_frontend_is_direct_camera_detection_wall(self):
        main = (ROOT / "services/frontend/core_v1/main.py").read_text(encoding="utf-8")
        wall = (ROOT / "services/frontend/core_v1/direct_detection_wall.py").read_text(encoding="utf-8")
        compile(wall, "direct_detection_wall.py", "exec")
        self.assertIn("direct_detection_wall", main)
        self.assertNotIn("simple_detection_wall", main)
        self.assertIn("CameraManager", wall)
        self.assertIn("StableYoloDetectorWorker", wall)
        self.assertIn("manager.stores.get", wall)
        self.assertIn("detector.results.get", wall)
        self.assertIn("Format_BGR888", wall)
        self.assertIn('"capture_output_width": 960', wall)
        self.assertIn('"capture_output_height": 540', wall)
        self.assertIn('"min_submit_interval_ms": 85', wall)
        self.assertIn("core.pop(\"rtsp_latency_ms\", None)", wall)
        self.assertNotIn("SmoothMmapFrameReader", wall)
        self.assertNotIn("SmoothFrameReader", wall)
        self.assertNotIn("requests.", wall)
        self.assertNotIn("StreamingResponse", wall)
        self.assertNotIn("cv2.imencode", wall)
        self.assertNotIn("QImage.fromData", wall)
        self.assertNotIn("SideBar", wall)
        self.assertNotIn("Enrollment", wall)

    def test_direct_mode_keeps_one_latest_frame_and_no_ipc_transport(self):
        wall = (ROOT / "services/frontend/core_v1/direct_detection_wall.py").read_text(encoding="utf-8")
        latest = (ROOT / "services/ml_service/core_v1/latest_frame.py").read_text(encoding="utf-8")
        self.assertIn("self._frame", latest)
        self.assertNotIn("self._history", latest)
        self.assertIn("self.store.get()", wall)
        self.assertNotIn("mmap", wall.lower())
        self.assertNotIn("jpeg", wall.lower())
        self.assertNotIn("http://", wall.lower())

    def test_direct_mode_reserves_gpu_headroom_without_changing_full_config(self):
        stable = (ROOT / "services/ml_service/core_v1/stable_detector.py").read_text(encoding="utf-8")
        wall = (ROOT / "services/frontend/core_v1/direct_detection_wall.py").read_text(encoding="utf-8")
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        self.assertIn("min_submit_interval_ms", stable)
        self.assertIn("_next_submit_monotonic", stable)
        self.assertIn('"min_submit_interval_ms": 85', wall)
        self.assertNotIn("min_submit_interval_ms", core["detector"])

    def test_direct_mode_uses_bounded_rtsp_latency_floor(self):
        wall = (ROOT / "services/frontend/core_v1/direct_detection_wall.py").read_text(encoding="utf-8")
        self.assertIn('floor_ms = 80 if codec in {"h265", "hevc"} else 60', wall)
        self.assertIn('camera["drop_on_latency"] = True', wall)
        self.assertIn('"postdecode_queue_buffers": 1', wall)
        self.assertIn('"max_pipeline_lag_ms": 500', wall)

    def test_capture_pipeline_remains_latest_only(self):
        deepstream = (ROOT / "services/ml_service/cameras/deepstream.py").read_text(encoding="utf-8")
        camera_worker = (ROOT / "services/ml_service/core_v1/camera_worker.py").read_text(encoding="utf-8")
        latest_frame = (ROOT / "services/ml_service/core_v1/latest_frame.py").read_text(encoding="utf-8")
        self.assertIn("leaky=downstream", deepstream)
        self.assertIn("drop=true max-buffers=1", deepstream)
        self.assertIn("single-slot frame store", camera_worker)
        self.assertIn("self._frame", latest_frame)
        self.assertNotIn("self._history", latest_frame)

    def test_detector_input_stays_small_while_display_is_960x540(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        height, width = [int(value) for value in core["detector"]["imgsz"]]
        self.assertEqual((height, width), (416, 736))
        stable = (ROOT / "services/ml_service/core_v1/stable_detector.py").read_text(encoding="utf-8")
        self.assertIn("cv2.resize", stable)
        self.assertIn("self.input_w", stable)
        self.assertIn("self.input_h", stable)

    def test_detector_is_person_only_and_freshness_gated(self):
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        config = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["detector"]
        self.assertIn('"classes": [0]', detector)
        self.assertLessEqual(float(config["max_submit_age_ms"]), 300.0)
        self.assertLessEqual(float(config["max_result_age_ms"]), 700.0)
        self.assertFalse(bool(config["roi_second_pass"]["enabled"]))

    def test_full_ui_face_reid_and_mmap_stack_remain_available_but_not_default(self):
        for relative in (
            "services/ml_service/core_v1/app.py",
            "services/ml_service/core_v1/simple_app.py",
            "services/ml_service/core_v1/mmap_publisher.py",
            "services/ml_service/core_v1/room_consensus_reid.py",
            "services/ml_service/core_v1/face_service_cuda.py",
            "services/frontend/core_v1/simple_detection_wall.py",
            "services/frontend/core_v1/operator_dashboard_mukammal.py",
            "services/frontend/core_v1/operator_dashboard_people_roster.py",
        ):
            self.assertTrue((ROOT / relative).exists(), relative)

    def test_full_tracker_ownership_code_is_untouched(self):
        tracking = (ROOT / "services/ml_service/core_v1/tracking_publisher.py").read_text(encoding="utf-8")
        ownership = (ROOT / "services/ml_service/core_v1/ownership_tracker.py").read_text(encoding="utf-8")
        self.assertIn("OwnershipLockedTracker", tracking)
        self.assertIn("ownership_quarantine", ownership)


if __name__ == "__main__":
    unittest.main()
