from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DetectionOnlyBaselineTests(unittest.TestCase):
    def test_config_contains_only_camera_detector_and_visual_tracker(self):
        payload = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))
        core = payload["core_v1"]
        self.assertEqual(core["profile"], "detection-only-baseline-v1")
        self.assertIn("detector", core)
        self.assertIn("visual_tracker", core)
        for forbidden in ("pose", "heatmap", "reid"):
            self.assertNotIn(forbidden, core)

    def test_optional_model_modules_are_removed(self):
        forbidden = [
            "services/ml_service/core_v1/global_identity.py",
            "services/ml_service/core_v1/reid_service.py",
            "services/ml_service/core_v1/reid_hardening.py",
            "services/ml_service/core_v1/spatial_calibration.py",
            "services/ml_service/core_v1/heatmap_publisher.py",
            "services/ml_service/core_v1/heatmap_publisher_v2.py",
            "services/ml_service/core_v1/heatmap_publisher_v3.py",
            "services/ml_service/core_v1/unified_detector.py",
            "services/ml_service/pose",
            "services/ml_service/heatmap",
            "config/room_mapping.yaml",
        ]
        for relative in forbidden:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_api_surface_is_detection_only(self):
        source = (ROOT / "services/ml_service/core_v1/app.py").read_text(encoding="utf-8")
        for required in ('@app.get("/health")', '@app.get("/detections")', '@app.get("/frame/{camera_id}")'):
            self.assertIn(required, source)
        for forbidden in ("/poses", "/heatmap", "/reid", "/room-mapping", "/overlays"):
            self.assertNotIn(forbidden, source)
        for forbidden_import in ("reid_service", "spatial_calibration", "services.ml_service.pose", "services.ml_service.heatmap"):
            self.assertNotIn(forbidden_import, source)

    def test_capture_pipeline_is_latest_only(self):
        deepstream = (ROOT / "services/ml_service/cameras/deepstream.py").read_text(encoding="utf-8")
        camera_worker = (ROOT / "services/ml_service/core_v1/camera_worker.py").read_text(encoding="utf-8")
        latest_frame = (ROOT / "services/ml_service/core_v1/latest_frame.py").read_text(encoding="utf-8")
        self.assertIn("leaky=downstream", deepstream)
        self.assertIn("drop=true max-buffers=1", deepstream)
        self.assertIn("postdecode_queue_buffers", deepstream)
        self.assertIn("LatestFrameStore is one slot only", camera_worker)
        self.assertIn("self._latest", latest_frame)

    def test_detector_is_person_only_and_freshness_gated(self):
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        config = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["detector"]
        self.assertIn('"classes": [0]', detector)
        self.assertLessEqual(float(config["max_submit_age_ms"]), 300.0)
        self.assertLessEqual(float(config["max_result_age_ms"]), 700.0)
        self.assertFalse(bool(config["roi_second_pass"]["enabled"]))

    def test_frontend_uses_detection_only_operator_console(self):
        main = (ROOT / "services/frontend/core_v1/main.py").read_text(encoding="utf-8")
        ui = (ROOT / "services/frontend/core_v1/operator_dashboard_detection.py").read_text(encoding="utf-8")
        self.assertIn("operator_dashboard_detection", main)
        self.assertIn('"/health"', ui)
        self.assertIn('"/detections"', ui)
        self.assertIn("self.heat.hide()", ui)
        self.assertIn("self.pose.hide()", ui)


if __name__ == "__main__":
    unittest.main()
