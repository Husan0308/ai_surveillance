from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DetectionTrackingReidBaselineTests(unittest.TestCase):
    def test_config_keeps_pose_and_heatmap_removed_but_enables_reid_sidepath(self):
        payload = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))
        core = payload["core_v1"]
        self.assertEqual(core["profile"], "detection-tracking-reid-smooth-v4")
        self.assertIn("detector", core)
        self.assertIn("visual_tracker", core)
        self.assertIn("reid", core)
        self.assertTrue(bool(core["reid"]["enabled"]))
        self.assertEqual(str(core["reid"]["device"]), "cpu")
        for forbidden in ("pose", "heatmap"):
            self.assertNotIn(forbidden, core)

        tracker = core["visual_tracker"]
        detector = core["detector"]
        self.assertEqual(tracker["assignment_solver"], "hungarian")
        self.assertTrue(bool(tracker["fuse_score"]))
        self.assertTrue(bool(tracker["ownership_lock"]))
        self.assertGreaterEqual(int(tracker["id_namespace_stride"]), 1000)
        self.assertLessEqual(float(detector["conf"]), float(tracker["byte_low_conf"]))
        self.assertGreaterEqual(float(tracker["new_track_min_conf"]), float(tracker["byte_high_conf"]))

    def test_detector_shape_is_stride_aligned_and_near_16_by_9(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        detector = core["detector"]
        height, width = [int(value) for value in detector["imgsz"]]
        self.assertEqual(height % 32, 0)
        self.assertEqual(width % 32, 0)
        self.assertLess(abs((width / height) - (16.0 / 9.0)), 0.02)
        self.assertLessEqual(height * width, 448 * 704)
        self.assertEqual(int(core["capture_output_width"]), width)
        self.assertEqual(int(core["capture_output_height"]), height)

    def test_pose_heatmap_and_legacy_reid_modules_stay_removed(self):
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
        self.assertTrue((ROOT / "services/ml_service/core_v1/global_reid.py").exists())
        self.assertTrue((ROOT / "services/ml_service/core_v1/reid_embedder.py").exists())

    def test_api_surface_keeps_detection_and_adds_only_reid(self):
        source = (ROOT / "services/ml_service/core_v1/app.py").read_text(encoding="utf-8")
        for required in (
            '@app.get("/health")',
            '@app.get("/detections")',
            '@app.get("/tracks")',
            '@app.get("/reid")',
            '@app.get("/frame/{camera_id}")',
        ):
            self.assertIn(required, source)
        for forbidden in ("/poses", "/heatmap", "/room-mapping", "/overlays"):
            self.assertNotIn(forbidden, source)
        self.assertIn("GlobalReIdCoordinator", source)
        self.assertIn("TrackingJpegPublisher", source)

    def test_capture_pipeline_is_latest_only(self):
        deepstream = (ROOT / "services/ml_service/cameras/deepstream.py").read_text(encoding="utf-8")
        camera_worker = (ROOT / "services/ml_service/core_v1/camera_worker.py").read_text(encoding="utf-8")
        latest_frame = (ROOT / "services/ml_service/core_v1/latest_frame.py").read_text(encoding="utf-8")
        self.assertIn("leaky=downstream", deepstream)
        self.assertIn("drop=true max-buffers=1", deepstream)
        self.assertIn("postdecode_queue_buffers", deepstream)
        self.assertIn("single-slot frame store", camera_worker)
        self.assertIn("self._frame", latest_frame)
        self.assertNotIn("self._history", latest_frame)
        self.assertNotIn("def get_frame", latest_frame)

    def test_detector_is_person_only_and_freshness_gated(self):
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        config = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["detector"]
        self.assertIn('"classes": [0]', detector)
        self.assertLessEqual(float(config["max_submit_age_ms"]), 300.0)
        self.assertLessEqual(float(config["max_result_age_ms"]), 700.0)
        self.assertFalse(bool(config["roi_second_pass"]["enabled"]))

    def test_frontend_remains_free_of_pose_and_heatmap_dependencies(self):
        main = (ROOT / "services/frontend/core_v1/main.py").read_text(encoding="utf-8")
        ui = (ROOT / "services/frontend/core_v1/operator_dashboard_detection.py").read_text(encoding="utf-8")
        self.assertIn("operator_dashboard_detection", main)
        self.assertIn('"/health"', ui)
        self.assertIn('"/detections"', ui)
        self.assertIn('"/tracks"', ui)
        self.assertIn("self.heat.hide()", ui)
        self.assertIn("self.pose.hide()", ui)

    def test_reid_is_cpu_sidepath_and_checkpoint_is_pinned(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        reid = core["reid"]
        self.assertEqual(reid["device"], "cpu")
        self.assertEqual(reid["model_name"], "osnet_x0_25")
        self.assertEqual(len(str(reid["model_sha256"])), 64)
        self.assertGreaterEqual(int(reid["min_samples"]), 3)
        self.assertGreater(float(reid["strong_similarity"]), float(reid["match_similarity"]))
        self.assertGreater(float(reid["prototype_update_similarity"]), float(reid["same_group_similarity"]))

    def test_tracking_publisher_keeps_local_ownership_lock(self):
        source = (ROOT / "services/ml_service/core_v1/tracking_publisher.py").read_text(encoding="utf-8")
        ownership = (ROOT / "services/ml_service/core_v1/ownership_tracker.py").read_text(encoding="utf-8")
        self.assertIn("OwnershipLockedTracker", source)
        self.assertIn("identity_provider.identity_for_track", source)
        self.assertIn("ownership_quarantine", ownership)
        self.assertIn("id_namespace_base", ownership)


if __name__ == "__main__":
    unittest.main()
