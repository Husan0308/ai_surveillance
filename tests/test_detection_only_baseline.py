from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DetectionTrackingReidBaselineTests(unittest.TestCase):
    def test_config_keeps_pose_heatmap_removed_and_face_is_bounded_cuda_sidepath(self):
        payload = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))
        core = payload["core_v1"]
        self.assertEqual(core["profile"], "detection-tracking-room-consensus-reid-face-cuda-v2")
        for required in ("detector", "visual_tracker", "reid", "face"):
            self.assertIn(required, core)
        self.assertTrue(bool(core["reid"]["enabled"]))
        self.assertEqual(str(core["reid"]["device"]), "cpu")
        self.assertTrue(bool(core["face"]["enabled"]))
        face = core["face"]
        self.assertEqual(face["provider"], "CUDAExecutionProvider")
        self.assertEqual(int(face["device_id"]), 0)
        self.assertTrue(bool(face["allow_cpu_fallback"]))
        self.assertLessEqual(int(face["gpu_mem_limit_mb"]), 1024)
        self.assertGreaterEqual(int(face["sample_interval_ms"]), 1000)
        self.assertLessEqual(int(face["max_people_per_camera"]), 2)
        self.assertEqual(face["model_pack"], "buffalo_m")
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
        height, width = [int(value) for value in core["detector"]["imgsz"]]
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
            "services/ml_service/core_v1/unified_detector.py",
            "services/ml_service/pose",
            "services/ml_service/heatmap",
            "config/room_mapping.yaml",
        ]
        for relative in forbidden:
            self.assertFalse((ROOT / relative).exists(), relative)
        for required in (
            "services/ml_service/core_v1/global_reid.py",
            "services/ml_service/core_v1/reid_embedder.py",
            "services/ml_service/core_v1/instant_reid.py",
            "services/ml_service/core_v1/room_consensus_reid.py",
            "services/ml_service/core_v1/face_service.py",
            "services/ml_service/core_v1/face_service_safe.py",
            "services/ml_service/core_v1/face_service_cuda.py",
        ):
            self.assertTrue((ROOT / required).exists(), required)

    def test_api_surface_keeps_detection_reid_and_adds_face_only(self):
        source = (ROOT / "services/ml_service/core_v1/app.py").read_text(encoding="utf-8")
        for required in (
            '@app.get("/health")',
            '@app.get("/detections")',
            '@app.get("/tracks")',
            '@app.get("/reid")',
            '@app.get("/faces")',
            '@app.post("/faces/enrollment/sample/{camera_id}/{track_id}")',
            '@app.post("/faces/enrollment/commit")',
            '@app.get("/faces/avatar/{person_id}")',
            '@app.get("/frame/{camera_id}")',
        ):
            self.assertIn(required, source)
        for forbidden in ("/poses", "/heatmap", "/room-mapping", "/overlays"):
            self.assertNotIn(forbidden, source)
        self.assertIn("RoomConsensusGlobalReIdCoordinator", source)
        self.assertIn("CudaFaceRecognitionService", source)
        self.assertIn("TrackingJpegPublisher", source)

    def test_cuda_face_service_verifies_real_onnx_sessions_and_preserves_safety(self):
        source = (ROOT / "services/ml_service/core_v1/face_service_cuda.py").read_text(encoding="utf-8")
        self.assertIn("SafeFaceRecognitionService", source)
        self.assertIn('import torch', source)
        self.assertIn('CUDAExecutionProvider', source)
        self.assertIn('CPUExecutionProvider', source)
        self.assertIn('ort.get_available_providers()', source)
        self.assertIn('get_providers', source)
        self.assertIn('gpu_mem_limit', source)
        self.assertIn('kSameAsRequested', source)
        self.assertIn('HEURISTIC', source)
        self.assertIn('cuda_verified', source)
        self.assertIn('cpu_fallback', source)

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

    def test_detector_is_person_only_and_freshness_gated(self):
        detector = (ROOT / "services/ml_service/core_v1/detector.py").read_text(encoding="utf-8")
        config = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["detector"]
        self.assertIn('"classes": [0]', detector)
        self.assertLessEqual(float(config["max_submit_age_ms"]), 300.0)
        self.assertLessEqual(float(config["max_result_age_ms"]), 700.0)
        self.assertFalse(bool(config["roi_second_pass"]["enabled"]))

    def test_frontend_keeps_mukammal_cuda_face_and_adds_worker_roster_adapter(self):
        main = (ROOT / "services/frontend/core_v1/main.py").read_text(encoding="utf-8")
        base_ui = (ROOT / "services/frontend/core_v1/operator_dashboard_mukammal.py").read_text(encoding="utf-8")
        face_ui = (ROOT / "services/frontend/core_v1/operator_dashboard_face.py").read_text(encoding="utf-8")
        cuda_ui = (ROOT / "services/frontend/core_v1/operator_dashboard_face_cuda.py").read_text(encoding="utf-8")
        roster_ui = (ROOT / "services/frontend/core_v1/operator_dashboard_people_roster.py").read_text(encoding="utf-8")
        self.assertIn("operator_dashboard_people_roster", main)
        self.assertIn("class Header", base_ui)
        self.assertIn("class SideBar", base_ui)
        self.assertIn("class CameraCard", base_ui)
        self.assertIn("SmoothFrameReader", base_ui)
        self.assertNotIn("class CameraSim", base_ui)
        self.assertNotIn("class SimPerson", base_ui)
        self.assertNotIn("random.uniform", base_ui)
        self.assertIn('"/faces"', face_ui)
        self.assertIn("InsightFace buffalo_m", cuda_ui)
        self.assertIn("CUDAExecutionProvider", cuda_ui)
        self.assertIn("768 MB ORT arena cap", cuda_ui)
        self.assertIn("class WorkerRosterPage", roster_ui)
        self.assertIn("class WorkerEnrollmentDialog", roster_ui)
        self.assertIn("MAX_WORKERS = 10", roster_ui)
        self.assertIn("return cuda.run()", roster_ui)

    def test_frontend_shares_one_mjpeg_reader_per_camera(self):
        ui = (ROOT / "services/frontend/core_v1/operator_dashboard_mukammal.py").read_text(encoding="utf-8")
        self.assertIn("class CameraFeed", ui)
        self.assertIn("One persistent MJPEG connection shared", ui)
        self.assertIn("hub.feeds[camera_id]", ui)
        self.assertIn("feed.surfaces", ui)
        self.assertNotIn("SmoothFrameReader(self.camera_id)", ui)

    def test_reid_uses_cross_domain_ain_and_room_consensus(self):
        reid = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["reid"]
        self.assertEqual(reid["device"], "cpu")
        self.assertEqual(reid["model_name"], "osnet_ain_x1_0")
        self.assertEqual(
            reid["model_sha256"],
            "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
        )
        self.assertGreaterEqual(int(reid["min_samples"]), 3)
        self.assertLessEqual(int(reid["min_crop_height_px"]), 50)
        self.assertGreater(float(reid["room_embedding_weight"]), float(reid["room_colour_weight"]))
        self.assertGreater(float(reid["room_pair_similarity"]), float(reid["room_single_similarity"]))
        self.assertGreater(float(reid["room_confirmed_merge_similarity"]), float(reid["room_pair_similarity"]))
        self.assertGreater(float(reid["same_camera_handoff_similarity"]), float(reid["same_group_similarity"]))

    def test_tracking_publisher_keeps_local_ownership_lock(self):
        source = (ROOT / "services/ml_service/core_v1/tracking_publisher.py").read_text(encoding="utf-8")
        ownership = (ROOT / "services/ml_service/core_v1/ownership_tracker.py").read_text(encoding="utf-8")
        self.assertIn("OwnershipLockedTracker", source)
        self.assertIn("identity_provider.identity_for_track", source)
        self.assertIn("ownership_quarantine", ownership)
        self.assertIn("id_namespace_base", ownership)


if __name__ == "__main__":
    unittest.main()
