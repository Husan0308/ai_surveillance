from __future__ import annotations

from pathlib import Path
import unittest

import yaml

from services.ml_service.cameras.smooth_gstreamer import _inject_gpu_scale


ROOT = Path(__file__).resolve().parents[1]


class CameraSmoothnessTests(unittest.TestCase):
    def test_config_uses_20fps_and_detector_sized_capture(self):
        core = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]
        self.assertEqual(int(core["display_fps"]), 20)
        self.assertEqual(int(core["capture_output_width"]), 736)
        self.assertEqual(int(core["capture_output_height"]), 416)
        self.assertEqual(list(core["detector"]["imgsz"]), [416, 736])
        self.assertGreaterEqual(int(core["max_pipeline_lag_samples"]), 12)

    def test_deepstream_scales_before_raw_host_mapping(self):
        source = (ROOT / "services/ml_service/cameras/deepstream.py").read_text(encoding="utf-8")
        self.assertIn("capture_output_width", source)
        self.assertIn("capture_output_height", source)
        self.assertIn("nvvideoconvert name=converter", source)
        self.assertIn("enable-last-sample=false", source)
        self.assertIn("gpu_scale_before_host_copy", source)

    def test_gstreamer_fallback_injects_nv_scale(self):
        base = (
            "nvv4l2decoder ! queue ! nvvideoconvert name=converter ! "
            "video/x-raw,format=BGRx ! appsink name=sink drop=true max-buffers=1 "
            "sync=false wait-on-eos=false"
        )
        payload = _inject_gpu_scale(
            base,
            {"capture_output_width": 736, "capture_output_height": 416},
        )
        self.assertIn("width=736,height=416,format=BGRx", payload)
        self.assertIn("enable-last-sample=false", payload)

    def test_detector_skips_duplicate_resize_at_capture_size(self):
        source = (ROOT / "services/ml_service/core_v1/stable_detector.py").read_text(encoding="utf-8")
        self.assertIn("same_size_prepare_skips", source)
        self.assertIn("frame.width) == self.input_w", source)
        self.assertIn("frame.height) == self.input_h", source)

    def test_ui_uses_one_persistent_mjpeg_connection_per_camera(self):
        reader = (ROOT / "services/frontend/core_v1/smooth_frame_reader.py").read_text(encoding="utf-8")
        wrapper = (ROOT / "services/frontend/core_v1/operator_dashboard_detection.py").read_text(encoding="utf-8")
        self.assertIn('f"/video/{self.camera_id}"', reader)
        self.assertNotIn('f"/frame/{self.camera_id}?after=', reader)
        self.assertIn("base.FrameReader = SmoothFrameReader", wrapper)

    def test_health_exposes_capture_and_decoder_telemetry(self):
        worker = (ROOT / "services/ml_service/core_v1/camera_worker.py").read_text(encoding="utf-8")
        metrics = (ROOT / "services/ml_service/core_v1/runtime_metrics.py").read_text(encoding="utf-8")
        self.assertIn("capture_stage", worker)
        self.assertIn("gpu_decoder_utilization_percent", metrics)


if __name__ == "__main__":
    unittest.main()
