from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import unittest
from unittest.mock import patch

from services.ml_service.core_v1 import jpeg_publisher


class _FakeTracker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = []
        self.visible_calls = []
        self.boxes = []

    def update(self, result, now, source_width, source_height):
        self.updates.append((result, now, source_width, source_height))

    def visible(self, now, *, target_time, max_observation_age_sec):
        self.visible_calls.append((now, target_time, max_observation_age_sec))
        return list(self.boxes)

    def metrics(self):
        return {}


class _DetectionStore:
    def __init__(self, result=None):
        self.result = result

    def get(self, camera_id):
        return self.result


def _result(frame_id, captured, produced=None):
    return SimpleNamespace(
        frame_id=frame_id,
        frame_captured_monotonic=captured,
        produced_monotonic=captured if produced is None else produced,
        boxes=(),
    )


class LatestJpegPublisherTimestampTests(unittest.TestCase):
    def _publisher(self, detections, *, max_age_ms=350, config=None):
        with patch.object(jpeg_publisher, "VisualTracker", _FakeTracker):
            return jpeg_publisher.LatestJpegPublisher(
                "CAM-01",
                object(),
                detections=detections,
                overlay_max_age_ms=max_age_ms,
                tracker_config=config,
            )

    def test_future_result_is_deferred_once_then_applied(self):
        result = _result(11, 10.100, 10.200)
        detections = _DetectionStore(result)
        publisher = self._publisher(detections)
        image = object()

        for _ in range(2):
            returned = publisher._draw_detection(
                image, 640, 480, 10.250, 10, 10.000
            )
            self.assertIs(returned, image)

        self.assertEqual(publisher.future_detection_deferrals, 1)
        self.assertEqual(publisher.visual_tracker.updates, [])
        self.assertEqual(
            publisher.visual_tracker.visible_calls[-1],
            (10.250, 10.000, 0.350),
        )

        publisher._draw_detection(image, 640, 480, 10.300, 11, 10.100)
        self.assertEqual(len(publisher.visual_tracker.updates), 1)
        self.assertIs(publisher.visual_tracker.updates[0][0], result)

    def test_newer_timestamp_is_deferred_even_if_frame_id_is_not(self):
        detections = _DetectionStore(_result(10, 10.100))
        publisher = self._publisher(detections)

        publisher._draw_detection(object(), 640, 480, 10.200, 10, 10.000)

        self.assertEqual(publisher.future_detection_deferrals, 1)
        self.assertEqual(publisher.visual_tracker.updates, [])

    def test_stale_result_is_rejected_once_per_unique_result(self):
        detections = _DetectionStore(_result(7, 10.000, 10.050))
        publisher = self._publisher(detections, max_age_ms=100)
        image = object()

        for _ in range(2):
            publisher._draw_detection(image, 640, 480, 10.300, 9, 10.200)

        self.assertEqual(publisher.stale_detection_rejects, 1)
        self.assertEqual(publisher.visual_tracker.updates, [])

        detections.result = _result(8, 10.050, 10.100)
        publisher._draw_detection(image, 640, 480, 10.300, 9, 10.200)
        self.assertEqual(publisher.stale_detection_rejects, 2)

    def test_tracker_motion_knobs_are_forwarded(self):
        config = {
            "size_velocity_damping": 0.91,
            "adaptive_error_low": 0.07,
            "adaptive_error_high": 0.23,
            "center_response_slow": 0.81,
            "center_response_fast": 0.96,
            "size_response": 0.44,
            "snap_distance_boxes": 0.57,
            "reversal_damping": 0.18,
        }
        publisher = self._publisher(_DetectionStore(), config=config)

        for key, value in config.items():
            self.assertEqual(publisher.visual_tracker.kwargs[key], value)

    def test_metrics_expose_unique_deferral_counter(self):
        publisher = self._publisher(_DetectionStore())
        publisher.future_detection_deferrals = 3

        self.assertEqual(publisher.metrics()["future_detection_deferrals"], 3)

    def test_previously_accepted_result_is_not_later_counted_as_stale(self):
        result = _result(7, 10.000, 10.050)
        publisher = self._publisher(_DetectionStore(result), max_age_ms=100)
        image = object()

        publisher._draw_detection(image, 640, 480, 10.070, 7, 10.070)
        self.assertEqual(len(publisher.visual_tracker.updates), 1)
        publisher._draw_detection(image, 640, 480, 10.250, 8, 10.250)

        self.assertEqual(publisher.stale_detection_rejects, 0)

    def test_invalid_and_huge_boxes_cannot_break_overlay_drawing(self):
        publisher = self._publisher(None)
        publisher.visual_tracker.boxes = [
            SimpleNamespace(x1=float("nan"), y1=0.0, x2=20.0, y2=20.0, confidence=0.9),
            SimpleNamespace(x1=0.0, y1=0.0, x2=20.0, y2=20.0, confidence=float("inf")),
            SimpleNamespace(x1=-1e300, y1=-1e300, x2=1e300, y2=1e300, confidence=0.9),
            SimpleNamespace(x1=-20.0, y1=10.0, x2=-10.0, y2=30.0, confidence=0.9),
        ]
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        with patch.object(jpeg_publisher.cv2, "rectangle") as rectangle, patch.object(
            jpeg_publisher.cv2, "putText"
        ) as put_text:
            returned = publisher._draw_detection(
                image,
                640,
                360,
                now=10.0,
                display_frame_id=1,
                display_frame_time=10.0,
            )

        self.assertIs(returned, image)
        rectangle.assert_called_once()
        put_text.assert_called_once()
        _draw_image, point1, point2, *_rest = rectangle.call_args.args
        for x, y in (point1, point2):
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, image.shape[1])
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, image.shape[0])

    def test_real_tracker_constructs_with_production_config(self):
        from pathlib import Path
        import yaml

        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "config/core_v1.yaml").read_text())
        tracker_config = config["core_v1"]["visual_tracker"]
        publisher = jpeg_publisher.LatestJpegPublisher(
            "CAM-01", object(), tracker_config=tracker_config
        )

        metrics = publisher.visual_tracker.metrics()
        self.assertEqual(metrics["algorithm"], "adaptive-kalman-byte-visual-v2")
        self.assertEqual(metrics["prediction_ms"], tracker_config["prediction_ms"])


if __name__ == "__main__":
    unittest.main()
