from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from services.ml_service.core_v1.detector import DetectionResult, PersonBox
from services.ml_service.core_v1.reid_service import ReIDCoordinator


class _DetectionStore:
    def __init__(self, result):
        self.result = result

    def snapshot(self):
        return {self.result.camera_id: self.result}


class _FrameStore:
    def __init__(self, frame):
        self.frame = frame
        self.requested = []

    def get_frame(self, frame_id):
        self.requested.append(frame_id)
        return self.frame if frame_id == self.frame.frame_id else None


class _OneIterationStop:
    def __init__(self):
        self.checks = 0

    def is_set(self):
        self.checks += 1
        return self.checks > 1

    def wait(self, _timeout):
        return False


class _AcceptingSelector:
    def __init__(self, crop):
        self.crop = crop
        self.calls = []

    def evaluate(self, frame, box):
        self.calls.append((frame, box))
        return SimpleNamespace(accepted=True, score=0.91, crop=self.crop)


class ReIDRealObservationBoundaryTests(unittest.TestCase):
    def test_reid_uses_exact_detector_frame_timestamp_box_and_crop(self):
        captured = 123.456
        detector_box = PersonBox(20.0, 15.0, 80.0, 130.0, 0.91)
        result = DetectionResult(
            camera_id="CAM-01",
            frame_id=42,
            frame_captured_monotonic=captured,
            produced_monotonic=captured + 0.20,
            boxes=(detector_box,),
        )
        source_frame = SimpleNamespace(
            camera_id="CAM-01",
            frame_id=42,
            captured_monotonic=captured,
            image=np.zeros((160, 100, 3), dtype=np.uint8),
        )
        frame_store = _FrameStore(source_frame)
        crop = np.ones((96, 48, 3), dtype=np.uint8)
        selector = _AcceptingSelector(crop)
        coordinator = ReIDCoordinator(
            {"CAM-01": frame_store},
            _DetectionStore(result),
            {
                "enabled": False,
                "min_track_hits": 1,
                "embed_cooldown_sec": 0.2,
                "refresh_sec": 0.2,
            },
        )
        jobs = []
        coordinator.selector = selector
        coordinator._enqueue_latest = jobs.append
        coordinator._stop = _OneIterationStop()

        coordinator._observe_loop()

        self.assertEqual(frame_store.requested, [42])
        self.assertEqual(selector.calls, [(source_frame, detector_box)])
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertEqual(job.frame_id, 42)
        self.assertEqual(job.observed_at, captured)
        self.assertIs(job.crop, crop)

    def test_missing_exact_frame_fails_closed(self):
        result = DetectionResult(
            camera_id="CAM-01",
            frame_id=99,
            frame_captured_monotonic=50.0,
            produced_monotonic=50.2,
            boxes=(PersonBox(1.0, 2.0, 30.0, 70.0, 0.9),),
        )
        missing_store = _FrameStore(SimpleNamespace(frame_id=98))
        coordinator = ReIDCoordinator(
            {"CAM-01": missing_store}, _DetectionStore(result), {"enabled": False}
        )
        jobs = []
        coordinator._enqueue_latest = jobs.append
        coordinator._stop = _OneIterationStop()

        coordinator._observe_loop()

        self.assertEqual(missing_store.requested, [99])
        self.assertEqual(jobs, [])
        self.assertEqual(coordinator.metrics()["frame_misses"], 1)


if __name__ == "__main__":
    unittest.main()
