from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.ml_service.core_v1.stable_detector import StableYoloDetectorWorker


class _Store:
    def get(self):
        return None, 0


class StableDetectorTests(unittest.TestCase):
    def test_missing_local_model_uses_ultralytics_checkpoint_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker = StableYoloDetectorWorker(
                {"CAM-01": _Store()},
                {
                    "model": "models/yolo26m.pt",
                    "model_fallback": "yolo26m.pt",
                    "batch_size": 1,
                },
                Path(tmp),
            )
            self.assertFalse(worker.model_local_exists)
            self.assertEqual(worker.model_source, "yolo26m.pt")
            self.assertFalse(worker.metrics()["pose_in_hot_path"])

    def test_existing_local_model_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "models" / "yolo26m.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"placeholder")

            worker = StableYoloDetectorWorker(
                {"CAM-01": _Store()},
                {"model": "models/yolo26m.pt", "batch_size": 1},
                root,
            )
            self.assertTrue(worker.model_local_exists)
            self.assertEqual(worker.model_source, str(model))


if __name__ == "__main__":
    unittest.main()
