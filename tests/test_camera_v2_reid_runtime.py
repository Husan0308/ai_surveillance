import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.camera_v2.qwen_reid import QwenVerdict, _parse_json_object, build_comparison_sheet
from services.camera_v2.reid_quality import crop_signature, hamming64, evaluate_crop_quality
from services.camera_v2.reid_runtime import CropJob, ReIdIdentityEngine


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    def embed_batch(self, crops):
        self.calls += 1
        out = []
        for crop in crops:
            mean = float(crop[..., 0].mean())
            v = np.array([1.0, mean / 255.0 + .01, 0, 0], dtype=np.float32)
            v /= np.linalg.norm(v)
            out.append(v)
        return np.stack(out)

    def metrics(self):
        return {"backend": "fake", "calls": self.calls}


class FakeQwen:
    enabled = False

    def verify(self, payload):
        return QwenVerdict("UNCERTAIN", 0, {}, 0)

    def metrics(self):
        return {"enabled": False}


class ReIdRuntimeTests(unittest.TestCase):
    def test_quality_accepts_clear_person_crop(self):
        crop = np.zeros((220, 90, 3), dtype=np.uint8)
        for y in range(crop.shape[0]):
            crop[y, :, 0] = (y * 7) % 255
            crop[y, :, 1] = (y * 11) % 255
            crop[y, :, 2] = (y * 17) % 255
        q = evaluate_crop_quality(
            crop,
            source_bbox=(100, 80, 190, 300),
            source_width=704,
            source_height=384,
            detector_confidence=.8,
            tracker_confidence=.8,
        )
        self.assertTrue(q.accepted)
        self.assertGreater(q.score, .34)

    def test_duplicate_hash_is_stable(self):
        crop = np.random.default_rng(4).integers(0, 255, (120, 60, 3), dtype=np.uint8)
        a = crop_signature(crop)
        b = crop_signature(crop.copy())
        self.assertEqual(hamming64(a, b), 0)

    def test_qwen_json_parser_handles_code_fence(self):
        row = _parse_json_object('```json\n{"verdict":"SAME","confidence":0.9}\n```')
        self.assertEqual(row["verdict"], "SAME")

    def test_comparison_sheet_encodes(self):
        import cv2
        img = np.full((180, 80, 3), 100, dtype=np.uint8)
        ok, j = cv2.imencode('.jpg', img)
        self.assertTrue(ok)
        sheet = build_comparison_sheet([j.tobytes()], [j.tobytes()])
        self.assertGreater(len(sheet), 1000)

    def test_engine_is_bounded_and_async(self):
        engine = ReIdIdentityEngine(
            {"CAM-01": "Devs", "CAM-04": "Devs"},
            {
                "min_samples": 3,
                "new_identity_confirm_samples": 3,
                "sample_interval_sec": .12,
                "min_crop_quality": .1,
            },
            embedder=FakeEmbedder(),
            qwen=FakeQwen(),
        )
        engine.start()
        try:
            engine.observe_tracks(
                "CAM-01",
                "Devs",
                [{
                    "object_id": 7,
                    "left": 100,
                    "top": 50,
                    "width": 80,
                    "height": 250,
                    "tracker_confidence": .8,
                }],
                now=10.0,
            )
            rng = np.random.default_rng(1)
            for i in range(3):
                crop = rng.integers(0, 255, (250, 80, 3), dtype=np.uint8)
                engine.submit_crop(
                    CropJob(
                        "CAM-01", 7, "Devs", crop, (100, 50, 180, 300),
                        704, 384, .9, .8, 0.0, 10.0 + i * .2,
                    )
                )
            deadline = time.time() + 2
            while time.time() < deadline and engine.binding_for_track("CAM-01", 7) is None:
                time.sleep(.02)
            self.assertIsNotNone(engine.binding_for_track("CAM-01", 7))
        finally:
            engine.stop()


if __name__ == '__main__':
    unittest.main()
