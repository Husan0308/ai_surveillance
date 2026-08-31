#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.camera_v11.step4_reid_quality_v1 import evaluate_reid_crop_quality


class Step4ReIDQualityV1Test(unittest.TestCase):
    @staticmethod
    def textured_frame() -> np.ndarray:
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
        return frame

    def test_valid_full_body_crop_accepted(self):
        decision = evaluate_reid_crop_quality(
            self.textured_frame(), (180, 35, 320, 360), 0.88
        )
        self.assertTrue(decision.accepted, decision)
        self.assertEqual(decision.reason, "accepted")
        self.assertIsNotNone(decision.crop_bgr)
        self.assertGreater(decision.crop_bgr.shape[0], decision.crop_bgr.shape[1])
        self.assertGreaterEqual(decision.quality_score, 0.25)

    def test_tiny_crop_rejected(self):
        decision = evaluate_reid_crop_quality(
            self.textured_frame(), (100, 100, 112, 132), 0.90
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "size")

    def test_heavily_clipped_edge_crop_rejected(self):
        decision = evaluate_reid_crop_quality(
            self.textured_frame(), (-120, -90, 230, 330), 0.91
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "edge")

    def test_extreme_aspect_ratio_rejected(self):
        decision = evaluate_reid_crop_quality(
            self.textured_frame(), (80, 100, 610, 205), 0.92
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "aspect")

    def test_very_blurred_crop_rejected(self):
        frame = np.full((360, 640, 3), 127, dtype=np.uint8)
        decision = evaluate_reid_crop_quality(frame, (180, 35, 320, 360), 0.90)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "blur")

    def test_invalid_empty_crop_rejected(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        decision = evaluate_reid_crop_quality(empty, (180, 35, 320, 360), 0.90)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "invalid")

    def test_frozen_step123_ci_guard(self):
        guard = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_camera_v11_frozen_step123_guard.py")],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(guard.returncode, 0, guard.stdout)
        self.assertIn("V11_FROZEN_STEP123_GUARD RESULT=PASS", guard.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
