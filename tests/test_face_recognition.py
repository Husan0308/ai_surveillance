from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from services.ml_service.core_v1.face_service_safe import (
    SafeFaceGallery,
    SafeFaceRecognitionService,
)


def unit(values):
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FaceRecognitionTests(unittest.TestCase):
    def gallery(self, root: Path):
        return SafeFaceGallery(
            root,
            {
                "data_dir": "data/faces",
                "db_path": "data/face_db.json",
                "match_similarity": 0.52,
                "strong_similarity": 0.68,
                "second_best_margin": 0.06,
                "strong_second_best_margin": 0.025,
            },
        )

    @staticmethod
    def samples(vector, count=3):
        return [
            {"embedding": unit(vector), "jpeg": b""}
            for _ in range(count)
        ]

    def test_distinct_person_matches_best_gallery_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = self.gallery(Path(directory))
            gallery.enroll("Husan", "Security", "EMP-1", self.samples([1, 0, 0]))
            gallery.enroll("Diyor", "IT", "EMP-2", self.samples([0, 1, 0]))

            match = gallery.match(unit([0.98, 0.10, 0.0]))
            self.assertIsNotNone(match)
            self.assertEqual(match.name, "Husan")
            self.assertGreater(match.similarity, 0.95)
            self.assertGreater(match.margin, 0.50)

    def test_two_high_near_equal_candidates_are_rejected_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            gallery = self.gallery(Path(directory))
            a = unit([1.0, 0.0, 0.0])
            b = unit([0.999, 0.0447, 0.0])
            gallery.enroll("Person A", "", "A", [{"embedding": a, "jpeg": b""}])
            gallery.enroll("Person B", "", "B", [{"embedding": b, "jpeg": b""}])
            query = unit(a + b)

            self.assertIsNone(gallery.match(query))

    def test_gallery_persists_multiple_prototypes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gallery = self.gallery(root)
            samples = [
                {"embedding": unit([1.0, 0.00, 0.0]), "jpeg": b""},
                {"embedding": unit([0.99, 0.08, 0.0]), "jpeg": b""},
                {"embedding": unit([0.98, -0.10, 0.0]), "jpeg": b""},
            ]
            person = gallery.enroll("Husan", "Security", "EMP-1", samples)
            self.assertEqual(person["samples"], 3)

            reloaded = self.gallery(root)
            rows = reloaded.list_people()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "Husan")
            self.assertEqual(rows[0]["samples"], 3)

    def service(self, root: Path):
        return SafeFaceRecognitionService(
            stores={},
            publishers={},
            config={
                "enabled": True,
                "data_dir": "data/faces",
                "db_path": "data/face_db.json",
                "enrollment_samples": 3,
                "enrollment_token_ttl_sec": 600,
                "enrollment_consistency_similarity": 0.35,
                "enrollment_max_outliers": 0,
                "match_similarity": 0.52,
                "strong_similarity": 0.68,
                "second_best_margin": 0.06,
                "strong_second_best_margin": 0.025,
            },
            root=root,
            base_identity=None,
        )

    def test_enrollment_rejects_tokens_from_different_tracks(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(Path(directory))
            now = time.monotonic()
            for index in range(3):
                service._enrollment_tokens[f"t{index}"] = {
                    "embedding": unit([1.0, 0.02 * index, 0.0]),
                    "jpeg": b"",
                    "created_mono": now,
                    "camera_id": "CAM-01",
                    "track_id": 10001 if index < 2 else 10002,
                }
            with self.assertRaisesRegex(ValueError, "changed person track"):
                service.commit_enrollment("Husan", "Security", "EMP-1", ["t0", "t1", "t2"])

    def test_enrollment_accepts_one_consistent_track(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(Path(directory))
            now = time.monotonic()
            vectors = [
                [1.0, 0.00, 0.0],
                [0.99, 0.06, 0.0],
                [0.98, -0.08, 0.0],
            ]
            for index, vector in enumerate(vectors):
                service._enrollment_tokens[f"t{index}"] = {
                    "embedding": unit(vector),
                    "jpeg": b"",
                    "created_mono": now,
                    "camera_id": "CAM-01",
                    "track_id": 10001,
                }
            person = service.commit_enrollment(
                "Husan", "Security", "EMP-1", ["t0", "t1", "t2"]
            )
            self.assertEqual(person["name"], "Husan")
            self.assertEqual(person["samples"], 3)
            self.assertEqual(len(service.gallery.list_people()), 1)

    def test_cuda_runtime_is_lazy_verified_and_keeps_cpu_fallback(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "services/ml_service/core_v1/face_service_cuda.py").read_text(encoding="utf-8")
        base_source = (root / "services/ml_service/core_v1/face_service.py").read_text(encoding="utf-8")

        self.assertIn("SafeFaceRecognitionService", source)
        self.assertIn('requested_provider', source)
        self.assertIn('CUDAExecutionProvider', source)
        self.assertIn('CPUExecutionProvider', source)
        self.assertIn('ort.get_available_providers()', source)
        self.assertIn('getattr(session, "get_providers", None)', source)
        self.assertIn('gpu_mem_limit', source)
        self.assertIn('cudnn_conv_algo_search', source)
        self.assertIn('import torch', source)
        self.assertIn("def _load_engine(self):", source)
        load_index = source.index("def _load_engine(self):")
        ort_index = source.index("import onnxruntime as ort")
        insightface_index = source.index("from insightface.app import FaceAnalysis")
        self.assertGreater(ort_index, load_index)
        self.assertGreater(insightface_index, load_index)
        # The reusable base remains free of top-level InsightFace imports.
        self.assertNotIn("from insightface.app import FaceAnalysis\n", base_source.split("class FaceRecognitionService", 1)[0])


if __name__ == "__main__":
    unittest.main()
