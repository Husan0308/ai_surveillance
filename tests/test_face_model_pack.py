from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from services.ml_service.core_v1.face_service_cuda import CudaFaceRecognitionService


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
OFFICIAL_SHA256 = "d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9"


class FaceModelPackTests(unittest.TestCase):
    @staticmethod
    def service(root: Path, *, download_if_missing=False):
        return CudaFaceRecognitionService(
            stores={},
            publishers={},
            config={
                "enabled": True,
                "provider": "CUDAExecutionProvider",
                "model_pack": "buffalo_m",
                "model_root": "models/insightface",
                "model_url": OFFICIAL_URL,
                "model_sha256": OFFICIAL_SHA256,
                "download_if_missing": download_if_missing,
                "data_dir": "data/faces",
                "db_path": "data/face_db.json",
            },
            root=root,
            base_identity=None,
        )

    def test_config_pins_official_buffalo_m_release_and_digest(self):
        face = yaml.safe_load((ROOT / "config/core_v1.yaml").read_text(encoding="utf-8"))["core_v1"]["face"]
        self.assertEqual(face["model_pack"], "buffalo_m")
        self.assertEqual(face["model_url"], OFFICIAL_URL)
        self.assertEqual(face["model_sha256"], OFFICIAL_SHA256)
        self.assertTrue(bool(face["download_if_missing"]))

    def test_pack_path_matches_faceanalysis_root_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self.service(root)
            self.assertEqual(
                service._pack_dir(),
                root / "models/insightface/models/buffalo_m",
            )

    def test_pack_requires_detection_and_recognition_models(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory)
            (pack / "det_2.5g.onnx").write_bytes(b"det")
            self.assertFalse(CudaFaceRecognitionService._pack_has_required_models(pack))
            (pack / "w600k_r50.onnx").write_bytes(b"rec")
            self.assertTrue(CudaFaceRecognitionService._pack_has_required_models(pack))

    def test_missing_pack_without_download_fails_with_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(Path(directory), download_if_missing=False)
            with self.assertRaisesRegex(RuntimeError, "missing or incomplete"):
                service._ensure_model_pack()

    def test_loader_enriches_blank_insightface_assertion(self):
        source = (ROOT / "services/ml_service/core_v1/face_service_cuda.py").read_text(encoding="utf-8")
        self.assertIn("self._ensure_model_pack()", source)
        self.assertIn("hashlib.sha256", source)
        self.assertIn("actual_sha256 != self.model_sha256", source)
        self.assertIn("InsightFace assertion while loading", source)
        self.assertIn("model_pack_ready", source)


if __name__ == "__main__":
    unittest.main()
