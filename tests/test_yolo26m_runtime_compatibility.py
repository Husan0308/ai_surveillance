import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.yolo26m_person.runtime import (
    _validate_gpu_provenance,
    validate_artifacts,
)


class Yolo26mRuntimeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.platform = {
            'deepstream': '9.1.0',
            'cuda_runtime': '13.2',
            'tensorrt': '10.16.1.11',
            'minimum_driver': '595.58.03',
            'image_tag': 'nvcr.io/nvidia/deepstream:9.1-samples-multiarch',
            'image': 'nvcr.io/nvidia/deepstream@sha256:pin',
            'platform': 'linux/amd64',
        }
        self.record = {
            'platform': self.platform.copy(),
            'gpu': 'NVIDIA GeForce RTX 3060, GPU-test, 595.91.07',
        }

    def test_forward_driver_update_on_same_gpu_is_accepted(self):
        _validate_gpu_provenance(
            self.record,
            self.platform,
            'NVIDIA GeForce RTX 3060, GPU-test, 595.99.02',
        )

    def test_different_gpu_or_uuid_is_rejected(self):
        for gpu in (
            'NVIDIA GeForce RTX 3060 Ti, GPU-test, 595.99.02',
            'NVIDIA GeForce RTX 3060, GPU-other, 595.99.02',
        ):
            with self.subTest(gpu=gpu), self.assertRaisesRegex(RuntimeError, 'GPU identity'):
                _validate_gpu_provenance(self.record, self.platform, gpu)

    def test_driver_below_build_driver_or_documented_minimum_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'predates'):
            _validate_gpu_provenance(
                self.record, self.platform,
                'NVIDIA GeForce RTX 3060, GPU-test, 595.90.00',
            )
        old_record = {
            **self.record,
            'gpu': 'NVIDIA GeForce RTX 3060, GPU-test, 595.50.00',
        }
        with self.assertRaisesRegex(RuntimeError, 'build driver'):
            _validate_gpu_provenance(
                old_record, self.platform,
                'NVIDIA GeForce RTX 3060, GPU-test, 595.99.02',
            )

    def test_platform_provenance_remains_exact(self):
        altered = {**self.platform, 'image': 'different-image-digest'}
        with self.assertRaisesRegex(RuntimeError, 'platform provenance'):
            _validate_gpu_provenance(
                self.record,
                altered,
                'NVIDIA GeForce RTX 3060, GPU-test, 595.99.02',
            )

    def test_artifact_hash_checks_remain_enforced_after_driver_update(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / '.runtime/models/yolo26m'
            config_dir = root / 'config/deepstream'
            model_dir.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            config_text = '''[property]
cluster-mode=2
warmup-engine=1
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
batch-size=6
parse-bbox-func-name=NvDsInferParseYolo26RawPerson
custom-lib-path=/models/libyolo26rawperson.so
model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine
'''
            (config_dir / 'config_infer_primary_yolo26m_raw_otm.txt').write_text(config_text)
            (root / 'config/deepstream-platform.json').write_text(
                json.dumps(self.platform)
            )
            assets = {
                'yolo26m_raw_otm.onnx': b'onnx-fixture',
                'yolo26m_raw_otm_b6_fp16.engine': b'engine-fixture',
                'libyolo26rawperson.so': b'parser-fixture',
            }
            for name, content in assets.items():
                (model_dir / name).write_bytes(content)
            record = {
                'platform': self.platform,
                'gpu': self.record['gpu'],
                'onnx_sha256': hashlib.sha256(assets['yolo26m_raw_otm.onnx']).hexdigest(),
                'engine_sha256': hashlib.sha256(assets['yolo26m_raw_otm_b6_fp16.engine']).hexdigest(),
                'parser_sha256': hashlib.sha256(assets['libyolo26rawperson.so']).hexdigest(),
            }
            (model_dir / 'engine_raw_otm.json').write_text(json.dumps(record))
            with patch(
                'scripts.yolo26m_person.runtime.subprocess.check_output',
                return_value='NVIDIA GeForce RTX 3060, GPU-test, 595.99.02',
            ):
                validate_artifacts(root)

            (model_dir / 'libyolo26rawperson.so').write_bytes(b'changed-parser')
            with patch(
                'scripts.yolo26m_person.runtime.subprocess.check_output',
                return_value='NVIDIA GeForce RTX 3060, GPU-test, 595.99.02',
            ), self.assertRaisesRegex(RuntimeError, 'libyolo26rawperson.so differs'):
                validate_artifacts(root)


if __name__ == '__main__':
    unittest.main()
