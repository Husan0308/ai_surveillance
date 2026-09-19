"""Refuse unproven or mismatched local TensorRT/parser artifacts before opening cameras."""
import hashlib
import json
import subprocess


def validate_artifacts(root):
    out=root/'.runtime/models/yolo26m'
    record=json.loads((out/'engine_raw_otm.json').read_text())
    current=json.loads((root/'config/deepstream-platform.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    if record['platform'] != current or record['gpu'] != gpu:
        raise RuntimeError('Engine deployment GPU/runtime differs from build provenance; rebuild locally')
    for name,key in [('yolo26m_raw_otm.onnx','onnx_sha256'),('yolo26m_raw_otm_b6_fp16.engine','engine_sha256'),('libyolo26rawperson.so','parser_sha256')]:
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != record[key]:
            raise RuntimeError(f'{name} differs from validated build provenance')
