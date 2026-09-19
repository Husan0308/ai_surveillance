"""Refuse unproven or mismatched local TensorRT/parser artifacts before opening cameras."""
import hashlib
import json
import subprocess


def _validate_nms_config(root):
    cfg = root/'config/deepstream/config_infer_primary_yolo26m_raw_otm.txt'
    text = cfg.read_text()
    required = {
        'cluster-mode=2': 'DeepStream NMS must be enabled',
        'warmup-engine=1': 'TensorRT engine warmup must be enabled',
        'nms-iou-threshold=0.70': 'NMS IoU threshold must remain 0.70',
        'pre-cluster-threshold=0.25': 'confidence threshold must remain 0.25',
        'batch-size=6': 'detector batch must remain 6',
        'parse-bbox-func-name=NvDsInferParseYolo26RawPerson': 'raw parser must be selected',
        'custom-lib-path=/models/libyolo26rawperson.so': 'raw parser library must be selected',
        'model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine': 'raw one-to-many engine must be selected',
    }
    missing = [why for needle, why in required.items() if needle not in text]
    if missing:
        raise RuntimeError('Unsafe YOLO26 DeepStream NMS config: ' + '; '.join(missing))


def validate_artifacts(root):
    _validate_nms_config(root)
    out=root/'.runtime/models/yolo26m'
    record=json.loads((out/'engine_raw_otm.json').read_text())
    current=json.loads((root/'config/deepstream-platform.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    if record['platform'] != current or record['gpu'] != gpu:
        raise RuntimeError('Engine deployment GPU/runtime differs from build provenance; rebuild locally')
    for name,key in [('yolo26m_raw_otm.onnx','onnx_sha256'),('yolo26m_raw_otm_b6_fp16.engine','engine_sha256'),('libyolo26rawperson.so','parser_sha256')]:
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != record[key]:
            raise RuntimeError(f'{name} differs from validated build provenance')
