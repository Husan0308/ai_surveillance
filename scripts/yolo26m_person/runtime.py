"""Refuse unproven or mismatched local TensorRT/parser artifacts before opening cameras."""
import hashlib
import json
import re
import subprocess


def _validate_nms_config(root):
    cfg = root/'config/deepstream/config_infer_primary_yolo26m_raw_otm.txt'
    text = cfg.read_text()
    required = {
        'cluster-mode=2': 'DeepStream NMS must be enabled',
        'warmup-engine=1': 'TensorRT engine warmup must be enabled',
        'nms-iou-threshold=0.45': 'NMS IoU threshold must remain 0.45',
        'pre-cluster-threshold=0.25': 'confidence threshold must remain 0.25',
        'batch-size=6': 'detector batch must remain 6',
        'parse-bbox-func-name=NvDsInferParseYolo26RawPerson': 'raw parser must be selected',
        'custom-lib-path=/models/libyolo26rawperson.so': 'raw parser library must be selected',
        'model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine': 'raw one-to-many engine must be selected',
    }
    missing = [why for needle, why in required.items() if needle not in text]
    if missing:
        raise RuntimeError('Unsafe YOLO26 DeepStream NMS config: ' + '; '.join(missing))


def _version_tuple(value):
    if not re.fullmatch(r'\d+(?:\.\d+)+', value):
        raise RuntimeError('Invalid NVIDIA driver version in engine provenance')
    return tuple(int(part) for part in value.split('.'))


def _gpu_provenance_fields(value, label):
    fields = [field.strip() for field in value.split(',')]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError(f'Invalid {label} GPU provenance')
    return fields[0], fields[1], fields[2]


def _validate_gpu_provenance(record, current_platform, current_gpu):
    """Allow forward driver updates on the same GPU, not metadata drift generally.

    TensorRT plans are bound to the TensorRT build/runtime and GPU hardware
    compatibility, not to an exact NVIDIA driver release string. The immutable
    platform/image record and exact GPU UUID remain strict; the driver must be
    at least both the build driver and the platform's documented minimum.
    Runtime deserialization is independently exercised by the deployment-image
    preflight before camera launch.
    """
    if record['platform'] != current_platform:
        raise RuntimeError('Engine DeepStream/CUDA/TensorRT platform provenance differs')

    built_name, built_uuid, built_driver = _gpu_provenance_fields(
        record['gpu'], 'build'
    )
    live_name, live_uuid, live_driver = _gpu_provenance_fields(
        current_gpu, 'runtime'
    )
    if built_name != live_name or built_uuid != live_uuid:
        raise RuntimeError('Engine deployment GPU identity differs from build provenance')

    minimum = _version_tuple(current_platform['minimum_driver'])
    built_version = _version_tuple(built_driver)
    live_version = _version_tuple(live_driver)
    if built_version < minimum:
        raise RuntimeError('Engine build driver is below the DeepStream platform minimum')
    if live_version < minimum:
        raise RuntimeError('Runtime NVIDIA driver is below the DeepStream platform minimum')
    if live_version < built_version:
        raise RuntimeError('Runtime NVIDIA driver predates the engine build driver')


def validate_artifacts(root):
    _validate_nms_config(root)
    out=root/'.runtime/models/yolo26m'
    record=json.loads((out/'engine_raw_otm.json').read_text())
    current=json.loads((root/'config/deepstream-platform.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    _validate_gpu_provenance(record, current, gpu)
    for name,key in [('yolo26m_raw_otm.onnx','onnx_sha256'),('yolo26m_raw_otm_b6_fp16.engine','engine_sha256'),('libyolo26rawperson.so','parser_sha256')]:
        if hashlib.sha256((out/name).read_bytes()).hexdigest() != record[key]:
            raise RuntimeError(f'{name} differs from validated build provenance')
