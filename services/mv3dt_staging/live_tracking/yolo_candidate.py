"""Fail-closed, offline-only YOLO26m candidate for Entrance Room appsrc smoke.

No imports of GPU modules. Never edits production DeepStream configs or models.
The frozen batch-6 TensorRT engine is only *proposed* for batch-2 use:
real nvinfer initialization must still prove runtime/profile compatibility.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / ".runtime/sixcam-approved-bbox-20261006-r5/models"
FROZEN_PGIE = ROOT / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
ASSETS = (
    "yolo26m_raw_otm.onnx",
    "yolo26m_raw_otm_b6_fp16.engine",
    "libyolo26rawperson.so",
    "labels.txt",
)
CONTAINER_CONFIG = "/workspace/pgie-yolo26m-b2.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_yolo_assets(model_dir: Path, *, expected: dict[str, str] | None = None,
                         root: Path = ROOT) -> dict[str, str]:
    """Reject symlink escapes, missing files, changed artifact bytes and bad config."""
    from scripts.build_six_camera_v13_wall_candidate import EXPECTED

    hashes = EXPECTED if expected is None else expected
    model_dir = Path(model_dir).resolve()
    allowed = (root / ".runtime").resolve()
    if not model_dir.is_relative_to(allowed):
        raise ValueError("YOLO artifacts must remain in this checkout's .runtime")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"frozen YOLO artifacts missing: {model_dir}")
    actual = {}
    for name in ASSETS:
        item = model_dir / name
        if not item.is_file() or not item.resolve().is_relative_to(allowed):
            raise ValueError(f"missing or unsafe YOLO model/parser asset: {name}")
        value = sha256_file(item)
        if value != hashes[name]:
            raise ValueError(f"frozen YOLO asset SHA256 mismatch: {name}")
        actual[name] = value

    pg = root / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt"
    if sha256_file(pg) != hashes["config_infer_primary_yolo26m_raw_otm.txt"]:
        raise ValueError("frozen YOLO nvinfer config SHA256 mismatch")
    return actual


def render_batch_two_config(template: str) -> str:
    """Only change explicit batch=6 to batch=2; no threshold/model changes."""
    mandatory = (
        "model-engine-file=/models/yolo26m_raw_otm_b6_fp16.engine",
        "custom-lib-path=/models/libyolo26rawperson.so",
        "parse-bbox-func-name=NvDsInferParseYolo26RawPerson",
        "onnx-file=/models/yolo26m_raw_otm.onnx",
        "interval=0",
        "gie-unique-id=1",
        "pre-cluster-threshold=0.25",
        "nms-iou-threshold=0.45",
    )
    if template.count("batch-size=6") != 1 or any(template.count(x) != 1 for x in mandatory):
        raise ValueError("frozen PGIE template contract changed; no silent fallback")
    return template.replace("batch-size=6", "batch-size=2", 1)


def stage_yolo_batch_two(model_dir: Path, output_dir: Path, *,
                         expected: dict[str, str] | None = None, root: Path = ROOT) -> dict:
    """Prepare a private, read-only-mounted candidate config for one offline run."""
    actual = validate_yolo_assets(model_dir, expected=expected, root=root)
    config_text = render_batch_two_config(
        (root / "config/deepstream/config_infer_primary_yolo26m_raw_otm.txt").read_text()
    )
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to((root / ".runtime").resolve()):
        # Isolated temp experiment output dirs under /tmp are also supported.
        if not output_dir.is_relative_to(Path("/tmp").resolve()):
            raise ValueError("candidate output must be under .runtime or /tmp")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_file = output_dir / "pgie-yolo26m-b2.txt"
    if config_file.exists():
        if config_file.read_text() != config_text:
            raise ValueError("existing staged YOLO config differs; do not overwrite")
    else:
        with config_file.open("x") as stream:
            stream.write(config_text)
    config_file.chmod(0o600)
    return {
        "name": "YOLO26m",
        "model_dir": str(Path(model_dir).resolve()),
        "config_file": str(config_file),
        "config_sha256": sha256_file(config_file),
        "model_asset_sha256": actual,
        "batch_size": 2,
        "person_class": 0,
        "validated": "static hashes/config only; engine runtime and detection quality NOT TESTED",
    }
