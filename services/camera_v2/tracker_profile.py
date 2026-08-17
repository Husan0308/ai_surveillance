from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .reid_engine import ensure_reid_engine

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"
REID_MODEL_DIR = RUNTIME_DIR / "models" / "reid"
REID_MODEL_NAME = "resnet50_market1501_aicity156.onnx"
REID_MODEL_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/reidentificationnet/"
    "versions/deployable_v1.2/files/resnet50_market1501_aicity156.onnx"
)
REID_MODEL_SHA256 = "0e21d09278508ec835955f422a9fdd3cd59b2a6ecdef98d705f388f33cebac2b"

_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    "enableBboxUnClipping": "1",
    "maxTargetsPerStream": "24",
    "minIouDiff4NewTarget": "0.90",
    "minTrackerConfidence": "0.18",
    "probationAge": "1",
    "maxShadowTrackingAge": "28",
    "earlyTerminationAge": "2",
}

_OPTIONAL_PATCHES: dict[str, str] = {
    "useColorNames": "0",
    "useHog": "1",
    "featureImgSizeLevel": "3",
    "searchRegionPaddingScale": "1",
    "associationMatcherType": "1",
    "tentativeDetectorConfidence": "0.22",
    "minMatchingScore4TentativeIou": "0.10",
    "minMatchingScore4Overall": "0.06",
    "minMatchingScore4SizeSimilarity": "0.05",
    "minMatchingScore4Iou": "0.02",
    "minMatchingScore4VisualSimilarity": "0.05",
    "usePrediction4Assoc": "1",
    "minTrackingConfidenceDuringInactive": "0.35",
    "minIou4TargetDuplicate": "0.98",
    "targetDuplicateRunInterval": "10",
}


def _deepstream_roots() -> list[Path]:
    roots = [Path("/opt/nvidia/deepstream/deepstream")]
    roots.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    output: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            output.append(root)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _auto_download_enabled() -> bool:
    value = os.environ.get("CAMERA_V2_REID_AUTO_DOWNLOAD", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _download_official_reid(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(
        "CAMERA_REID model missing; downloading NVIDIA TAO ReIdentificationNet v1.2 "
        f"to {destination}",
        flush=True,
    )
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        request = urllib.request.Request(
            REID_MODEL_URL,
            headers={"User-Agent": "camera-v2-reid/1.2"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if tmp_path.stat().st_size < 80 * 1024 * 1024:
            raise RuntimeError(
                f"downloaded ReID model is too small: {tmp_path.stat().st_size} bytes"
            )
        digest = _sha256(tmp_path)
        if digest != REID_MODEL_SHA256:
            raise RuntimeError(
                "ReID model SHA256 mismatch: "
                f"expected={REID_MODEL_SHA256} got={digest}"
            )
        tmp_path.replace(destination)
        print(
            f"CAMERA_REID model ready: {destination} "
            f"({destination.stat().st_size / (1024 * 1024):.1f} MiB)",
            flush=True,
        )
        return destination.resolve()
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def resolve_reid_model() -> Path:
    override = os.environ.get("CAMERA_V2_REID_MODEL", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
        raise RuntimeError(f"CAMERA_V2_REID_MODEL does not exist: {candidate}")

    local_model = REID_MODEL_DIR / REID_MODEL_NAME
    if local_model.exists() and local_model.is_file():
        digest = _sha256(local_model)
        if digest == REID_MODEL_SHA256:
            return local_model.resolve()
        if not _auto_download_enabled():
            raise RuntimeError(
                f"Camera V2 ReID model failed SHA256 verification: {local_model}\n"
                "Run: python scripts/setup_camera_v2_reid.py --force"
            )
        print(f"CAMERA_REID replacing corrupt model: {local_model}", flush=True)
        local_model.unlink(missing_ok=True)

    for root in _deepstream_roots():
        candidate = root / "samples/models/Tracker" / REID_MODEL_NAME
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    if _auto_download_enabled():
        try:
            return _download_official_reid(local_model)
        except Exception as exc:
            raise RuntimeError(
                "Camera V2 could not auto-install the NVIDIA TAO ReID model.\n"
                f"Reason: {exc}\n"
                "Retry manually with:\n"
                "  python scripts/setup_camera_v2_reid.py --force\n"
                "Or disable ReID temporarily with:\n"
                "  CAMERA_V2_REID=0 python -m services.camera_v2.person_tracking_final"
            ) from exc

    searched = [local_model]
    searched.extend(root / "samples/models/Tracker" / REID_MODEL_NAME for root in _deepstream_roots())
    raise RuntimeError(
        "Camera V2 ReID model is missing. Install the NVIDIA TAO model first:\n"
        "  python scripts/setup_camera_v2_reid.py\n"
        "Searched:\n  - " + "\n  - ".join(str(p) for p in searched)
    )


def _section_header(line: str) -> str | None:
    if not line or line[0].isspace():
        return None
    body = line.split("#", 1)[0].strip()
    if body.endswith(":") and body[:-1].strip():
        return body[:-1].strip()
    return None


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = None
    for index, line in enumerate(lines):
        if _section_header(line) == section:
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if _section_header(lines[index]) is not None:
            end = index
            break
    return start, end


def _ensure_section(lines: list[str], section: str) -> tuple[int, int]:
    bounds = _section_bounds(lines, section)
    if bounds is not None:
        return bounds
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(f"{section}:")
    return len(lines) - 1, len(lines)


def _set_section_key(lines: list[str], section: str, key: str, value: str) -> None:
    start, end = _ensure_section(lines, section)
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if stripped.startswith(key + ":"):
            indent = lines[index][: len(lines[index]) - len(stripped)] or "  "
            comment = ""
            if "#" in stripped:
                comment = "  #" + stripped.split("#", 1)[1]
            lines[index] = f"{indent}{key}: {value}{comment}"
            return
    lines.insert(end, f"  {key}: {value}")


def _remove_section_keys(lines: list[str], section: str, keys: set[str]) -> None:
    bounds = _section_bounds(lines, section)
    if bounds is None:
        return
    start, end = bounds
    for index in range(end - 1, start, -1):
        stripped = lines[index].lstrip()
        if any(stripped.startswith(key + ":") for key in keys):
            del lines[index]


def _set_or_insert_target_management(lines: list[str], key: str, value: str) -> bool:
    _set_section_key(lines, "TargetManagement", key, value)
    return True


def _configure_reid(lines: list[str]) -> Path:
    model = resolve_reid_model()

    # GTX 1050 Ti has 4 GB and also drives the desktop. A large TensorRT ReID batch
    # is the wrong trade-off here: it increases engine-build and runtime memory, and
    # the build was observed aborting when NvMultiObjectTracker tried to construct
    # the engine after the live camera pipeline had already allocated GPU memory.
    # Batch=1 plus periodic embeddings is sufficient for the global identity layer.
    batch_size = max(1, min(4, int(os.environ.get("CAMERA_V2_REID_BATCH", "1"))))
    extraction_interval = max(-1, int(os.environ.get("CAMERA_V2_REID_INTERVAL", "5")))
    workspace_mb = max(64, min(512, int(os.environ.get("CAMERA_V2_REID_WORKSPACE_MB", "256"))))
    engine = Path(
        os.environ.get(
            "CAMERA_V2_REID_ENGINE",
            str(REID_MODEL_DIR / f"{model.stem}_b{batch_size}_gpu0_fp16.engine"),
        )
    ).expanduser().resolve()
    engine.parent.mkdir(parents=True, exist_ok=True)

    # Build before CameraDetectionV2 creates six decoders, EGL surfaces and the
    # YOLO worker. This converts a crash-prone in-pipeline TensorRT build into a
    # bounded offline-style build and lets nvtracker only deserialize at startup.
    if batch_size == 1:
        ensure_reid_engine(model, engine, workspace_mb=workspace_mb)
    elif not engine.exists():
        raise RuntimeError(
            "CAMERA_V2_REID_BATCH > 1 requires a prebuilt matching TensorRT engine. "
            "Use the default batch=1 on GTX 1050 Ti or set CAMERA_V2_REID_ENGINE "
            "to a compatible prebuilt plan."
        )

    trajectory = {
        "useUniqueID": "1",
        "enableReAssoc": "1",
        "reidExtractionInterval": str(extraction_interval),
        "minMatchingScore4ReidSimilarity": "0.68",
        "matchingScoreWeight4ReidSimilarity": "0.85",
        "minTrackletMatchingScore": "0.42",
        "maxTrackletMatchingTimeSearchRange": "20",
    }
    for key, value in trajectory.items():
        _set_section_key(lines, "TrajectoryManagement", key, value)

    _remove_section_keys(
        lines,
        "ReID",
        {"tltEncodedModel", "tltModelKey", "onnxFile", "modelEngineFile"},
    )

    reid = {
        "reidType": "2",
        "batchSize": str(batch_size),
        "workspaceSize": str(workspace_mb),
        "reidFeatureSize": "256",
        "reidHistorySize": "24",
        "inferDims": "[3, 256, 128]",
        "networkMode": "1",
        "inputOrder": "0",
        "colorFormat": "0",
        "offsets": "[123.6750, 116.2800, 103.5300]",
        "netScaleFactor": "0.01735207",
        "addFeatureNormalization": "1",
        "keepAspc": "1",
        "minVisibility4GalleryUpdate": "0.60",
        "outputReidTensor": "1",
        "onnxFile": json.dumps(str(model)),
        "modelEngineFile": json.dumps(str(engine)),
    }
    for key, value in reid.items():
        _set_section_key(lines, "ReID", key, value)
    return model


def prepare_sparse_tracker_config(stock: Path) -> Path:
    stock = Path(stock)
    if not stock.exists():
        raise RuntimeError(f"NvDCF stock config not found: {stock}")

    patches = {**_REQUIRED_PATCHES, **_OPTIONAL_PATCHES}
    lines = stock.read_text(encoding="utf-8").splitlines()
    patched: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        replaced = False
        for key, value in patches.items():
            if stripped.startswith(key + ":"):
                comment = ""
                if "#" in stripped:
                    comment = "  #" + stripped.split("#", 1)[1]
                output.append(f"{indent}{key}: {value}{comment}")
                patched.add(key)
                replaced = True
                break
        if not replaced:
            output.append(line)

    if "maxTargetsPerStream" not in patched:
        _set_or_insert_target_management(output, "maxTargetsPerStream", "24")
        patched.add("maxTargetsPerStream")

    missing_required = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing_required:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing_required)
        )

    _set_section_key(output, "TargetManagement", "outputShadowTracks", "0")

    reid_enabled = os.environ.get("CAMERA_V2_REID", "1").strip().lower() not in {"0", "false", "no", "off"}
    model = None
    if reid_enabled:
        model = _configure_reid(output)
    else:
        _set_section_key(output, "TrajectoryManagement", "enableReAssoc", "0")
        _set_section_key(output, "ReID", "reidType", "0")
        _set_section_key(output, "ReID", "outputReidTensor", "0")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    optional_applied = sorted(set(_OPTIONAL_PATCHES) & patched)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2 low-memory NvDCF profile for GTX 1050 Ti 4GB.",
        "# maxTargetsPerStream=24; close-person admission tuned; shadow output stays internal.",
        "# ReID=" + (f"enabled model={model.name}" if model else "disabled"),
        "# ReID engine is prebuilt before the live pipeline on the 4 GB GPU.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
