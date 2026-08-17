from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"
REID_MODEL_DIR = RUNTIME_DIR / "models" / "reid"
REID_MODEL_NAME = "resnet50_market1501_aicity156.onnx"

# 4 GB Pascal fixed-CCTV pedestrian profile.
#
# NvMultiObjectTracker pre-allocates GPU memory from
# streams * maxTargetsPerStream, so keep the target pool realistic for an office.
# The detector remains high resolution; NvDCF itself is intentionally lightweight.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    "enableBboxUnClipping": "1",
    "maxTargetsPerStream": "24",
    # New-target creation uses: create when max IoU to existing targets is LOWER
    # than minIouDiff4NewTarget. A high value therefore keeps two strongly
    # overlapping, but genuinely different, people eligible for separate tracks.
    "minIouDiff4NewTarget": "0.90",
    # Restore the known-stable lifecycle thresholds. Realtime OSD gaps are handled
    # downstream by a bounded display bridge rather than by weakening NvDCF state.
    "minTrackerConfidence": "0.18",
    "probationAge": "1",
    "maxShadowTrackingAge": "28",
    "earlyTerminationAge": "2",
}

# Only patch keys actually present in the NVIDIA max_perf sample. Cascaded
# association keeps low-score YOLO person candidates useful for recovering an
# existing target, while the visual feature footprint is kept small enough for a
# GTX 1050 Ti running six decoded streams plus YOLO26m.
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
    # Do not let NvDCF's periodic duplicate-track cleanup collapse two people that
    # are almost on top of each other. 0.98 means only near-identical track boxes
    # are eligible for duplicate removal; run it less frequently as well.
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


def resolve_reid_model() -> Path:
    override = os.environ.get("CAMERA_V2_REID_MODEL", "").strip()
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates.append(REID_MODEL_DIR / REID_MODEL_NAME)
    for root in _deepstream_roots():
        candidates.append(root / "samples/models/Tracker" / REID_MODEL_NAME)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    searched = "\n  - ".join(str(p) for p in candidates)
    raise RuntimeError(
        "Camera V2 ReID model is missing. Install the NVIDIA TAO model first:\n"
        "  python scripts/setup_camera_v2_reid.py\n"
        "Searched:\n  - " + searched
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


def _set_or_insert_target_management(lines: list[str], key: str, value: str) -> bool:
    _set_section_key(lines, "TargetManagement", key, value)
    return True


def _configure_reid(lines: list[str]) -> Path:
    model = resolve_reid_model()
    batch_size = max(1, min(16, int(os.environ.get("CAMERA_V2_REID_BATCH", "8"))))
    extraction_interval = max(-1, int(os.environ.get("CAMERA_V2_REID_INTERVAL", "5")))
    workspace_mb = max(64, min(512, int(os.environ.get("CAMERA_V2_REID_WORKSPACE_MB", "256"))))
    engine = Path(
        os.environ.get(
            "CAMERA_V2_REID_ENGINE",
            str(REID_MODEL_DIR / f"{model.stem}_b{batch_size}_gpu0_fp16.engine"),
        )
    ).expanduser().resolve()
    engine.parent.mkdir(parents=True, exist_ok=True)

    # Re-association is deliberately conservative. NvDCF handles within-camera
    # continuity; the Python GlobalReIDManager fuses identities across cameras.
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

    # NVIDIA TAO ReIdentificationNet v1.2: ResNet-50, 256-D embedding,
    # 3x256x128 RGB input. L2 normalization is explicitly enabled because the raw
    # model output is not normalized. Small batch/history keep 4 GB VRAM bounded.
    reid = {
        "reidType": "2",
        "batchSize": str(batch_size),
        "workspaceSize": str(workspace_mb),
        "reidFeatureSize": "256",
        "reidHistorySize": "32",
        "inferDims": "[3, 256, 128]",
        "networkMode": "1",
        "inputOrder": "0",
        "colorFormat": "0",
        "offsets": "[123.6750, 116.2800, 103.5300]",
        "netScaleFactor": "0.01735207",
        "addFeatureNormalization": "1",
        "keepAspc": "1",
        "minVisibility4GalleryUpdate": "0.55",
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

    # maxTargetsPerStream is critical to memory usage. Some NVIDIA sample revisions
    # omit it, so insert it explicitly under TargetManagement when needed.
    if "maxTargetsPerStream" not in patched:
        _set_or_insert_target_management(output, "maxTargetsPerStream", "24")
        patched.add("maxTargetsPerStream")

    missing_required = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing_required:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing_required)
        )

    # Shadow-history remains internal to NvDCF. The live OSD bridge below the tracker
    # is bounded independently, so we do not export long-lived shadow metadata here.
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
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
