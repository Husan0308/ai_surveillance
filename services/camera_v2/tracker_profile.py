from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"

# Local NvDCF tuning only. ReID/re-association is forcibly disabled below.
#
# IMPORTANT PERFORMANCE CONTRACT
# ------------------------------
# This profile is generated from NVIDIA's config_tracker_NvDCF_max_perf.yml and
# must stay a max-performance visual tracker.  An older Camera V2 patch silently
# changed it to HOG-only + feature level 3.  HOG is 18 channels and level 3 uses
# a larger feature image, so that override made the visual tracker substantially
# more expensive on every frame of every camera.  On the GTX 1050 Ti this showed
# up as 100-250 ms wall stalls once six streams + TRT were active.
#
# ColorNames-only + feature level 2 follows NVIDIA's lightweight NvDCF profile:
# enough appearance information for local continuity, while YOLO remains the
# authority that periodically corrects geometry/classification.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    "enableBboxUnClipping": "0",
    "maxTargetsPerStream": "24",
    "minIouDiff4NewTarget": "0.72",
    "minTrackerConfidence": "0.10",
    "probationAge": "1",
    "maxShadowTrackingAge": "100",
    "earlyTerminationAge": "4",
}

_OPTIONAL_PATCHES: dict[str, str] = {
    # NVIDIA max-perf style visual features. Do not turn HOG back on here unless
    # a measured tracking-accuracy regression justifies the GPU cost.
    "useColorNames": "1",
    "useHog": "0",
    "useHighPrecisionFeature": "0",
    "featureImgSizeLevel": "2",
    "searchRegionPaddingScale": "1",
    "associationMatcherType": "1",
    "tentativeDetectorConfidence": "0.20",
    "minMatchingScore4TentativeIou": "0.12",
    "minMatchingScore4Overall": "0.08",
    "minMatchingScore4SizeSimilarity": "0.08",
    "minMatchingScore4Iou": "0.03",
    "minMatchingScore4VisualSimilarity": "0.08",
    "usePrediction4Assoc": "1",
    "minTrackingConfidenceDuringInactive": "0.08",
    "minIou4TargetDuplicate": "0.90",
    "targetDuplicateRunInterval": "1",
}


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


def _disable_reid(lines: list[str]) -> None:
    """Make the generated NvDCF profile strictly camera-local."""
    _set_section_key(lines, "TrajectoryManagement", "enableReAssoc", "0")
    _set_section_key(lines, "ReID", "reidType", "0")
    _set_section_key(lines, "ReID", "outputReidTensor", "0")
    _remove_section_keys(
        lines,
        "ReID",
        {"tltEncodedModel", "tltModelKey", "onnxFile", "modelEngineFile"},
    )


def prepare_sparse_tracker_config(stock: Path) -> Path:
    """Generate the low-memory camera-local NvDCF profile used by the live wall."""
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
        _set_section_key(output, "TargetManagement", "maxTargetsPerStream", "24")
        patched.add("maxTargetsPerStream")

    # Some stock versions may omit optional visual keys. Insert the performance
    # contract into the correct sections rather than silently falling back.
    visual_keys = {
        "useColorNames": "1",
        "useHog": "0",
        "useHighPrecisionFeature": "0",
        "featureImgSizeLevel": "2",
    }
    for key, value in visual_keys.items():
        if key not in patched:
            _set_section_key(output, "VisualTracker", key, value)
            patched.add(key)

    missing = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing)
        )

    # Shadow output is real NvDCF current-frame localization. It lets the visual
    # tracker bridge sparse detector observations without a Python sticky/ghost box.
    _set_section_key(output, "TargetManagement", "outputShadowTracks", "1")
    _disable_reid(output)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2 local NvDCF MAX-PERF profile; cross-camera ReID is absent.",
        "# VisualTracker: ColorNames=1 HOG=0 featureImgSizeLevel=2 highPrecision=0.",
        "# maxShadowTrackingAge=100; outputShadowTracks=1; continuity-first tracking.",
        "# Shadow output is real NvDCF metadata, not a fabricated hold box.",
        "# TrajectoryManagement.enableReAssoc=0; ReID.reidType=0; outputReidTensor=0.",
        "# Do not edit: regenerated at runtime.",
    ]
    text = "\n".join(header + output) + "\n"
    for required in (
        "useColorNames: 1",
        "useHog: 0",
        "featureImgSizeLevel: 2",
        "maxShadowTrackingAge: 100",
        "outputShadowTracks: 1",
    ):
        if required not in text:
            raise RuntimeError(f"NvDCF max-perf contract missing {required}")
    SPARSE_CONFIG.write_text(text, encoding="utf-8")
    print(
        "CAMERA_NVDCF_MAX_PERF useColorNames=1 useHog=0 featureImgSizeLevel=2 "
        "highPrecision=0 outputShadowTracks=1 verified=1",
        flush=True,
    )
    return SPARSE_CONFIG
