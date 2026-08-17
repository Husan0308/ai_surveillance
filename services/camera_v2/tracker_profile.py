from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"

# Local NvDCF tuning only. ReID/re-association is forcibly disabled below.
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
    """Make the generated NvDCF profile strictly camera-local.

    This is unconditional so stale CAMERA_V2_REID* shell variables cannot silently
    re-enable model loading, TensorRT engines, galleries or trajectory reassociation.
    """
    _set_section_key(lines, "TrajectoryManagement", "enableReAssoc", "0")
    _set_section_key(lines, "ReID", "reidType", "0")
    _set_section_key(lines, "ReID", "outputReidTensor", "0")
    _remove_section_keys(
        lines,
        "ReID",
        {"tltEncodedModel", "tltModelKey", "onnxFile", "modelEngineFile"},
    )


def prepare_sparse_tracker_config(stock: Path) -> Path:
    """Generate the low-memory local NvDCF profile used by the live wall."""
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

    missing = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing)
        )

    # Shadow tracks remain internal; only real NvDCF outputs reach OSD/heatmap.
    _set_section_key(output, "TargetManagement", "outputShadowTracks", "0")
    _disable_reid(output)

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    optional_applied = sorted(set(_OPTIONAL_PATCHES) & patched)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2 local NvDCF profile; cross-camera ReID is intentionally absent.",
        "# maxTargetsPerStream=24; close-person admission tuned; shadow output internal.",
        "# TrajectoryManagement.enableReAssoc=0; ReID.reidType=0; outputReidTensor=0.",
        "# No ReID model, TensorRT engine, gallery, sidecar or room topology is loaded.",
        "# Optional patches applied: "
        + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
