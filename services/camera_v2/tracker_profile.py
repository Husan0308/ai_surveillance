from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"

# Production fixed-CCTV pedestrian profile.
#
# Important design choice: shadow tracks are kept INTERNAL to NvDCF for recovery,
# but are not promoted to live OSD. NVIDIA explicitly warns that overly permissive
# inactive/shadow output can create lingering ghost bboxes after a person leaves.
# Continuity is instead improved at the real detector/association/tracker levels.
_REQUIRED_PATCHES: dict[str, str] = {
    # Keep low-score pedestrian detections available for association. New false
    # targets are controlled by probation + early termination below.
    "minDetectorConfidence": "0.05",
    "enableBboxUnClipping": "1",
    # Reject near-duplicate new targets around an existing target.
    "minIouDiff4NewTarget": "0.14",
    # NVIDIA's optimized PeopleNet+NvDCF example is around 0.21. Use a slightly
    # more permissive value for our sparse external YOLO corrections.
    "minTrackerConfidence": "0.18",
    "probationAge": "1",
    "maxShadowTrackingAge": "40",
    "earlyTerminationAge": "2",
}

# Apply only when the selected DeepStream sample config exposes the key.
# Cascaded association is the key improvement: low-confidence detections are used
# to recover existing ACTIVE targets instead of being discarded, similar in spirit
# to ByteTrack's second-stage low-score association.
_OPTIONAL_PATCHES: dict[str, str] = {
    "useColorNames": "1",
    "useHog": "1",
    "featureImgSizeLevel": "5",
    # At 20 FPS a walking person moves only a small distance per video frame.
    # NVIDIA notes that too-large search regions reduce effective target feature
    # resolution, so keep padding at 1 while using level-5 (48x48) features.
    "searchRegionPaddingScale": "1",
    "associationMatcherType": "1",
    "tentativeDetectorConfidence": "0.22",
    "minMatchingScore4TentativeIou": "0.10",
    "minMatchingScore4Overall": "0.06",
    "minMatchingScore4SizeSimilarity": "0.05",
    "minMatchingScore4Iou": "0.02",
    "minMatchingScore4VisualSimilarity": "0.05",
    "usePrediction4Assoc": "1",
    # If an older NvDCF profile exposes this legacy parameter, do not set it near
    # zero; NVIDIA warns that very low inactive-output thresholds cause ghosts.
    "minTrackingConfidenceDuringInactive": "0.35",
}


def _set_or_insert_target_management(lines: list[str], key: str, value: str) -> bool:
    section = None
    for index, line in enumerate(lines):
        if line.strip() == "TargetManagement:":
            section = index
            break
    if section is None:
        return False

    for index in range(section + 1, len(lines)):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not lines[index].startswith((" ", "\t")) and stripped.endswith(":"):
            lines.insert(index, f"  {key}: {value}")
            return True
    lines.append(f"  {key}: {value}")
    return True


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

    missing_required = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing_required:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing_required)
        )

    # Shadow history may still be useful internally to NvDCF, but live rendering of
    # its misc history is intentionally disabled. This is the direct fix for stale
    # giant rectangles / ghost bboxes seen in the previous build.
    shadow_found = False
    for i, line in enumerate(output):
        if line.lstrip().startswith("outputShadowTracks:"):
            indent = line[: len(line) - len(line.lstrip())]
            output[i] = f"{indent}outputShadowTracks: 0"
            shadow_found = True
            break
    if not shadow_found:
        _set_or_insert_target_management(output, "outputShadowTracks", "0")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    optional_applied = sorted(set(_OPTIONAL_PATCHES) & patched)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2 production pedestrian tracking profile.",
        "# No synthetic/shadow OSD boxes: detector + cascaded NvDCF association only.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
