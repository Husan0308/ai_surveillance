from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"

# 4 GB Pascal fixed-CCTV pedestrian profile.
#
# NvMultiObjectTracker pre-allocates GPU memory from
# streams * maxTargetsPerStream, so keep the target pool realistic for an office.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    "enableBboxUnClipping": "1",
    "maxTargetsPerStream": "24",
    # A very low value suppresses a new target whenever it overlaps an existing
    # target even modestly. 0.50 is NVIDIA's documented default and lets a second
    # shoulder-to-shoulder person become a separate target when appropriate.
    "minIouDiff4NewTarget": "0.50",
    # Keep the visual tracker alive through short partial occlusions instead of
    # dropping the bbox as soon as DCF confidence softens.
    "minTrackerConfidence": "0.14",
    "probationAge": "1",
    # At ~20 FPS this gives roughly two seconds for recovery/re-association.
    "maxShadowTrackingAge": "40",
    "earlyTerminationAge": "2",
}

# Only patch keys actually present in the NVIDIA max_perf sample. Cascaded
# association keeps low-score YOLO person candidates useful for recovering an
# existing target, while the visual feature footprint stays realistic for a
# GTX 1050 Ti running six decoded streams plus YOLO26m.
_OPTIONAL_PATCHES: dict[str, str] = {
    "useColorNames": "0",
    "useHog": "1",
    "featureImgSizeLevel": "3",
    "searchRegionPaddingScale": "1",
    "associationMatcherType": "1",
    "tentativeDetectorConfidence": "0.10",
    "minMatchingScore4TentativeIou": "0.10",
    "minMatchingScore4Overall": "0.06",
    "minMatchingScore4SizeSimilarity": "0.05",
    "minMatchingScore4Iou": "0.02",
    "minMatchingScore4VisualSimilarity": "0.05",
    "usePrediction4Assoc": "1",
    "minTrackingConfidenceDuringInactive": "0.28",
    # Duplicate cleanup should require almost exact overlap so two nearby people
    # are not collapsed into one track.
    "minIou4TargetDuplicate": "0.94",
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

    if "maxTargetsPerStream" not in patched:
        if not _set_or_insert_target_management(output, "maxTargetsPerStream", "24"):
            raise RuntimeError("NvDCF config has no TargetManagement section")
        patched.add("maxTargetsPerStream")

    missing_required = sorted(set(_REQUIRED_PATCHES) - patched)
    if missing_required:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch required keys: "
            + ", ".join(missing_required)
        )

    # Shadow state is kept internally for recovery, but we do not emit synthetic
    # shadow-history boxes to the live OSD/UI.
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
        "# Camera V2 low-memory NvDCF profile for GTX 1050 Ti 4GB.",
        "# close-person separation + short-occlusion recovery tuned; no synthetic OSD boxes.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG