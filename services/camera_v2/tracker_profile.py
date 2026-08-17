from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"

# 4 GB Pascal fixed-CCTV pedestrian profile.
#
# NvMultiObjectTracker pre-allocates GPU memory from
# streams * maxTargetsPerStream, so keep the target pool realistic for an office.
# The detector remains high resolution; NvDCF itself is intentionally lightweight.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.05",
    # Do not extrapolate a full bbox outside the camera FOV. New partial people at
    # the image edge are admission-gated before NvDCF, and existing tracks can rely
    # on normal shadow tracking while they leave the frame.
    "enableBboxUnClipping": "0",
    "maxTargetsPerStream": "24",
    # NVIDIA's 0.5 default is much safer for two nearby people. 0.14 treated many
    # overlapping person detections as duplicates of an existing target.
    "minIouDiff4NewTarget": "0.50",
    "minTrackerConfidence": "0.14",
    "probationAge": "1",
    "maxShadowTrackingAge": "38",
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
    "minTrackingConfidenceDuringInactive": "0.28",
}


def _set_or_insert_target_management(lines: list[str], key: str, value: str) -> bool:
    section = None
    for index, line in enumerate(lines):
        if line.strip() == "TargetManagement:":
            section = index
            break
    if section is None:
        return False

    # Replace when already present in TargetManagement.
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

    # maxTargetsPerStream is critical to memory usage. Some NVIDIA sample revisions
    # omit it, so insert it explicitly under TargetManagement when needed.
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

    # Never render shadow-history/synthetic tracks. They may stay inside NvDCF for
    # recovery, but live OSD contains only real current-frame tracked objects.
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
        "# maxTargetsPerStream=24; no synthetic/shadow OSD boxes.",
        "# Partial edge detections are gated before tracker admission; bbox unclipping is disabled.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
