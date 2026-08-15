from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"


# Fixed-CCTV pedestrian tracking profile for sparse external YOLO26m detections.
#
# The critical continuity setting is outputShadowTracks=1. DeepStream normally
# keeps a low-confidence NvDCF target alive internally in Shadow Tracking mode but
# suppresses its current-frame NvDsObjectMeta downstream. We consume the official
# NVDS_TRACKER_SHADOW_LIST_META in the native bridge and render that current-frame
# shadow bbox, so a person does not blink merely because tracker confidence dips.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.06",
    "enableBboxUnClipping": "1",
    "minIouDiff4NewTarget": "0.15",
    "minTrackerConfidence": "0.00",
    "probationAge": "0",
    "maxShadowTrackingAge": "70",
    "earlyTerminationAge": "5",
}

_OPTIONAL_PATCHES: dict[str, str] = {
    "useColorNames": "1",
    "useHog": "1",
    "featureImgSizeLevel": "4",
    "searchRegionPaddingScale": "2",
    "minTrackingConfidenceDuringInactive": "0.00",
    "processNoiseVar4Loc": "4.0",
    "processNoiseVar4Vel": "1.0",
    "measurementNoiseVar4Detector": "1.5",
    "measurementNoiseVar4Tracker": "3.0",
}


def _insert_target_management_key(lines: list[str], key: str, value: str) -> bool:
    """Insert a DeepStream TargetManagement key if the stock sample omits it."""
    section = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "TargetManagement:":
            section = index
            break
    if section is None:
        return False

    # DeepStream sample YAML uses two-space indentation inside sections.
    insert_at = len(lines)
    for index in range(section + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")) and stripped.endswith(":"):
            insert_at = index
            break
    lines.insert(insert_at, f"  {key}: {value}")
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
            prefix = key + ":"
            if stripped.startswith(prefix):
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

    # outputShadowTracks is not present in every NVIDIA sample config, so insert it
    # under TargetManagement when absent instead of treating absence as an error.
    shadow_present = any(line.lstrip().startswith("outputShadowTracks:") for line in output)
    if shadow_present:
        output = [
            (line[: len(line) - len(line.lstrip())] + "outputShadowTracks: 1")
            if line.lstrip().startswith("outputShadowTracks:")
            else line
            for line in output
        ]
    elif not _insert_target_management_key(output, "outputShadowTracks", "1"):
        raise RuntimeError("NvDCF config has no TargetManagement section for outputShadowTracks")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    optional_applied = sorted(set(_OPTIONAL_PATCHES) & patched)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2 continuous-person NvDCF profile.",
        "# outputShadowTracks=1 is consumed by native_meta_bridge for no-blink OSD.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
