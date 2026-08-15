from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"


# Sparse external YOLO26m + fixed CCTV camera tuning.
#
# Important behavior:
# - a low minIouDiff4NewTarget rejects a new detector box if it significantly
#   overlaps an existing target, reducing duplicate boxes;
# - a low minTrackerConfidence keeps a walking/partially-occluded person visible
#   instead of immediately moving the target to shadow mode (where current-frame
#   output is suppressed);
# - shadow age stays long enough to recover after short occlusions/missed YOLO
#   frames without keeping ghosts forever.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.08",
    "enableBboxUnClipping": "1",
    "minIouDiff4NewTarget": "0.15",
    "minTrackerConfidence": "0.06",
    "probationAge": "0",
    "maxShadowTrackingAge": "70",
    "earlyTerminationAge": "4",
}

# Accuracy/performance knobs are patched only when the selected DeepStream sample
# profile contains them. The balanced `perf` profile is preferred by the runtime.
_OPTIONAL_PATCHES: dict[str, str] = {
    "useColorNames": "1",
    "useHog": "1",
    "featureImgSizeLevel": "4",
    "minTrackingConfidenceDuringInactive": "0.06",
}


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

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    optional_applied = sorted(set(_OPTIONAL_PATCHES) & patched)
    header = [
        f"# Auto-generated from {stock.name}.",
        "# Camera V2: frequent sparse YOLO26m + continuous NvDCF pedestrian tracking.",
        "# Tuned to reduce duplicate targets and bbox blink during walking/occlusion.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
