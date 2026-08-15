from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"


# Fixed-CCTV pedestrian tracking profile for sparse external YOLO26m detections.
#
# NVIDIA documents two behaviors that matter for our office cameras:
# 1) NvDCF can suppress downstream current-frame output when tracker confidence
#    falls below minTrackerConfidence and a target goes to shadow mode;
# 2) dynamic / abrupt motion needs state-estimator measurements to be trusted more,
#    otherwise the fused bbox can visibly lag behind the actual target.
_REQUIRED_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.06",
    "enableBboxUnClipping": "1",
    "minIouDiff4NewTarget": "0.15",
    # Do not hide a valid DCF target merely because correlation confidence dips for
    # a few frames while a person walks, raises an arm, turns, or is half occluded.
    "minTrackerConfidence": "0.00",
    "probationAge": "0",
    "maxShadowTrackingAge": "70",
    "earlyTerminationAge": "5",
}

# These parameters exist in the DeepStream NvDCF perf/accuracy sample profiles.
# They are intentionally optional so slightly different DeepStream 7.x configs
# remain usable. Lower measurement noise = react faster to measured motion; larger
# search region helps a fast walker remain inside the DCF crop on the next frame.
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
        "# Camera V2 fast-pedestrian NvDCF tuning.",
        "# Goal: less shadow-output blink and less state-estimator lag.",
        "# Optional patches applied: " + (", ".join(optional_applied) if optional_applied else "none"),
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
