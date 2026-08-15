from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / ".runtime" / "camera_v2"
SPARSE_CONFIG = RUNTIME_DIR / "config_tracker_NvDCF_sparse.yml"


# Sparse external YOLO needs a different target-creation profile than the stock
# DeepStream max_perf example, which assumes detector metadata arrives much more
# frequently. Values are intentionally conservative for a GTX 1050 Ti camera wall.
_PATCHES: dict[str, str] = {
    "minDetectorConfidence": "0.10",
    "enableBboxUnClipping": "1",
    "minIouDiff4NewTarget": "0.22",
    "minTrackerConfidence": "0.08",
    "probationAge": "0",
    "maxShadowTrackingAge": "60",
    "earlyTerminationAge": "8",
}


def prepare_sparse_tracker_config(stock: Path) -> Path:
    stock = Path(stock)
    if not stock.exists():
        raise RuntimeError(f"NvDCF stock config not found: {stock}")

    lines = stock.read_text(encoding="utf-8").splitlines()
    patched: set[str] = set()
    output: list[str] = []

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        replaced = False
        for key, value in _PATCHES.items():
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

    missing = sorted(set(_PATCHES) - patched)
    if missing:
        raise RuntimeError(
            "NvDCF stock config format is unexpected; could not patch: " + ", ".join(missing)
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated from DeepStream config_tracker_NvDCF_max_perf.yml.",
        "# Tuned for sparse external YOLO26m detections on Camera V2.",
        "# Do not edit: regenerated at runtime.",
    ]
    SPARSE_CONFIG.write_text("\n".join(header + output) + "\n", encoding="utf-8")
    return SPARSE_CONFIG
