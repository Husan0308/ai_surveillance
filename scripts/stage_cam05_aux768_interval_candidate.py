"""Stage an isolated verifier-latency candidate with a reduced 768-GIE cadence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-context448-batch6-20261008-r57"
INTERVAL = int(os.getenv("SIXCAM_AUX_768_INTERVAL", "1"))
OUT = ROOT / f".runtime/sixcam-cam05-context448-batch6-auxi{INTERVAL}-20261008-r58"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if not 1 <= INTERVAL <= 2:
        raise ValueError("diagnostic auxiliary interval must be 1 or 2")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    manifest = json.loads((PARENT / "build-manifest.json").read_text())
    for binary_name, key in (("cam-six-yolo-local6-posttracker", "binary_sha256"),
                             ("cam-six-yolo-local6-direct-gatea", "direct_file_binary_sha256")):
        if sha256(PARENT / "bin" / binary_name) != manifest[key]:
            raise RuntimeError(f"r57 binary does not match manifest: {binary_name}")
    if manifest["detector_profile"].get("aux_interval") != 0:
        raise RuntimeError("parent must use 768 supplemental inference on every frame")

    shutil.copytree(PARENT, OUT)
    config_name = manifest["detector_profile"]["aux_config"]
    config = OUT / "config" / config_name
    original = config.read_text()
    if original.count("interval=0") != 1:
        raise RuntimeError("expected exactly one 768-GIE interval=0 setting")
    config.write_text(original.replace("interval=0", f"interval={INTERVAL}", 1))
    digest = sha256(config)
    manifest["detector_profile"]["aux_interval"] = INTERVAL
    manifest["detector_profile"]["aux_config_sha256"] = digest
    manifest["config_sha256"][config_name] = digest
    manifest["candidate_label"] = (
        f"{OUT.name}: r57 context batch-6 verifier with 768 auxiliary interval={INTERVAL}"
    )
    manifest["runtime"] = manifest["runtime"].replace(
        "768 supplemental batch-6 GIE interval=0", f"768 supplemental batch-6 GIE interval={INTERVAL}")
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_binary_sha256": manifest["binary_sha256"],
        "change": f"isolated 768 auxiliary GIE cadence interval 0 -> {INTERVAL}",
        "reason": "test whether auxiliary GPU contention delays same-frame person verification",
        "invariants": [
            "frozen YOLO26m 640 primary and >=0.25 NvDCF input unchanged",
            "NvDCF config and thresholds unchanged",
            "448px batch-6 verifier engine, strict matching and 26ms deadline unchanged",
            "no camera-specific location/frame rule introduced",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "STAGED_ISOLATED_CADENCE_DIAGNOSTIC", "candidate": str(OUT.relative_to(ROOT)),
            "binary_sha256": manifest["binary_sha256"], "aux_interval": INTERVAL,
            "aux_config_sha256": digest}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
