"""Stage an isolated candidate that runs the supplemental 896 GIE each frame.

The frozen 640 primary, parser, thresholds, tracker, and calibration are copied
unchanged.  This candidate exists to test whether interval-2 supplemental
detection cadence caused gaps in the reviewed CAM-05 seated-person miss clip.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-aux448-pretracker-verify-20261008-r41b"
OUT = ROOT / ".runtime/sixcam-cam05-aux448-every-frame-20261008-r42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite isolated candidate: {OUT}")
    manifest_path = PARENT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    profile = manifest["detector_profile"]
    if profile.get("aux_interval") != 2:
        raise RuntimeError("parent must be the reviewed interval-2 auxiliary candidate")
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != manifest["binary_sha256"]:
        raise RuntimeError("parent live binary does not match its manifest")
    if sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != manifest[
            "direct_file_binary_sha256"]:
        raise RuntimeError("parent direct-file binary does not match its manifest")

    shutil.copytree(PARENT, OUT)
    aux_config = OUT / "config/config_infer_aux_yolo26m_896.txt"
    text = aux_config.read_text()
    if text.count("interval=2") != 1:
        raise RuntimeError("expected exactly one interval=2 entry in the staged auxiliary config")
    aux_config.write_text(text.replace("interval=2", "interval=0", 1))

    profile["aux_interval"] = 0
    profile["aux_config_sha256"] = sha256(aux_config)
    manifest["config_sha256"][aux_config.name] = sha256(aux_config)
    manifest["candidate_label"] = (
        f"{OUT.name}: r41b detector/tracker/verifier with 896 auxiliary inference every frame"
    )
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch=6 primary unchanged; "
        "YOLO26m 896 supplemental batch-6 GIE interval=0 with exact-PTS merge and duplicate "
        "suppression; only >=0.25 primary detections enter normal NvDCF unchanged; 448 FP16 "
        "context verifier gates canonical target birth/low-confidence publication; local NvDCF type2"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "parent_live_binary_sha256": manifest["binary_sha256"],
        "parent_direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "change": "auxiliary 896 inference interval changed from 2 to 0 in isolated candidate config",
        "reason": "test whether sparse supplemental proposals cause repeated CAM-05 seated-person box gaps",
        "invariants": [
            "YOLO26m 640 primary engine/parser/config/thresholds/preprocessing unchanged",
            "only >=0.25 primary detections enter normal NvDCF unchanged",
            "NvDCF profile and thresholds unchanged",
            "448 verifier and exact frame/PTS join unchanged",
            "frozen V13 and calibration remain byte-identical",
        ],
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_DIAGNOSTIC_CANDIDATE",
        "candidate": str(OUT.relative_to(ROOT)),
        "live_binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": profile["primary_engine_sha256"],
        "primary_parser_sha256": profile["primary_parser_sha256"],
        "aux_engine_sha256": profile["aux_engine_sha256"],
        "aux_interval": profile["aux_interval"],
        "aux_config_sha256": profile["aux_config_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
