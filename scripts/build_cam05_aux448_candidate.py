"""Create an isolated 640+896 candidate using the reviewed 448 verifier.

The candidate preserves the 896 proposal path required by the CAM-05 miss and
disables optional per-frame shadow-context refresh, which was creating extra
verifier load. Primary >=0.25 NvDCF inputs and all detector/tracker parameters
remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / ".runtime/sixcam-cam05-cascade-20261008-r38-shadow-context-448-b2"
OUT = ROOT / ".runtime/sixcam-cam05-aux448-no-shadow-refresh-20261008-r40"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> dict:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {OUT}")
    manifest_path = PARENT / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(PARENT / "bin/cam-six-yolo-local6-posttracker") != manifest.get("binary_sha256"):
        raise RuntimeError("parent live binary does not match its manifest")
    if sha256(PARENT / "bin/cam-six-yolo-local6-direct-gatea") != manifest.get("direct_file_binary_sha256"):
        raise RuntimeError("parent direct-file binary does not match its manifest")
    if manifest.get("detector_profile", {}).get("aux_interval") != 2:
        raise RuntimeError("expected the reviewed interval-2 896 auxiliary profile")
    shutil.copytree(PARENT, OUT)
    manifest["candidate_label"] = (
        f"{OUT.name}: frozen 640 primary + interval-2 896 supplemental proposals; "
        "448 context verification; per-frame shadow refresh disabled"
    )
    manifest["shadow_context_refresh_enabled"] = False
    manifest["yolo_context_verifier"]["use"] = (
        "current-frame local person confirmation for 0.23-0.25 primary sidecar and "
        "new 640-primary/896-auxiliary target births; never changes >=0.25 primary NvDCF input"
    )
    manifest["yolo_context_verifier"]["scheduling"]["policy"] = (
        "three persistent batch-2 TensorRT contexts; bounded per-camera queues; exact source frame/PTS; "
        "26ms deadline; late results fail closed; optional per-frame shadow refresh disabled"
    )
    manifest["runtime"] = (
        "DeepStream 9.1; frozen YOLO26m FP16 640x640 batch=6 primary; interval-2 896 supplemental "
        "batch-6 GIE with exact-PTS merge and duplicate suppression; only >=0.25 primary detections "
        "enter normal NvDCF unchanged; 448 FP16 context verifier gates canonical birth/low-confidence "
        "publication; per-frame shadow refresh disabled to bound verifier demand; local NvDCF type2; "
        "no MV3DT/calibration"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(PARENT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "change": "manifest-only runtime policy: shadow_context_refresh_enabled true -> false",
        "reason": "reduce nonessential repeated shadow-crop verifier requests; low-confidence fallback and target-birth verification remain enabled",
    }
    (OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "status": "STAGED_EXPERIMENTAL",
        "candidate": str(OUT.relative_to(ROOT)),
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": manifest["detector_profile"]["primary_engine_sha256"],
        "auxiliary_engine_sha256": manifest["detector_profile"]["aux_engine_sha256"],
        "context_engine_sha256": manifest["yolo_context_verifier"]["engine_sha256"],
        "shadow_context_refresh_enabled": False,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
