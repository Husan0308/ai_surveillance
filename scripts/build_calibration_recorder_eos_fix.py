#!/usr/bin/env python3
"""Build an isolated recorder-EOS repair from the currently accepted r5 candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from scripts.build_calibration_recording_candidate import (
    FROZEN_V13,
    FROZEN_V13_SHA256,
    PROTECTED,
    compile_binary,
    sha256,
    source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = ROOT / ".runtime/sixcam-calibration-recording-20261009-r5"
DEFAULT_OUTPUT = ROOT / ".runtime/sixcam-calibration-recording-20261009-r6-eos-fix"
EXPECTED_PARENT_BINARY = "54edebdb4c2764b819d57014bafab17549ed94f080e0fa43c7bb3cd5cf080604"


def main(parent: Path = DEFAULT_PARENT, output: Path = DEFAULT_OUTPUT) -> dict:
    parent, output = parent.resolve(), output.resolve()
    if not parent.is_relative_to(ROOT / ".runtime") or not output.is_relative_to(ROOT / ".runtime"):
        raise ValueError("candidate parent/output must remain under .runtime")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate: {output}")
    manifest_path = parent / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binary = parent / "bin/cam-six-yolo-local6-posttracker"
    if sha256(binary) != EXPECTED_PARENT_BINARY or manifest.get("binary_sha256") != EXPECTED_PARENT_BINARY:
        raise RuntimeError("r5 parent executable does not match the validated binary SHA256")
    if sha256(FROZEN_V13) != FROZEN_V13_SHA256:
        raise RuntimeError("frozen V13 reference changed")
    for name, (relative, expected) in PROTECTED.items():
        if sha256(parent / relative) != expected:
            raise RuntimeError(f"r5 protected artifact mismatch: {name}")

    shutil.copytree(parent, output)
    compile_binary("cam-six-yolo-local6-posttracker", direct_file=False, output=output)
    compile_binary("cam-six-yolo-local6-direct-gatea", direct_file=True, output=output)

    manifest_path = output / "build-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["binary_sha256"] = sha256(output / "bin/cam-six-yolo-local6-posttracker")
    manifest["direct_file_binary_sha256"] = sha256(output / "bin/cam-six-yolo-local6-direct-gatea")
    manifest["source_sha256"] = source_hashes()
    manifest["candidate_label"] = "r6 EOS-finalization and graceful recorder-cancellation repair; r5 analytics behavior retained"
    manifest["calibration_recording"]["eos_finalization"] = (
        "schedule EOS exactly once; set sent flag only when callback pushes EOS; drain encoder/qtmux before branch removal"
    )
    manifest["calibration_recording"]["operator_cancel"] = (
        "session-scoped STOP command requests downstream EOS; early stop is REVIEW_REQUIRED, never COMPLETE"
    )
    manifest["build_provenance"] = {
        "parent_candidate": str(parent.relative_to(ROOT)),
        "parent_binary_sha256": EXPECTED_PARENT_BINARY,
        "changes": [
            "fixed EOS scheduled/sent flag ordering that prevented qtmux finalization",
            "added session-scoped native STOP command with explicit REVIEW_REQUIRED semantics for early cancellation",
            "host controller catches Ctrl+C, asks for native EOS, and keeps evidence child alive for orderly close",
        ],
        "invariants": [
            "YOLO, NvDCF, canonical output, preview, source ownership, and calibration unchanged from r5",
            "same six RTSP sources; no additional source owner",
            "full-duration recording is COMPLETE only after duration, frame-count, and queue-integrity checks pass",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "BUILT_ISOLATED_EOS_FIX_CANDIDATE",
        "candidate": str(output.relative_to(ROOT)),
        "parent_binary_sha256": EXPECTED_PARENT_BINARY,
        "binary_sha256": manifest["binary_sha256"],
        "direct_file_binary_sha256": manifest["direct_file_binary_sha256"],
        "primary_engine_sha256": PROTECTED["primary_engine"][1],
        "primary_parser_sha256": PROTECTED["primary_parser"][1],
        "local_tracker_sha256": PROTECTED["local_tracker"][1],
        "frozen_v13_sha256": FROZEN_V13_SHA256,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(main(args.parent, args.output), indent=2))
