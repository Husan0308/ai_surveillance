#!/usr/bin/env python3
"""Verify the scoped production profile still matches the accepted prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_values(path: Path) -> list[float]:
    text = path.read_text()
    match = re.search(r"projectionMatrix_3x4_w2p:\s*(.*?)\n\s*modelInfo:", text, re.S)
    if not match:
        raise ValueError(f"missing projection matrix in {path}")
    return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", match.group(1))]


def verify(repo: Path, binary: Path | None) -> dict:
    manifest = json.loads((repo / "services/mv3dt_room/asset_manifest.json").read_text())
    profile = repo / "config/mv3dt_dev_room"
    errors: list[str] = []
    checked: dict[str, str] = {}
    for relative, expected in {**manifest["validated_assets"], **manifest["accepted_config_sha256"]}.items():
        path = (repo / "services/mv3dt_room" / relative) if relative.startswith("native/") or relative.endswith(".py") else profile / relative
        if not path.exists():
            errors.append(f"missing {path}")
            continue
        actual = sha256(path)
        checked[str(path.relative_to(repo))] = actual
        if actual != expected:
            errors.append(f"hash mismatch {path}: {actual} != {expected}")

    msgconv = (profile / "config_msgconv.txt").read_text()
    if sorted(re.findall(r"^id=(CAM-\d+)$", msgconv, re.M)) != ["CAM-01", "CAM-04"]:
        errors.append("message conversion scope is not exactly CAM-01 and CAM-04")
    for forbidden in ("CAM-02", "CAM-03", "CAM-05", "CAM-06"):
        if forbidden in msgconv:
            errors.append(f"forbidden camera in room-pair config: {forbidden}")

    pgie = (profile / "config_pgie.txt").read_text()
    ds = (profile / "config_deepstream.txt").read_text()
    required = (
        "resnet34_peoplenet.onnx_b2_gpu0_fp16.engine",
        "batch-size=2",
        "interval=0",
        "filter-out-class-ids=1;2",
        "infer-dims=3;544;960",
    )
    for token in required:
        if token not in pgie + ds:
            errors.append(f"missing PN2.6.3 invariant: {token}")
    if "live-source=0" not in ds:
        errors.append("offline profile must retain live-source=0")
    tracker = (profile / "config_tracker.yml").read_text()
    for token in ("ObjectModelProjection:", "MultiViewAssociator:", "VisualTracker:", "enableReAssoc: 1"):
        if token not in tracker:
            errors.append(f"missing accepted MV3DT invariant: {token}")

    existing = repo / ".runtime/mv3dt/calibration/dev-room-cam01-cam04-vggt-v1/camInfo"
    for index in ("00", "01"):
        scoped = profile / "camInfo" / f"cam_{index}.yml"
        old = existing / f"camInfo_{index}.yml"
        if scoped.exists() and old.exists():
            if projection_values(scoped) != projection_values(old):
                errors.append(f"VGGT projection matrix changed for camera {index}")

    if binary is not None:
        if not binary.exists():
            errors.append(f"accepted MV3DT binary missing: {binary}")
        elif sha256(binary) != manifest["accepted_binary_sha256"]:
            errors.append(f"accepted MV3DT binary hash mismatch: {binary}")

    result = {
        "ok": not errors,
        "scope": manifest["scope"],
        "checked": checked,
        "binary": str(binary) if binary else None,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.repo, args.binary)["ok"] else 1)


if __name__ == "__main__":
    main()
