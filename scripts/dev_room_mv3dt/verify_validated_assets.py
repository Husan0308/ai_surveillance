#!/usr/bin/env python3
"""Verify the scoped production profile still matches the accepted prototype."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


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


def exact_sha256(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("an exact lowercase SHA256 is required; wildcards are not allowed")
    return value


def resolve_binary(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def load_staging_profile(repo: Path, path: Path) -> dict:
    """Explicit candidate pins are not an acceptance/rebaseline operation."""
    path = resolve_binary(repo, path)
    payload = json.loads(path.read_text())
    if payload.get("status") != "staged_pending_live_recall":
        raise ValueError("staging profile must explicitly remain pending live recall")
    exact_sha256(payload["source_sha256"])
    exact_sha256(payload["binary_sha256"])
    image = payload["toolchain_image"]
    if not isinstance(image, str) or not re.fullmatch(r"nvcr\.io/nvidia/deepstream@sha256:[0-9a-f]{64}", image):
        raise ValueError("staging runtime image must be digest-pinned")
    for relative, expected in payload.get("frozen_inputs_sha256", {}).items():
        input_path = Path(relative)
        if not input_path.parts or input_path.is_absolute() or ".." in input_path.parts or input_path.parts[0] not in {"config", "services"}:
            raise ValueError("staging frozen inputs must be repository config/service paths")
        exact_sha256(expected)
        if sha256(repo / input_path) != expected:
            raise ValueError(f"staging frozen input hash mismatch: {relative}")
    return {**payload, "binary": resolve_binary(repo, payload["binary"])}


def verify(
    repo: Path,
    binary: Path | None = None,
    *,
    candidate_source_sha256: str | None = None,
    candidate_binary_sha256: str | None = None,
) -> dict:
    manifest = json.loads((repo / "services/mv3dt_room/asset_manifest.json").read_text())
    profile = repo / "config/mv3dt_dev_room"
    errors: list[str] = []
    checked: dict[str, str] = {}
    staging = candidate_source_sha256 is not None or candidate_binary_sha256 is not None
    if staging:
        if binary is None or candidate_source_sha256 is None or candidate_binary_sha256 is None:
            raise ValueError("staging requires an explicit binary and both source/binary SHA256 pins")
        exact_sha256(candidate_source_sha256)
        exact_sha256(candidate_binary_sha256)
    if binary is None:
        runtime = yaml.safe_load((profile / "runtime.yaml").read_text())
        binary = resolve_binary(repo, runtime["runtime"]["binary"])
    else:
        binary = resolve_binary(repo, binary)
    accepted_assets = dict(manifest["validated_assets"])
    if staging:
        # Only this deliberately changed native compilation unit is staged.
        # Identity, bbox correction, inference/tracker config, and calibration
        # remain protected by their original accepted hashes and invariants.
        accepted_assets["native/deepstream_test5_app_main.c"] = candidate_source_sha256
    for relative, expected in {**accepted_assets, **manifest["accepted_config_sha256"]}.items():
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

    expected_binary = candidate_binary_sha256 if staging else manifest["accepted_binary_sha256"]
    actual_binary = None
    if not binary.is_file():
        errors.append(f"MV3DT binary missing: {binary}")
    else:
        actual_binary = sha256(binary)
        if actual_binary != expected_binary:
            errors.append(f"MV3DT binary hash mismatch: {binary}: {actual_binary} != {expected_binary}")

    result = {
        "ok": not errors,
        "scope": manifest["scope"],
        "checked": checked,
        "binary": str(binary),
        "binary_sha256": actual_binary,
        "expected_binary_sha256": expected_binary,
        "asset_role": "staging_pending_live_recall" if staging else "accepted_production",
        "staging_hash_override": staging,
        "production_accepted": not staging and not errors,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--staging-profile", type=Path,
                        help="Explicit exact candidate pins; never marks production accepted")
    args = parser.parse_args()
    if args.staging_profile:
        if args.binary:
            parser.error("choose --binary or --staging-profile, not both")
        staging = load_staging_profile(args.repo, args.staging_profile)
        result = verify(args.repo, staging["binary"],
                        candidate_source_sha256=staging["source_sha256"],
                        candidate_binary_sha256=staging["binary_sha256"])
    else:
        result = verify(args.repo, args.binary)
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
