#!/usr/bin/env python3
"""F0 inventory/checkpoint only. Does not launch services or accept assets.

Records hashes, not configuration contents, so camera credentials and secrets
never enter the evidence. Existing evidence directories cannot be overwritten.
Use --verify to detect protected-file/artifact/ref drift after a freeze stage.
Known production manifest mismatches are recorded, never silently rebaselined.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

PROTECTED_PREFIXES = (
    "config/mv3dt_dev_room/", "services/mv3dt_room/native/",
)
PROTECTED_FILES = (
    "config/cameras.yaml", "services/mv3dt_room/asset_manifest.json",
    "config/deepstream/config_tracker_NvDCF_yolo26m_retention_v13_probation0.yml",
    "docs/NVDCF_CAUSAL_ACCEPTANCE_20261002.md",
)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "exists": path.is_file(),
            "sha256": sha(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None}


def freeze_refs(root: Path) -> str:
    return git(root, "for-each-ref", "--format=%(refname) %(objectname)",
               "refs/heads/freeze/", "refs/remotes/origin/freeze/")


def manifest_inventory(root: Path, binary: Path) -> dict:
    manifest = json.loads((root / "services/mv3dt_room/asset_manifest.json").read_text())
    checks = {}
    for name, expected in manifest["validated_assets"].items():
        record = file_record(root / "services/mv3dt_room" / name)
        checks[record["path"]] = dict(record, declared_sha256=expected,
                                     matches_declaration=record["sha256"] == expected)
    for name, expected in manifest["accepted_config_sha256"].items():
        record = file_record(root / "config/mv3dt_dev_room" / name)
        checks[record["path"]] = dict(record, declared_sha256=expected,
                                     matches_declaration=record["sha256"] == expected)
    record = file_record(binary)
    checks[record["path"]] = dict(record, declared_sha256=manifest["accepted_binary_sha256"],
                                 matches_declaration=record["sha256"] == manifest["accepted_binary_sha256"])
    return {"checks": checks, "mismatch_count": sum(not r["matches_declaration"] for r in checks.values()),
            "asset_acceptance_claimed": False}


def capture(root: Path, output: Path, artifacts: list[Path]) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("F0 evidence already exists; use --verify, never overwrite it")
    tracked = git(root, "ls-files", "-z").rstrip("\0").split("\0")
    protected = sorted({p for p in tracked if p in PROTECTED_FILES
                        or p.startswith(PROTECTED_PREFIXES)})
    important = sorted({p for p in tracked if p.startswith(("config/", "services/mv3dt_room/", "services/camera_v11/"))
                        or p in ("scripts/start_full_live_stack.sh", "scripts/setup_v91_py310_runtime_deps.sh",
                                 ".env.example", "docs/NVDCF_CAUSAL_ACCEPTANCE_20261002.md")})
    # Deliberately do not evaluate environment variables or load camera URLs.
    runtime = (root / "config/mv3dt_dev_room/runtime.yaml").read_text()
    match = re.search(r"^  binary: ([^\n]+)$", runtime, re.MULTILINE)
    if not match:
        raise ValueError("cannot resolve configured runtime binary without guessing")
    binary = Path(match[1].strip())
    if not binary.is_absolute():
        binary = root / binary
    acceptance = root / ".runtime/nvdcf-causal-20261002/final-acceptance.json"
    defaults = [binary, acceptance]
    models = root / ".runtime/yolo26m-full-pipeline-20261002-8gkAUa2W/models/yolo26m"
    defaults.extend(models / n for n in ("model.engine", "model.onnx", "parser.so", "nvinfer.txt", "build.json"))
    if acceptance.is_file():
        accepted = json.loads(acceptance.read_text())
        for dataset in accepted.get("datasets", {}).values():
            run = Path(dataset["run"])
            defaults.extend(run / n for n in ("run/config_tracker.yml", "run/config_pgie.txt", "execution_assets.json"))
    all_artifacts = sorted({p.resolve() for p in [*defaults, *artifacts]})
    report = {
        "schema_version": 1, "freeze": "F0", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "branch": git(root, "branch", "--show-current").strip(),
        "head": git(root, "rev-parse", "HEAD").strip(),
        "git_status": git(root, "status", "--porcelain=v1"),
        "worktrees": git(root, "worktree", "list", "--porcelain"),
        "recent_commits": git(root, "log", "-10", "--format=%H %s"),
        "freeze_refs": freeze_refs(root), "audit_interpreter": sys.executable,
        "project_venv_exists": (root / ".venv/bin/python").exists(),
        "protected_hashes": {p: file_record(root / p) for p in protected},
        "important_hashes": {p: file_record(root / p) for p in important},
        "artifact_hashes": {str(p): file_record(p) for p in all_artifacts},
        "manifest_inventory": manifest_inventory(root, binary),
        "production_changed": False, "camera_test": "NOT_RUN_F0_ONLY",
        "known_blockers": [
            {"stage": "F1", "issue": "Project runtime ownership/dependency preflight not frozen; no project .venv",
             "evidence": "project_venv_exists"},
            {"stage": "F1", "issue": "OSNet interpreter points into Trash",
             "evidence": "scripts/dev_room_mv3dt/run_room_pair.py:44"},
            {"stage": "F1", "issue": "Launcher command substitution can capture bootstrap status stdout",
             "evidence": "scripts/start_full_live_stack.sh:bootstrap_full_stack_python"},
            {"stage": "F3", "issue": "Launcher hard-codes 8000/8001; actual deployment port ownership must be reconciled",
             "evidence": "scripts/start_full_live_stack.sh; calibration service previously owned 8000"},
            {"stage": "F4/F5/F6", "issue": "PeopleNet live profile, six-camera YOLO/stable policy, and YOLO/V13 replay are distinct stacks",
             "evidence": "config/mv3dt_dev_room/runtime.yaml; README.md; docs/NVDCF_CAUSAL_ACCEPTANCE_20261002.md"},
            {"stage": "F4/F8", "issue": "Accepted-manifest/source/binary drift exists; do not promote or edit hashes to mask it",
             "evidence": "manifest_inventory"},
            {"stage": "F9", "issue": "Absolute metric scale lacks measured physical anchor; defer calibration until runtime stable",
             "evidence": str(acceptance)},
            {"stage": "F5/F6/F8", "issue": "V13 evidence is replay-only, not completed live integration acceptance",
             "evidence": "docs/NVDCF_CAUSAL_ACCEPTANCE_20261002.md"},
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify(report: dict) -> dict:
    changed = []
    for group in ("protected_hashes", "artifact_hashes"):
        for name, expected in report[group].items():
            actual = file_record(Path(expected["path"]))
            if actual["sha256"] != expected["sha256"] or actual["exists"] != expected["exists"]:
                changed.append({"group": group, "file": name, "before": expected, "after": actual})
    root = Path(report["root"])
    if freeze_refs(root) != report["freeze_refs"]:
        changed.append({"group": "freeze_refs", "before": report["freeze_refs"], "after": freeze_refs(root)})
    return {"ok": not changed, "changed": changed,
            "protected_count": len(report["protected_hashes"]), "artifact_count": len(report["artifact_hashes"]),
            "production_asset_gate_pass_claimed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify(json.loads((args.output / "baseline.json").read_text()))
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    report = capture(args.root, args.output, args.artifact)
    print(json.dumps({k: report[k] for k in ("freeze", "branch", "head", "git_status", "project_venv_exists")}, indent=2))
    print(f"Protected files: {len(report['protected_hashes'])}; manifest mismatches recorded: {report['manifest_inventory']['mismatch_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
