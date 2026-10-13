#!/usr/bin/env python3
"""Reconcile an auth-recovery scenario verdict with its generic soak subresult."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_r11_auth_quarantine_recovery import combine_auth_test_outcome


def audit(run_dir: Path) -> dict:
    auth = json.loads((run_dir / "auth-recovery-verification.json").read_text(encoding="utf-8"))
    generic = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    plan_path = run_dir / "auth-recovery-test-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else {}
    generic_exit = int(plan.get("generic_soak_exit_code", plan.get("result_code", 1)))
    outcome = combine_auth_test_outcome(auth, generic.get("status"), generic_exit)
    return {
        "schema_version": 1,
        "candidate_binary_sha256": auth.get("binary_sha256"),
        "scenario": "localhost RTSP 401 quarantine, credential correction, and app-owned source recovery",
        "scenario_status": outcome["status"],
        "scenario_exit_code": outcome["result_code"],
        "scenario_checks": outcome["scenario_checks"],
        "failed_scenario_checks": outcome["failed_scenario_checks"],
        "generic_soak_status": outcome["generic_soak_status"],
        "generic_soak_exit_code": outcome["generic_soak_exit_code"],
        "candidate_process_exit_code": generic.get("exit_code"),
        "generic_soak_is_authoritative_for_this_scenario": False,
        "scope_note": "The specialized auth verdict uses the explicit 401/quarantine/recovery contract; generic continuous-FPS flags remain separately visible and are not suppressed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scenario_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
