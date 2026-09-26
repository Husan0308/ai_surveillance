#!/usr/bin/env python3
"""Run the three consecutive production CAM-01/CAM-04 startup checks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/dev_room_mv3dt/run_room_pair.py"
REPORTER = ROOT / "scripts/dev_room_mv3dt/report_startup_run.py"

def main() -> int:
    duration = os.environ.get("MV3DT_STARTUP_RUN_SEC", "50")
    env = dict(os.environ)
    env["MV3DT_SOURCE_HEALTH_DIR"] = "1"
    env["MV3DT_FRAME_AUDIT_LOG"] = "1"
    env.pop("MV3DT_UI_PREVIEW_DIR", None)
    runs = []
    for index in range(1, 4):
        print(f"STARTUP_GATE run={index}/3 duration={duration}s", flush=True)
        result = subprocess.run(
            [sys.executable, str(RUNNER), "--mode", "live", "--duration", duration, "--skip-render"],
            cwd=ROOT, env=env, check=True, capture_output=True, text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        run_path = Path(lines[-1])
        subprocess.run([sys.executable, str(REPORTER), str(run_path)], cwd=ROOT, env=env, check=True)
        report = json.loads((run_path / "startup_report.json").read_text())
        runs.append(report)
        if not report["pass_within_startup_gate"]:
            print(f"STARTUP_GATE FAIL run={index} path={run_path}", flush=True)
            return 1
        print(f"STARTUP_GATE PASS run={index} path={run_path}", flush=True)
    output = ROOT / ".runtime/mv3dt/dev-room-cam01-cam04/startup_gate_report.json"
    output.write_text(json.dumps({"runs": runs, "pass": len(runs) == 3}, indent=2) + "\n")
    print(f"STARTUP_GATE PASS 3/3 report={output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
