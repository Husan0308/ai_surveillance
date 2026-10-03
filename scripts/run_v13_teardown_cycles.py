"""Run repeated occupied F5 live teardown cycles against one pinned candidate build."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_room_candidate import ROOT


def _rtsp_sockets() -> list[str]:
    result = subprocess.run(["ss", "-Htnp"], capture_output=True, text=True, check=False)
    return [line for line in result.stdout.splitlines()
            if ":554" in line and ("ESTAB" in line or "ESTABLISHED" in line)]


def _candidate_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=False,
    )
    return [name for name in result.stdout.splitlines()
            if name.startswith("ai-surveillance-v13-candidate-")]


def cycle_pass(row: dict) -> bool:
    result = row.get("result") or {}
    return (
        row.get("runner_returncode") == 0
        and result.get("status") == "PASS"
        and result.get("native_exit") == 0
        and result.get("identity_exit") == 0
        and result.get("capture_exit") == 0
        and not row.get("shutdown_backtrace_present")
        and not row.get("rtsp_sockets_after")
        and not row.get("candidate_containers_after")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    if args.cycles < 1:
        raise SystemExit("--cycles must be >= 1")
    output_root = args.output_root.resolve()
    output_root.relative_to(ROOT / ".runtime")
    output_root.mkdir(parents=True, exist_ok=False)

    native_build = args.native_build.resolve()
    if not native_build.is_file():
        raise SystemExit(f"missing native build record: {native_build}")

    rows: list[dict] = []
    for index in range(1, args.cycles + 1):
        cycle_dir = output_root / f"cycle-{index:02d}"
        command = [
            sys.executable, "-B", "-m", "scripts.run_v13_room_candidate",
            "--dataset", "live",
            "--duration", str(args.duration),
            "--shutdown-debug",
            "--native-build", str(native_build),
            "--output", str(cycle_dir),
        ]
        started = time.time()
        proc = subprocess.run(command, cwd=ROOT)
        result_path = cycle_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        row = {
            "cycle": index,
            "started_epoch": started,
            "runner_returncode": proc.returncode,
            "result": result,
            "shutdown_backtrace_present": (cycle_dir / "logs/shutdown-backtrace.log").exists(),
            "rtsp_sockets_after": _rtsp_sockets(),
            "candidate_containers_after": _candidate_containers(),
        }
        row["pass"] = cycle_pass(row)
        rows.append(row)
        (output_root / "teardown-cycles.json").write_text(
            json.dumps({
                "required_cycles": args.cycles,
                "completed_cycles": len(rows),
                "status": "PASS" if len(rows) == args.cycles and all(r["pass"] for r in rows) else "FAIL",
                "cycles": rows,
            }, indent=2) + "\n"
        )
        print(json.dumps({
            "cycle": index,
            "pass": row["pass"],
            "native_exit": (result or {}).get("native_exit"),
            "status": (result or {}).get("status"),
            "shutdown_backtrace_present": row["shutdown_backtrace_present"],
            "rtsp_sockets_after": len(row["rtsp_sockets_after"]),
            "candidate_containers_after": row["candidate_containers_after"],
        }), flush=True)
        if not row["pass"]:
            raise SystemExit(1)

    print(json.dumps({
        "status": "PASS",
        "cycles": args.cycles,
        "evidence": str(output_root / "teardown-cycles.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
