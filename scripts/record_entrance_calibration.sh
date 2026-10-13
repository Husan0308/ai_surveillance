#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.runtime/full-stack-venv/bin/python"
RUN_DIR="${FULL_STACK_OUT:-$ROOT/.runtime/full-stack-calibration-20261009-r6-eos-fix}"
EXPECTED_BINARY_SHA256="6d3bf5613bebb0322c7c924b354f5f3c6097aa7b9750467c2d5c83ae58f817bd"
EXPECTED_ENGINE_SHA256="e220664bbbd5b57cc67f6f94fa6ba601a722df6f1ff9e58ab082039fe1765c16"
EXPECTED_TRACKER_SHA256="5a14896a3afa0159a874e6904987ff44504741c13119a43cbc076219f987d355"
CONTROL="$RUN_DIR/analytics/calibration-recording-command.tsv"
STATUS="$RUN_DIR/analytics/calibration-recording-status.json"

if [[ ! -x "$PY" ]]; then
  printf 'ERROR: project Python runtime is unavailable: %s\n' "$PY" >&2
  exit 2
fi

"$PY" - "$RUN_DIR" "$EXPECTED_BINARY_SHA256" "$EXPECTED_ENGINE_SHA256" "$EXPECTED_TRACKER_SHA256" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
expected_binary, expected_engine, expected_tracker = sys.argv[2:]
launch_path = run_dir / "launch.json"
status_path = run_dir / "analytics/calibration-recording-status.json"
if not launch_path.is_file() or not status_path.is_file():
    raise SystemExit("ERROR: the validated 180-second candidate is not running; no recording started")
launch = json.loads(launch_path.read_text(encoding="utf-8"))
if launch.get("analytics_binary_sha256") != expected_binary:
    raise SystemExit("ERROR: active executable is not the validated 180-second candidate; no recording started")
if launch.get("yolo_engine_sha256") != expected_engine:
    raise SystemExit("ERROR: active YOLO engine hash differs; no recording started")
if launch.get("local_tracker_sha256") != expected_tracker:
    raise SystemExit("ERROR: active local NvDCF profile differs; no recording started")
if not launch.get("yolo_context_verifier_enabled"):
    raise SystemExit("ERROR: active candidate runtime profile differs; no recording started")
status = json.loads(status_path.read_text(encoding="utf-8"))
if status.get("state") != "IDLE":
    raise SystemExit(f"ERROR: native recorder is not IDLE ({status.get('state')}); no recording started")
print("Candidate executable, YOLO, tracker, context-verifier mode, and recorder IDLE verified.")
PY

if [[ -e "$CONTROL" ]]; then
  printf 'ERROR: a recorder command is already pending; no recording started.\n' >&2
  exit 2
fi

socket_count="$(ss -Htnp state established '( dport = :554 )' | wc -l)"
if [[ "$socket_count" != "6" ]]; then
  printf 'ERROR: expected six established RTSP sockets, found %s; no recording started.\n' "$socket_count" >&2
  exit 2
fi

container_count="$(docker ps --filter name=surv-sixcam-analytics --format '{{.Names}}' | wc -l)"
if [[ "$container_count" != "1" ]]; then
  printf 'ERROR: expected exactly one six-camera analytics owner, found %s; no recording started.\n' "$container_count" >&2
  exit 2
fi

"$PY" - <<'PY'
import httpx

ml = httpx.get("http://127.0.0.1:8101/health", timeout=3).json()
api = httpx.get("http://127.0.0.1:8100/health", timeout=3).json()
cameras = httpx.get("http://127.0.0.1:8100/api/v1/cameras", timeout=3).json().get("cameras", [])
if ml.get("status") != "ok" or ml.get("online_camera_count") != 6:
    raise SystemExit("ERROR: ML is not healthy for six cameras; no recording started")
if api.get("status") != "ok" or api.get("ml_status") != "ok":
    raise SystemExit("ERROR: API/ML health check failed; no recording started")
if len(cameras) != 6 or any(not camera.get("online") for camera in cameras):
    raise SystemExit("ERROR: API does not report all six cameras online; no recording started")
if any(camera.get("pipeline_errors", 0) or camera.get("reconnects", 0) for camera in cameras):
    raise SystemExit("ERROR: a camera has pipeline errors or reconnects; no recording started")
print("ML/API healthy; all six camera sources online with no pipeline errors/reconnects.")
PY

SESSION_ID="entrance-$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="$ROOT/.runtime/human-validation-$SESSION_ID"
printf 'Recorder preflight passed. Running this command is your authorization to begin the 180-second session.\n'
printf 'Do not move until the controller prints the red REC STARTED message.\n'
exec "$PY" -B -m scripts.calibration_recording_controller \
  --room entrance \
  --run-dir "$RUN_DIR" \
  --evidence-dir "$EVIDENCE_DIR" \
  --session-id "$SESSION_ID" \
  --duration-sec 180
