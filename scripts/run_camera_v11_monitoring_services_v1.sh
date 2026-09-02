#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
COMMON_GIT_DIR="$(git rev-parse --path-format=absolute --git-common-dir)"
PRIMARY_ROOT="$(dirname "$COMMON_GIT_DIR")"
APP_PY="${V11_MONITORING_SERVICE_PYTHON:-$PRIMARY_ROOT/.venv/bin/python}"
[[ -x "$APP_PY" ]] || { printf 'V11_MONITORING_SERVICES RESULT=FAIL reason=python_missing path=%s\n' "$APP_PY" >&2; exit 1; }
"$APP_PY" -c 'import fastapi, httpx, uvicorn, yaml' || {
  printf 'V11_MONITORING_SERVICES RESULT=FAIL reason=dependencies_missing python=%s\n' "$APP_PY" >&2
  exit 1
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ML_HOST="${ML_HOST:-127.0.0.1}" ML_PORT="${ML_PORT:-8001}"
export API_HOST="${API_HOST:-127.0.0.1}" API_PORT="${API_PORT:-8000}"
export ML_SERVICE_URL="${ML_SERVICE_URL:-http://127.0.0.1:${ML_PORT}}"

ml_pid=""
api_pid=""
stopping=0
cleanup() {
  trap - INT TERM EXIT
  stopping=1
  for pid in "$ml_pid" "$api_pid"; do [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "$ml_pid" "$api_pid"; do [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true; done
}
handle_signal() { stopping=1; exit 0; }
trap handle_signal INT TERM
trap cleanup EXIT

start_ml() { "$APP_PY" -u -m services.ml_service.app.main & ml_pid="$!"; }
start_api() { "$APP_PY" -u -m services.api_service.app.main & api_pid="$!"; }
start_ml
start_api
printf 'V11_MONITORING_SERVICES RESULT=STARTED ml_pid=%s api_pid=%s ml_port=%s api_port=%s\n' \
  "$ml_pid" "$api_pid" "$ML_PORT" "$API_PORT"

while (( ! stopping )); do
  stopped_pid=""
  set +e
  wait -n -p stopped_pid "$ml_pid" "$api_pid"
  status=$?
  set -e
  (( stopping )) && break
  sleep 0.5
  if [[ "$stopped_pid" == "$ml_pid" ]]; then
    start_ml
    printf 'V11_MONITORING_SERVICES RESULT=RESTARTED service=ml_service pid=%s previous_status=%s\n' "$ml_pid" "$status"
  elif [[ "$stopped_pid" == "$api_pid" ]]; then
    start_api
    printf 'V11_MONITORING_SERVICES RESULT=RESTARTED service=api_service pid=%s previous_status=%s\n' "$api_pid" "$status"
  fi
done
