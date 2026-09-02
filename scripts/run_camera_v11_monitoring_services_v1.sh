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

children=()
cleanup() {
  trap - INT TERM EXIT
  for pid in "${children[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${children[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM EXIT

"$APP_PY" -u -m services.ml_service.app.main &
children+=("$!")
"$APP_PY" -u -m services.api_service.app.main &
children+=("$!")
printf 'V11_MONITORING_SERVICES RESULT=STARTED ml_pid=%s api_pid=%s ml_port=%s api_port=%s\n' \
  "${children[0]}" "${children[1]}" "$ML_PORT" "$API_PORT"

set +e
wait -n "${children[@]}"
status=$?
set -e
printf 'V11_MONITORING_SERVICES RESULT=STOPPING child_status=%s\n' "$status"
exit "$status"
