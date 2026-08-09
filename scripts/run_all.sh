#!/usr/bin/env bash
set -u
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SURVEILLANCE_PYTHON:-$repo_dir/venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then echo "Python environment not found: $python_bin" >&2; exit 1; fi
pids=()
shutdown() {
  trap - INT TERM EXIT
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap shutdown INT TERM EXIT
cd "$repo_dir" || exit 1
"$python_bin" -m services.api_service.app & pids+=("$!")
for _ in {1..40}; do curl -fsS http://127.0.0.1:8000/api/v1/ready >/dev/null 2>&1 && break; sleep .25; done
curl -fsS http://127.0.0.1:8000/api/v1/ready >/dev/null || { echo "API did not become ready" >&2; exit 1; }
"$python_bin" -m services.ml_service.app & pids+=("$!")
for _ in {1..120}; do curl -fsS http://127.0.0.1:8001/ready >/dev/null 2>&1 && break; sleep .5; done
curl -fsS http://127.0.0.1:8001/health >/dev/null || { echo "ML control endpoint did not become ready" >&2; exit 1; }
"$python_bin" -m services.frontend.main & pids+=("$!")
wait "${pids[-1]}"
