#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/setup_camera_v2_reid.py

# The original ReID smoke test contains assertions for the fixed six-camera
# office pairing. Keep that stronger preflight when all six are enabled, but do
# not block Settings users who intentionally disable/delete a camera.
ACTIVE_CAMERAS="$(python - <<'PY'
import yaml
raw = yaml.safe_load(open('config/cameras.yaml', encoding='utf-8')) or {}
print(sum(1 for row in (raw.get('cameras') or []) if bool(row.get('enabled', True))))
PY
)"

if [[ "$ACTIVE_CAMERAS" == "6" ]]; then
  python scripts/preflight_camera_v2_reid.py
else
  python - <<'PY'
from services.camera_v2.native_bridge import ensure_bridge
path = ensure_bridge()
print(f"CAMERA_V2_DYNAMIC_PREFLIGHT native=PASS path={path}")
PY
  echo "CAMERA_V2_DYNAMIC_PREFLIGHT active_cameras=${ACTIVE_CAMERAS} strict_pair_topology=SKIP"
fi

python scripts/preflight_sentinel_ui.py

exec python -m services.camera_v2.monitor_ui
