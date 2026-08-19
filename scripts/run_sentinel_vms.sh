#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.19-r5"

# Fail immediately if the checkout contains a stale camera dialog, fake tile
# occupancy badges, broken fullscreen/hover behavior, native stale-page bleed,
# or a regressed ankle heatmap contract.
python scripts/preflight_sentinel_ui.py

# Resolve/warm bounded side paths only after the UI/source-schema contract passes.
python scripts/setup_camera_v2_reid.py

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
from services.camera_v2.pose_heatmap_bridge import ensure_pose_heatmap_bridge
native = ensure_bridge()
pose = ensure_pose_heatmap_bridge()
print(f"CAMERA_V2_DYNAMIC_PREFLIGHT native=PASS path={native}")
print(f"CAMERA_V2_DYNAMIC_PREFLIGHT pose_heatmap=PASS path={pose} bbox_anchor=OFF")
PY
  echo "CAMERA_V2_DYNAMIC_PREFLIGHT active_cameras=${ACTIVE_CAMERAS} strict_pair_topology=SKIP"
fi

exec python -m services.camera_v2.monitor_ui
