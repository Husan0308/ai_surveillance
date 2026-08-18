#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

HEAD_SHA="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
echo "SENTINEL_BUILD branch=${BRANCH} head=${HEAD_SHA} expected_ui=2026.08.18-r3"

# Fail immediately if the checkout still contains the old Camera dialog or if
# Known/Unknown/fullscreen/ankle-heatmap wiring regressed. This runs before any
# model warmup or six-camera RTSP startup so stale local code cannot hide behind
# a long initialization sequence.
python scripts/preflight_sentinel_ui.py

# Resolve/warm the bounded CPU side paths (pose + ReID) only after the UI/source
# schema contract is known-good.
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
from services.camera_v2.pose_heatmap_bridge import ensure_pose_heatmap_bridge
native = ensure_bridge()
pose = ensure_pose_heatmap_bridge()
print(f"CAMERA_V2_DYNAMIC_PREFLIGHT native=PASS path={native}")
print(f"CAMERA_V2_DYNAMIC_PREFLIGHT pose_heatmap=PASS path={pose} bbox_anchor=OFF")
PY
  echo "CAMERA_V2_DYNAMIC_PREFLIGHT active_cameras=${ACTIVE_CAMERAS} strict_pair_topology=SKIP"
fi

exec python -m services.camera_v2.monitor_ui
