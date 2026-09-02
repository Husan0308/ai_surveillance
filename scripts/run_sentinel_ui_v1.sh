#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
BRANCH_EXPECTED="rebuild/service-architecture-v11-monitoring-realtime-v1-20260902"
UI_CAMERAS="${V11_UI_STAGE_CAMERAS:-CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06}"
case "$UI_CAMERAS" in
  CAM-01|CAM-01,CAM-02|CAM-01,CAM-02,CAM-03|CAM-01,CAM-02,CAM-03,CAM-04|CAM-01,CAM-02,CAM-03,CAM-04,CAM-05|CAM-01,CAM-02,CAM-03,CAM-04,CAM-05,CAM-06) ;;
  *) printf 'V11_SENTINEL_UI_PREFLIGHT RESULT=FAIL reason=invalid_staged_camera_list value=%s\n' "$UI_CAMERAS" >&2; exit 1 ;;
esac
LAST_CAMERA="${UI_CAMERAS##*,}"
STAGE_TAG="CAM01_${LAST_CAMERA//-/}"
APP_PY="${V11_UI_PYTHON:-$HOME/ai_surveillance/.venv/bin/python}"
[[ -x "$APP_PY" ]] || APP_PY="$(command -v python3)"
fail(){ printf 'V11_SENTINEL_UI_%s_PREFLIGHT RESULT=FAIL reason=%s\n' "$STAGE_TAG" "$*" >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || fail "DISPLAY_empty"
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || true)"
[[ "$CURRENT_BRANCH" == "$BRANCH_EXPECTED" ]] || fail "wrong_branch=${CURRENT_BRANCH:-detached} expected=$BRANCH_EXPECTED"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SENTINEL_CAMERA_IDS="$UI_CAMERAS"
export SENTINEL_LIVE_PREVIEW_CAMERAS="$UI_CAMERAS"
IFS=',' read -ra ui_camera_rows <<< "$UI_CAMERAS"
preview_paths=()
for camera_id in "${ui_camera_rows[@]}"; do
  key="V11_UI_PREVIEW_PATH_${camera_id//-/}"
  slug="${camera_id,,}"; slug="${slug//-/}"
  default_path="/dev/shm/v11_ui_preview_${slug}_v1.bin"
  if [[ -z "${!key:-}" ]]; then export "$key=$default_path"; fi
  preview_paths+=("${!key}")
done
CAMERA_STATE="$($APP_PY - <<'PYUI'
import os
import PySide6
import services.frontend.sentinel_v1.ui as ui
from services.frontend.sentinel_v1.data import CAMERAS
expected = os.environ["SENTINEL_CAMERA_IDS"].split(",")
ids = [camera.id for camera in CAMERAS]
if ids != expected:
    raise SystemExit(f"unexpected_camera_cards={ids} expected={expected}")
if tuple(expected) != ui.LIVE_PREVIEW_CAMERAS:
    raise SystemExit(f"unexpected_live_previews={ui.LIVE_PREVIEW_CAMERAS}")
print(",".join(ids))
PYUI
)" || fail "PySide6_ui_or_camera_state"
printf 'V11_SENTINEL_UI_%s_PREFLIGHT RESULT=PASS cards=%s count=%s demo_cameras=0 preview_paths=%s\n' "$STAGE_TAG" "$CAMERA_STATE" "${#ui_camera_rows[@]}" "$(IFS=,; echo "${preview_paths[*]}")"
exec "$APP_PY" -m services.frontend.sentinel_v1.main
