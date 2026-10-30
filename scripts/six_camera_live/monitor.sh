#!/usr/bin/env bash
# Manual daily operation of the existing DeepStream + Camera Wall launcher.
set -euo pipefail
monitor_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$monitor_root"
monitor_python="$monitor_root/.runtime/full-stack-venv/bin/python"

case "${1:-}" in
  start)
    shift
    usage() {
      echo 'Usage: bash scripts/six_camera_live/monitor.sh start [--verify-fullscreen] [--fullscreen-camera CAM-XX]... [--duration SECONDS]' >&2
      exit 2
    }
    verify_fullscreen=0
    duration_seen=0
    duration_args=()
    fullscreen_camera_args=()
    while (($#)); do
      case "$1" in
        --verify-fullscreen)
          if ((verify_fullscreen)); then usage; fi
          verify_fullscreen=1
          shift
          ;;
        --fullscreen-camera)
          if (($# < 2)); then usage; fi
          case "$2" in
            CAM-01|CAM-02|CAM-03|CAM-04|CAM-05|CAM-06) ;;
            *) usage ;;
          esac
          if [[ " ${fullscreen_camera_args[*]} " == *" $2 "* ]]; then usage; fi
          fullscreen_camera_args+=(--fullscreen-camera "$2")
          verify_fullscreen=1
          shift 2
          ;;
        --duration)
          if ((duration_seen)) || (($# < 2)) || [[ ! "$2" =~ ^[0-9]+$ ]]; then usage; fi
          duration_seen=1
          duration_args+=(--duration "$2")
          shift 2
          ;;
        *)
          usage
          ;;
      esac
    done
    export DISPLAY="${DISPLAY:-:0}"
    monitor_xauthority="/run/user/$(id -u)/gdm/Xauthority"
    if [[ -z "${XAUTHORITY:-}" && -r "$monitor_xauthority" ]]; then
      export XAUTHORITY="$monitor_xauthority"
    fi
    monitor_viewer="$monitor_root/.runtime/deepstream-rtsp-pause-suppression-reporting-20261030-r1/sources/apps/sample_apps/deepstream-app/deepstream-app"
    if ! printf '%s  %s\n' \
      '2602a28a191e3b8671e5e9e4d4fda1fc98d428b70b2d4235bfea8684c269483e' \
      "$monitor_viewer" | sha256sum --check --status; then
      echo 'START BLOCKED: validated DeepStream viewer is missing or changed.' >&2
      exit 2
    fi
    fullscreen_args=()
    if ((verify_fullscreen)); then
      umask 077
      runtime_dir="$monitor_root/.runtime"
      if [[ -L "$runtime_dir" ]]; then
        echo 'START BLOCKED: private runtime directory cannot be a symlink.' >&2
        exit 2
      fi
      mkdir -p -m 700 -- "$runtime_dir"
      if [[ ! -d "$runtime_dir" || ! -O "$runtime_dir" ]]; then
        echo 'START BLOCKED: private runtime directory is unavailable or not user-owned.' >&2
        exit 2
      fi
      sequence_dir="$(mktemp -d -- "$runtime_dir/fullscreen-sequence-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
      export V11_UI_FULLSCREEN_SEQUENCE_OUTPUT="$sequence_dir/fullscreen-sequence.jsonl"
      export V11_UI_FULLSCREEN_CAPTURE_MODE=qt-widget
      fullscreen_args=(--verify-fullscreen)
      fullscreen_args+=("${fullscreen_camera_args[@]}")
    else
      # An ambient variable must not silently replace the normal daily wall.
      unset V11_UI_FULLSCREEN_SEQUENCE_OUTPUT V11_UI_FULLSCREEN_CAPTURE_MODE
    fi
    export SIX_LIVE_SUPPRESS_SHUTDOWN_PAUSE=1
    exec "$monitor_python" -m services.gpu.runtime.live_launcher \
      --run --monitor --authorize-physical-rtsp --network bridge \
      --infer-config "$monitor_root/.runtime/cam-all-live-fontfix-20261026-r1/infer_config.txt" \
      --models-root "$monitor_root/.runtime/six-source-recorded-validation-20261024-r1" \
      --expected-engine-sha256 '3b75d51182b9f4679c328ac339e5d063dde681874f83e388792a933f34e9e3d2' \
      --mosaic-preview --mosaic-viewer-binary "$monitor_viewer" \
      --publish-ml-telemetry \
      "${fullscreen_args[@]}" "${duration_args[@]}"
    ;;
  stop)
    exec "$monitor_python" -m services.gpu.runtime.live_launcher --stop-monitoring
    ;;
  *)
    echo 'Usage: bash scripts/six_camera_live/monitor.sh {start|stop}' >&2
    exit 2
    ;;
esac
