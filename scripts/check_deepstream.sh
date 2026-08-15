#!/usr/bin/env bash
set -euo pipefail

plugins=(nvurisrcbin nvstreammux nvmultistreamtiler nveglglessink)

for plugin in "${plugins[@]}"; do
  if gst-inspect-1.0 "$plugin" >/dev/null 2>&1; then
    echo "[OK] $plugin"
  else
    echo "[MISSING] $plugin"
    exit 1
  fi
done

if ! python3 - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
print("[OK] Python GStreamer bindings:", Gst.version_string())
PY
then
  cat <<'MSG'
[ERROR] Python cannot import gi/GStreamer bindings.

Ubuntu/Kubuntu fix:
  sudo apt update
  sudo apt install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0 python3-venv

If you use the ML virtualenv, recreate it so it can see system packages:
  deactivate 2>/dev/null || true
  rm -rf .venv-ml
  python3 -m venv --system-site-packages .venv-ml
  source .venv-ml/bin/activate
MSG
  exit 1
fi
