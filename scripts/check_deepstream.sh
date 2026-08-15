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

python3 - <<'PY'
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
Gst.init(None)
print("[OK] Python GStreamer bindings")
PY
