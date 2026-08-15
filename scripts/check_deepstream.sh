#!/usr/bin/env bash
set -euo pipefail

plugins=(nvurisrcbin nvstreammux nvvideoconvert nvmultistreamtiler nveglglessink appsink tee queue)

for plugin in "${plugins[@]}"; do
  if gst-inspect-1.0 "$plugin" >/dev/null 2>&1; then
    echo "[OK] $plugin"
  else
    echo "[MISSING] $plugin"
    exit 1
  fi
done

python3 - <<'PY'
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except ModuleNotFoundError as exc:
    raise SystemExit("[MISSING] Python gi bindings. Install python3-gi and create the ML venv with --system-site-packages.") from exc

Gst.init(None)
print(f"[OK] Python GStreamer bindings: {Gst.version_string()}")
PY
