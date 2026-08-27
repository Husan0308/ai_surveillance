#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
PY310="${CAMERA_V90_PYTHON:-$ROOT/.venv-trt86/bin/python}"

if [[ ! -x "$PY310" ]]; then
  echo "V90_PROBE FAIL stage=python310 reason=missing path=$PY310"
  exit 2
fi

"$PY310" - <<'PY'
from __future__ import annotations

import os
import re
import sys


def libs() -> list[str]:
    out: list[str] = []
    try:
        with open('/proc/self/maps', 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if 'libnvinfer' not in line:
                    continue
                path = line.split()[-1]
                if path not in out:
                    out.append(path)
    except OSError:
        pass
    return out


def majors(paths: list[str]) -> list[int]:
    found: set[int] = set()
    for path in paths:
        m = re.search(r'libnvinfer(?:_plugin)?\.so\.(\d+)', path)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)

print(f"V90_PROBE python={sys.version.split()[0]} executable={sys.executable}")
if sys.version_info[:2] != (3, 10):
    print(f"V90_PROBE FAIL stage=python310 reason=wrong_python got={sys.version_info.major}.{sys.version_info.minor}")
    raise SystemExit(10)

try:
    import gi
except Exception as exc:
    print(f"V90_PROBE FAIL stage=gi_py310 type={type(exc).__name__} error={exc}")
    print("V90_PROBE next=install-pygobject-into-.venv-trt86")
    raise SystemExit(20)

try:
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
except Exception as exc:
    print(f"V90_PROBE FAIL stage=gstreamer_py310 type={type(exc).__name__} error={exc}")
    raise SystemExit(21)

print("V90_PROBE gst=OK")
tracker = Gst.ElementFactory.make('nvtracker', 'v90_tracker_probe')
print(f"V90_PROBE nvtracker={'OK' if tracker is not None else 'FAIL'}")
if tracker is None:
    raise SystemExit(22)

before = libs()
print("V90_PROBE nvinfer_before=" + (";".join(before) if before else "none"))

try:
    import tensorrt as trt
except Exception as exc:
    print(f"V90_PROBE FAIL stage=tensorrt_py310 type={type(exc).__name__} error={exc}")
    raise SystemExit(30)

print(f"V90_PROBE tensorrt={trt.__version__}")
if not str(trt.__version__).startswith('8.6.1'):
    print(f"V90_PROBE FAIL stage=tensorrt_version expected=8.6.1 got={trt.__version__}")
    raise SystemExit(31)

after = libs()
print("V90_PROBE nvinfer_after=" + (";".join(after) if after else "none"))
loaded_majors = majors(after)
print("V90_PROBE nvinfer_majors=" + (",".join(map(str, loaded_majors)) if loaded_majors else "none"))
if len(loaded_majors) > 1:
    print("V90_PROBE FAIL stage=nvinfer_collision reason=multiple_major_versions")
    raise SystemExit(40)
if loaded_majors and loaded_majors != [8]:
    print(f"V90_PROBE FAIL stage=nvinfer_runtime reason=unexpected_major majors={loaded_majors}")
    raise SystemExit(41)

print("V90_PROBE PASS same_process_feasible=1 python=3.10 gst=1 nvtracker=1 trt86=1")
PY
