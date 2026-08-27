#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(*cmd: str) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return (p.stdout or "") + (p.stderr or "")


def section(name: str) -> None:
    print(f"\n===== {name} =====")


def main() -> int:
    section("versions")
    print(run("gst-launch-1.0", "--version").strip())
    print(run("deepstream-app", "--version-all").strip())

    for plugin in ("nvvideoconvert", "nvv4l2decoder", "nvurisrcbin", "nveglglessink"):
        section(f"gst-inspect {plugin}")
        text = run("gst-inspect-1.0", plugin)
        keep = []
        capture = False
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("Filename") or s.startswith("Version") or s.startswith("Source module") or s.startswith("Package") or s.startswith("Origin URL"):
                keep.append(line)
            if s == "Element Properties:":
                capture = True
                keep.append(line)
                continue
            if capture:
                keep.append(line)
        print("\n".join(keep) if keep else text[:8000])

    section("plugin files")
    candidates = [
        "/opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/libnvdsgst_video_*",
        "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libnvdsgst_video_*",
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnv*",
        "/usr/lib/x86_64-linux-gnu/gstreamer-1.0/libnvdsgst*",
    ]
    for pattern in candidates:
        print(f"pattern={pattern}")
        print(run("bash", "-lc", f"ls -l {pattern} 2>/dev/null || true").strip())

    section("environment")
    for key in ("GST_PLUGIN_PATH", "GST_PLUGIN_SYSTEM_PATH", "LD_LIBRARY_PATH", "PATH"):
        print(f"{key}={os.environ.get(key, '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
