from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from services.ml_service.app.config import load_settings


def run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
        text = (result.stdout or "") + (result.stderr or "")
        return result.returncode, text.strip()
    except Exception as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print("CAMERA_V2 PREFLIGHT")
    print(f"python={sys.version.split()[0]}")
    print(f"display={os.environ.get('DISPLAY', '-')}")
    print(f"session={os.environ.get('XDG_SESSION_TYPE', '-')}")

    try:
        settings = load_settings()
        print(f"cameras={len(settings.cameras)} ids={[c.camera_id for c in settings.cameras]}")
        if len(settings.cameras) != 6:
            failures.append(f"expected 6 cameras, found {len(settings.cameras)}")
        missing_creds = [c.camera_id for c in settings.cameras if not c.username or not c.password]
        if missing_creds:
            warnings.append("RTSP credentials missing for: " + ", ".join(missing_creds))
    except Exception as exc:
        failures.append(f"config: {type(exc).__name__}: {exc}")

    if shutil.which("nvidia-smi"):
        code, out = run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        ])
        if code == 0:
            print("gpu=" + out.replace("\n", " | "))
        else:
            failures.append("nvidia-smi failed: " + out)
    else:
        failures.append("nvidia-smi not found")

    if not shutil.which("gst-inspect-1.0"):
        failures.append("gst-inspect-1.0 not found")
    else:
        required = ["nvurisrcbin", "nvstreammux", "nvmultistreamtiler", "nveglglessink", "queue"]
        for plugin in required:
            code, _ = run(["gst-inspect-1.0", plugin])
            state = "OK" if code == 0 else "MISSING"
            print(f"plugin {plugin}={state}")
            if code != 0:
                failures.append(f"missing plugin: {plugin}")

    rmem_path = Path("/proc/sys/net/core/rmem_max")
    if rmem_path.exists():
        try:
            rmem = int(rmem_path.read_text().strip())
            print(f"net.core.rmem_max={rmem}")
            if rmem < 8 * 1024 * 1024:
                warnings.append(
                    "net.core.rmem_max is below 8 MiB; UDP RTSP may not get the requested receive buffer"
                )
        except Exception as exc:
            warnings.append(f"could not read rmem_max: {exc}")

    if not os.environ.get("DISPLAY") and os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland":
        warnings.append("DISPLAY is unset; nveglglessink may not be able to open a visible window")

    for warning in warnings:
        print("WARNING: " + warning)
    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        print("PREFLIGHT=FAIL")
        return 1

    print("PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
