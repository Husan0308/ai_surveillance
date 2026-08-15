from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def roots() -> list[Path]:
    values = [Path("/opt/nvidia/deepstream/deepstream")]
    values.extend(sorted(Path("/opt/nvidia/deepstream").glob("deepstream-*"), reverse=True))
    out: list[Path] = []
    seen: set[str] = set()
    for p in values:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def main() -> int:
    print("CAMERA_V2 TRACKING PREFLIGHT")
    detection = subprocess.run([sys.executable, str(ROOT / "scripts/preflight_detection_v2.py")], check=False)
    if detection.returncode != 0:
        print("FAIL: detection preflight failed")
        print("TRACKING_PREFLIGHT=FAIL")
        return 1

    failures: list[str] = []
    code, detail = run(["gst-inspect-1.0", "nvtracker"])
    print(f"plugin nvtracker={'OK' if code == 0 else 'MISSING'}")
    if code != 0:
        failures.append("DeepStream nvtracker plugin is unavailable: " + detail[-800:])

    lib = next(
        (p / "lib/libnvds_nvmultiobjecttracker.so" for p in roots() if (p / "lib/libnvds_nvmultiobjecttracker.so").exists()),
        None,
    )
    cfg = next(
        (
            p / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml"
            for p in roots()
            if (p / "samples/configs/deepstream-app/config_tracker_NvDCF_max_perf.yml").exists()
        ),
        None,
    )
    print(f"nvdcf_lib={lib or 'NOT_FOUND'}")
    print(f"nvdcf_config={cfg or 'NOT_FOUND'}")
    if lib is None:
        failures.append("libnvds_nvmultiobjecttracker.so not found")
    if cfg is None:
        failures.append("config_tracker_NvDCF_max_perf.yml not found")

    if lib is not None:
        code, out = run(["ldd", str(lib)])
        missing = [line.strip() for line in out.splitlines() if "not found" in line]
        if code != 0:
            failures.append("ldd failed for NvDCF library")
        elif missing:
            failures.append("NvDCF dependencies missing: " + " | ".join(missing))
        else:
            print("nvdcf_dependencies=OK")

    if failures:
        for failure in failures:
            print("FAIL: " + failure)
        print("TRACKING_PREFLIGHT=FAIL")
        return 1
    print("TRACKING_PREFLIGHT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
