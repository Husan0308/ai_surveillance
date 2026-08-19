from __future__ import annotations

from getpass import getpass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _upsert(lines: list[str], key: str, value: str) -> list[str]:
    prefix = key + "="
    replacement = f"{key}={_quote(value)}"
    found = False
    output: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            if not found:
                output.append(replacement)
                found = True
            continue
        output.append(line)
    if not found:
        output.append(replacement)
    return output


def main() -> int:
    print("RTSP credentials will be saved only to .env (gitignored).")
    username = input("NVR/RTSP username: ").strip()
    if not username:
        print("Cancelled: username is empty.")
        return 1
    password = getpass("NVR/RTSP password: ")
    if not password:
        print("Cancelled: password is empty.")
        return 1

    existing = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    lines = _upsert(existing, "SURVEILLANCE_RTSP_USERNAME", username)
    lines = _upsert(lines, "SURVEILLANCE_RTSP_PASSWORD", password)
    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.chmod(ENV_PATH, 0o600)
    print(f"Saved credentials to {ENV_PATH} with mode 600. Password was not printed.")
    print("Next:")
    print("  python scripts/preflight_sentinel_ui.py")
    print("  python scripts/preflight_camera_v2_core.py")
    print("  bash scripts/run_sentinel_vms.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
